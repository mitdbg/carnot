"""Orchestrator — walks a Plan's compute chain, dispatches per-op subagent functions.

Subagents keep their `(op: OpNode, prev, ctx)` interface; the orchestrator
translates Branch/ComputeNode → OpNode at dispatch time. OpNode is just a
dispatch envelope, not an AST node.

A Plan is one or more ComputeNodes ending in a final aggregator (see dsl.py).
- Single-compute (legacy flat): run the final compute's branches, then the
  final compute. Recovery on MissingData fires once at the final compute.
- Multi-compute: run each intermediate ComputeNode's branches + intermediate
  compute in parallel (each producing list[AnnotatedValue]). Concatenate
  intermediate outputs and feed them to the final aggregator. Intermediates
  do NOT get recovery; only the final compute does.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from skunk.common import HarnessContext
from skunk.dsl import (
    AnnotatedValue,
    Branch,
    ComputeNode,
    DocHandle,
    FormattedString,
    LookupBranch,
    OpNode,
    Plan,
    RetrieveBranch,
)
from skunk.subagents import compute, extract, lookup_external, retrieve
from skunk.subagents.base import MissingData, StepFailed, SubagentFn

_SUBAGENT_REGISTRY: dict[str, SubagentFn] = {
    "retrieve": retrieve.run,
    "extract": extract.run,
    "lookup_external": lookup_external.run,
    "compute": compute.run,
}


@dataclass
class StepTrace:
    op: str
    args: dict
    input_desc: str
    output_desc: str
    elapsed_s: float
    error: str | None = None
    input_full: str = ""   # full repr (no truncation), for trace dump
    output_full: str = ""  # full repr (no truncation), for trace dump
    step_idx: int = 0


@dataclass
class QuestionTrace:
    question: str
    answer: str | None = None
    failed: bool = False
    failure_reason: str | None = None
    steps: list[StepTrace] = field(default_factory=list)

    def pretty(self) -> str:
        lines = [f"Question: {self.question}"]
        if self.failed:
            lines.append(f"FAILED: {self.failure_reason}")
        else:
            lines.append(f"Answer: {self.answer}")
        lines.append(f"Steps ({len(self.steps)}):")
        for s in self.steps:
            status = f"ERROR: {s.error}" if s.error else "OK"
            lines.append(f"  [{s.op}] {status} ({s.elapsed_s:.2f}s)")
            lines.append(f"    in:  {s.input_desc}")
            lines.append(f"    out: {s.output_desc}")
        return "\n".join(lines)


def execute(plan: Plan, ctx: HarnessContext) -> QuestionTrace:
    """Walk the plan's compute chain. Legacy single-compute = data → final compute.
    Decomposed = parallel intermediates → final aggregator compute."""
    trace = QuestionTrace(question=ctx.question)
    try:
        if not plan.computes:
            raise StepFailed("orchestrator", "plan has no computes")
        if len(plan.computes) == 1:
            final = plan.computes[0]
            prev = _run_data_phase(final.branches, ctx, trace)
        else:
            intermediates = plan.computes[:-1]
            final = plan.computes[-1]
            prev = _run_intermediates_parallel(intermediates, ctx, trace)
        result = _run_compute_with_recovery(prev, plan, ctx, trace, final)
        trace.answer = result.text
    except StepFailed as e:
        trace.failed = True
        trace.failure_reason = str(e)
    except Exception as e:
        trace.failed = True
        trace.failure_reason = f"unexpected error: {type(e).__name__}: {e}"
    return trace


def _run_data_phase(branches: list[Branch], ctx: HarnessContext, trace: QuestionTrace) -> Any:
    if not branches:
        raise StepFailed("orchestrator", "plan has no branches")
    if len(branches) == 1:
        return _run_branch(branches[0], ctx, trace)
    return _run_parallel(branches, ctx, trace)


def _run_branch(branch: Branch, ctx: HarnessContext, trace: QuestionTrace) -> list[AnnotatedValue]:
    """Execute a single branch and return its entries."""
    if isinstance(branch, RetrieveBranch):
        ret_args: dict[str, Any] = {"concept": branch.concept, "period": branch.period}
        if branch.source_bulletin:
            ret_args["source_bulletin"] = branch.source_bulletin
        doc = _run_op(OpNode(op="retrieve", args=ret_args), None, ctx, trace)
        ext_args: dict[str, Any] = {"visual_only": True} if branch.visual_only else {}
        return _run_op(OpNode(op="extract", args=ext_args), doc, ctx, trace)
    if isinstance(branch, LookupBranch):
        return _run_op(OpNode(op="lookup_external", args={"nl": branch.nl}), None, ctx, trace)
    raise TypeError(f"Unknown branch type: {type(branch).__name__}")


def _run_parallel(branches: list[Branch], ctx: HarnessContext, trace: QuestionTrace) -> list[AnnotatedValue]:
    """Best-effort: drop failed branches, flatten survivors into one entry list.

    Only re-raises if every branch failed; then the first error propagates.
    Downstream compute() decides whether the survivors are sufficient (and may
    raise MissingData to trigger the orchestrator's recovery path).
    """
    results: list[Any] = [None] * len(branches)
    errors: list[tuple[int, StepFailed]] = []
    with ThreadPoolExecutor(max_workers=ctx.config.max_parallel_workers) as pool:
        futures = {pool.submit(_run_branch, b, ctx, trace): i for i, b in enumerate(branches)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except StepFailed as e:
                errors.append((i, e))
    for i, e in errors:
        ctx.emit("orchestrator", "parallel branch failed", branch_idx=i, error=str(e))
    if len(errors) == len(branches):
        raise errors[0][1]
    return [entry for branch in results if branch is not None for entry in branch]


def _run_intermediates_parallel(
    nodes: list[ComputeNode], ctx: HarnessContext, trace: QuestionTrace,
) -> list[AnnotatedValue]:
    """For each non-final ComputeNode: run its data phase, then its intermediate
    compute. Returns the concatenated list of AnnotatedValues across survivors.
    Failed intermediates are skipped (best-effort), mirroring _run_parallel."""
    results: list[list[AnnotatedValue] | None] = [None] * len(nodes)
    errors: list[tuple[int, Exception]] = []
    with ThreadPoolExecutor(max_workers=ctx.config.max_parallel_workers) as pool:
        futures = {
            pool.submit(_run_intermediate_node, node, ctx, trace): i
            for i, node in enumerate(nodes)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except (StepFailed, MissingData) as e:
                errors.append((i, e))
    for i, e in errors:
        ctx.emit("orchestrator", "intermediate compute failed",
                 node_idx=i, task=nodes[i].task, error=str(e))
    if len(errors) == len(nodes):
        # All intermediates failed — surface the first.
        first = errors[0][1]
        if isinstance(first, MissingData):
            raise StepFailed("compute", f"all intermediates missing data: {first.reason}")
        raise first
    return [entry for r in results if r is not None for entry in r]


def _run_intermediate_node(
    node: ComputeNode, ctx: HarnessContext, trace: QuestionTrace,
) -> list[AnnotatedValue]:
    """Run one intermediate ComputeNode's branches then its compute. No recovery."""
    prev = _run_data_phase(node.branches, ctx, trace)
    return _run_op(
        OpNode(op="compute", args={"task": node.task, "final": False}),
        prev, ctx, trace,
    )


def _summarize_prev(prev: Any) -> list[str]:
    """Descriptions-only summary of an AnnotatedValue list for the recovery prompt.
    Drops raw values/vectors/tables to keep the prompt bounded."""
    if not isinstance(prev, list):
        return []
    out: list[str] = []
    for av in prev:
        if isinstance(av, AnnotatedValue):
            out.append(f"{av.kind} | {av.unit} | {av.description}")
    return out


def _run_compute_with_recovery(
    prev: Any, plan: Plan, ctx: HarnessContext, trace: QuestionTrace,
    final_node: ComputeNode,
) -> FormattedString:
    """Run the final compute with a bounded recovery loop. On MissingData, ask
    the recovery planner for one or more supplemental branches (given the
    failure reason and a descriptions-only summary of current prev), run them
    in parallel, append to prev, and retry compute. Bounded by
    ctx.config.recovery_max_rounds (total compute calls ≤ max_rounds + 1)."""
    compute_args: dict[str, Any] = {"final": True}
    if final_node.task:
        compute_args["task"] = final_node.task
    compute_op = OpNode(op="compute", args=compute_args)

    from skunk.planner import plan_recovery

    max_rounds = ctx.config.recovery_max_rounds
    current_prev: list[Any] = list(prev) if isinstance(prev, list) else prev
    for round_idx in range(max_rounds + 1):
        try:
            return _run_op(compute_op, current_prev, ctx, trace)
        except MissingData as e:
            if round_idx == max_rounds:
                raise StepFailed(
                    "compute",
                    f"still missing after {max_rounds} recovery round(s): {e.reason}",
                ) from e
            ctx.emit("orchestrator", "compute reported MISSING; attempting recovery",
                     round=round_idx + 1, reason=e.reason)
            prev_summary = _summarize_prev(current_prev)
            try:
                extra_branches = plan_recovery(
                    ctx.question, plan, e.reason, prev_summary, ctx,
                )
            except Exception as planner_err:
                raise StepFailed(
                    "compute",
                    f"missing data and recovery planner failed: {e.reason}; planner: {planner_err}",
                ) from planner_err
            if not extra_branches:
                raise StepFailed(
                    "compute", f"missing data and recovery declined: {e.reason}",
                ) from e
            ctx.emit("orchestrator", "running recovery branches",
                     round=round_idx + 1, count=len(extra_branches),
                     branches=[repr(b) for b in extra_branches])
            if len(extra_branches) == 1:
                extra_value = _run_branch(extra_branches[0], ctx, trace)
            else:
                extra_value = _run_parallel(extra_branches, ctx, trace)
            current_prev = list(current_prev) + extra_value


def _run_op(op: OpNode, prev: Any, ctx: HarnessContext, trace: QuestionTrace) -> Any:
    fn = _SUBAGENT_REGISTRY.get(op.op)
    if fn is None:
        raise StepFailed(op.op, f"No subagent registered for '{op.op}'")

    step_idx = len(trace.steps) + 1
    input_desc = _describe_value(prev)
    input_full = _full_repr(prev)
    ctx.emit("_step", "begin", step_idx=step_idx, op=op.op, args=op.args, input=input_desc)

    t0 = time.perf_counter()
    try:
        result = fn(op, prev, ctx)
    except (StepFailed, MissingData) as e:
        elapsed = time.perf_counter() - t0
        err_str = f"MissingData: {e.reason}" if isinstance(e, MissingData) else str(e)
        ctx.emit("_step", "error", step_idx=step_idx, error=err_str, elapsed_s=round(elapsed, 2))
        trace.steps.append(StepTrace(op=op.op, args=op.args, input_desc=input_desc,
                                     output_desc="(failed)", elapsed_s=elapsed, error=err_str,
                                     input_full=input_full, output_full="(failed)", step_idx=step_idx))
        raise
    elapsed = time.perf_counter() - t0
    output_desc = _describe_value(result)
    output_full = _full_repr(result)
    ctx.emit("_step", "done", step_idx=step_idx, output=output_desc, elapsed_s=round(elapsed, 2))
    trace.steps.append(StepTrace(op=op.op, args=op.args, input_desc=input_desc,
                                 output_desc=output_desc, elapsed_s=elapsed,
                                 input_full=input_full, output_full=output_full, step_idx=step_idx))
    return result


def _describe_value(v: Any) -> str:
    if v is None:
        return "(none)"
    if isinstance(v, DocHandle):
        return f"DocHandle({len(v.refs)} refs): {v.desc}"
    if isinstance(v, list) and v and isinstance(v[0], AnnotatedValue):
        descriptions = [e.description for e in v]
        return f"[{len(descriptions)} entries: {descriptions}]"
    if isinstance(v, FormattedString):
        return f"FormattedString: {v.text!r}"
    if isinstance(v, list):
        return f"list({len(v)} branches)"
    s = repr(v)
    return s[:100] + ("..." if len(s) > 100 else "")


def _full_repr(v: Any) -> str:
    """Untruncated repr for trace dumps."""
    if v is None:
        return "(none)"
    if isinstance(v, DocHandle):
        refs = "\n    ".join(repr(r) for r in v.refs)
        return f"DocHandle(desc={v.desc!r}, {len(v.refs)} refs):\n    {refs}" if v.refs else f"DocHandle(empty, desc={v.desc!r})"
    if isinstance(v, list) and v and isinstance(v[0], AnnotatedValue):
        return repr(v)
    if isinstance(v, FormattedString):
        return f"FormattedString(text={v.text!r})"
    if isinstance(v, list):
        parts = [f"  [{i}] {_full_repr(x)}" for i, x in enumerate(v)]
        return "list:\n" + "\n".join(parts)
    return repr(v)
