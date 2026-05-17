"""Orchestrator — walks a Plan, dispatches per-op operator functions.

Operators keep their `(op: OpNode, prev, ctx)` interface; the orchestrator
translates Branch → OpNode at dispatch time. OpNode is just a dispatch
envelope, not an AST node.

A Plan has a flat list of branches feeding a single compute step. The
orchestrator:
  1. Runs the data phase (each Branch in parallel via _run_branch).
  2. Feeds the merged AnnotatedValues to compute, which returns the
     final answer string.

Recovery
--------
When compute reports MissingData, `execute()` builds a one-shot recovery
lesson summarizing what was tried + what's missing, injects it into a
copy of `ctx.prompt_overrides` (targeting the planner only), calls
`planner.plan()` to produce a fresh Plan, and re-executes from scratch.
Bounded by `config.recovery_max_rounds`.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from skunk.common import HarnessContext
from skunk.plan import (
    AnnotatedValue,
    Branch,
    LookupBranch,
    PageRef,
    Plan,
    RetrieveBranch,
)
from skunk.prompt_overrides import PromptOverride
from skunk import compute, extract, lookup_external, retrieve
from skunk.operator import MissingData, OpNode, OperatorFn, StepFailed

_OPERATOR_REGISTRY: dict[str, OperatorFn] = {
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
    """Walk the plan: data phase → compute. On MissingData from compute,
    inject a recovery lesson into ctx and re-plan; re-execute up to
    `ctx.config.recovery_max_rounds` times before failing."""
    trace = QuestionTrace(question=ctx.question)
    max_rounds = ctx.config.recovery_max_rounds
    current_plan = plan

    try:
        prev: Any = None
        for round_idx in range(max_rounds + 1):
            # Subagents read top-level constraints (units_out, etc.) off ctx.plan;
            # refresh per round so they see the current plan.
            ctx.plan = current_plan
            if not current_plan.branches:
                raise StepFailed("orchestrator", "plan has no branches")
            try:
                prev = _run_data_phase(current_plan.branches, ctx, trace)
                result = _run_compute(prev, ctx, trace, current_plan)
                trace.answer = result
                return trace
            except MissingData as e:
                if round_idx == max_rounds:
                    raise StepFailed(
                        "compute",
                        f"still missing after {max_rounds} recovery round(s): {e.reason}",
                    ) from e
                prev_summary = _summarize_prev(prev)
                ctx.emit("orchestrator", "compute reported MISSING; replanning",
                         round=round_idx + 1, reason=e.reason)
                lesson = _build_recovery_lesson(current_plan, e.reason, prev_summary)
                recovery_ctx = ctx.with_extra_override(PromptOverride(
                    section="lessons", targets=("planner",), content=lesson,
                ))
                from skunk.plan import PlannerExecutor
                try:
                    current_plan = PlannerExecutor().plan(ctx.question, recovery_ctx)
                except Exception as planner_err:
                    raise StepFailed(
                        "compute",
                        f"missing data and replanning failed: {e.reason}; planner: {planner_err}",
                    ) from planner_err
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
        ret_args: dict[str, Any] = {"key": branch.key, "period": branch.period}
        doc = _run_op(OpNode(op="retrieve", args=ret_args), None, ctx, trace)
        ext_args: dict[str, Any] = {"key": branch.key, "period": branch.period}
        if branch.visual_only:
            ext_args["visual_only"] = True
        if branch.value_kind:
            ext_args["value_kind"] = branch.value_kind
        return _run_op(OpNode(op="extract", args=ext_args), doc, ctx, trace)
    if isinstance(branch, LookupBranch):
        lk_args: dict[str, Any] = {"target": branch.target}
        if branch.src:
            lk_args["src"] = branch.src
        return _run_op(OpNode(op="lookup_external", args=lk_args), None, ctx, trace)
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


def _run_compute(prev: Any, ctx: HarnessContext, trace: QuestionTrace, plan: Plan) -> str:
    """Run compute and return the final answer string. Lets MissingData
    propagate up to execute() for recovery handling. `plan.task` is the
    planner's `computation.task` (free-form NL; may be empty for direct
    lookups). `plan.qualifiers` is `computation.qualifiers` — short
    directives the compute operator must apply."""
    compute_args: dict[str, Any] = {}
    if plan.task:
        compute_args["task"] = plan.task
    if plan.qualifiers:
        compute_args["qualifiers"] = list(plan.qualifiers)
    return _run_op(OpNode(op="compute", args=compute_args), prev, ctx, trace)


def _branch_summary(b: Branch) -> str:
    """One-line description of a branch for the recovery lesson."""
    if isinstance(b, LookupBranch):
        s = f"lookup_external target={b.target!r}"
        if b.src:
            s += f" src={b.src!r}"
        return s
    if isinstance(b, RetrieveBranch):
        bits = [f"retrieve key={b.key!r} period={b.period!r}"]
        if b.visual_only:
            bits.append("visual_only=True")
        return " ".join(bits)
    return f"<unknown branch type: {type(b).__name__}>"


def _build_recovery_lesson(
    prev_plan: Plan, missing_reason: str, prev_summary: list[str],
) -> str:
    """One-shot lesson appended to the planner's prompt when re-planning after
    a MissingData failure. Targets `planner` only; built and injected by the
    orchestrator at recovery time."""
    branches_block = "\n".join(f"  - {_branch_summary(b)}" for b in prev_plan.branches) or "  (none)"
    prev_block = "\n".join(f"  - {line}" for line in prev_summary) or "  (empty)"
    return (
        "RECOVERY ROUND. A previous plan was generated for this question and "
        "executed, but compute reported MISSING data. Use this round to produce "
        "a NEW plan that addresses the gap. Consider different keys, periods, "
        "or visual_only flags than what was tried.\n"
        f"Previous plan branches:\n{branches_block}\n"
        f"What was retrieved (descriptions only):\n{prev_block}\n"
        f"Compute reported MISSING: {missing_reason}"
    )


def _run_op(op: OpNode, prev: Any, ctx: HarnessContext, trace: QuestionTrace) -> Any:
    fn = _OPERATOR_REGISTRY.get(op.op)
    if fn is None:
        raise StepFailed(op.op, f"No operator registered for '{op.op}'")

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
    if isinstance(v, list) and v and isinstance(v[0], PageRef):
        return f"[{len(v)} page refs]"
    if isinstance(v, list) and v and isinstance(v[0], AnnotatedValue):
        descriptions = [e.description for e in v]
        return f"[{len(descriptions)} entries: {descriptions}]"
    if isinstance(v, str):
        return f"str: {v!r}"
    if isinstance(v, list):
        return f"list({len(v)} branches)"
    s = repr(v)
    return s[:100] + ("..." if len(s) > 100 else "")


def _full_repr(v: Any) -> str:
    """Untruncated repr for trace dumps."""
    if v is None:
        return "(none)"
    if isinstance(v, list) and v and isinstance(v[0], PageRef):
        refs = "\n    ".join(repr(r) for r in v)
        return f"page refs ({len(v)}):\n    {refs}"
    if isinstance(v, list) and v and isinstance(v[0], AnnotatedValue):
        return repr(v)
    if isinstance(v, str):
        return f"str: {v!r}"
    if isinstance(v, list):
        parts = [f"  [{i}] {_full_repr(x)}" for i, x in enumerate(v)]
        return "list:\n" + "\n".join(parts)
    return repr(v)
