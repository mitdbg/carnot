"""Orchestrator — walks the DSL AST, dispatches per-op subagent functions."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from skunk.common.context import HarnessContext
from skunk.dsl import ChainNode, DocHandle, FormattedString, OpNode, ParallelNode, TypedValue
from skunk.subagents import SUBAGENT_REGISTRY
from skunk.subagents.base import StepFailed

_MAX_WORKERS = 4


@dataclass
class StepTrace:
    op: str
    args: dict
    input_desc: str
    output_desc: str
    elapsed_s: float
    error: str | None = None


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


def execute(chain: ChainNode, ctx: HarnessContext) -> QuestionTrace:
    trace = QuestionTrace(question=ctx.question)
    try:
        result = _execute_chain(chain, prev=None, ctx=ctx, trace=trace)
        if isinstance(result, FormattedString):
            trace.answer = result.text
        elif isinstance(result, TypedValue):
            trace.answer = str(result.value)
        elif isinstance(result, DocHandle):
            trace.answer = result.desc
        else:
            trace.answer = str(result)
    except StepFailed as e:
        trace.failed = True
        trace.failure_reason = str(e)
    return trace


def _execute_chain(chain: ChainNode, prev: Any, ctx: HarnessContext, trace: QuestionTrace) -> Any:
    current = prev
    for step in chain.steps:
        if isinstance(step, OpNode):
            current = _run_op(step, current, ctx, trace)
        elif isinstance(step, ParallelNode):
            current = _run_parallel(step, current, ctx, trace)
        else:
            raise ValueError(f"Unknown step type: {type(step).__name__}")
    return current


def _run_op(op: OpNode, prev: Any, ctx: HarnessContext, trace: QuestionTrace) -> Any:
    fn = SUBAGENT_REGISTRY.get(op.op)
    if fn is None:
        raise StepFailed(op.op, f"No subagent registered for '{op.op}'")

    input_desc = _describe_value(prev)
    t0 = time.perf_counter()
    try:
        result = fn(op, prev, ctx)
    except StepFailed as e:
        elapsed = time.perf_counter() - t0
        trace.steps.append(StepTrace(op=op.op, args=op.args, input_desc=input_desc,
                                     output_desc="(failed)", elapsed_s=elapsed, error=str(e)))
        raise
    elapsed = time.perf_counter() - t0
    trace.steps.append(StepTrace(op=op.op, args=op.args, input_desc=input_desc,
                                 output_desc=_describe_value(result), elapsed_s=elapsed))
    return result


def _run_parallel(node: ParallelNode, prev: Any, ctx: HarnessContext, trace: QuestionTrace) -> list[Any]:
    results: list[Any] = [None] * len(node.branches)
    errors: list[StepFailed] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_execute_chain, branch, prev, ctx, trace): i
                   for i, branch in enumerate(node.branches)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except StepFailed as e:
                errors.append(e)
    if errors:
        raise errors[0]
    return results


def _describe_value(v: Any) -> str:
    if v is None:
        return "(none)"
    if isinstance(v, DocHandle):
        return f"DocHandle({len(v.refs)} refs): {v.desc}"
    if isinstance(v, TypedValue):
        raw = repr(v.value)
        truncated = raw[:100] + ("..." if len(raw) > 100 else "")
        return f"TypedValue({v.dtype}): {truncated} — {v.desc}"
    if isinstance(v, FormattedString):
        return f"FormattedString: {v.text!r}"
    if isinstance(v, list):
        return f"list({len(v)} branches)"
    s = repr(v)
    return s[:100] + ("..." if len(s) > 100 else "")
