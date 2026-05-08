"""Orchestrator — walks the AST, dispatches per-op subagents.

Features:
- Branch-level parallelism (concurrent.futures)
- Speculative pre-warm: next subagent dispatched eagerly with input description
- StepFailed bubbles up; question is marked failed with a full trace
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from skunk.dsl import (
    ChainNode, DocHandle, FormattedString, OpNode, ParallelNode, TypedValue
)
from skunk.subagents import SUBAGENT_REGISTRY
from skunk.subagents.base import HarnessContext, StepFailed, Subagent

# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(self, max_workers: int = 4):
        self._registry = {
            op: cls() for op, cls in SUBAGENT_REGISTRY.items()
        }
        self._max_workers = max_workers

    def execute(self, chain: ChainNode, ctx: HarnessContext) -> QuestionTrace:
        trace = QuestionTrace(question=ctx.question)
        try:
            result = self._execute_chain(chain, prev=None, ctx=ctx, trace=trace)
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

    # ------------------------------------------------------------------

    def _execute_chain(
        self,
        chain: ChainNode,
        prev: Any,
        ctx: HarnessContext,
        trace: QuestionTrace,
    ) -> Any:
        current = prev
        steps = chain.steps
        for i, step in enumerate(steps):
            # Speculative pre-warm for next step
            next_step = steps[i + 1] if i + 1 < len(steps) else None
            if next_step is not None and isinstance(next_step, OpNode):
                input_desc = _describe_value(current)
                self._prewarm_async(next_step, input_desc)

            if isinstance(step, OpNode):
                current = self._run_op(step, current, ctx, trace)
            elif isinstance(step, ParallelNode):
                current = self._run_parallel(step, current, ctx, trace)
            else:
                raise ValueError(f"Unknown step type: {type(step).__name__}")

        return current

    def _run_op(
        self,
        op: OpNode,
        prev: Any,
        ctx: HarnessContext,
        trace: QuestionTrace,
    ) -> Any:
        subagent = self._registry.get(op.op)
        if subagent is None:
            raise StepFailed(op.op, f"No subagent registered for op '{op.op}'")

        input_desc = _describe_value(prev)
        t0 = time.perf_counter()
        error: str | None = None
        result: Any = None

        try:
            result = subagent.run(op, prev, ctx)
        except StepFailed as e:
            error = str(e)
            elapsed = time.perf_counter() - t0
            trace.steps.append(StepTrace(
                op=op.op, args=op.args,
                input_desc=input_desc,
                output_desc="(failed)",
                elapsed_s=elapsed,
                error=error,
            ))
            raise

        elapsed = time.perf_counter() - t0
        trace.steps.append(StepTrace(
            op=op.op, args=op.args,
            input_desc=input_desc,
            output_desc=_describe_value(result),
            elapsed_s=elapsed,
        ))
        return result

    def _run_parallel(
        self,
        node: ParallelNode,
        prev: Any,
        ctx: HarnessContext,
        trace: QuestionTrace,
    ) -> list[Any]:
        """Execute parallel branches concurrently; return list of results."""
        branch_results: list[Any] = [None] * len(node.branches)
        errors: list[StepFailed] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_to_idx = {
                pool.submit(self._execute_chain, branch, prev, ctx, trace): idx
                for idx, branch in enumerate(node.branches)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    branch_results[idx] = future.result()
                except StepFailed as e:
                    errors.append(e)

        if errors:
            raise errors[0]

        return branch_results

    def _prewarm_async(self, op: OpNode, input_desc: str) -> None:
        """Fire-and-forget prewarm in background thread."""
        subagent = self._registry.get(op.op)
        if subagent is None:
            return
        import threading
        t = threading.Thread(
            target=_safe_prewarm,
            args=(subagent, op, input_desc),
            daemon=True,
        )
        t.start()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _describe_value(v: Any) -> str:
    if v is None:
        return "(none)"
    if isinstance(v, DocHandle):
        return f"DocHandle({len(v.refs)} refs): {v.desc}"
    if isinstance(v, TypedValue):
        val_repr = repr(v.value)[:100]
        return f"TypedValue({v.dtype}): {val_repr} — {v.desc}"
    if isinstance(v, FormattedString):
        return f"FormattedString: {v.text!r}"
    if isinstance(v, list):
        return f"list({len(v)} branches)"
    return repr(v)[:100]


def _safe_prewarm(subagent: Subagent, op: OpNode, input_desc: str) -> None:
    try:
        subagent.prewarm(op, input_desc)
    except Exception:
        pass
