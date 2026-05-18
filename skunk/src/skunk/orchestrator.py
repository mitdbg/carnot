"""Orchestrator — per-question execution driver.

`Orchestrator(ctx)` is the per-NL-question lifecycle object. It owns the
harness context, one instance of every operator class
(`PlannerPromptedCall`, `RetrieveExecutor`, `ExtractExecutor`,
`LookupExternalPromptedCall`, `ComputeExecutor`), the `Plan` being executed,
and the `QuestionTrace` (stats). One instance per question.

A Plan has a flat list of branches feeding a single compute step. The
orchestrator:
  1. Calls the planner to produce the Plan.
  2. Runs each branch via `_run_branch` (in parallel when there is more than one).
  3. Feeds the merged AnnotatedValues to compute, which returns the
     final answer string.

`MissingData` propagates out of `.execute()` to the caller; no replan is
attempted.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from skunk.compute import ComputeExecutor
from skunk.errors import MissingData, StepFailed
from skunk.extract import ExtractExecutor
from skunk.lookup_external import LookupExternalPromptedCall
from skunk.models import AnnotatedValue, HarnessContext
from skunk.plan import Branch, Plan, PlannerPromptedCall
from skunk.retrieve import RetrieveExecutor
from skunk.trace import QuestionTrace, StepTrace, describe_value, full_repr


class Orchestrator:
    """One Orchestrator per NL question. Owns the ctx, an instance of each
    operator class, the Plan being executed, and the stats/trace.

    Lifecycle:
        orch = Orchestrator(ctx)              # planner runs inside execute()
        answer = orch.execute()               # returns the answer string ("" on failure)
        orch.current_plan                     # the Plan produced by the planner (None before execute)
        orch.trace                            # QuestionTrace populated during execute()
    """

    def __init__(self, ctx: HarnessContext):
        self._ctx = ctx
        self._current_plan: Plan | None = None
        self._planner = PlannerPromptedCall()
        self._retriever = RetrieveExecutor()
        self._extractor = ExtractExecutor()
        self._looker = LookupExternalPromptedCall()
        self._computer = ComputeExecutor()
        self._trace = QuestionTrace(question=ctx.question)

    @property
    def current_plan(self) -> Plan | None:
        return self._current_plan

    @property
    def trace(self) -> QuestionTrace:
        return self._trace

    def execute(self) -> str:
        """Resolve the plan, walk the data phase, run compute. Returns the
        final answer string (empty string on failure — read `self.trace` for
        the full record including `failed` / `failure_reason`). Catches
        `StepFailed` and records it on the trace; `MissingData` and any
        other exception propagate to the caller."""
        try:
            self._current_plan = self._run_op(
                "planner",
                {},
                self._ctx.question,
                lambda prev, ctx: self._planner.plan(prev, ctx),
            )
            plan = self._current_plan
            if not plan.branches:
                raise StepFailed("orchestrator", "plan has no branches")
            if len(plan.branches) == 1:
                prev = self._run_branch(plan.branches[0])
            else:
                prev = self._run_parallel(plan.branches)

            self._trace.answer = self._run_op(
                "compute",
                {"plan": plan},
                prev,
                self._computer.run,
            )
        except StepFailed as e:
            self._trace.failed = True
            self._trace.failure_reason = str(e)
        return self._trace.answer or ""

    def _run_branch(self, branch: Branch) -> list[AnnotatedValue]:
        """Execute a single branch and return its entries."""
        match branch.kind:
            case "retrieve":
                ret_args: dict[str, Any] = {"key": branch.key, "period": branch.period}
                doc = self._run_op("retrieve", ret_args, None, self._retriever.run)
                return self._run_op(
                    "extract", {"branch": branch}, doc, self._extractor.run
                )
            case "lookup_external":
                return self._run_op(
                    "lookup_external", {"branch": branch}, None, self._looker.run
                )

    def _run_parallel(self, branches: list[Branch]) -> list[AnnotatedValue]:
        """Best-effort: drop failed branches, flatten survivors into one entry list.

        Only re-raises if every branch failed; then the first error propagates.
        Downstream compute() decides whether the survivors are sufficient (and may
        raise MissingData, which propagates to the caller).
        """
        results: list[Any] = [None] * len(branches)
        errors: list[tuple[int, StepFailed]] = []
        with ThreadPoolExecutor(
            max_workers=self._ctx.config.max_parallel_workers
        ) as pool:
            futures = {
                pool.submit(self._run_branch, b): i for i, b in enumerate(branches)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except StepFailed as e:
                    errors.append((i, e))
        for i, e in errors:
            self._ctx.emit(
                "orchestrator", "parallel branch failed", branch_idx=i, error=str(e)
            )
        if len(errors) == len(branches):
            raise errors[0][1]
        return [entry for branch in results if branch is not None for entry in branch]

    def _run_op(
        self,
        op_name: str,
        args: dict[str, Any],
        prev: Any,
        fn: Callable[..., Any],
    ) -> Any:
        """Invoke `fn(prev, ctx, **args)`, recording timing and a `StepTrace`
        entry. Re-raises `StepFailed` / `MissingData` after recording."""
        step_idx = len(self._trace.steps) + 1
        input_desc = describe_value(prev)
        input_full = full_repr(prev)
        self._ctx.emit(
            "_step", "begin", step_idx=step_idx, op=op_name, args=args, input=input_desc
        )

        t0 = time.perf_counter()
        try:
            result = fn(prev, self._ctx, **args)
        except (StepFailed, MissingData) as e:
            elapsed = time.perf_counter() - t0
            err_str = (
                f"MissingData: {e.reason}" if isinstance(e, MissingData) else str(e)
            )
            self._ctx.emit(
                "_step",
                "error",
                step_idx=step_idx,
                error=err_str,
                elapsed_s=round(elapsed, 2),
            )
            self._trace.steps.append(
                StepTrace(
                    op=op_name,
                    args=args,
                    input_desc=input_desc,
                    output_desc="(failed)",
                    elapsed_s=elapsed,
                    error=err_str,
                    input_full=input_full,
                    output_full="(failed)",
                    step_idx=step_idx,
                )
            )
            raise
        elapsed = time.perf_counter() - t0
        output_desc = describe_value(result)
        output_full = full_repr(result)
        self._ctx.emit(
            "_step",
            "done",
            step_idx=step_idx,
            output=output_desc,
            elapsed_s=round(elapsed, 2),
        )
        self._trace.steps.append(
            StepTrace(
                op=op_name,
                args=args,
                input_desc=input_desc,
                output_desc=output_desc,
                elapsed_s=elapsed,
                input_full=input_full,
                output_full=output_full,
                step_idx=step_idx,
            )
        )
        return result
