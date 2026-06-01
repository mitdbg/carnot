"""Orchestrator — per-question execution driver. One instance per NL question.

A Plan is a flat list of branches feeding a single compute step: the planner
produces the Plan, each branch runs via `_run_branch` (parallel when >1), and the
merged AnnotatedValues feed compute, which returns the answer string.

If compute raises `MissingData`, a bounded replan loop (`config.recovery_max_rounds`)
re-invokes the planner with the prior plan + `prev` + the missing signal; only the
newly-added branches execute. After the budget is spent, `MissingData` propagates."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from skunk.compute import ComputeExecutor
from skunk.errors import MissingData, StepFailed
from skunk.extract import ExtractExecutor
from skunk.lookup_external import LookupExternal
from skunk.common import AnnotatedValue, HarnessContext
from skunk.plan import Branch, Plan, Planner
from skunk.question_explainer import ConceptExplanation, QuestionExplainer
from skunk.retrieve import RetrieveDispatcher
from skunk.trace import QuestionTrace, StepTrace, describe_value, full_repr

@dataclass
class BranchOutcome:
    """Per-branch result. `entries` is None on failure, `error` None on success.
    Threaded to the replanner so it can rephrase failed branches."""
    branch: Branch
    entries: list[AnnotatedValue] | None
    error: str | None


def _entries(outcomes: list[BranchOutcome]) -> list[AnnotatedValue]:
    """Flatten the successful entries from a list of outcomes."""
    return [e for o in outcomes if o.entries for e in o.entries]


class Orchestrator:
    """One per NL question. Owns the ctx, an instance of each operator class, the
    Plan, and the trace. `execute()` runs the planner + data phase + compute and
    returns the answer string (`""` on failure; see `.trace` for the full record)."""

    def __init__(self, ctx: HarnessContext):
        self._ctx = ctx
        self._current_plan: Plan | None = None
        self._planner = Planner()
        self._retriever = RetrieveDispatcher(ctx.config)
        self._extractor = ExtractExecutor()
        self._looker = LookupExternal()
        self._explainer = QuestionExplainer()
        self._computer = ComputeExecutor()
        self._trace = QuestionTrace(question=ctx.question)
        # Per-branch outcomes across the initial plan + every replan round.
        self._outcomes: list[BranchOutcome] = []

    @property
    def current_plan(self) -> Plan | None:
        return self._current_plan

    @property
    def trace(self) -> QuestionTrace:
        return self._trace

    def execute(self) -> str:
        """Resolve the plan, walk the data phase, run compute. `StepFailed` is
        caught and recorded on the trace; other exceptions propagate."""
        try:
            self._current_plan = self.execute_with_tracing(
                "planner",
                lambda: self._planner.plan(self._ctx.question, self._ctx),
            )
            plan = self._current_plan
            if not plan.branches:
                raise StepFailed("orchestrator", "plan has no branches")
            concept_explanations = self._explain_concepts()
            self._outcomes = self._run_branches(plan.branches)
            prev = _entries(self._outcomes)

            max_rounds = self._ctx.config.recovery_max_rounds
            for attempt in range(max_rounds + 1):
                try:
                    self._trace.answer = self.execute_with_tracing(
                        "compute",
                        lambda: self._computer.run(
                            prev, self._ctx, plan=plan,
                            concept_explanations=concept_explanations,
                        ),
                    )
                    break
                except MissingData as e:
                    if attempt == max_rounds:
                        raise
                    plan, prev = self._replan_round(plan, prev, e)
        except StepFailed as e:
            self._trace.failed = True
            self._trace.failure_reason = str(e)
        return self._trace.answer or ""

    def _replan_round(
        self, plan: Plan, prev: list[AnnotatedValue], missing: MissingData
    ) -> tuple[Plan, list[AnnotatedValue]]:
        """One replan round: call planner.replan, diff branches, execute the
        additions, and adopt new computation/presentation. Re-raises the
        `MissingData` if the replanner produced no useful diff."""
        failed_branches = [(o.branch, o.error) for o in self._outcomes if o.error]
        new_plan = self.execute_with_tracing(
            "replanner",
            lambda: self._planner.replan(
                self._ctx, plan, prev, failed_branches, missing.reason, missing.missing
            ),
        )
        new_branches = [b for b in new_plan.branches if b not in plan.branches]
        framing_changed = (
            new_plan.computation != plan.computation
            or new_plan.presentation != plan.presentation
        )
        if not new_branches and not framing_changed:
            self._ctx.emit("orchestrator", "replanner declined; no diff to execute")
            raise missing

        plan = new_plan.model_copy(
            update={"branches": [*plan.branches, *new_branches]}
        )
        self._current_plan = plan
        if new_branches:
            try:
                new_outcomes = self._run_branches(new_branches)
                self._outcomes = [*self._outcomes, *new_outcomes]
                prev = prev + _entries(new_outcomes)
            except StepFailed as e:
                # All replan-added branches failed. Keep the original `prev`; if
                # framing also didn't change, this round added nothing — bail.
                self._ctx.emit(
                    "orchestrator",
                    "all replan branches failed; keeping prior prev",
                    error=str(e),
                )
                if not framing_changed:
                    raise missing from None
        return plan, prev

    def _explain_concepts(self) -> list[ConceptExplanation]:
        """Run the question-explainer (traced). Returns [] when there are no
        non-obvious concepts or the reply is malformed — best-effort, never fatal."""
        return self.execute_with_tracing(
            "question_explainer",
            lambda: self._explainer.run(self._ctx, question=self._ctx.question),
        )

    def _run_branch(self, branch: Branch) -> list[AnnotatedValue]:
        """Execute a single branch and return its entries."""
        match branch.kind:
            case "retrieve":
                doc = self.execute_with_tracing(
                    "retrieve",
                    lambda: self._retriever.run(self._ctx, branch),
                )
                return self.execute_with_tracing(
                    "extract",
                    lambda: self._extractor.run(doc, self._ctx, branch),
                )
            case "lookup_external":
                return self.execute_with_tracing(
                    "lookup_external",
                    lambda: self._looker.run(self._ctx, branch),
                )

    def _run_branches(self, branches: list[Branch]) -> list[BranchOutcome]:
        """Run branches in parallel, returning one outcome per branch in order.
        Failures become `BranchOutcome(entries=None, error=...)`. Re-raises
        `StepFailed` only when EVERY branch failed."""
        if len(branches) == 1:
            # Fast path: let StepFailed propagate (matches "all failed" for n=1).
            entries = self._run_branch(branches[0])
            return [BranchOutcome(branch=branches[0], entries=entries, error=None)]
        outcomes: list[BranchOutcome | None] = [None] * len(branches)
        with ThreadPoolExecutor(
            max_workers=self._ctx.config.max_parallel_workers
        ) as pool:
            futures = {
                pool.submit(self._run_branch, b): i for i, b in enumerate(branches)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    outcomes[i] = BranchOutcome(
                        branch=branches[i], entries=future.result(), error=None
                    )
                except StepFailed as e:
                    self._ctx.emit(
                        "orchestrator", "parallel branch failed",
                        branch_idx=i, error=str(e),
                    )
                    outcomes[i] = BranchOutcome(
                        branch=branches[i], entries=None, error=str(e),
                    )
        result = [o for o in outcomes if o is not None]
        if all(o.error for o in result):
            # All failed → propagate the first error; the caller decides what to do.
            raise StepFailed(
                "orchestrator",
                f"all {len(branches)} branches failed; first error: {result[0].error}",
            )
        return result

    def execute_with_tracing(self, op_name: str, fn: Callable[[], Any]) -> Any:
        """Run `fn()` and record one `StepTrace` (op name, timing, result/error).

        `fn` is a zero-arg thunk closing over the actual operator call, so each
        operator runs with its own natural signature. The `begin` event delimits
        this step's operator emits in the trace dump. Re-raises `StepFailed` /
        `MissingData` after recording."""
        step_idx = len(self._trace.steps) + 1
        # `_step`/`begin` is the join key between the two observability systems:
        # it delimits this step's operator emits in the event stream so the
        # trace dump can group events under their StepTrace (eval/util.py). It is
        # caller-owned frame infrastructure, not an operator boundary log.
        self._ctx.emit("_step", "begin", step_idx=step_idx, op=op_name)
        t0 = time.perf_counter()
        try:
            result = fn()
        except (StepFailed, MissingData) as e:
            err = f"MissingData: {e.reason}" if isinstance(e, MissingData) else str(e)
            self._trace.steps.append(StepTrace(
                op=op_name, output_desc="(failed)", output_full="(failed)",
                elapsed_s=time.perf_counter() - t0, error=err, step_idx=step_idx,
            ))
            raise
        self._trace.steps.append(StepTrace(
            op=op_name, output_desc=describe_value(result), output_full=full_repr(result),
            elapsed_s=time.perf_counter() - t0, step_idx=step_idx,
        ))
        return result
