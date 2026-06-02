"""Orchestrator — per-question execution driver. One instance per NL question.

A Plan is a flat list of branches feeding a single compute step: the planner
produces the Plan, the branches run via `_run_branches` (parallel when >1), and the
merged AnnotatedValues feed compute, which returns the answer string.

If compute raises `MissingData`, a bounded replan loop (`config.recovery_max_rounds`)
re-invokes the planner with the prior plan + accumulated entries + the missing
signal; only the newly-added branches execute. After the budget is spent,
`MissingData` propagates."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from skunk.compute import ComputeOp
from skunk.errors import MissingData, StepFailed
from skunk.extract import ExtractOp
from skunk.lookup_external import LookupExternalOp
from skunk.common import AnnotatedValue, HarnessContext
from skunk.plan import Branch, Plan, Planner
from skunk.question_explainer import QuestionExplainer
from skunk.retrieve import RetrieveOp
from skunk.result import ExecutionResult, describe_value, full_repr


@dataclass
class BranchOutcome:
    """Per-branch result. `entries` is None on failure, `error` None on success.
    `error` keeps the whole `StepFailed` (not `str(e)`) so its `.diagnostic` —
    the operator's rich account of what it tried and what blocked it — survives
    to the replanner, which renders it untruncated to choose a workable pivot."""

    branch: Branch
    entries: list[AnnotatedValue] | None
    error: StepFailed | None


class Orchestrator:
    """One per NL question. The orchestrator drives the work forward by invoking the planner and stepping through the plan.
    It is also responsible for catching any execution errors and replanning when necessary."""

    def __init__(self, ctx: HarnessContext):
        self._ctx = ctx
        self._current_plan: Plan | None = None
        self._planner = Planner()
        self._retrieve = RetrieveOp(ctx.config)
        self._extract = ExtractOp()
        self._lookup = LookupExternalOp()
        self._explainer = QuestionExplainer()
        self._compute = ComputeOp()
        self._result = ExecutionResult(question=ctx.question)

        self._step_lock = threading.Lock()
        self._next_step_idx = 1

    @property
    def current_plan(self) -> Plan | None:
        return self._current_plan

    @property
    def result(self) -> ExecutionResult:
        return self._result

    @property
    def n_steps(self) -> int:
        """How many operator steps were allocated (each emits one boundary event)."""
        return self._next_step_idx - 1

    def execute(self) -> str:
        """Resolve the plan, walk the data phase, run compute, return the answer."""
        with ThreadPoolExecutor(
            max_workers=self._ctx.config.max_parallel_workers
        ) as pool:
            explain_future = pool.submit(
                self._execute_with_tracing,
                "question_explainer",
                lambda: self._explainer.run(self._ctx, question=self._ctx.question),
            )
            # Initial pass: plan, run every branch, then join the explainer —
            # it ran concurrently with planning + branch execution.
            plan = self._execute_with_tracing(
                "planner",
                lambda: self._planner.plan(self._ctx.question, self._ctx),
            )
            outcomes = self._run_branches(plan.branches, pool)
            explanations = explain_future.result()

            replans_remaining = self._ctx.config.recovery_max_rounds
            while True:
                self._current_plan = plan
                entries = [e for o in outcomes if o.entries for e in o.entries]
                try:
                    self._result.answer = self._execute_with_tracing(
                        "compute",
                        lambda: self._compute.run(
                            entries,
                            self._ctx,
                            plan,
                            concept_explanations=explanations,
                        ),
                    )
                    return self._result.answer
                except MissingData as e:
                    if replans_remaining == 0:
                        raise
                    replans_remaining -= 1
                    # Rebind out of the except clause: the `as` name is del'd at
                    # block end, so the replanner lambda below can't close over it.
                    missing = e
                    # Replan against the failure; adopt the replanner's
                    # computation/presentation, but run only its new branches.
                    failed = [(o.branch, o.error) for o in outcomes if o.error]
                    replanned = self._execute_with_tracing(
                        "replanner",
                        lambda: self._planner.replan(
                            self._ctx, plan, entries, failed,
                            missing.reason, missing.missing,
                        ),
                    )
                    new_branches = [
                        b for b in replanned.branches if b not in plan.branches
                    ]
                    merged = replanned.model_copy(
                        update={"branches": [*plan.branches, *new_branches]}
                    )
                    if merged == plan:
                        # No usable diff; the budget can't help.
                        self._ctx.emit(
                            "orchestrator", "replanner declined; no diff to execute"
                        )
                        raise
                    outcomes += self._run_branches(new_branches, pool)
                    plan = merged

    def _run_branch(self, branch: Branch) -> list[AnnotatedValue]:
        """Run one branch's operator pipeline (traced): a retrieve branch is
        retrieve → extract; a lookup_external branch is a single call."""
        if branch.kind == "retrieve":
            doc = self._execute_with_tracing(
                "retrieve", lambda: self._retrieve.run(self._ctx, branch)
            )
            return self._execute_with_tracing(
                "extract", lambda: self._extract.run(doc, self._ctx, branch)
            )
        return self._execute_with_tracing(
            "lookup_external", lambda: self._lookup.run(self._ctx, branch)
        )

    def _run_branches(
        self, branches: list[Branch], pool: ThreadPoolExecutor
    ) -> list[BranchOutcome]:
        outcomes: list[BranchOutcome | None] = [None] * len(branches)
        futures = {pool.submit(self._run_branch, b): i for i, b in enumerate(branches)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                outcomes[i] = BranchOutcome(
                    branch=branches[i], entries=future.result(), error=None
                )
            except StepFailed as e:
                self._ctx.emit(
                    "orchestrator",
                    "parallel branch failed",
                    branch_idx=i,
                    error=str(e),
                )
                outcomes[i] = BranchOutcome(
                    branch=branches[i],
                    entries=None,
                    error=e,
                )
        return [o for o in outcomes if o is not None]

    def _execute_with_tracing[T](self, op_name: str, fn: Callable[[], T]) -> T:
        # Allocate the step id up front and atomically so concurrent branches
        # get distinct, stable ids. `ctx.step` scopes it on this thread for the
        # duration of the call, stamping every emit (the operator's internals AND
        # the boundary event below) with this (step_idx, op) — the join key the
        # trace dump groups on. The boundary is a caller-owned `("orchestrator",
        # "step")` event: the orchestrator logs each operator's op/timing/result,
        # so operators never self-report their own boundaries.
        with self._step_lock:
            step_idx = self._next_step_idx
            self._next_step_idx += 1
        t0 = time.perf_counter()
        with self._ctx.step(step_idx, op_name):
            try:
                result = fn()
            except (StepFailed, MissingData) as e:
                err = f"MissingData: {e.reason}" if isinstance(e, MissingData) else str(e)
                self._ctx.emit(
                    "orchestrator", "step",
                    elapsed_s=round(time.perf_counter() - t0, 3),
                    output_desc="(failed)", output_full="(failed)", error=err,
                )
                raise
            self._ctx.emit(
                "orchestrator", "step",
                elapsed_s=round(time.perf_counter() - t0, 3),
                output_desc=describe_value(result), output_full=full_repr(result),
            )
        return result
