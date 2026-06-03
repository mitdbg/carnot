from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from skunk.compute import ComputeOp
from skunk.config import SkunkConfig
from skunk.errors import MissingData, StepFailed
from skunk.extract import ExtractOp
from skunk.lookup_external import LookupExternalOp
from skunk.common import AnnotatedValue, ExecutionContext, LLMClient
from skunk.plan import Branch, Plan, PlanDiff, Planner
from skunk.prompted_call import PromptOverride
from skunk.question_explainer import QuestionExplainer
from skunk.retrieve import RetrieveOp
from skunk.result import ExecutionResult, describe_value


@dataclass
class BranchOutcome:
    branch: Branch
    entries: list[AnnotatedValue] | None
    error: StepFailed | None


class Orchestrator:
    """One per NL question. The orchestrator drives the work forward by invoking the planner and stepping through the plan.
    It is also responsible for catching any execution errors and replanning when necessary."""

    def __init__(
        self,
        question: str,
        *,
        uid: str | None = None,
        verbose: bool = False,
        log_path: str | None = None,
        config: SkunkConfig | None = None,
        prompt_overrides: tuple[PromptOverride, ...] = (),
        llm_client: LLMClient | None = None,
    ):
        self._ctx = ExecutionContext(
            question=question,
            uid=uid,
            verbose=verbose,
            log_path=log_path,
            config=config or SkunkConfig.from_env(),
            prompt_overrides=prompt_overrides,
            llm_client=llm_client,
        )
        self._current_plan: Plan | None = None
        self._planner = Planner()
        self._retrieve = RetrieveOp(self._ctx.config)
        self._extract = ExtractOp()
        self._lookup = LookupExternalOp()
        self._explainer = QuestionExplainer()
        self._compute = ComputeOp()
        self._result = ExecutionResult(question=question)

    @property
    def ctx(self) -> ExecutionContext:
        return self._ctx

    @property
    def current_plan(self) -> Plan | None:
        return self._current_plan

    @property
    def result(self) -> ExecutionResult:
        return self._result

    async def execute(self) -> str:
        explain_task = asyncio.create_task(
            self._execute_with_tracing(
                "question_explainer",
                lambda: self._explainer.run(self._ctx, question=self._ctx.question),
            )
        )
        try:
            plan = await self._execute_with_tracing(
                "planner",
                lambda: self._planner.plan(self._ctx.question, self._ctx),
            )
            outcomes = await self._run_branches(plan.branches)
            explanations = await explain_task
        finally:
            explain_task.cancel()

        attempt = 0
        while True:
            self._current_plan = plan
            entries = [e for o in outcomes if o.entries for e in o.entries]
            try:
                self._result.answer = await self._execute_with_tracing(
                    "compute",
                    lambda: self._compute.run(
                        entries,
                        self._ctx,
                        concept_explanations=explanations,
                    ),
                )
                return self._result.answer
            except MissingData as e:
                attempt += 1
                if attempt > self._ctx.config.recovery_max_rounds:
                    # recovery budget exhausted
                    raise
                # Bind before the lambda: Python deletes the `except` variable on
                # block exit, and the lambda is also opaque to ruff's use analysis.
                reason, missing = e.reason, e.missing
                failed = [(o.branch, o.error) for o in outcomes if o.error]
                diff = await self._execute_with_tracing(
                    "replanner",
                    lambda: self._planner.replan(
                        self._ctx, plan, entries, failed,
                        reason, missing,
                    ),
                )
                plan, outcomes = await self._apply_diff(plan, outcomes, diff)

    async def _apply_diff(
        self, plan: Plan, outcomes: list[BranchOutcome], diff: PlanDiff
    ) -> tuple[Plan, list[BranchOutcome]]:
        """Apply a replan `PlanDiff`, keeping `outcomes[i]` aligned with the rewritten
        `plan.branches[i]`. `PlanDiff.apply` does the pure plan rewrite and reports
        which prior indices were KEPT; the orchestrator carries those branches'
        already-gathered outcomes forward, runs the added branches, and appends them.
        Compute therefore never sees data from a dropped branch."""
        new_plan, kept = diff.apply(plan)
        dropped = sorted(set(range(len(plan.branches))) - set(kept))
        if dropped:
            self._ctx.emit(f"replan_dropped n_dropped={len(dropped)} indices={dropped}")
        kept_outcomes = [outcomes[i] for i in kept]
        new_outcomes = await self._run_branches(diff.add) if diff.add else []
        return new_plan, [*kept_outcomes, *new_outcomes]

    async def _run_branch(self, branch: Branch) -> list[AnnotatedValue]:
        if branch.kind == "retrieve":
            doc = await self._execute_with_tracing(
                "retrieve", lambda: self._retrieve.run(self._ctx, branch)
            )
            return await self._execute_with_tracing(
                "extract", lambda: self._extract.run(doc, self._ctx, branch)
            )
        return await self._execute_with_tracing(
            "lookup_external", lambda: self._lookup.run(self._ctx, branch)
        )

    async def _run_branches(self, branches: list[Branch]) -> list[BranchOutcome]:
        results = await asyncio.gather(
            *(self._run_branch(b) for b in branches), return_exceptions=True
        )
        outcomes: list[BranchOutcome] = []
        for i, (branch, res) in enumerate(zip(branches, results)):
            if isinstance(res, StepFailed):
                self._ctx.emit(f"parallel_branch_failed branch_idx={i} error={str(res)!r}")
                outcomes.append(BranchOutcome(branch=branch, entries=None, error=res))
            elif isinstance(res, BaseException):
                raise res
            else:
                outcomes.append(BranchOutcome(branch=branch, entries=res, error=None))
        return outcomes

    async def _execute_with_tracing[T](self, op_name: str, fn: Callable[[], Awaitable[T]]) -> T:
        t0 = time.perf_counter()
        with self._ctx.step(op_name):
            try:
                result = await fn()
            except (StepFailed, MissingData) as e:
                err = f"MissingData: {e.reason}" if isinstance(e, MissingData) else str(e)
                self._ctx.emit(
                    f"step elapsed_s={round(time.perf_counter() - t0, 3)} "
                    f"output=(failed) error={err!r}"
                )
                raise
            self._ctx.emit(
                f"step elapsed_s={round(time.perf_counter() - t0, 3)} "
                f"output={describe_value(result)!r}"
            )
        return result
