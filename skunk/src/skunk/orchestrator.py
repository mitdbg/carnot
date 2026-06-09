from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast
from skunk.compute import ComputeOp
from skunk.config import SkunkConfig
from skunk.errors import MissingData, StepFailed
from skunk.extract import ExtractOp
from skunk.lookup_external import LookupExternalOp
from skunk.common import AnnotatedValue, ExecutionContext, PageRef
from skunk.llm_client import LLMClient
from skunk.plan import Branch, Plan, PlanDiff, Planner, RetrieveBranch
from skunk.prompted_call import PromptOverride
from skunk.question_explainer import QuestionExplainer
from skunk.retrieve import RetrieveOp
from skunk.result import ExecutionResult, describe_value, summarize_value


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
        # Stable per-question branch identity. `_branch_ids[i]` is the id of the
        # current plan's branch `i`; a replan carries kept branches' ids forward and
        # allocates fresh ids for added branches (`PlanDiff.apply` makes new branch
        # COPIES, so positional indices are not stable across revisions). The trace
        # viewer keys plan-branch ↔ operator-step on this id to render revision tabs.
        self._branch_ids: list[int] = []
        self._next_branch_id = 0
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
            self._branch_ids = self._alloc_branch_ids(len(plan.branches))
            self._emit_plan(plan, "initial")
            outcomes = await self._run_branches(plan.branches, self._branch_ids)
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
                        self._ctx,
                        plan,
                        entries,
                        failed,
                        reason,
                        missing,
                    ),
                )
                plan, outcomes = await self._apply_diff(plan, outcomes, diff)
                self._emit_plan(plan, "replan", reason=reason, missing=missing)

    def _alloc_branch_ids(self, n: int) -> list[int]:
        """Allocate `n` fresh, monotonically-increasing branch ids (stable for the
        question's lifetime — see `_branch_ids`)."""
        ids = list(range(self._next_branch_id, self._next_branch_id + n))
        self._next_branch_id += n
        return ids

    async def _apply_diff(
        self, plan: Plan, outcomes: list[BranchOutcome], diff: PlanDiff
    ) -> tuple[Plan, list[BranchOutcome]]:
        """Apply a replan `PlanDiff`, keeping `outcomes[i]` aligned with the rewritten
        `plan.branches[i]`. `PlanDiff.apply` does the pure plan rewrite and reports
        which prior indices were KEPT; the orchestrator carries those branches'
        already-gathered outcomes forward, runs the added branches, and appends them.
        Compute therefore never sees data from a dropped branch. Stable branch ids
        ride along: kept branches keep theirs, added branches get fresh ones."""
        new_plan, kept = diff.apply(plan)
        dropped = sorted(set(range(len(plan.branches))) - set(kept))
        if dropped:
            self._ctx.emit(f"replan_dropped n_dropped={len(dropped)} indices={dropped}")
        kept_outcomes = [outcomes[i] for i in kept]
        kept_ids = [self._branch_ids[i] for i in kept]
        added_ids = self._alloc_branch_ids(len(diff.add))
        self._branch_ids = [*kept_ids, *added_ids]
        new_outcomes = await self._run_branches(diff.add, added_ids) if diff.add else []
        return new_plan, [*kept_outcomes, *new_outcomes]

    def _emit_plan(
        self,
        plan: Plan,
        label: str,
        *,
        reason: str | None = None,
        missing: list[str] | None = None,
    ) -> None:
        """Emit the current plan as a structured `kind="plan"` event — one per
        revision — so the trace viewer can render the revision tabs. Each branch
        carries its stable `branch_id` (so the viewer ties it to its operator steps),
        and a replan revision carries the `reason`/`missing` that triggered it. Out of
        any step (the plan describes the whole question, not one node)."""
        branches = []
        for bid, b in zip(self._branch_ids, plan.branches):
            entry = b.model_dump(mode="json")
            entry["branch_id"] = bid
            branches.append(entry)
        data: dict = {"label": label, "branches": branches}
        if reason is not None:
            data["reason"] = reason
        if missing is not None:
            data["missing"] = missing
        self._ctx.emit(
            f"plan label={label} branches={len(plan.branches)}",
            kind="plan",
            data=data,
        )

    async def _run_retrieve_phase(
        self, branches: list[RetrieveBranch]
    ) -> list[list[PageRef] | StepFailed]:
        """Unified multi-scan retrieve for every retrieve branch at once: their candidate
        pages are deduped and the LLM semantic filter scans each unique page at most once,
        judging it against all branches' targets, then routes the survivors back per branch.
        Returns each branch's routed page refs in input order, or — if the whole sweep fails
        — the `StepFailed` to attribute to every retrieve branch so each replans on its own.
        Not tied to one branch's trace step (it spans them), so `branch_id` is left unset."""
        if not branches:
            return []
        try:
            return list(
                await self._execute_with_tracing(
                    "retrieve",
                    lambda: self._retrieve.run_all(self._ctx, branches),
                )
            )
        except StepFailed as e:
            return [e] * len(branches)

    async def _run_branches(
        self, branches: list[Branch], branch_ids: list[int]
    ) -> list[BranchOutcome]:
        # Global retrieve phase: all retrieve branches share one deduped semantic-filter
        # sweep, then each branch's routed refs feed its own extract. Lookup branches are
        # independent and run in the per-branch tail below.
        retrieve_pos = [i for i, b in enumerate(branches) if b.kind == "retrieve"]
        docs = await self._run_retrieve_phase(
            [cast(RetrieveBranch, branches[i]) for i in retrieve_pos]
        )
        docs_by_pos: dict[int, list[PageRef] | StepFailed] = dict(
            zip(retrieve_pos, docs)
        )

        async def _tail(pos: int) -> list[AnnotatedValue]:
            branch, bid = branches[pos], branch_ids[pos]
            if branch.kind == "retrieve":
                doc = docs_by_pos[pos]
                if isinstance(doc, StepFailed):
                    raise doc
                doc = await self._maybe_select(cast(RetrieveBranch, branch), doc, bid)
                return await self._execute_with_tracing(
                    "extract",
                    lambda: self._extract.run(doc, self._ctx, branch),
                    branch_id=bid,
                )
            return await self._execute_with_tracing(
                "lookup_external",
                lambda: self._lookup.run(self._ctx, branch),
                branch_id=bid,
            )

        results = await asyncio.gather(
            *(_tail(i) for i in range(len(branches))),
            return_exceptions=True,
        )
        outcomes: list[BranchOutcome] = []
        for bid, (branch, res) in zip(branch_ids, zip(branches, results)):
            if isinstance(res, StepFailed):
                self._ctx.emit(
                    f"parallel_branch_failed branch_id={bid} error={str(res)!r}",
                    data={"branch_id": bid},
                )
                outcomes.append(BranchOutcome(branch=branch, entries=None, error=res))
            elif isinstance(res, BaseException):
                raise res
            else:
                outcomes.append(BranchOutcome(branch=branch, entries=res, error=None))
        return outcomes

    async def _maybe_select(
        self, branch: RetrieveBranch, doc: list[PageRef], bid: int
    ) -> list[PageRef]:
        """Intermediate selection stage (opt-in, page-index path only). Between the semantic
        filter and extract, `block_select` (`config.block_select`) narrows the filter's survivors
        to what this branch actually needs: a tournament of small packed group calls that reduces
        the candidate blocks to the few relevant ones (block granularity). Runs once per retrieve
        branch, isolated and in parallel with its siblings (called from `_tail`, fanned out by
        `_run_branches`). A no-op for golden / search-agent retrieval, when disabled, or when no
        candidate ref resolves to a catalog block (nothing to narrow). Otherwise the selector runs
        and its result is used directly — there is NO degrade-to-unfiltered-refs fallback: a
        selector failure propagates and fails the branch loudly rather than silently flooding
        extract with every candidate."""
        cfg = self._ctx.config
        if (
            not cfg.block_select
            or not doc
            or cfg.golden_pages is not None
            or cfg.retriever != "page_index_old"
        ):
            return doc
        from skunk.page_index.block_select import BlockSelectAgent

        agent = BlockSelectAgent(doc, str(cfg.pdf_dir))
        if not agent.has_candidates():
            return doc
        return await self._execute_with_tracing(
            "block_select",
            lambda: agent.select(self._ctx, self._ctx.question, branch),
            branch_id=bid,
        )

    async def _execute_with_tracing[T](
        self,
        op_name: str,
        fn: Callable[[], Awaitable[T]],
        *,
        branch_id: int | None = None,
    ) -> T:
        t0 = time.perf_counter()
        with self._ctx.step(op_name):
            try:
                result = await fn()
            except (StepFailed, MissingData) as e:
                err = (
                    f"MissingData: {e.reason}" if isinstance(e, MissingData) else str(e)
                )
                elapsed = round(time.perf_counter() - t0, 3)
                self._ctx.emit(
                    f"step elapsed_s={elapsed} output=(failed) error={err!r}",
                    kind="step",
                    data={"branch_id": branch_id, "elapsed_s": elapsed, "error": err},
                )
                raise
            elapsed = round(time.perf_counter() - t0, 3)
            self._ctx.emit(
                f"step elapsed_s={elapsed} output={describe_value(result)!r}",
                kind="step",
                data={
                    "branch_id": branch_id,
                    "elapsed_s": elapsed,
                    "summary": summarize_value(result),
                },
            )
        return result
