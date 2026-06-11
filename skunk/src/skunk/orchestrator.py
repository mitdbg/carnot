from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import cast
from skunk.compute import ComputeOp
from skunk.config import SkunkConfig
from skunk.errors import MissingData, StepFailed
from skunk.extract import ExtractOp
from skunk.human import HumanAssist
from skunk.lookup_external import LookupExternalOp
from skunk.common import AnnotatedValue, BlockRef, ExecutionContext, traced_step
from skunk.llm_client import LLMClient
from skunk.plan import Branch, Plan, PlanDiff, Planner, RetrieveBranch
from skunk.prompted_call import PromptOverride
from skunk.question_explainer import QuestionExplainer
from skunk.retrieve import RetrieveOp
from skunk.result import ExecutionResult


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
        # Human-in-the-loop middleware (inert unless a SKUNK_HUMAN_* flag is set). Gates on
        # ctx.config per call, so constructing it unconditionally is free when disabled.
        self._human = HumanAssist()
        self._explainer = QuestionExplainer()
        self._compute = ComputeOp()
        self._result = ExecutionResult(question=question)
        # Deduped union of every BlockRef the retrieve phase produced this question (first-seen
        # order, accumulated across the initial sweep and any replan sweeps). Exposed via
        # `retrieved_blocks` so the eval harness can cache and replay a run without re-paying
        # retrieval. Empty under golden/replay bypass (retrieve never runs).
        self._retrieved_blocks: list[BlockRef] = []

    @property
    def ctx(self) -> ExecutionContext:
        return self._ctx

    @property
    def retrieved_blocks(self) -> list[BlockRef]:
        """Deduped union of blocks from every retrieve sweep this question (empty under
        golden/replay bypass). Pages are derivable via each block's `member_refs`."""
        return self._retrieved_blocks

    @property
    def current_plan(self) -> Plan | None:
        return self._current_plan

    @property
    def result(self) -> ExecutionResult:
        return self._result

    async def execute(self) -> str:
        explain_task = asyncio.create_task(
            traced_step(
                self._ctx,
                "question_explainer",
                lambda: self._explainer.run(self._ctx, question=self._ctx.question),
            )
        )
        try:
            plan = await traced_step(
                self._ctx,
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
                self._result.answer = await traced_step(
                    self._ctx,
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
                diff = await traced_step(
                    self._ctx,
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
        self, branches: list[RetrieveBranch], branch_ids: list[int]
    ) -> list[list[BlockRef] | StepFailed]:
        """Unified multi-scan retrieve for every retrieve branch at once: their candidate
        pages are deduped and the LLM semantic filter scans each unique page at most once,
        judging it against all branches' targets, then routes the survivors back per branch.
        On the page-index path the routed survivors are further narrowed by the backend's
        intrinsic `block_select` stage before they are returned. Returns each branch's
        blocks in input order, or — if the whole sweep fails — the `StepFailed` to
        attribute to every retrieve branch so each replans on its own.

        Tracing is backend-aware: the page-index sweep spans all branches, so it is one
        phase `retrieve` step with no `branch_id`; the search-agent backend runs an
        independent agent per branch, so `run_all` emits a per-branch `retrieve` step (each
        carrying its `branch_id` + retrieved `pages`) and we do not add a phase step here."""
        if not branches:
            return []
        try:
            if (
                str(self._ctx.config.retriever) == "search_agent"
                and self._ctx.config.golden_pages is None
            ):
                docs: list[list[BlockRef] | StepFailed] = list(
                    await self._retrieve.run_all(self._ctx, branches, branch_ids)
                )
            else:
                docs = list(
                    await traced_step(
                        self._ctx,
                        "retrieve",
                        lambda: self._retrieve.run_all(self._ctx, branches, branch_ids),
                    )
                )
        except StepFailed as e:
            return [e] * len(branches)

        # Accumulate the deduped block union (cache seam). Replan sweeps extend the same
        # list, so rebuild the seen-set from the current list each call rather than carrying
        # a persistent set that would outlive the sweep it was built for.
        seen_blocks = set(self._retrieved_blocks)
        for doc in docs:
            if isinstance(doc, StepFailed):
                continue
            for blk in doc:
                if blk not in seen_blocks:
                    seen_blocks.add(blk)
                    self._retrieved_blocks.append(blk)
        return docs

    async def _run_branches(
        self, branches: list[Branch], branch_ids: list[int]
    ) -> list[BranchOutcome]:
        # Global retrieve phase: all retrieve branches share one deduped semantic-filter
        # sweep, then each branch's routed refs feed its own extract. Lookup branches are
        # independent and run in the per-branch tail below.
        retrieve_pos = [i for i, b in enumerate(branches) if b.kind == "retrieve"]
        docs = await self._run_retrieve_phase(
            [cast(RetrieveBranch, branches[i]) for i in retrieve_pos],
            [branch_ids[i] for i in retrieve_pos],
        )
        docs_by_pos: dict[int, list[BlockRef] | StepFailed] = dict(
            zip(retrieve_pos, docs)
        )

        async def _tail(pos: int) -> list[AnnotatedValue]:
            branch, bid = branches[pos], branch_ids[pos]
            if branch.kind == "retrieve":
                # Already post-block-select: `_run_retrieve_phase` returned the branch's blocks
                # (page-index selections, or whole-page blocks for golden / search-agent). Extract
                # reads those and feeds each block's pages whole.
                doc = docs_by_pos[pos]
                if isinstance(doc, StepFailed):
                    raise doc
                entries = await traced_step(
                    self._ctx,
                    "extract",
                    lambda: self._extract.run(doc, self._ctx, branch),
                    branch_id=bid,
                )
                # Human verifies/produces the extracted value(s) (figure or OCR/table read)
                # only when the policy opts in — no step (or prompt) on the default path.
                if self._human.wants_verify(branch, entries, self._ctx):
                    entries = await traced_step(
                        self._ctx,
                        "human_verify",
                        lambda: self._human.verify_extract(
                            entries, doc, branch, self._ctx
                        ),
                        branch_id=bid,
                    )
                return entries
            # Human performs the external lookup when the flag is on; else the lookup agent.
            if self._human.wants_lookup(branch, self._ctx):
                return await traced_step(
                    self._ctx,
                    "human_lookup",
                    lambda: self._human.human_lookup(branch, self._ctx),
                    branch_id=bid,
                )
            return await traced_step(
                self._ctx,
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
