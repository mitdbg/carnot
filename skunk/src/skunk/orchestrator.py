from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import cast
from skunk.compute import ComputeOp
from skunk.config import SkunkConfig
from skunk.errors import MissingData, StepFailed
from skunk.lookup_external import LookupExternalOp
from skunk.common import (
    AnnotatedValue,
    BlockRef,
    ExecutionContext,
    Final,
    NeedsMore,
    SemPoolEntry,
    traced_step,
)
from skunk.llm_client import LLMClient
from skunk.plan import AttemptRecord, Branch, Plan, Planner, RetrieveBranch
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
        # current sweep's branch `i`; every sweep (initial or replan) allocates
        # fresh ids — branch identity dies at the end of its sweep, and a plan
        # event's branches belong to that revision only. The trace viewer keys
        # plan-branch ↔ operator-step on this id to render revision tabs.
        self._branch_ids: list[int] = []
        self._next_branch_id = 0
        self._planner = Planner()
        self._retrieve = RetrieveOp(self._ctx.config)
        self._lookup = LookupExternalOp()
        self._explainer = QuestionExplainer()
        self._compute = ComputeOp()
        self._result = ExecutionResult(question=question)
        # Deduped union of every BlockRef the retrieve phase produced this question (first-seen
        # order, accumulated across the initial sweep and any replan sweeps). Exposed via
        # `retrieved_blocks` so the eval harness can cache and replay a run without re-paying
        # retrieval. Empty under golden/replay bypass (retrieve never runs).
        self._retrieved_blocks: list[BlockRef] = []
        # Deduped union of every branch's sem-filter candidate pool (same lifecycle as
        # `_retrieved_blocks`). Exposed via `sem_pool` so the eval harness can persist it
        # into the retrieval cache for replayable coverage repair.
        self._sem_pool: list[SemPoolEntry] = []

    @property
    def ctx(self) -> ExecutionContext:
        return self._ctx

    @property
    def retrieved_blocks(self) -> list[BlockRef]:
        """Deduped union of blocks from every retrieve sweep this question (empty under
        golden/replay bypass). Pages are derivable via each block's `member_refs`."""
        return self._retrieved_blocks

    @property
    def sem_pool(self) -> list[SemPoolEntry]:
        """Deduped union of sem-filter candidate pools across retrieve sweeps (empty under
        golden/replay bypass and on the search-agent path)."""
        return self._sem_pool

    @property
    def current_plan(self) -> Plan | None:
        """The latest plan revision. After a replan this holds only that sweep's
        branches (replans compose fresh plans; earlier revisions live in the
        `kind="plan"` events)."""
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

        # Recovery state: the value POOL is what compute sees — compute's explicit
        # `keep` whitelist + committed computed intermediates + each sweep's new
        # values; everything compute does not keep is dropped. ATTEMPTS records
        # every branch's fate across sweeps. Branch outcomes don't outlive their
        # sweep — the pool is the only data carried across rounds.
        pool: list[AnnotatedValue] = [
            e for o in outcomes if o.entries for e in o.entries
        ]
        attempts = [
            AttemptRecord(0, o.branch, len(o.entries or []), o.error) for o in outcomes
        ]
        needs: NeedsMore | None = None
        round_idx = 0
        sweep_added = True  # the initial sweep always reaches compute

        while True:
            self._current_plan = plan
            if sweep_added or needs is None:
                outcome = await traced_step(
                    self._ctx,
                    "compute",
                    lambda: self._compute.run(
                        pool,
                        self._ctx,
                        concept_explanations=explanations,
                        round_idx=round_idx,
                    ),
                )
                if isinstance(outcome, Final):
                    self._result.answer = outcome.answer
                    return outcome.answer
                needs = outcome
                keep_set = set(needs.keep)
                discarded = [e for i, e in enumerate(pool) if i not in keep_set]
                pool = [e for i, e in enumerate(pool) if i in keep_set] + list(
                    needs.committed
                )
                self._ctx.emit(
                    f"pool_update round={round_idx} "
                    f"kept={len(pool) - len(needs.committed)} "
                    f"discarded={len(discarded)} committed={len(needs.committed)} "
                    f"pool_size={len(pool)}",
                    data={
                        "discarded": [e.description for e in discarded],
                        "committed": [e.description for e in needs.committed],
                    },
                )
            else:
                # Replan sweep produced zero new values: compute on an unchanged
                # pool is a known no-op, so reuse the prior NeedsMore and go
                # straight to the next replan (the new failure diagnostics are
                # already in `attempts`). Still consumes a recovery round.
                self._ctx.emit(
                    f"compute_short_circuit round={round_idx} missing={needs.missing!r}"
                )

            round_idx += 1
            if round_idx > self._ctx.config.recovery_max_rounds:
                self._ctx.emit(
                    f"recovery_exhausted rounds={round_idx - 1} "
                    f"missing={needs.missing!r}"
                )
                raise MissingData(needs.missing_reason, needs.missing)

            # Bind before the lambda (opaque to ruff's use analysis).
            reason, missing = needs.missing_reason, needs.missing
            plan = await traced_step(
                self._ctx,
                "replanner",
                lambda: self._planner.replan(
                    self._ctx,
                    pool,
                    attempts,
                    reason,
                    missing,
                ),
            )
            ids = self._alloc_branch_ids(len(plan.branches))
            self._branch_ids = ids
            self._emit_plan(plan, "replan", reason=reason, missing=missing)
            outcomes = await self._run_branches(plan.branches, ids)
            attempts += [
                AttemptRecord(round_idx, o.branch, len(o.entries or []), o.error)
                for o in outcomes
            ]
            new_entries = [e for o in outcomes if o.entries for e in o.entries]
            pool += new_entries
            sweep_added = bool(new_entries)

    def _alloc_branch_ids(self, n: int) -> list[int]:
        """Allocate `n` fresh, monotonically-increasing branch ids (stable for the
        question's lifetime — see `_branch_ids`)."""
        ids = list(range(self._next_branch_id, self._next_branch_id + n))
        self._next_branch_id += n
        return ids

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
    ) -> tuple[list[list[BlockRef] | StepFailed], list[list[SemPoolEntry]]]:
        """Unified multi-scan retrieve for every retrieve branch at once: their candidate
        pages are deduped and the LLM semantic filter scans each unique page at most once,
        judging it against all branches' targets, then routes the survivors back per branch.
        On the page-index path the survivors are left whole — the block-selection
        tournament (`run_select_pipeline`) narrows them downstream. Returns each branch's blocks in
        input order, or — if the whole sweep fails — the `StepFailed` to attribute to every
        retrieve branch. The shared `retrieve` sweep carries no `branch_id`."""
        if not branches:
            return [], []
        try:
            sem_survivors, base_pools = await traced_step(
                self._ctx,
                "retrieve",
                lambda: self._retrieve.run_all(self._ctx, branches),
            )
        except StepFailed as e:
            return [e] * len(branches), [[] for _ in branches]

        # Page-index path: survivors pass through whole — the selection agent narrows them.
        # golden / search-agent paths return final blocks from run_all.
        if (
            str(self._ctx.config.retriever) == "page_index"
            and self._ctx.config.golden_pages is None
        ):
            docs: list[list[BlockRef] | StepFailed] = [
                StepFailed(
                    "retrieve",
                    f"semantic filter kept no blocks for branch {b.key!r}",
                )
                if not brs
                else brs
                for b, brs in zip(branches, sem_survivors)
            ]
            pools = self._retrieve.build_survivor_pools(
                self._ctx, cast(list[list[BlockRef]], sem_survivors)
            )
        else:
            docs = sem_survivors
            pools = base_pools

        # Accumulate the deduped block + pool unions (cache seam). Replan sweeps extend the
        # same lists, so rebuild the seen-sets from the current lists each call rather than
        # carrying persistent sets that would outlive the sweep they were built for.
        seen_blocks = set(self._retrieved_blocks)
        for doc in docs:
            if isinstance(doc, StepFailed):
                continue
            for blk in doc:
                if blk not in seen_blocks:
                    seen_blocks.add(blk)
                    self._retrieved_blocks.append(blk)
        seen_pool = {e.ref for e in self._sem_pool}
        for pool in pools:
            for entry in pool:
                if entry.ref not in seen_pool:
                    seen_pool.add(entry.ref)
                    self._sem_pool.append(entry)
        return docs, pools

    async def _run_branches(
        self, branches: list[Branch], branch_ids: list[int]
    ) -> list[BranchOutcome]:
        # Global retrieve phase: all retrieve branches share one deduped semantic-filter
        # sweep, then each branch's routed refs feed its own extract. Lookup branches are
        # independent and run in the per-branch tail below.
        retrieve_pos = [i for i, b in enumerate(branches) if b.kind == "retrieve"]
        docs, pools = await self._run_retrieve_phase(
            [cast(RetrieveBranch, branches[i]) for i in retrieve_pos]
        )
        docs_by_pos: dict[int, list[BlockRef] | StepFailed] = dict(
            zip(retrieve_pos, docs)
        )
        pools_by_pos: dict[int, list[SemPoolEntry]] = dict(zip(retrieve_pos, pools))

        # Select → extract runs as ONE shared pipeline over all retrieve branches,
        # launched as a task so lookup branches proceed concurrently. Each retrieve
        # tail awaits the shared task and picks out its branch's result. The pipeline
        # does its own per-branch traced_steps, so the tail doesn't wrap it again.
        pipeline: asyncio.Task | None = None
        if retrieve_pos:
            from skunk.block_select import run_select_pipeline

            pipeline = asyncio.create_task(
                run_select_pipeline(
                    self._ctx,
                    [cast(RetrieveBranch, branches[i]) for i in retrieve_pos],
                    [docs_by_pos[i] for i in retrieve_pos],
                    [pools_by_pos.get(i, []) for i in retrieve_pos],
                    [branch_ids[i] for i in retrieve_pos],
                )
            )

        async def _tail(pos: int) -> list[AnnotatedValue]:
            branch, bid = branches[pos], branch_ids[pos]
            if branch.kind == "retrieve":
                assert pipeline is not None
                res = (await pipeline)[retrieve_pos.index(pos)]
                if isinstance(res, StepFailed):
                    raise res
                return res
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
