from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import cast
from skunk.compute import ComputeOp
from skunk.config import SkunkConfig
from skunk.data_prep import DataPrepOp
from skunk.errors import MissingData, StepFailed
from skunk.lookup_external import LookupExternalOp
from skunk.common import (
    AnnotatedValue,
    BranchRetrieval,
    ExecutionContext,
    Final,
    NeedsMore,
    PageRef,
    traced_step,
)
from skunk.llm_client import LLMClient
from skunk.plan import (
    AttemptRecord,
    Branch,
    Plan,
    Planner,
    RetrieveBranch,
)
from skunk.prompted_call import PromptOverride
from skunk.question_explainer import QuestionExplainer
from skunk.retrieve import RetrieveOp
from skunk.result import ExecutionResult


@dataclass
class BranchOutcome:
    branch: Branch
    entries: list[AnnotatedValue] | None
    error: StepFailed | None
    pages: list[PageRef] = field(default_factory=list)


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
        self._data_prep = DataPrepOp()
        self._result = ExecutionResult(question=question)
        # Deduped union of every page the retrieve phase produced this question (first-seen
        # order, accumulated across the initial sweep and any replan sweeps). Exposed via
        # `retrieved_pages` for the eval harness's `likely_pages` / retrieval-recall reporting.
        # Empty under golden bypass (retrieve never runs).
        self._retrieved_pages: list[PageRef] = []

    @property
    def ctx(self) -> ExecutionContext:
        return self._ctx

    @property
    def retrieved_pages(self) -> list[PageRef]:
        """Deduped union of pages from every retrieve sweep this question (empty under
        golden bypass)."""
        return self._retrieved_pages

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

        # Recovery state: the value POOL is what compute sees. On a NeedsMore the next pool
        # is exactly what compute chose to keep (re-stated inputs + any partial computation,
        # drop-by-default); each replan sweep then appends its newly gathered values. ATTEMPTS
        # records every branch's fate across sweeps. Branch outcomes don't outlive their
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
                pool = await self._prep_pool(pool)
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
                # Drop-by-default: the next pool IS exactly what compute chose to keep
                # (re-stated inputs + any partial computation); everything else is dropped.
                prev_size = len(pool)
                pool = list(needs.keep)
                self._ctx.emit(
                    f"pool_update round={round_idx} prev={prev_size} kept={len(pool)}",
                    data={"kept": [e.description for e in pool]},
                )
            else:
                # Replan sweep produced zero new values: compute on an unchanged pool is a
                # known no-op, so reuse the prior NeedsMore and go straight to the next
                # replan (the new failure diagnostics are already in `attempts`). Still
                # consumes a recovery round. Emit an EMPTY `compute` step (no model call)
                # anyway so the trace viewer gets one compute node per plan revision — its
                # revision tabs are keyed to the compute-delimited segments, so a skipped
                # compute would drop a tab.
                async def _short_circuit(_round: int = round_idx, _needs: NeedsMore = needs) -> None:
                    self._ctx.emit(
                        f"compute_short_circuit round={_round} missing={_needs.missing!r}"
                    )

                await traced_step(self._ctx, "compute", _short_circuit)

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
                lambda: self._planner.replan(self._ctx, pool, attempts, reason, missing),
            )

            ids = self._alloc_branch_ids(len(plan.branches))
            self._branch_ids = ids
            self._emit_plan(
                plan, "replan", reason=reason, missing=missing, recovery_round=round_idx
            )
            outcomes = await self._run_branches(plan.branches, ids)
            attempts += [
                AttemptRecord(round_idx, o.branch, len(o.entries or []), o.error)
                for o in outcomes
            ]
            new_entries = [e for o in outcomes if o.entries for e in o.entries]
            pool += new_entries
            sweep_added = bool(new_entries)

    async def _prep_pool(
        self, pool: list[AnnotatedValue]
    ) -> list[AnnotatedValue]:
        """Run the data-prep gate over the pool just before a compute call: dedup corpus reprints
        and coalesce same-series values across everything gathered so far. Fails safe via
        `DataPrepOp.run` (returns its input unchanged on any error), so it never starves compute of
        inputs."""
        if not pool:
            return pool
        return await traced_step(
            self._ctx,
            "data_prep",
            lambda: self._data_prep.run(pool, self._ctx),
        )

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
        recovery_round: int | None = None,
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
        if recovery_round is not None:
            data["recovery_round"] = recovery_round
        self._ctx.emit(
            f"plan label={label} branches={len(plan.branches)}",
            kind="plan",
            data=data,
        )

    async def _run_retrieve_phase(
        self,
        branches: list[RetrieveBranch],
        branch_ids: list[int],
    ) -> list[BranchRetrieval | StepFailed]:
        """Retrieve for every retrieve branch at once via the search-agent backend, returning
        one normalized `BranchRetrieval` (its whole pages) per branch — or the `StepFailed` to
        attribute to it. `RetrieveOp.run_all` owns all backend dispatch. `branch_ids` lets the
        search-agent backend emit a per-branch `retrieve` step."""
        if not branches:
            return []
        try:
            results = await traced_step(
                self._ctx,
                "retrieve",
                lambda: self._retrieve.run_all(self._ctx, branches, branch_ids),
            )
        except StepFailed as e:
            return [e] * len(branches)

        # Accumulate the deduped page union (for `likely_pages` / retrieval-recall reporting).
        # Replan sweeps extend the same list, so rebuild the seen-set from the current list each
        # call rather than carrying a persistent set that would outlive the sweep it was built for.
        seen_pages = set(self._retrieved_pages)
        for r in results:
            if isinstance(r, StepFailed):
                continue
            for p in r.pages:
                if p not in seen_pages:
                    seen_pages.add(p)
                    self._retrieved_pages.append(p)
        return results

    async def _run_branches(
        self,
        branches: list[Branch],
        branch_ids: list[int],
    ) -> list[BranchOutcome]:
        # Global retrieve phase: all retrieve branches share one deduped semantic-filter
        # sweep, then each branch's routed refs feed its own extract. Lookup branches are
        # independent and run in the per-branch tail below.
        retrieve_pos = [i for i, b in enumerate(branches) if b.kind == "retrieve"]
        retrievals = await self._run_retrieve_phase(
            [cast(RetrieveBranch, branches[i]) for i in retrieve_pos],
            [branch_ids[i] for i in retrieve_pos],
        )
        retr_by_pos: dict[int, BranchRetrieval | StepFailed] = dict(
            zip(retrieve_pos, retrievals)
        )

        # Extract runs as ONE shared sweep over all retrieve branches, launched as a task so
        # lookup branches proceed concurrently. Each retrieve tail awaits the shared task and
        # picks out its branch's result. Retrievals are whole pages (golden / search-agent);
        # `run_extract` reads each unique page once and does its own per-branch traced_steps,
        # so the tail doesn't wrap them again.
        pipeline: asyncio.Task | None = None
        if retrieve_pos:
            from skunk.extract import run_extract

            sub_branches = [cast(RetrieveBranch, branches[i]) for i in retrieve_pos]
            sub_ids = [branch_ids[i] for i in retrieve_pos]
            sub_retrievals = [retr_by_pos[i] for i in retrieve_pos]
            pipeline = asyncio.create_task(
                run_extract(self._ctx, sub_branches, sub_retrievals, sub_ids)
            )

        def _branch_pages(pos: int) -> list[PageRef]:
            r = retr_by_pos.get(pos)
            return list(r.pages) if isinstance(r, BranchRetrieval) else []

        async def _tail(pos: int) -> list[AnnotatedValue]:
            branch, bid = branches[pos], branch_ids[pos]
            if branch.kind == "retrieve":
                assert pipeline is not None
                res = (await pipeline)[retrieve_pos.index(pos)]
                if isinstance(res, StepFailed):
                    raise res
                return res
            # External lookup.
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
        for pos, (bid, (branch, res)) in enumerate(
            zip(branch_ids, zip(branches, results))
        ):
            pages = _branch_pages(pos)
            if isinstance(res, StepFailed):
                self._ctx.emit(
                    f"parallel_branch_failed branch_id={bid} error={str(res)!r}",
                    data={"branch_id": bid},
                )
                outcomes.append(
                    BranchOutcome(
                        branch=branch, entries=None, error=res, pages=pages
                    )
                )
            elif isinstance(res, BaseException):
                raise res
            else:
                outcomes.append(
                    BranchOutcome(
                        branch=branch, entries=res, error=None, pages=pages
                    )
                )
        return outcomes
