from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import cast
from skunk.compute import ComputeOp
from skunk.config import SkunkConfig
from skunk.data_prep import DataPrepOp
from skunk.errors import MissingData, StepFailed
from skunk.human import (
    POOL_REVIEW_BRANCH_ID,
    BrokerChannel,
    ConsoleChannel,
    HumanAssist,
    apply_overrides,
)
from skunk.lookup_external import LookupExternalOp
from skunk.common import (
    AnnotatedValue,
    BranchRetrieval,
    ExecutionContext,
    Final,
    HumanInterventionHandler,
    HumanReviewRegister,
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
from skunk.question_explainer import ConceptExplanation, QuestionExplainer
from skunk.retrieve import RetrieveOp
from skunk.result import ExecutionResult


@dataclass
class BranchOutcome:
    branch: Branch
    entries: list[AnnotatedValue] | None
    error: StepFailed | None
    pages: list[PageRef] = field(default_factory=list)


@dataclass
class RecomputeState:
    """The minimal snapshot needed to revise an answer after a human review resolves, WITHOUT
    re-planning or re-retrieving: the per-branch extracted entries (keyed by stable branch id),
    any extra (human-recovery) entries, and the concept explanations compute was given. A
    recompute swaps a branch's entries for the human's correction and re-runs ONLY compute —
    deterministic and cheap, so the revision reflects exactly the human's edit (no planner drift).
    """

    question: str
    order: list[int]
    entries_by_branch: dict[int, list[AnnotatedValue]]
    extra_entries: list[AnnotatedValue]
    explanations: list[ConceptExplanation]

    def to_jsonable(self) -> dict:
        return {
            "question": self.question,
            "order": list(self.order),
            "entries_by_branch": {
                str(bid): [e.model_dump(mode="json") for e in entries]
                for bid, entries in self.entries_by_branch.items()
            },
            "extra_entries": [e.model_dump(mode="json") for e in self.extra_entries],
            "explanations": [c.model_dump() for c in self.explanations],
        }

    @classmethod
    def from_jsonable(cls, data: dict) -> RecomputeState:
        return cls(
            question=str(data["question"]),
            order=[int(b) for b in data.get("order", [])],
            entries_by_branch={
                int(bid): [AnnotatedValue.model_validate(e) for e in entries]
                for bid, entries in data.get("entries_by_branch", {}).items()
            },
            extra_entries=[
                AnnotatedValue.model_validate(e) for e in data.get("extra_entries", [])
            ],
            explanations=[
                ConceptExplanation.model_validate(c)
                for c in data.get("explanations", [])
            ],
        )


async def recompute_answer(
    state: RecomputeState,
    overrides: dict[int, list[dict]],
    ctx: ExecutionContext,
) -> str:
    """Re-run ONLY compute over the first attempt's per-branch entries with human `overrides`
    (keyed by branch id) applied — the deterministic revision path. Each override is the human's
    source-indexed review items for that branch; `apply_overrides` rebuilds the branch's entries
    from them (honoring edits, deletes, and provenance). Branches with no override keep their
    original entries. No re-plan / re-retrieve / re-extract. One compute pass: a `Final` yields
    the revised answer, an unresolved `NeedsMore` surfaces as `MissingData`."""
    merged: list[AnnotatedValue] = []
    for bid in state.order:
        cached = state.entries_by_branch.get(bid, [])
        if bid in overrides:
            merged.extend(apply_overrides(overrides[bid], cached))
        else:
            merged.extend(cached)
    merged.extend(state.extra_entries)
    outcome = await ComputeOp().run(
        merged, ctx, concept_explanations=state.explanations
    )
    if isinstance(outcome, Final):
        return outcome.answer
    raise MissingData(outcome.missing_reason, outcome.missing)


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
        human_intervention_handler: HumanInterventionHandler | None = None,
        human_review_register: HumanReviewRegister | None = None,
    ):
        self._ctx = ExecutionContext(
            question=question,
            uid=uid,
            verbose=verbose,
            log_path=log_path,
            config=config or SkunkConfig.from_env(),
            prompt_overrides=prompt_overrides,
            llm_client=llm_client,
            human_intervention_handler=human_intervention_handler,
            human_review_register=human_review_register,
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
        # Human-in-the-loop middleware (inert unless a SKUNK_HUMAN_* flag is set). Gates on
        # ctx.config per call, so constructing it unconditionally is free when disabled.
        # Channel by transport: under the competition server a handler is present → route
        # human requests through the async broker/web UI; for a local CLI run fall back to
        # the blocking console.
        human_channel = (
            BrokerChannel(self._ctx.human_intervention_handler)
            if self._ctx.human_intervention_handler is not None
            else ConsoleChannel()
        )
        self._human = HumanAssist(channel=human_channel)
        self._explainer = QuestionExplainer()
        self._compute = ComputeOp()
        self._data_prep = DataPrepOp()
        self._result = ExecutionResult(question=question)
        # Deduped union of every page the retrieve phase produced this question (first-seen
        # order, accumulated across the initial sweep and any replan sweeps). Exposed via
        # `retrieved_pages` for the eval harness's `likely_pages` / retrieval-recall reporting.
        # Empty under golden bypass (retrieve never runs).
        self._retrieved_pages: list[PageRef] = []
        # Snapshot for an optimistic-review recompute (the per-branch entries that produced the
        # answer); set on every compute attempt, None until then. The server stores it on the
        # task so a later human resolve can revise the answer via `recompute_answer`.
        self._recompute_state: RecomputeState | None = None

    @property
    def ctx(self) -> ExecutionContext:
        return self._ctx

    @property
    def retrieved_pages(self) -> list[PageRef]:
        """Deduped union of pages from every retrieve sweep this question (empty under
        golden bypass)."""
        return self._retrieved_pages

    @property
    def recompute_state(self) -> RecomputeState | None:
        """Snapshot of the per-branch entries that produced the answer (None until the first
        compute attempt). The server stores it so a later human-review resolve can revise the
        answer deterministically via `recompute_answer`. See `RecomputeState`."""
        return self._recompute_state

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
                pre_clean_pool = pool
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
                    # Final answer: NOW snapshot the pool for recompute and open the single human
                    # review of it. Registering only on success (not before every compute) means a
                    # superseded/replanning round never has an open pool review — so a human can't
                    # be mid-edit on a review the orchestrator is about to discard, and the
                    # recompute snapshot always matches the data behind the answer. Page attribution
                    # uses the PRE-clean pool (single doc_id each); the cleaned pool's coalesced
                    # multi-doc values can't resolve to a PDF.
                    self._capture_recompute_pool(pool, explanations)
                    if self._human.wants_pool_review(self._ctx):
                        self._human.register_pool_review(
                            pool, self._ctx, source_values=pre_clean_pool
                        )
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

            # === Replan, then HITL approval of the proposed plan ===
            # Replan FIRST (no human steer) to produce a proposed plan, then — when a handler is
            # wired — show the human the data-prep output, the previous plan, what compute reported
            # missing, and the proposed plan, and block for approval. On rejection, re-run the
            # replan with their feedback injected and execute THAT directly (no second approval).
            previous_plan = plan
            handler = self._ctx.human_intervention_handler

            async def _replan(guidance: str | None) -> Plan:
                return await traced_step(
                    self._ctx,
                    "replanner",
                    lambda: self._planner.replan(
                        self._ctx, pool, attempts, reason, missing, human_guidance=guidance
                    ),
                )

            plan = await _replan(None)
            if handler is not None:
                self._emit_plan(
                    previous_plan,
                    "human_pending",
                    reason=reason,
                    missing=missing,
                    recovery_round=round_idx,
                )
                feedback = await self._request_replan_approval(
                    handler, previous_plan, plan, pool, reason, missing, round_idx
                )
                if feedback:
                    plan = await _replan(feedback)

            ids = self._alloc_branch_ids(len(plan.branches))
            self._branch_ids = ids
            self._emit_plan(
                plan, "replan", reason=reason, missing=missing, recovery_round=round_idx
            )
            # The first replan may surface in-agent `request_human` help (figure / search
            # tool), gated to round 1 like the pre-merge HITL path.
            self._ctx.human_intervention_enabled = (
                round_idx == 1 and handler is not None
            )
            try:
                outcomes = await self._run_branches(plan.branches, ids)
            finally:
                self._ctx.human_intervention_enabled = False
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
        inputs.

        Cleaning runs every round, but the human review of the cleaned pool is opened only once
        compute reaches a Final answer (see the recovery loop) — so a superseded/replanning round
        never has an open review the human could be mid-editing."""
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

    def _review_mode(self) -> str:
        """How a wanted human review is serviced for this run:
        - "optimistic": a register hook is wired (under the competition server) — register an
          open review and keep the LLM result, non-blocking. Whether a review is wanted at all
          is the per-stage policy's job (the SKUNK_HUMAN_* gates); this only picks the transport.
        - "blocking":   no register hook (local CLI) — await the human on the console / async
          broker as before.
        """
        if self._ctx.human_review_register is None:
            return "blocking"
        return "optimistic"

    def _capture_recompute_pool(
        self,
        pool: list[AnnotatedValue],
        explanations: list[ConceptExplanation],
    ) -> None:
        """Snapshot the cleaned pool compute reads + explanations that produced this answer, so a
        resolve of the data-prep pool review can recompute. The pool is stored under the sentinel
        `POOL_REVIEW_BRANCH_ID`; `recompute_answer` applies the human's edited pool onto it and
        re-runs ONLY compute (no replan / re-retrieve / re-extract)."""
        self._recompute_state = RecomputeState(
            question=self._ctx.question,
            order=[POOL_REVIEW_BRANCH_ID],
            entries_by_branch={POOL_REVIEW_BRANCH_ID: list(pool)},
            extra_entries=[],
            explanations=list(explanations),
        )

    def _replan_approval_guidance(
        self,
        previous_plan: Plan,
        proposed_plan: Plan,
        pool: list[AnnotatedValue],
        reason: str,
        missing: list[str],
        round_idx: int,
    ) -> dict:
        """The payload the replan-approval review renders: the data-prep agent's output (the
        cleaned pool compute read), the previous plan, what compute reported missing, and the
        proposed new plan the human is approving. Branch ids are attached to the previous plan
        (its `self._branch_ids`); the proposed plan has none yet (allocated after approval)."""

        def _branches(plan: Plan, *, with_ids: bool) -> list[dict]:
            if with_ids:
                return [
                    {"branch_id": bid, **branch.model_dump(mode="json")}
                    for bid, branch in zip(self._branch_ids, plan.branches)
                ]
            return [branch.model_dump(mode="json") for branch in plan.branches]

        return {
            "recovery_round": round_idx,
            "reason": reason,
            "missing": missing,
            "data_prep_output": [
                entry.model_dump(
                    mode="json",
                    include={"description", "value", "unit", "kind", "doc_id", "pages"},
                )
                for entry in pool
            ],
            "previous_plan": _branches(previous_plan, with_ids=True),
            "proposed_plan": _branches(proposed_plan, with_ids=False),
        }

    async def _request_replan_approval(
        self,
        handler: HumanInterventionHandler,
        previous_plan: Plan,
        proposed_plan: Plan,
        pool: list[AnnotatedValue],
        reason: str,
        missing: list[str],
        round_idx: int,
    ) -> str:
        """Block on the human to approve the proposed replan. Shows the data-prep output, the
        previous plan, the compute MissingData, and the proposed plan. Returns the steer feedback:
        an empty string means APPROVE (execute the proposed plan as-is); a non-empty string means
        REJECT — the caller re-runs the replan with it injected and executes that without asking
        again."""
        guidance = self._replan_approval_guidance(
            previous_plan, proposed_plan, pool, reason, missing, round_idx
        )
        instructions = (
            "Review the proposed plan. Approve it to execute as-is, or reject with feedback "
            f"telling the replanner what to change. Compute reported: {reason}"
        )
        if missing:
            instructions += f"\nStill missing: {', '.join(missing)}"
        self._ctx.emit(
            f"replan_approval_request round={round_idx} missing={missing!r} "
            f"proposed_branches={len(proposed_plan.branches)}",
            kind="user",
        )
        response = await handler(
            "replan_approval",
            instructions,
            f"Question: {self._ctx.question}",
            [],
            guidance,
        )
        feedback = str(response.get("response", "")).strip()
        self._ctx.emit(
            f"replan_approval_resolved round={round_idx} "
            f"decision={'rejected' if feedback else 'approved'}",
            kind="observation",
        )
        return feedback

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
                # Optimistic mode reviews the data-prep output once per round (see
                # `_prep_pool`), not each extract. Blocking mode (local CLI) still verifies the
                # value(s) per branch in place when the policy opts in.
                if (
                    self._review_mode() == "blocking"
                    and self._human.wants_verify(
                        cast(RetrieveBranch, branch), res, self._ctx
                    )
                ):
                    pages = _branch_pages(pos)
                    res = await traced_step(
                        self._ctx,
                        "human_verify",
                        lambda: self._human.verify_extract(
                            res, pages, cast(RetrieveBranch, branch), self._ctx
                        ),
                        branch_id=bid,
                    )
                return res
            # External lookup. Its value(s) are rolled into the data-prep agent's output and
            # reviewed (when HITL is on) in the single data-prep pool review — there is no
            # per-lookup human hook.
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
