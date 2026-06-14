from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import cast
from skunk.compute import ComputeOp
from skunk.config import SkunkConfig
from skunk.errors import MissingData, StepFailed
from skunk.human import (
    BrokerChannel,
    ConsoleChannel,
    HumanAssist,
    apply_overrides,
)
from skunk.lookup_external import LookupExternalOp
from skunk.common import (
    AnnotatedValue,
    BlockRef,
    BranchRetrieval,
    ExecutionContext,
    Final,
    HumanInterventionHandler,
    HumanReviewRegister,
    NeedsMore,
    traced_step,
)
from skunk.llm_client import LLMClient
from skunk.plan import (
    AttemptRecord,
    Branch,
    LookupBranch,
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
    blocks: list[BlockRef] = field(default_factory=list)


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


def _merge_branch_outcome(previous: BranchOutcome, new: BranchOutcome) -> BranchOutcome:
    """Merge a human-directed re-retrieval's outcome into a branch's prior outcome: take the
    new outcome on any failure, else union both sets of entries + blocks (deduped)."""
    if previous.error is not None or previous.entries is None:
        return new
    if new.error is not None or new.entries is None:
        return new
    entries: list[AnnotatedValue] = []
    seen_entries: set[str] = set()
    for entry in [*previous.entries, *new.entries]:
        key = entry.model_dump_json()
        if key not in seen_entries:
            seen_entries.add(key)
            entries.append(entry)
    blocks = list(dict.fromkeys([*previous.blocks, *new.blocks]))
    return BranchOutcome(
        branch=new.branch,
        entries=entries,
        error=None,
        blocks=blocks,
    )


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
        # Branch ids actually executed in the latest sweep (vs carried forward) — drives the
        # `execution_status` the HITL guidance shows a reviewer.
        self._last_executed_branch_ids: set[int] = set()
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
        self._result = ExecutionResult(question=question)
        # Deduped union of every BlockRef the retrieve phase produced this question (first-seen
        # order, accumulated across the initial sweep and any replan sweeps). Exposed via
        # `retrieved_blocks` for the eval harness's `likely_pages` / retrieval-recall reporting.
        # Empty under golden bypass (retrieve never runs).
        self._retrieved_blocks: list[BlockRef] = []
        # Snapshot for an optimistic-review recompute (the per-branch entries that produced the
        # answer); set on every compute attempt, None until then. The server stores it on the
        # task so a later human resolve can revise the answer via `recompute_answer`.
        self._recompute_state: RecomputeState | None = None

    @property
    def ctx(self) -> ExecutionContext:
        return self._ctx

    @property
    def retrieved_blocks(self) -> list[BlockRef]:
        """Deduped union of blocks from every retrieve sweep this question (empty under
        golden bypass). Pages are derivable via each block's `member_refs`."""
        return self._retrieved_blocks

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
        # Human-recovery state: values a human supplied during mandatory intervention
        # (kept in the pool like any other gathered value), and the missing-id lists each
        # resolved — threaded to the replanner so it treats those ids as answered.
        human_entries: list[AnnotatedValue] = []
        human_resolutions: list[list[str]] = []
        needs: NeedsMore | None = None
        round_idx = 0
        sweep_added = True  # the initial sweep always reaches compute

        while True:
            self._current_plan = plan
            # Snapshot BEFORE compute so a human review can revise even a task whose compute
            # fails (re-running compute over a human correction may then succeed).
            self._capture_recompute_state(outcomes, human_entries, explanations)
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

            # === HITL: mandatory human intervention on missing data ===
            # When a handler is wired, ask the human for the missing data (and optionally let
            # them scope a branch's source documents for a re-retrieval) BEFORE spending a
            # replan. If their input alone resolves compute, return without replanning.
            handler = self._ctx.human_intervention_handler
            if handler is not None:
                self._emit_plan(
                    plan,
                    "human_pending",
                    reason=reason,
                    missing=missing,
                    recovery_round=round_idx,
                )
                entries_snapshot = [
                    *[e for o in outcomes if o.entries for e in o.entries],
                    *human_entries,
                ]
                outcomes, directed_entries, human_entry = (
                    await self._request_missing_from_human(
                        handler, plan, outcomes, entries_snapshot, reason, missing, round_idx
                    )
                )
                pool += directed_entries
                if human_entry is not None:
                    pool.append(human_entry)
                    human_entries.append(human_entry)
                    human_resolutions.append(list(missing))
                if directed_entries or human_entry is not None:
                    # Try compute over the human-augmented pool before spending a replan.
                    self._capture_recompute_state(outcomes, human_entries, explanations)
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
                    pool = list(needs.keep)
                    reason, missing = needs.missing_reason, needs.missing

            # Map each human-provided value still in the pool to the missing ids it resolved,
            # so the replanner treats those ids as answered (identity-keyed: keep-filtering
            # retains the same objects, so id() survives across rounds).
            pool_pos = {id(e): i for i, e in enumerate(pool)}
            human_resolution_refs = [
                (pool_pos[id(he)], resolved)
                for he, resolved in zip(human_entries, human_resolutions)
                if id(he) in pool_pos
            ]

            plan = await traced_step(
                self._ctx,
                "replanner",
                lambda: self._planner.replan(
                    self._ctx,
                    pool,
                    attempts,
                    reason,
                    missing,
                    human_resolution_refs,
                ),
            )
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

    def _capture_recompute_state(
        self,
        outcomes: list[BranchOutcome],
        human_entries: list[AnnotatedValue],
        explanations: list[ConceptExplanation],
    ) -> None:
        """Snapshot the per-branch entries (keyed by stable branch id) + recovery entries +
        explanations that produced this answer, so a later human-review resolve can recompute."""
        self._recompute_state = RecomputeState(
            question=self._ctx.question,
            order=list(self._branch_ids),
            entries_by_branch={
                bid: list(outcome.entries or [])
                for bid, outcome in zip(self._branch_ids, outcomes)
            },
            extra_entries=list(human_entries),
            explanations=list(explanations),
        )

    def _missing_data_guidance(
        self,
        plan: Plan,
        outcomes: list[BranchOutcome],
        entries: list[AnnotatedValue],
        reason: str,
        missing: list[str],
        round_idx: int,
    ) -> tuple[dict, list[dict[str, str | int | None]]]:
        """Build the rich per-branch context a human reviewer needs to supply missing data,
        plus the `likely_pages` (the question's gathered + retrieved source pages, capped)."""
        likely_pages: list[dict[str, str | int | None]] = []
        seen_pages: set[tuple[str | None, int | None]] = set()
        for entry in entries:
            for page in entry.pages:
                key = (entry.bulletin, page)
                if key not in seen_pages:
                    seen_pages.add(key)
                    likely_pages.append({"bulletin": entry.bulletin, "page": page})
        for block in self._retrieved_blocks:
            for ref in block.member_refs:
                key = (ref.month, ref.page)
                if key not in seen_pages:
                    seen_pages.add(key)
                    likely_pages.append({"bulletin": ref.month, "page": ref.page})
        likely_pages = likely_pages[:8]

        def _blocks_payload(blocks: list[BlockRef]) -> list[dict]:
            return [
                {
                    "bulletin": block_ref.page.month,
                    "page": block_ref.page.page,
                    "block_index": block_ref.block_index,
                    "member_pages": [
                        {"bulletin": ref.month, "page": ref.page}
                        for ref in block_ref.member_refs
                    ],
                    "kind": block_ref.block.kind if block_ref.block else None,
                    "title": block_ref.block.title if block_ref.block else None,
                    "column_headers": (
                        block_ref.block.column_headers if block_ref.block else []
                    ),
                    "row_headers": (
                        block_ref.block.row_headers if block_ref.block else []
                    ),
                    "summary": block_ref.block.summary if block_ref.block else None,
                }
                for block_ref in blocks
            ]

        guidance = {
            "recovery_round": round_idx,
            "previous_round_plan": [
                {
                    "branch_id": branch_id,
                    **branch.model_dump(mode="json"),
                    "searched": (
                        {
                            "key": branch.key,
                            "period": branch.period,
                            "as_of": branch.as_of,
                            "visual_only": branch.visual_only,
                        }
                        if branch.kind == "retrieve"
                        else {"target": branch.target, "src": branch.src}
                    ),
                    "blocks": _blocks_payload(outcome.blocks),
                    "output": (
                        {
                            "type": "values",
                            "values": [
                                {
                                    "description": entry.description,
                                    "unit": entry.unit,
                                    "value_kind": entry.kind,
                                    "value": entry.value,
                                    "index_name": entry.index_name,
                                    "row_name": entry.row_name,
                                    "col_name": entry.col_name,
                                    "bulletin": entry.bulletin,
                                    "pages": list(entry.pages),
                                    "source_block_page": entry.source_block_page,
                                    "source_block_index": entry.source_block_index,
                                }
                                for entry in outcome.entries
                            ],
                        }
                        if outcome.entries is not None
                        else None
                    ),
                    "output_step": (
                        "extract" if branch.kind == "retrieve" else "lookup_external"
                    ),
                    "status": ("failed" if outcome.error is not None else "ok"),
                    "execution_status": (
                        "executed"
                        if branch_id in self._last_executed_branch_ids
                        else "carried_forward"
                    ),
                    "outcome_status": (
                        "failed" if outcome.error is not None else "succeeded"
                    ),
                    "error": (
                        str(outcome.error) if outcome.error is not None else None
                    ),
                    "considered_pages": (
                        outcome.error.details.get("considered_pages", [])
                        if outcome.error is not None
                        else []
                    ),
                }
                for branch_id, branch, outcome in zip(
                    self._branch_ids, plan.branches, outcomes
                )
            ],
            "partial_plan": [
                {"branch_id": branch_id, **branch.model_dump(mode="json")}
                for branch_id, branch in zip(self._branch_ids, plan.branches)
            ],
            "gathered_values": [
                {
                    "description": entry.description,
                    "value": entry.value,
                    "unit": entry.unit,
                    "bulletin": entry.bulletin,
                    "pages": list(entry.pages),
                }
                for entry in entries
            ],
            "failed_branches": [
                {
                    "branch_id": branch_id,
                    "branch": outcome.branch.model_dump(mode="json"),
                    "error": str(outcome.error),
                    "considered_pages": outcome.error.details.get(
                        "considered_pages", []
                    ),
                }
                for branch_id, outcome in zip(self._branch_ids, outcomes)
                if outcome.error is not None
            ],
            "likely_pages": likely_pages,
            "missing": missing,
            "reason": reason,
        }
        return guidance, likely_pages

    async def _request_missing_from_human(
        self,
        handler: HumanInterventionHandler,
        plan: Plan,
        outcomes: list[BranchOutcome],
        entries: list[AnnotatedValue],
        reason: str,
        missing: list[str],
        round_idx: int,
    ) -> tuple[list[BranchOutcome], list[AnnotatedValue], AnnotatedValue | None]:
        """Ask the human for the missing data (mandatory intervention). Optionally re-runs
        human-directed retrieval (scoping named branches to named bulletins), merging the
        result into `outcomes`. Returns the updated outcomes, any new pool entries from the
        directed retrieval, and an optional human-authored `AnnotatedValue`. Raises on an
        empty response."""
        guidance, likely_pages = self._missing_data_guidance(
            plan, outcomes, entries, reason, missing, round_idx
        )
        source_docs = [
            f"Treasury Bulletin {page['bulletin']} PDF page {page['page']}"
            for page in likely_pages
            if page["bulletin"] and page["page"] is not None
        ]
        instructions = (
            "Provide the missing information needed to answer the question. "
            f"Failure reason: {reason}"
        )
        if missing:
            instructions += f"\nMissing values: {', '.join(missing)}"
        response = await handler(
            "missing_data",
            instructions,
            f"Question: {self._ctx.question}",
            source_docs,
            guidance,
        )
        response_text = str(response.get("response", "")).strip()
        current_retrieve_ids = {
            branch_id
            for branch_id, branch in zip(self._branch_ids, plan.branches)
            if branch.kind == "retrieve"
        }
        retrieval_directives: dict[int, list[str]] = {}
        for item in response.get("retrieval_directives", []):
            if not isinstance(item, dict):
                continue
            branch_id = item.get("branch_id")
            documents = item.get("documents")
            if (
                not isinstance(branch_id, int)
                or branch_id not in current_retrieve_ids
                or not isinstance(documents, list)
            ):
                continue
            bulletins: list[str] = []
            for document in documents:
                match = re.fullmatch(
                    r"Treasury Bulletin (\d{4}-(?:0[1-9]|1[0-2])) PDF",
                    str(document).strip(),
                )
                if match and match.group(1) not in bulletins:
                    bulletins.append(match.group(1))
            if bulletins:
                retrieval_directives[branch_id] = bulletins
        if not response_text and not retrieval_directives:
            raise RuntimeError("human intervention returned an empty response")
        response_sources = [
            str(item).strip()
            for item in response.get("source_docs", [])
            if str(item).strip()
        ]

        new_entries: list[AnnotatedValue] = []
        if retrieval_directives:
            rerun_positions = [
                i
                for i, (branch_id, branch) in enumerate(
                    zip(self._branch_ids, plan.branches)
                )
                if branch_id in retrieval_directives and branch.kind == "retrieve"
            ]
            if rerun_positions:
                rerun_branches = [plan.branches[i] for i in rerun_positions]
                rerun_ids = [self._branch_ids[i] for i in rerun_positions]
                rerun_outcomes = await self._run_branches(
                    rerun_branches,
                    rerun_ids,
                    document_scopes=retrieval_directives,
                )
                outcomes = list(outcomes)
                for position, rerun in zip(rerun_positions, rerun_outcomes):
                    merged = _merge_branch_outcome(outcomes[position], rerun)
                    outcomes[position] = merged
                    for e in rerun.entries or []:
                        new_entries.append(e)
                self._last_executed_branch_ids = set(rerun_ids)
                self._ctx.emit(
                    "human_directed_retrieval",
                    kind="observation",
                    data={
                        "directives": [
                            {"branch_id": bid, "bulletins": retrieval_directives[bid]}
                            for bid in rerun_ids
                        ]
                    },
                )

        human_entry: AnnotatedValue | None = None
        if response_text:
            description = (
                f"Human-provided value for: {', '.join(missing)}"
                if missing
                else f"Human-provided information for: {reason}"
            )
            if response_sources:
                description += f" (sources: {', '.join(response_sources)})"
            human_entry = AnnotatedValue(description=description, value=response_text)

        self._ctx.emit(
            "mandatory_human_intervention_resolved",
            kind="observation",
            data={
                "response": response_text,
                "source_docs": response_sources,
                "retrieval_directives": [
                    {"branch_id": bid, "bulletins": bulletins}
                    for bid, bulletins in retrieval_directives.items()
                ],
                "missing": missing,
            },
        )
        return outcomes, new_entries, human_entry

    async def _run_retrieve_phase(
        self,
        branches: list[RetrieveBranch],
        branch_ids: list[int],
        document_scopes: list[list[str] | None] | None = None,
    ) -> list[BranchRetrieval | StepFailed]:
        """Unified multi-scan retrieve for every retrieve branch at once: their candidate
        pages are deduped and the LLM semantic filter scans each unique page at most once,
        judging it against all branches' targets, then routes the survivors back per branch.
        Returns one normalized `BranchRetrieval` per branch (or the `StepFailed` to attribute
        to it) — `RetrieveOp.run_all` owns all backend dispatch and the `pre_selected` flag.
        `branch_ids` lets the search-agent backend emit a per-branch `retrieve` step;
        `document_scopes` hard-scopes a branch's corpus to human-required bulletins (HITL).
        The shared page-index `retrieve` sweep carries no `branch_id`."""
        if not branches:
            return []
        try:
            results = await traced_step(
                self._ctx,
                "retrieve",
                lambda: self._retrieve.run_all(
                    self._ctx, branches, branch_ids, document_scopes=document_scopes
                ),
            )
        except StepFailed as e:
            return [e] * len(branches)

        # Accumulate the deduped block union (for `likely_pages` / retrieval-recall reporting).
        # Replan sweeps extend the same list, so rebuild the seen-set from the current list each
        # call rather than carrying a persistent set that would outlive the sweep it was built for.
        seen_blocks = set(self._retrieved_blocks)
        for r in results:
            if isinstance(r, StepFailed):
                continue
            for blk in r.blocks:
                if blk not in seen_blocks:
                    seen_blocks.add(blk)
                    self._retrieved_blocks.append(blk)
        return results

    async def _run_branches(
        self,
        branches: list[Branch],
        branch_ids: list[int],
        *,
        document_scopes: dict[int, list[str]] | None = None,
    ) -> list[BranchOutcome]:
        # Global retrieve phase: all retrieve branches share one deduped semantic-filter
        # sweep, then each branch's routed refs feed its own extract. Lookup branches are
        # independent and run in the per-branch tail below. `document_scopes` (keyed by
        # stable branch id) hard-scopes a retrieve branch's corpus to human-required
        # bulletins — the HITL "annotate this branch's source documents" recovery action.
        self._last_executed_branch_ids = set(branch_ids)
        retrieve_pos = [i for i, b in enumerate(branches) if b.kind == "retrieve"]
        retrievals = await self._run_retrieve_phase(
            [cast(RetrieveBranch, branches[i]) for i in retrieve_pos],
            [branch_ids[i] for i in retrieve_pos],
            document_scopes=[
                (document_scopes or {}).get(branch_ids[i]) for i in retrieve_pos
            ],
        )
        retr_by_pos: dict[int, BranchRetrieval | StepFailed] = dict(
            zip(retrieve_pos, retrievals)
        )

        # Select → extract runs as ONE shared pipeline over all retrieve branches,
        # launched as a task so lookup branches proceed concurrently. Each retrieve
        # tail awaits the shared task and picks out its branch's result. Selection and
        # extraction are two stages (`run_select` then `run_extract`); each does its own
        # per-branch traced_steps, so the tail doesn't wrap them again.
        pipeline: asyncio.Task | None = None
        if retrieve_pos:
            from skunk.block_extract import run_extract
            from skunk.block_select import run_select

            sub_branches = [cast(RetrieveBranch, branches[i]) for i in retrieve_pos]
            sub_ids = [branch_ids[i] for i in retrieve_pos]
            sub_retrievals = [retr_by_pos[i] for i in retrieve_pos]

            async def _select_extract() -> list[list[AnnotatedValue] | StepFailed]:
                selections = await run_select(
                    self._ctx, sub_branches, sub_retrievals, sub_ids
                )
                return await run_extract(
                    self._ctx, sub_branches, selections, sub_ids
                )

            pipeline = asyncio.create_task(_select_extract())

        def _branch_blocks(pos: int) -> list[BlockRef]:
            r = retr_by_pos.get(pos)
            return list(r.blocks) if isinstance(r, BranchRetrieval) else []

        async def _tail(pos: int) -> list[AnnotatedValue]:
            branch, bid = branches[pos], branch_ids[pos]
            if branch.kind == "retrieve":
                assert pipeline is not None
                res = (await pipeline)[retrieve_pos.index(pos)]
                if isinstance(res, StepFailed):
                    raise res
                # Human verifies / produces the extracted value(s) (figure or OCR/table read)
                # only when the policy opts in — no step (or prompt) on the default path.
                if self._human.wants_verify(
                    cast(RetrieveBranch, branch), res, self._ctx
                ):
                    blocks = _branch_blocks(pos)
                    if self._review_mode() == "optimistic":
                        # Register an open review and keep the LLM entries — the question
                        # completes without blocking; a human resolve drives a recompute.
                        self._human.register_verify(
                            res, blocks, cast(RetrieveBranch, branch), bid, self._ctx
                        )
                    else:
                        res = await traced_step(
                            self._ctx,
                            "human_verify",
                            lambda: self._human.verify_extract(
                                res, blocks, cast(RetrieveBranch, branch), self._ctx
                            ),
                            branch_id=bid,
                        )
                return res
            # External lookup. Optimistic: run the lookup agent, then register a review of
            # its result. Blocking (local CLI): the human performs the lookup in place when
            # the flag is on; else the agent does.
            lookup_branch = cast("LookupBranch", branch)
            mode = self._review_mode()
            if self._human.wants_lookup(lookup_branch, self._ctx) and mode == "optimistic":
                entries = await traced_step(
                    self._ctx,
                    "lookup_external",
                    lambda: self._lookup.run(self._ctx, branch),
                    branch_id=bid,
                )
                self._human.register_lookup(entries, lookup_branch, bid, self._ctx)
                return entries
            if self._human.wants_lookup(lookup_branch, self._ctx) and mode == "blocking":
                return await traced_step(
                    self._ctx,
                    "human_lookup",
                    lambda: self._human.human_lookup(lookup_branch, self._ctx),
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
        for pos, (bid, (branch, res)) in enumerate(
            zip(branch_ids, zip(branches, results))
        ):
            blocks = _branch_blocks(pos)
            if isinstance(res, StepFailed):
                self._ctx.emit(
                    f"parallel_branch_failed branch_id={bid} error={str(res)!r}",
                    data={"branch_id": bid},
                )
                outcomes.append(
                    BranchOutcome(
                        branch=branch, entries=None, error=res, blocks=blocks
                    )
                )
            elif isinstance(res, BaseException):
                raise res
            else:
                outcomes.append(
                    BranchOutcome(
                        branch=branch, entries=res, error=None, blocks=blocks
                    )
                )
        return outcomes
