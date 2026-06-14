from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast
from skunk.compute import ComputeOp
from skunk.config import SkunkConfig
from skunk.errors import MissingData, StepFailed
from skunk.extract import ExtractOp
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
    ExecutionContext,
    HumanInterventionHandler,
    HumanReviewRegister,
    traced_step,
)
from skunk.llm_client import LLMClient
from skunk.plan import Branch, Plan, PlanDiff, Planner, RetrieveBranch
from skunk.prompted_call import PromptOverride
from skunk.question_explainer import ConceptExplanation, QuestionExplainer
from skunk.retrieve import RetrieveOp
from skunk.result import ExecutionResult

if TYPE_CHECKING:
    from skunk.search_agent import SearchAgent


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
    original entries. No re-plan / re-retrieve / re-extract."""
    merged: list[AnnotatedValue] = []
    for bid in state.order:
        cached = state.entries_by_branch.get(bid, [])
        if bid in overrides:
            merged.extend(apply_overrides(overrides[bid], cached))
        else:
            merged.extend(cached)
    merged.extend(state.extra_entries)
    return await ComputeOp().run(merged, ctx, concept_explanations=state.explanations)


def _merge_branch_outcome(previous: BranchOutcome, new: BranchOutcome) -> BranchOutcome:
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
        # current plan's branch `i`; a replan carries kept branches' ids forward and
        # allocates fresh ids for added branches (`PlanDiff.apply` makes new branch
        # COPIES, so positional indices are not stable across revisions). The trace
        # viewer keys plan-branch ↔ operator-step on this id to render revision tabs.
        self._branch_ids: list[int] = []
        self._next_branch_id = 0
        self._last_executed_branch_ids: set[int] = set()
        self._planner = Planner()
        self._retrieve = RetrieveOp(self._ctx.config)
        self._extract = ExtractOp()
        self._lookup = LookupExternalOp()
        # Human-in-the-loop middleware (inert unless a SKUNK_HUMAN_* flag is set). Gates on
        # ctx.config per call, so constructing it unconditionally is free when disabled.
        # Channel by transport: under the competition server a handler is present → route
        # human requests through the async broker/web UI; for a local CLI run fall back to
        # the blocking console. (Selecting by handler-presence also keeps the SKUNK_HUMAN_*
        # flags from hanging a server worker on stdin.)
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
        # `retrieved_blocks` so the eval harness can cache and replay a run without re-paying
        # retrieval. Empty under golden/replay bypass (retrieve never runs).
        self._retrieved_blocks: list[BlockRef] = []
        # Live SearchAgent per retrieve branch, keyed by stable branch id (search_agent backend
        # only). Registered as each branch's agent is built (see RetrieveOp._run_search_agent) so
        # that on a downstream MissingData we can resume the SAME agent with the missing-data
        # feedback and a fresh step budget — instead of replanning a brand-new agent from scratch.
        self._retrieve_agents: dict[int, "SearchAgent"] = {}
        # Snapshot for an optimistic-review recompute (the per-branch entries that produced the
        # answer); set on every successful compute, None until then. The server stores it on the
        # task so a later human resolve can revise the answer via `recompute_answer`.
        self._recompute_state: RecomputeState | None = None

    @property
    def ctx(self) -> ExecutionContext:
        return self._ctx

    @property
    def recompute_state(self) -> RecomputeState | None:
        """The snapshot needed to revise this answer after a human review resolves (None until a
        compute has succeeded). See `RecomputeState` / `recompute_answer`."""
        return self._recompute_state

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
            self._last_executed_branch_ids = set(self._branch_ids)
            self._emit_plan(plan, "initial")
            outcomes = await self._run_branches(plan.branches, self._branch_ids)
            explanations = await explain_task
        finally:
            explain_task.cancel()

        attempt = 0
        human_entries: list[AnnotatedValue] = []
        human_resolutions: list[list[str]] = []
        while True:
            self._current_plan = plan
            entries = [
                *[e for o in outcomes if o.entries for e in o.entries],
                *human_entries,
            ]
            # Snapshot BEFORE compute so a human review can revise even a task whose compute
            # fails (the snapshot holds the branch entries; re-running compute over a human's
            # correction may then succeed). Captured each loop iteration to reflect the latest
            # entries/recovery state.
            self._capture_recompute_state(outcomes, human_entries, explanations)
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
                handler = self._ctx.human_intervention_handler
                if handler is None:
                    # Autonomously replan. The planner returns a PlanDiff that may add/drop
                    # branches AND/OR resume an existing retrieve branch with its own feedback
                    # (`_apply_diff` runs the resumes against the retained agents). Bounded by
                    # recovery_max_rounds exactly like the human-assisted path below.
                    failed = [(o.branch, o.error) for o in outcomes if o.error]
                    diff = await traced_step(
                        self._ctx,
                        "replanner",
                        lambda: self._planner.replan(
                            self._ctx, plan, entries, failed, reason, missing, []
                        ),
                    )
                    plan, outcomes = await self._apply_diff(plan, outcomes, diff)
                    self._emit_plan(
                        plan,
                        "replan",
                        reason=reason,
                        missing=missing,
                        recovery_round=attempt,
                    )
                    continue
                self._emit_plan(
                    plan,
                    "human_pending",
                    reason=reason,
                    missing=missing,
                    recovery_round=attempt,
                )
                likely_pages: list[dict[str, str | int | None]] = []
                seen_pages: set[tuple[str | None, int | None]] = set()
                for entry in entries:
                    for page in entry.pages:
                        key = (entry.bulletin, page)
                        if key not in seen_pages:
                            seen_pages.add(key)
                            likely_pages.append(
                                {"bulletin": entry.bulletin, "page": page}
                            )
                for block in self._retrieved_blocks:
                    for ref in block.member_refs:
                        key = (ref.month, ref.page)
                        if key not in seen_pages:
                            seen_pages.add(key)
                            likely_pages.append(
                                {"bulletin": ref.month, "page": ref.page}
                            )
                likely_pages = likely_pages[:8]
                source_docs = [
                    f"Treasury Bulletin {page['bulletin']} PDF page {page['page']}"
                    for page in likely_pages
                    if page["bulletin"] and page["page"] is not None
                ]
                guidance = {
                    "recovery_round": attempt,
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
                                else {
                                    "target": branch.target,
                                    "src": branch.src,
                                }
                            ),
                            "blocks": [
                                {
                                    "bulletin": block_ref.page.month,
                                    "page": block_ref.page.page,
                                    "block_index": block_ref.block_index,
                                    "member_pages": [
                                        {"bulletin": ref.month, "page": ref.page}
                                        for ref in block_ref.member_refs
                                    ],
                                    "kind": (
                                        block_ref.block.kind
                                        if block_ref.block
                                        else None
                                    ),
                                    "title": (
                                        block_ref.block.title
                                        if block_ref.block
                                        else None
                                    ),
                                    "column_headers": (
                                        block_ref.block.column_headers
                                        if block_ref.block
                                        else []
                                    ),
                                    "row_headers": (
                                        block_ref.block.row_headers
                                        if block_ref.block
                                        else []
                                    ),
                                    "summary": (
                                        block_ref.block.summary
                                        if block_ref.block
                                        else None
                                    ),
                                }
                                for block_ref in outcome.blocks
                            ],
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
                                            "source_block_page": (
                                                entry.source_block_page
                                            ),
                                            "source_block_index": (
                                                entry.source_block_index
                                            ),
                                        }
                                        for entry in outcome.entries
                                    ],
                                }
                                if outcome.entries is not None
                                else None
                            ),
                            "output_step": (
                                "extract"
                                if branch.kind == "retrieve"
                                else "lookup_external"
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
                                str(outcome.error)
                                if outcome.error is not None
                                else None
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
                        {
                            "branch_id": branch_id,
                            **branch.model_dump(mode="json"),
                        }
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
                retrieval_directives: dict[int, list[str]] = {}
                current_retrieve_ids = {
                    branch_id
                    for branch_id, branch in zip(self._branch_ids, plan.branches)
                    if branch.kind == "retrieve"
                }
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
                    bulletins = []
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
                if retrieval_directives:
                    rerun_positions = [
                        i
                        for i, (branch_id, branch) in enumerate(
                            zip(self._branch_ids, plan.branches)
                        )
                        if branch_id in retrieval_directives
                        and branch.kind == "retrieve"
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
                        for position, outcome in zip(rerun_positions, rerun_outcomes):
                            outcomes[position] = _merge_branch_outcome(
                                outcomes[position], outcome
                            )
                        self._last_executed_branch_ids = set(rerun_ids)
                        self._ctx.emit(
                            "human_directed_retrieval",
                            kind="observation",
                            data={
                                "directives": [
                                    {
                                        "branch_id": branch_id,
                                        "bulletins": retrieval_directives[branch_id],
                                    }
                                    for branch_id in rerun_ids
                                ]
                            },
                        )
                if response_text:
                    description = (
                        f"Human-provided value for: {', '.join(missing)}"
                        if missing
                        else f"Human-provided information for: {reason}"
                    )
                    if response_sources:
                        description += f" (sources: {', '.join(response_sources)})"
                    human_entry = AnnotatedValue(
                        description=description,
                        value=response_text,
                    )
                    human_entries.append(human_entry)
                    human_resolutions.append(list(missing))
                entries = [
                    *[
                        entry
                        for outcome in outcomes
                        if outcome.entries
                        for entry in outcome.entries
                    ],
                    *human_entries,
                ]
                self._ctx.emit(
                    "mandatory_human_intervention_resolved",
                    kind="observation",
                    data={
                        "response": response_text,
                        "source_docs": response_sources,
                        "retrieval_directives": [
                            {
                                "branch_id": branch_id,
                                "bulletins": bulletins,
                            }
                            for branch_id, bulletins in retrieval_directives.items()
                        ],
                        "missing": missing,
                    },
                )
                self._capture_recompute_state(outcomes, human_entries, explanations)
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
                except MissingData as post_human_error:
                    reason = post_human_error.reason
                    missing = post_human_error.missing
                    if retrieval_directives:
                        self._ctx.emit(
                            "human_annotation_retry_incomplete",
                            kind="observation",
                            data={
                                "recovery_round": attempt,
                                "branch_ids": sorted(retrieval_directives),
                                "reason": reason,
                                "missing": missing,
                            },
                        )
                        # A branch annotation is itself the recovery action for this
                        # round: it reruns the same stable branch with updated retrieval
                        # context. Only a later, unannotated round may structurally
                        # rewrite the plan.
                        attempt += 1
                        if attempt > self._ctx.config.recovery_max_rounds:
                            raise

                self._emit_plan(
                    plan,
                    "replan_pending",
                    reason=reason,
                    missing=missing,
                    recovery_round=attempt,
                )
                failed = [(o.branch, o.error) for o in outcomes if o.error]
                first_human_index = len(entries) - len(human_entries)
                human_resolution_refs = [
                    (first_human_index + i, resolved_missing)
                    for i, resolved_missing in enumerate(human_resolutions)
                ]
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
                        human_resolution_refs,
                    ),
                )
                self._ctx.human_intervention_enabled = (
                    attempt == 1 and self._ctx.human_intervention_handler is not None
                )
                try:
                    plan, outcomes = await self._apply_diff(plan, outcomes, diff)
                finally:
                    self._ctx.human_intervention_enabled = False
                self._emit_plan(
                    plan,
                    "replan",
                    reason=reason,
                    missing=missing,
                    recovery_round=attempt,
                )

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
        already-gathered outcomes forward, runs any planner-chosen `resume` ops against the
        kept branches' retained agents, runs the added branches, and appends them. Compute
        therefore never sees data from a dropped branch. Stable branch ids ride along: kept
        branches keep theirs, added branches get fresh ones."""
        new_plan, kept = diff.apply(plan)
        dropped = sorted(set(range(len(plan.branches))) - set(kept))
        if dropped:
            self._ctx.emit(f"replan_dropped n_dropped={len(dropped)} indices={dropped}")
        kept_outcomes = [outcomes[i] for i in kept]
        kept_ids = [self._branch_ids[i] for i in kept]

        # Planner-chosen resume: continue a kept retrieve branch's own agent with the
        # planner's feedback (in parallel), replacing that branch's outcome in place. Resume
        # ops on a dropped branch, an unknown index, or a branch with no retained agent (e.g.
        # a lookup branch) are skipped with a note rather than failing the recovery round.
        kept_pos_by_prior = {prior_i: j for j, prior_i in enumerate(kept)}
        resume_targets: list[tuple[int, int, str]] = []  # (kept_pos, branch_id, feedback)
        for op in diff.resume:
            j = kept_pos_by_prior.get(op.index)
            bid = kept_ids[j] if j is not None else None
            if j is None or bid not in self._retrieve_agents:
                self._ctx.emit(
                    f"resume_skipped index={op.index} "
                    f"reason={'dropped/unknown index' if j is None else 'no retained agent'!r}"
                )
                continue
            resume_targets.append((j, bid, op.feedback))
        if resume_targets:
            resumed = await asyncio.gather(
                *(
                    self._resume_one(kept_outcomes[j], self._retrieve_agents[bid], fb, bid)
                    for j, bid, fb in resume_targets
                )
            )
            for (j, _, _), outcome in zip(resume_targets, resumed):
                kept_outcomes[j] = outcome

        added_ids = self._alloc_branch_ids(len(diff.add))
        self._branch_ids = [*kept_ids, *added_ids]
        self._last_executed_branch_ids = set(added_ids) | {
            bid for _, bid, _ in resume_targets
        }
        new_outcomes = await self._run_branches(diff.add, added_ids) if diff.add else []
        return new_plan, [*kept_outcomes, *new_outcomes]

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
        document_scopes: list[list[str] | None] | None = None,
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
                    await self._retrieve.run_all(
                        self._ctx,
                        branches,
                        branch_ids,
                        document_scopes=document_scopes,
                        agent_sink=self._retrieve_agents,
                    )
                )
            else:
                docs = list(
                    await traced_step(
                        self._ctx,
                        "retrieve",
                        lambda: self._retrieve.run_all(
                            self._ctx,
                            branches,
                            branch_ids,
                            document_scopes=document_scopes,
                        ),
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

    @staticmethod
    def _extract_summary_metadata(doc: list[BlockRef]) -> dict:
        """Per-block metadata attached to an `extract` step for the trace viewer."""
        return {
            "blocks": [
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
                for block_ref in doc
            ]
        }

    async def _resume_one(
        self, prior: BranchOutcome, agent: "SearchAgent", feedback: str, bid: int
    ) -> BranchOutcome:
        """Continue one retrieve branch's `agent` with the planner's `feedback` (fresh, smaller
        step budget), re-run extract over the UNION of its prior + newly-found blocks, and
        return the replacement outcome. A resume/extract failure keeps the entries + blocks the
        branch already had (so a previously successful branch never loses data it found) while
        carrying the new error."""
        branch = prior.branch
        try:
            new_blocks = await traced_step(
                self._ctx,
                "retrieve",
                lambda: self._retrieve.resume_search_agent(self._ctx, agent, feedback),
                branch_id=bid,
            )
        except StepFailed as e:
            return BranchOutcome(
                branch=branch, entries=prior.entries, error=e, blocks=prior.blocks
            )
        # Union prior + resumed blocks (first-seen order), and extend the deduped
        # corpus-wide union the cache/replay seam reads from.
        union: list[BlockRef] = list(prior.blocks)
        seen = set(union)
        corpus_seen = set(self._retrieved_blocks)
        for blk in new_blocks:
            if blk not in seen:
                seen.add(blk)
                union.append(blk)
            if blk not in corpus_seen:
                corpus_seen.add(blk)
                self._retrieved_blocks.append(blk)
        try:
            entries = await traced_step(
                self._ctx,
                "extract",
                lambda: self._extract.run(union, self._ctx, branch),
                branch_id=bid,
                summary_metadata=self._extract_summary_metadata(union),
            )
        except StepFailed as e:
            return BranchOutcome(
                branch=branch, entries=prior.entries, error=e, blocks=union
            )
        return BranchOutcome(branch=branch, entries=entries, error=None, blocks=union)

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

    def _review_mode(self) -> str:
        """How a wanted human review is serviced for this run:
        - "optimistic": a register hook is wired (under the competition server) — register an
          open review and keep the LLM result, non-blocking. Whether a review is wanted at all
          is the per-stage policy's job (the SKUNK_HUMAN_* gates); this only picks the transport.
        - "blocking":   no register hook (local CLI) — await the human on the console as before.
        """
        if self._ctx.human_review_register is None:
            return "blocking"
        return "optimistic"

    async def _run_branches(
        self,
        branches: list[Branch],
        branch_ids: list[int],
        *,
        document_scopes: dict[int, list[str]] | None = None,
    ) -> list[BranchOutcome]:
        # Global retrieve phase: all retrieve branches share one deduped semantic-filter
        # sweep, then each branch's routed refs feed its own extract. Lookup branches are
        # independent and run in the per-branch tail below.
        retrieve_pos = [i for i, b in enumerate(branches) if b.kind == "retrieve"]
        docs = await self._run_retrieve_phase(
            [cast(RetrieveBranch, branches[i]) for i in retrieve_pos],
            [branch_ids[i] for i in retrieve_pos],
            document_scopes=[
                (document_scopes or {}).get(branch_ids[i]) for i in retrieve_pos
            ],
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
                    summary_metadata=self._extract_summary_metadata(doc),
                )
                # Human verifies/produces the extracted value(s) (figure or OCR/table read)
                # only when the policy opts in — no step (or prompt) on the default path.
                if self._human.wants_verify(branch, entries, self._ctx):
                    mode = self._review_mode()
                    if mode == "optimistic":
                        # Register an open review and keep the LLM entries — the question
                        # completes without blocking; a human resolve drives a recompute later.
                        self._human.register_verify(
                            entries, doc, branch, bid, self._ctx
                        )
                    elif mode == "blocking":
                        entries = await traced_step(
                            self._ctx,
                            "human_verify",
                            lambda: self._human.verify_extract(
                                entries, doc, branch, self._ctx
                            ),
                            branch_id=bid,
                        )
                return entries
            # External lookup. Optimistic: always run the lookup agent, then register a review
            # of its result (the human confirms/corrects). Blocking (local CLI): the human
            # performs the lookup in place when the flag is on; else the agent does.
            mode = self._review_mode()
            if self._human.wants_lookup(branch, self._ctx) and mode == "optimistic":
                entries = await traced_step(
                    self._ctx,
                    "lookup_external",
                    lambda: self._lookup.run(self._ctx, branch),
                    branch_id=bid,
                )
                self._human.register_lookup(entries, branch, bid, self._ctx)
                return entries
            if self._human.wants_lookup(branch, self._ctx) and mode == "blocking":
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
        for pos, (bid, branch, res) in enumerate(zip(branch_ids, branches, results)):
            selected = docs_by_pos.get(pos)
            blocks = selected if isinstance(selected, list) else []
            if isinstance(res, StepFailed):
                self._ctx.emit(
                    f"parallel_branch_failed branch_id={bid} error={str(res)!r}",
                    data={"branch_id": bid},
                )
                outcomes.append(
                    BranchOutcome(
                        branch=branch,
                        entries=None,
                        error=res,
                        blocks=blocks,
                    )
                )
            elif isinstance(res, BaseException):
                raise res
            else:
                outcomes.append(
                    BranchOutcome(
                        branch=branch,
                        entries=res,
                        error=None,
                        blocks=blocks,
                    )
                )
        return outcomes
