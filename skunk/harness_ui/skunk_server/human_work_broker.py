"""Optimistic human-review coordination (single-operator, slim).

The pipeline runs WITHOUT blocking on a human: as a task executes, table/vector extracts,
figure reads, and external lookups register an open `HumanReview` (via `register_review`,
wrapped into the agent's `human_review_register` hook). The task completes optimistically and
becomes submittable. When the operator resolves a review in the web UI (`resolve_review`), the
broker schedules a background RECOMPUTE: it re-runs only `compute` over the first attempt's
cached entries with the human's correction swapped in (no re-plan / re-retrieve), records the
revised answer as a superseding candidate, and best-effort resubmits if the round is still open.

This deliberately drops the old claim/release/assignment/worker-registry machinery — there is
one operator, and reviews are advisory (the LLM answer already stands).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from skunk_server.domain import AnswerCandidate, HumanReview, QuestionTask, TaskStatus
from skunk_server.task_registry import TaskRegistry

logger = logging.getLogger(__name__)

# (recompute_state_json, {branch_id: raw_response_json}) -> revised answer text.
RecomputeFn = Callable[[dict[str, Any], dict[int, str]], Awaitable[str]]
# task_id -> None; best-effort resubmit of the latest (revised) candidate.
SubmitFn = Callable[[str], Awaitable[None]]


class HumanWorkBroker:
    def __init__(
        self,
        registry: TaskRegistry,
        recompute_fn: RecomputeFn | None,
        submit_fn: SubmitFn,
        publish_status: Callable[[], None],
    ) -> None:
        self._registry = registry
        self._recompute_fn = recompute_fn
        self._submit_fn = submit_fn
        self._publish_status = publish_status
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ---- Registration (called from a worker loop, fire-and-forget) ---------------------

    def register_review(
        self,
        task_id: str,
        attempt_id: str,
        kind: str,
        instructions: str,
        context: str | None,
        source_docs: list[str] | None,
        guidance: dict[str, Any] | None,
    ) -> str | None:
        """Open a review and return its id WITHOUT blocking the branch. Safe to call from a
        worker thread/loop: the registry is locked and the status publish is scheduled on the
        server loop."""
        try:
            review = self._registry.create_review(
                task_id,
                attempt_id,
                kind,
                instructions,
                context,
                list(source_docs or []),
                guidance,
            )
        except KeyError:
            return None  # task vanished (round closed) — nothing to review
        self._notify()
        return review.review_id

    # ---- Resolution (called from the server loop, via the API route) -------------------

    def resolve_review(
        self,
        review_id: str,
        response: str,
        source_docs: list[str],
    ) -> tuple[QuestionTask, HumanReview]:
        task, review = self._registry.resolve_review(review_id, response, source_docs)
        # A non-empty correction on a task that completed a compute → recompute in the
        # background. Accept-as-is (empty response) keeps the optimistic answer untouched.
        if (
            (response or "").strip()
            and task.recompute_state is not None
            and self._recompute_fn is not None
        ):
            self._registry.set_revising(task.task_id, True)  # UI shows "Revising…"
            self._schedule(self._recompute_and_record(task.task_id, review.attempt_id))
        self._notify()
        return task, review

    async def _recompute_and_record(self, task_id: str, attempt_id: str) -> None:
        try:
            task = self._registry.get(task_id)
            if task is None or task.recompute_state is None:
                return
            overrides = self._registry.resolved_overrides(task_id)
            if not overrides:
                return
            try:
                answer = await self._recompute_fn(task.recompute_state, overrides)
            except Exception:
                logger.exception("recompute failed for %s", task_id)
                return
            # Source docs the human consulted, for the revised candidate's provenance.
            seen: set[str] = set()
            revised_docs: list[str] = []
            for review in task.reviews:
                for doc in [*review.source_docs, *review.response_source_docs]:
                    if doc and doc not in seen:
                        seen.add(doc)
                        revised_docs.append(doc)
            # Submission type stays "agent" (the default) — the answer is still mostly
            # LLM-generated; the human only corrected a value. As an AGENT submission, Cup
            # enforces a reasoning floor (AGENT_MIN_REASONING_CHARS=100), so the revised
            # candidate must carry a substantial reasoning: a note about the human revision
            # prepended to the prior answer's reasoning (the original structured payload). The
            # note alone clears the floor, so this is valid even for a task that had no prior
            # candidate (e.g. a FAILED attempt the human is fixing).
            prior = task.latest_candidate
            prior_reasoning = (
                prior.reasoning if prior and prior.reasoning else ""
            ).strip()
            note = (
                "Answer revised through human-in-the-loop review: a human reviewer confirmed or "
                "corrected the extracted/looked-up value(s) against the source pages, and the "
                "computation was re-run over the corrected inputs."
            )
            reasoning = f"{note}\n\n{prior_reasoning}" if prior_reasoning else note
            candidate = AnswerCandidate(
                attempt_id=attempt_id,
                answer_text=answer,
                reasoning=reasoning,
                source_docs=(revised_docs or (prior.source_docs if prior else []))[:64],
            )
            updated = self._registry.add_revised_candidate(task_id, candidate)
            # Best-effort resubmit: only while the round is open and the task is (re)submittable.
            if (
                updated is not None
                and updated.status == TaskStatus.READY
                and self._registry.round_state().status == "ACTIVE"
            ):
                try:
                    await self._submit_fn(task_id)
                except Exception:
                    logger.exception("best-effort resubmit failed for %s", task_id)
        finally:
            # Clear the revising flag on every path (success, no-op, or failure) and refresh.
            self._registry.set_revising(task_id, False)
            self._notify()

    def cancel_all(self) -> None:
        self._registry.cancel_active_reviews()
        self._notify()

    # ---- internals ---------------------------------------------------------------------

    def _schedule(self, coro: Awaitable[None]) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(lambda: asyncio.ensure_future(coro))

    def _notify(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._publish_status)
