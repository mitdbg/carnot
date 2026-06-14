"""Human-review coordination (single-operator, slim). Two transports share one review lifecycle:

The pipeline runs WITHOUT blocking on a human: as a task executes, table/vector extracts,
figure reads, and external lookups register an open `HumanReview` (via `register_review`,
wrapped into the agent's `human_review_register` hook). The task completes optimistically and
becomes submittable. When the operator resolves a review in the web UI (`resolve_review`), the
broker schedules a background RECOMPUTE: it re-runs only `compute` over the first attempt's
cached entries with the human's correction swapped in (no re-plan / re-retrieve) and records the
revised answer as a superseding candidate. It never submits — nothing is sent optimistically; the
revised answer waits for an operator click or the deadline sweep, like any other READY answer.

- BLOCKING (`await_intervention` → the agent's `human_intervention_handler`, wired when
  `SKUNK_HUMAN_BLOCKING` is set): the branch SUSPENDS on the review until a human resolves it,
  then the response rides back to the orchestrator, which corrects the value inline and runs
  compute (no recompute — the correction happens before compute). `resolve_review` auto-detects a
  pending blocking review and wakes its branch instead of recomputing.

This deliberately drops the old claim/release/assignment/worker-registry machinery.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from skunk_server.domain import AnswerCandidate, HumanReview, QuestionTask
from skunk_server.task_registry import TaskRegistry

logger = logging.getLogger(__name__)

# (recompute_state_json, {branch_id: raw_response_json}) -> revised answer text.
RecomputeFn = Callable[[dict[str, Any], dict[int, str]], Awaitable[str]]


class HumanWorkBroker:
    def __init__(
        self,
        registry: TaskRegistry,
        recompute_fn: RecomputeFn | None,
        publish_status: Callable[[], None],
    ) -> None:
        self._registry = registry
        self._recompute_fn = recompute_fn
        self._publish_status = publish_status
        self._loop: asyncio.AbstractEventLoop | None = None
        # Blocking transport: review_id -> (worker_loop, future). The branch coroutine (on a
        # worker loop) awaits the future; `resolve_review` (server loop) sets its result
        # cross-loop. Empty unless `await_intervention` is wired (SKUNK_HUMAN_BLOCKING).
        self._pending: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Future]] = {}
        self._pending_lock = threading.Lock()

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

    # ---- Blocking transport (awaited on a worker loop) ---------------------------------

    async def await_intervention(
        self,
        task_id: str,
        attempt_id: str,
        kind: str,
        instructions: str,
        context: str | None,
        source_docs: list[str] | None,
        guidance: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Open a review and SUSPEND this branch until a human resolves it in the web UI, then
        return the human's response (`{"response": <raw>}`, empty = accept-as-is) to the
        orchestrator. Awaited on the worker's event loop, so only this branch's coroutine
        suspends — the loop keeps running every other branch/question. Wired (instead of
        `register_review`) when blocking is enabled."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        try:
            review = self._registry.create_review(
                task_id, attempt_id, kind, instructions, context, list(source_docs or []), guidance
            )
        except KeyError:
            return {"response": ""}  # task vanished (round closing) — accept the model as-is
        with self._pending_lock:
            self._pending[review.review_id] = (loop, fut)
        self._notify()
        try:
            response = await fut
        except asyncio.CancelledError:
            with self._pending_lock:
                self._pending.pop(review.review_id, None)
            raise
        return {"response": response}

    # ---- Resolution (called from the server loop, via the API route) -------------------

    def resolve_review(
        self,
        review_id: str,
        response: str,
        source_docs: list[str],
    ) -> tuple[QuestionTask, HumanReview]:
        task, review = self._registry.resolve_review(review_id, response, source_docs)
        with self._pending_lock:
            pending = self._pending.pop(review_id, None)
        if pending is not None:
            # BLOCKING review: wake the suspended branch with the human's response (NO recompute —
            # the branch corrects the value inline). Set the result on the loop that owns the future.
            worker_loop, fut = pending

            def _fulfill() -> None:
                if not fut.done():
                    fut.set_result(response)

            try:
                worker_loop.call_soon_threadsafe(_fulfill)
            except RuntimeError:
                pass  # worker loop already gone (round torn down) — the branch was cancelled
            self._notify()
            return task, review
        # OPTIMISTIC: a non-empty correction on a task that completed a compute → recompute in the
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
            # Record the revised answer as the new latest candidate and leave the task READY.
            # We deliberately do NOT submit here: nothing is ever submitted optimistically.
            # The revised answer waits for an explicit operator click or the deadline sweep
            # (SubmissionCoordinator), the same as any other READY answer.
            self._registry.add_revised_candidate(task_id, candidate)
        finally:
            # Clear the revising flag on every path (success, no-op, or failure) and refresh.
            self._registry.set_revising(task_id, False)
            self._notify()

    def discard_reviews(self, task_id: str) -> None:
        """Cancel one task's still-open optimistic reviews — called from a worker loop when the
        orchestrator takes a replan (the replan supersedes the branches those reviews belong to).
        Safe from a worker thread: the registry is locked and the status publish is scheduled on
        the server loop."""
        if self._registry.cancel_task_reviews(task_id):
            self._notify()

    def cancel_all(self) -> None:
        self._registry.cancel_active_reviews()
        # Wake any branches suspended on a blocking review so they unblock (CancelledError)
        # instead of hanging until process exit.
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for worker_loop, fut in pending:
            try:
                worker_loop.call_soon_threadsafe(fut.cancel)
            except RuntimeError:
                pass
        self._notify()

    # ---- internals ---------------------------------------------------------------------

    def _schedule(self, coro: Awaitable[None]) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(lambda: asyncio.ensure_future(coro))

    def _notify(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._publish_status)
