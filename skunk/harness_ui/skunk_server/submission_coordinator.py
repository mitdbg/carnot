"""Validation and Cup submission coordination for agent answers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from cup_kit.protocol import (
    MAX_ANSWER_TEXT_CHARS,
    MAX_REASONING_CHARS,
    MAX_SOURCE_DOCS_ENTRIES,
    MAX_SOURCE_DOC_REF_CHARS,
)

from skunk_server.competition_adapter import CompetitionAdapter
from skunk_server.domain import TaskStatus
from skunk_server.task_queues import TaskQueues
from skunk_server.task_registry import TaskRegistry

logger = logging.getLogger(__name__)

# How many seconds before the round deadline the server auto-submits every un-submitted
# READY answer, so a missed manual click never loses an answer. An answer that first turns
# READY inside this final window is submitted immediately instead of waiting for the sweep.
DEADLINE_SUBMIT_LEAD_S = 5.0


class SubmissionCoordinator:
    def __init__(
        self,
        registry: TaskRegistry,
        queues: TaskQueues,
        adapter: CompetitionAdapter,
        publish_status: Callable[[], None],
    ) -> None:
        self._registry = registry
        self._queues = queues
        self._adapter = adapter
        self._publish_status = publish_status
        self._locks: dict[str, asyncio.Lock] = {}
        # One in-flight deadline sweep per active round; replaced/cancelled on round change.
        self._sweep_task: asyncio.Task | None = None

    def handle_agent_completion(self, task_id: str, outcome: str) -> None:
        # Called (on the main loop) whenever a task's status changes. Manual submission is the
        # primary path, so a freshly-READY task is only auto-submitted here if we're already in
        # the final pre-deadline window (the scheduled sweep snapshots READY tasks once, so an
        # answer that arrives after it must self-submit). Every outcome refreshes the stream.
        if outcome == "ready" and self._in_final_window():
            asyncio.create_task(self._auto_submit_ready(task_id))
        self._publish_status()

    def on_round_active(self, round_num: int) -> None:
        # Called on the main loop when a round becomes/stays ACTIVE. (Re)schedule the one-shot
        # deadline sweep for this round, cancelling any sweep left over from a prior round.
        if self._sweep_task is not None and not self._sweep_task.done():
            self._sweep_task.cancel()
        self._sweep_task = asyncio.create_task(self._deadline_sweep(round_num))

    def _in_final_window(self) -> bool:
        round_state = self._registry.round_state()
        if round_state.status != "ACTIVE" or round_state.ends_at is None:
            return False
        return datetime.now(timezone.utc) >= round_state.ends_at - timedelta(seconds=DEADLINE_SUBMIT_LEAD_S)

    async def _deadline_sweep(self, round_num: int) -> None:
        # Sleep until DEADLINE_SUBMIT_LEAD_S before the round deadline, then submit every task
        # still sitting in READY. ends_at is read from the registry (not passed in) so a bare
        # round_state refresh can't strand the sweep on a stale deadline.
        try:
            ends_at = self._registry.round_state().ends_at
            if ends_at is None:
                logger.warning("round %s has no deadline; auto-submit safety net inactive", round_num)
                return
            delay = (ends_at - datetime.now(timezone.utc)).total_seconds() - DEADLINE_SUBMIT_LEAD_S
            if delay > 0:
                await asyncio.sleep(delay)
            round_state = self._registry.round_state()
            if round_state.round_num != round_num or round_state.status != "ACTIVE":
                return
            ready_ids = [
                task.task_id
                for task in self._registry.list_tasks()
                if task.round_num == round_num and task.status == TaskStatus.READY
            ]
            if ready_ids:
                logger.info("deadline sweep submitting %d READY task(s) for round %s", len(ready_ids), round_num)
            await asyncio.gather(
                *(self._auto_submit_ready(task_id) for task_id in ready_ids),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("deadline sweep failed for round %s", round_num)

    async def _auto_submit_ready(self, task_id: str) -> None:
        try:
            await self.submit_ready(task_id)
        except Exception:
            logger.exception("auto-submit failed for %s", task_id)

    async def submit_ready(self, task_id: str) -> None:
        async with self._locks.setdefault(task_id, asyncio.Lock()):
            existing = self._registry.get(task_id)
            if existing is None or existing.latest_candidate is None:
                raise KeyError(task_id)
            self._validate(
                existing.latest_candidate.answer_text,
                existing.latest_candidate.reasoning,
                existing.latest_candidate.source_docs,
            )
            task, candidate, submission = self._registry.begin_candidate_submission(task_id)
            self._publish_status()
            try:
                response = await self._adapter.submit(submission, candidate)
            except Exception as error:
                self._registry.record_submission_error(
                    task.task_id,
                    submission.local_submission_id,
                    str(error),
                )
                self._queues.enqueue_ready(task.task_id)
                self._publish_status()
                raise
            if response.accepted:
                self._registry.record_submission_accepted(
                    task.task_id,
                    submission.local_submission_id,
                    response.submission_id,
                    response.tokens_remaining,
                    response.score.correct,
                    response.score.points_awarded,
                )
            else:
                self._registry.record_submission_rejected(
                    task.task_id,
                    submission.local_submission_id,
                    response.reason.value,
                    response.tokens_remaining,
                )
                self._queues.enqueue_ready(task.task_id)
            self._publish_status()

    @staticmethod
    def _validate(answer: str, reasoning: str, source_docs: list[str]) -> None:
        if not answer.strip():
            raise ValueError("answer must not be empty")
        if len(answer) > MAX_ANSWER_TEXT_CHARS:
            raise ValueError("answer exceeds Cup limit")
        if len(reasoning) > MAX_REASONING_CHARS:
            raise ValueError("reasoning exceeds Cup limit")
        if len(source_docs) > MAX_SOURCE_DOCS_ENTRIES:
            raise ValueError("too many source documents")
        if any(not item.strip() or len(item) > MAX_SOURCE_DOC_REF_CHARS for item in source_docs):
            raise ValueError("invalid source document reference")
