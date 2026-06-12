"""Validation and Cup submission coordination for agent answers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from cup_kit.protocol import (
    MAX_ANSWER_TEXT_CHARS,
    MAX_REASONING_CHARS,
    MAX_SOURCE_DOCS_ENTRIES,
    MAX_SOURCE_DOC_REF_CHARS,
)

from skunk_server.competition_adapter import CompetitionAdapter
from skunk_server.task_queues import TaskQueues
from skunk_server.task_registry import TaskRegistry

logger = logging.getLogger(__name__)


class SubmissionCoordinator:
    def __init__(
        self,
        registry: TaskRegistry,
        queues: TaskQueues,
        adapter: CompetitionAdapter,
        publish_status: Callable[[], None],
        auto_submit: bool,
    ) -> None:
        self._registry = registry
        self._queues = queues
        self._adapter = adapter
        self._publish_status = publish_status
        self._auto_submit = auto_submit
        self._locks: dict[str, asyncio.Lock] = {}

    def handle_agent_completion(self, task_id: str, outcome: str) -> None:
        # Called (on the main loop) whenever a task's status changes. A READY task is
        # auto-submitted to cup; every outcome refreshes the lightweight status stream.
        if outcome == "ready" and self._auto_submit:
            asyncio.create_task(self._auto_submit_ready(task_id))
        self._publish_status()

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
