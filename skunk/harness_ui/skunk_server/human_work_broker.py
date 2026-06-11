"""Human task assignment and first-valid-action coordination."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from skunk_server.domain import (
    AnswerCandidate,
    HumanAssignment,
    HumanIntervention,
    QuestionTask,
)
from skunk_server.human_worker_registry import HumanWorkerRegistry
from skunk_server.task_queues import TaskQueues
from skunk_server.task_registry import TaskConflict, TaskRegistry


class HumanWorkBroker:
    def __init__(
        self,
        registry: TaskRegistry,
        queues: TaskQueues,
        workers: HumanWorkerRegistry,
    ) -> None:
        self._registry = registry
        self._queues = queues
        self._workers = workers
        self._pending: dict[
            str,
            tuple[asyncio.AbstractEventLoop, asyncio.Future[dict[str, Any]]],
        ] = {}
        self._pending_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_change: Callable[[], Awaitable[None]] | None = None
        self._registry.set_intervention_cancel_callback(self._cancel_ids)

    def start(
        self,
        loop: asyncio.AbstractEventLoop,
        on_change: Callable[[], Awaitable[None]],
    ) -> None:
        self._loop = loop
        self._on_change = on_change

    async def request_intervention(
        self,
        task_id: str,
        attempt_id: str,
        kind: str,
        instructions: str,
        context: str | None,
        source_docs: list[str],
        guidance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        waiter_loop = asyncio.get_running_loop()
        pending: asyncio.Future[dict[str, Any]] = waiter_loop.create_future()
        with self._pending_lock:
            intervention = self._registry.create_intervention(
                task_id,
                attempt_id,
                kind,
                instructions,
                context,
                source_docs,
                guidance,
            )
            self._pending[intervention.intervention_id] = (waiter_loop, pending)
        self._notify()
        return await pending

    def claim_intervention(
        self,
        worker_id: str,
        intervention_id: str,
    ) -> HumanIntervention:
        if self._workers.get(worker_id) is None:
            raise KeyError(worker_id)
        return self._registry.claim_intervention(intervention_id, worker_id)

    def release_intervention(
        self,
        worker_id: str,
        intervention_id: str,
    ) -> HumanIntervention:
        return self._registry.release_intervention(intervention_id, worker_id)

    def resolve_intervention(
        self,
        worker_id: str,
        intervention_id: str,
        response: str,
        source_docs: list[str],
        retrieval_directives: list[dict] | None = None,
    ) -> tuple[QuestionTask, HumanIntervention]:
        task, intervention = self._registry.resolve_intervention(
            intervention_id,
            worker_id,
            response,
            source_docs,
            retrieval_directives,
        )
        with self._pending_lock:
            waiter = self._pending.pop(intervention_id, None)
        if waiter is None:
            raise TaskConflict("human intervention waiter is no longer active")
        waiter_loop, pending = waiter
        waiter_loop.call_soon_threadsafe(
            pending.set_result,
            {
                "response": intervention.response,
                "source_docs": list(intervention.response_source_docs),
                "retrieval_directives": list(
                    intervention.response_retrieval_directives
                ),
            },
        )
        return task, intervention

    def cancel_all(self) -> None:
        self._cancel_ids(self._registry.cancel_active_interventions())

    def claim(self, worker_id: str, task_id: str) -> HumanAssignment:
        if self._workers.get(worker_id) is None:
            raise KeyError(worker_id)
        return self._registry.create_assignment(task_id, worker_id)

    def release(self, worker_id: str, assignment_id: str) -> HumanAssignment:
        return self._registry.release_assignment(assignment_id, worker_id)

    def retry(
        self,
        worker_id: str,
        assignment_id: str,
        task_version: int,
        feedback: str,
    ) -> QuestionTask:
        task = self._registry.retry_from_assignment(
            assignment_id,
            worker_id,
            task_version,
            feedback,
        )
        self._queues.enqueue_agent(task.task_id)
        return task

    def direct_answer(
        self,
        worker_id: str,
        assignment_id: str,
        task_version: int,
        answer_text: str,
        reasoning: str,
        source_docs: list[str],
    ) -> tuple[QuestionTask, AnswerCandidate]:
        return self._registry.human_answer_from_assignment(
            assignment_id,
            worker_id,
            task_version,
            answer_text,
            reasoning,
            source_docs,
        )

    def _cancel_ids(self, intervention_ids: list[str]) -> None:
        with self._pending_lock:
            waiters = [
                self._pending.pop(intervention_id, None)
                for intervention_id in intervention_ids
            ]
        for waiter in waiters:
            if waiter is None:
                continue
            waiter_loop, pending = waiter
            if not pending.done():
                waiter_loop.call_soon_threadsafe(
                    pending.set_exception,
                    RuntimeError("human intervention was cancelled"),
                )
        if intervention_ids:
            self._notify()

    def _notify(self) -> None:
        if self._loop is None or self._on_change is None:
            return
        self._loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._on_change())
        )
