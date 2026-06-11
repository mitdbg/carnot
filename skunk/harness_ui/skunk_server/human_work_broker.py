"""Human task assignment and first-valid-action coordination."""

from __future__ import annotations

from skunk_server.domain import AnswerCandidate, HumanAssignment, QuestionTask
from skunk_server.human_worker_registry import HumanWorkerRegistry
from skunk_server.task_queues import TaskQueues
from skunk_server.task_registry import TaskRegistry


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
