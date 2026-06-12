"""Thread-safe authoritative task registry and transition rules."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime

from skunk_server.domain import (
    AnswerCandidate,
    AssignmentStatus,
    Attempt,
    FailureRecord,
    HumanAssignment,
    HumanIntervention,
    HumanInterventionStatus,
    QuestionTask,
    RoundState,
    SubmissionRecord,
    SubmissionStatus,
    TaskStatus,
    to_jsonable,
    utc_now,
)

MAX_ATTEMPT_EVENTS = 80
MAX_EVENT_MESSAGE_CHARS = 240
MAX_EVENT_DATA_CHARS = 2000


class TaskConflict(ValueError):
    """Raised when a command targets a stale or no-longer-actionable task."""


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, QuestionTask] = {}
        self._assignments: dict[str, HumanAssignment] = {}
        self._interventions: dict[str, HumanIntervention] = {}
        self._round = RoundState()
        self._lock = threading.RLock()
        self._intervention_cancel_callback: Callable[[list[str]], None] | None = None
        self._round_close_callback: Callable[[int], None] | None = None

    def set_intervention_cancel_callback(
        self,
        callback: Callable[[list[str]], None],
    ) -> None:
        self._intervention_cancel_callback = callback

    def set_round_close_callback(self, callback: Callable[[int], None]) -> None:
        """Register a hook fired when a round closes — the agent worker pool uses it to
        cancel that round's still-running reasoners so the workers are freed for the next
        round (rather than staying blocked on un-cancellable in-flight work)."""
        self._round_close_callback = callback

    def create_task(
        self,
        round_num: int,
        question_id: str,
        prompt: str,
        round_ends_at: datetime | None = None,
    ) -> tuple[QuestionTask, bool]:
        task_id = f"{round_num}:{question_id}"
        with self._lock:
            existing = self._tasks.get(task_id)
            if existing is not None:
                return existing, False
            task = QuestionTask(
                task_id=task_id,
                round_num=round_num,
                question_id=question_id,
                prompt=prompt,
                round_ends_at=round_ends_at,
            )
            self._tasks[task_id] = task
            self._set_status(task, TaskStatus.QUEUED)
            return task, True

    def get(self, task_id: str) -> QuestionTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> list[QuestionTask]:
        with self._lock:
            return list(self._tasks.values())

    def round_state(self) -> RoundState:
        with self._lock:
            return self._round

    def update_round(
        self,
        *,
        round_num: int,
        status: str,
        ends_at: datetime | None = None,
        opens_at: datetime | None = None,
        resubmits_left: int | None = None,
        connection_status: str | None = None,
        event: str = "",
    ) -> None:
        with self._lock:
            self._round.round_num = round_num
            self._round.status = status
            self._round.ends_at = ends_at
            self._round.opens_at = opens_at
            if resubmits_left is not None:
                self._round.resubmits_left = resubmits_left
            if connection_status is not None:
                self._round.connection_status = connection_status
            if event:
                self._round.last_event = event

    def set_connection(self, status: str, event: str = "") -> None:
        with self._lock:
            self._round.connection_status = status
            if event:
                self._round.last_event = event

    def begin_attempt(self, task_id: str, worker_id: str) -> Attempt | None:
        with self._lock:
            task = self._require_task(task_id)
            if task.status not in {TaskStatus.QUEUED, TaskStatus.RETRY_QUEUED}:
                return None
            attempt = Attempt(
                task_id=task_id,
                attempt_number=len(task.attempts) + 1,
                worker_id=worker_id,
                feedback=task.pending_retry_feedback,
                context_feedback=list(task.cup_feedback),
                previous_attempt_ids=[item.attempt_id for item in task.attempts],
            )
            task.pending_retry_feedback = None
            task.attempts.append(attempt)
            task.current_attempt_id = attempt.attempt_id
            self._set_status(task, TaskStatus.PROCESSING)
            return attempt

    def complete_attempt(self, task_id: str, attempt_id: str, candidate: AnswerCandidate) -> bool:
        with self._lock:
            task = self._require_task(task_id)
            if task.current_attempt_id != attempt_id or task.status != TaskStatus.PROCESSING:
                return False
            task.attempts[-1].completed_at = utc_now()
            task.answer_candidates.append(candidate)
            task.current_attempt_id = None
            self._set_status(task, TaskStatus.READY)
            return True

    def fail_attempt(self, task_id: str, attempt_id: str, failure: FailureRecord) -> bool:
        with self._lock:
            task = self._require_task(task_id)
            if task.current_attempt_id != attempt_id or task.status != TaskStatus.PROCESSING:
                return False
            task.attempts[-1].completed_at = utc_now()
            task.failures.append(failure)
            task.current_attempt_id = None
            self._set_status(task, TaskStatus.FAILED)
            return True

    def append_attempt_event(
        self,
        task_id: str,
        attempt_id: str,
        event: dict,
    ) -> bool:
        with self._lock:
            task = self._require_task(task_id)
            for attempt in task.attempts:
                if attempt.attempt_id == attempt_id:
                    attempt.events.append(_compact_event(event))
                    if len(attempt.events) > MAX_ATTEMPT_EVENTS:
                        del attempt.events[:len(attempt.events) - MAX_ATTEMPT_EVENTS]
                    task.updated_at = utc_now()
                    return True
            return False

    def create_intervention(
        self,
        task_id: str,
        attempt_id: str,
        kind: str,
        instructions: str,
        context: str | None,
        source_docs: list[str],
        guidance: dict | None = None,
    ) -> HumanIntervention:
        with self._lock:
            task = self._require_task(task_id)
            if task.current_attempt_id != attempt_id:
                raise TaskConflict("human intervention targets a stale attempt")
            if task.status not in {TaskStatus.PROCESSING, TaskStatus.AWAIT_HUMAN}:
                raise TaskConflict(
                    f"task cannot request human intervention; status={task.status}"
                )
            intervention = HumanIntervention(
                task_id=task_id,
                attempt_id=attempt_id,
                kind=kind,
                instructions=instructions,
                context=context,
                source_docs=source_docs,
                guidance=dict(guidance or {}),
            )
            task.human_interventions.append(intervention)
            self._interventions[intervention.intervention_id] = intervention
            if task.status == TaskStatus.PROCESSING:
                self._set_status(task, TaskStatus.AWAIT_HUMAN)
            else:
                task.updated_at = utc_now()
            return intervention

    def claim_intervention(
        self,
        intervention_id: str,
        worker_id: str,
    ) -> HumanIntervention:
        with self._lock:
            intervention = self._require_intervention(intervention_id)
            task = self._require_task(intervention.task_id)
            if (
                task.current_attempt_id != intervention.attempt_id
                or task.status != TaskStatus.AWAIT_HUMAN
            ):
                raise TaskConflict("human intervention is no longer actionable")
            if intervention.status == HumanInterventionStatus.CLAIMED:
                if intervention.claimed_by == worker_id:
                    return intervention
                raise TaskConflict("human intervention is claimed by another worker")
            if intervention.status != HumanInterventionStatus.PENDING:
                raise TaskConflict(
                    f"human intervention is not pending; status={intervention.status}"
                )
            intervention.status = HumanInterventionStatus.CLAIMED
            intervention.claimed_by = worker_id
            intervention.claimed_at = utc_now()
            task.updated_at = utc_now()
            return intervention

    def release_intervention(
        self,
        intervention_id: str,
        worker_id: str,
    ) -> HumanIntervention:
        with self._lock:
            intervention = self._require_intervention(intervention_id)
            if intervention.status != HumanInterventionStatus.CLAIMED:
                raise TaskConflict(
                    f"human intervention is not claimed; status={intervention.status}"
                )
            if intervention.claimed_by != worker_id:
                raise TaskConflict("human intervention belongs to another worker")
            intervention.status = HumanInterventionStatus.PENDING
            intervention.claimed_by = None
            intervention.claimed_at = None
            self._require_task(intervention.task_id).updated_at = utc_now()
            return intervention

    def resolve_intervention(
        self,
        intervention_id: str,
        worker_id: str,
        response: str,
        source_docs: list[str],
        retrieval_directives: list[dict] | None = None,
    ) -> tuple[QuestionTask, HumanIntervention]:
        cleaned = response.strip()
        directives = list(retrieval_directives or [])
        if not cleaned and not directives:
            raise ValueError(
                "human intervention response or retrieval directive must not be empty"
            )
        with self._lock:
            intervention = self._require_intervention(intervention_id)
            task = self._require_task(intervention.task_id)
            if intervention.status != HumanInterventionStatus.CLAIMED:
                raise TaskConflict(
                    f"human intervention is not claimed; status={intervention.status}"
                )
            if intervention.claimed_by != worker_id:
                raise TaskConflict("human intervention belongs to another worker")
            if (
                task.current_attempt_id != intervention.attempt_id
                or task.status != TaskStatus.AWAIT_HUMAN
            ):
                raise TaskConflict("human intervention is no longer actionable")
            intervention.status = HumanInterventionStatus.RESOLVED
            intervention.response = cleaned or None
            intervention.response_source_docs = list(source_docs)
            intervention.response_retrieval_directives = directives
            intervention.resolved_at = utc_now()
            unresolved = any(
                item.attempt_id == intervention.attempt_id
                and item.status
                in {
                    HumanInterventionStatus.PENDING,
                    HumanInterventionStatus.CLAIMED,
                }
                for item in task.human_interventions
            )
            if not unresolved:
                self._set_status(task, TaskStatus.PROCESSING)
            else:
                task.updated_at = utc_now()
            return task, intervention

    def create_assignment(self, task_id: str, worker_id: str) -> HumanAssignment:
        with self._lock:
            task = self._require_task(task_id)
            if task.status not in {TaskStatus.READY, TaskStatus.FAILED}:
                raise TaskConflict(f"task is not claimable; status={task.status}")
            for assignment in task.assignments:
                if assignment.worker_id == worker_id and assignment.status == AssignmentStatus.ACTIVE:
                    return assignment
            assignment = HumanAssignment(
                task_id=task_id,
                worker_id=worker_id,
                task_version=task.version,
                source_kind=task.status.value,
            )
            task.assignments.append(assignment)
            self._assignments[assignment.assignment_id] = assignment
            task.updated_at = utc_now()
            return assignment

    def release_assignment(self, assignment_id: str, worker_id: str) -> HumanAssignment:
        with self._lock:
            assignment = self._require_assignment(assignment_id)
            if assignment.worker_id != worker_id:
                raise TaskConflict("assignment belongs to another worker")
            if assignment.status != AssignmentStatus.ACTIVE:
                raise TaskConflict(f"assignment is not active; status={assignment.status}")
            assignment.status = AssignmentStatus.RELEASED
            assignment.completed_at = utc_now()
            return assignment

    def retry_from_assignment(
        self,
        assignment_id: str,
        worker_id: str,
        task_version: int,
        feedback: str,
    ) -> QuestionTask:
        cleaned = feedback.strip()
        if not cleaned:
            raise ValueError("retry feedback must not be empty")
        with self._lock:
            task, assignment = self._validate_action(assignment_id, worker_id, task_version)
            self._finish_assignment(task, assignment)
            task.pending_retry_feedback = cleaned
            self._set_status(task, TaskStatus.RETRY_QUEUED)
            return task

    def human_answer_from_assignment(
        self,
        assignment_id: str,
        worker_id: str,
        task_version: int,
        answer_text: str,
        reasoning: str,
        source_docs: list[str],
    ) -> tuple[QuestionTask, AnswerCandidate]:
        cleaned = answer_text.strip()
        if not cleaned:
            raise ValueError("answer must not be empty")
        with self._lock:
            task, assignment = self._validate_action(assignment_id, worker_id, task_version)
            if assignment.source_kind != TaskStatus.FAILED.value:
                raise TaskConflict("direct human answers are only valid for failed tasks")
            self._finish_assignment(task, assignment)
            attempt = Attempt(
                task_id=task.task_id,
                attempt_number=len(task.attempts) + 1,
                worker_id=worker_id,
                previous_attempt_ids=[item.attempt_id for item in task.attempts],
                completed_at=utc_now(),
            )
            candidate = AnswerCandidate(
                attempt_id=attempt.attempt_id,
                answer_text=cleaned,
                reasoning=reasoning,
                source_docs=source_docs,
                submission_type="human",
            )
            task.attempts.append(attempt)
            task.answer_candidates.append(candidate)
            self._set_status(task, TaskStatus.SUBMITTING)
            return task, candidate

    def begin_candidate_submission(
        self,
        task_id: str,
        *,
        assignment_id: str | None = None,
        worker_id: str | None = None,
        task_version: int | None = None,
    ) -> tuple[QuestionTask, AnswerCandidate, SubmissionRecord]:
        with self._lock:
            task = self._require_task(task_id)
            if assignment_id is not None:
                if worker_id is None or task_version is None:
                    raise ValueError("worker_id and task_version are required")
                task, assignment = self._validate_action(assignment_id, worker_id, task_version)
                if assignment.source_kind != TaskStatus.READY.value:
                    raise TaskConflict("generated candidate submission requires a READY assignment")
                self._finish_assignment(task, assignment)
            elif task.status != TaskStatus.READY:
                raise TaskConflict(f"task is not ready for auto-submit; status={task.status}")
            candidate = task.latest_candidate
            if candidate is None:
                raise TaskConflict("task has no answer candidate")
            submission = SubmissionRecord(
                task_id=task_id,
                candidate_id=candidate.candidate_id,
                submission_type=candidate.submission_type,
            )
            task.submissions.append(submission)
            self._set_status(task, TaskStatus.SUBMITTING)
            return task, candidate, submission

    def begin_candidate_submission_after_human_answer(
        self,
        task_id: str,
    ) -> SubmissionRecord:
        with self._lock:
            task = self._require_task(task_id)
            if task.status != TaskStatus.SUBMITTING:
                raise TaskConflict(f"task is not submitting; status={task.status}")
            candidate = task.latest_candidate
            if candidate is None or candidate.submission_type != "human":
                raise TaskConflict("task has no direct human answer")
            if any(
                item.candidate_id == candidate.candidate_id
                and item.status == SubmissionStatus.PENDING
                for item in task.submissions
            ):
                raise TaskConflict("human answer already has a pending submission")
            submission = SubmissionRecord(
                task_id=task_id,
                candidate_id=candidate.candidate_id,
                submission_type=candidate.submission_type,
            )
            task.submissions.append(submission)
            task.updated_at = utc_now()
            return submission

    def record_submission_accepted(
        self,
        task_id: str,
        local_submission_id: str,
        cup_submission_id: str,
        tokens_remaining: int,
        correct: bool | None = None,
        points_awarded: float | None = None,
    ) -> None:
        with self._lock:
            task = self._require_task(task_id)
            submission = self._find_submission(task, local_submission_id)
            submission.status = SubmissionStatus.ACCEPTED
            submission.cup_submission_id = cup_submission_id
            submission.correct = correct
            submission.points_awarded = points_awarded
            if correct is not None:
                task.cup_feedback.append(
                    f"Cup score: correct={correct}, points_awarded={points_awarded or 0.0}"
                )
            self._round.resubmits_left = tokens_remaining
            self._set_status(task, TaskStatus.SUBMITTED)

    def record_submission_rejected(
        self,
        task_id: str,
        local_submission_id: str,
        reason: str,
        tokens_remaining: int,
    ) -> None:
        with self._lock:
            task = self._require_task(task_id)
            submission = self._find_submission(task, local_submission_id)
            submission.status = SubmissionStatus.REJECTED
            submission.rejection_reason = reason
            task.cup_feedback.append(f"Cup submission rejected: {reason}")
            self._round.resubmits_left = tokens_remaining
            self._set_status(task, TaskStatus.READY)

    def record_submission_error(
        self,
        task_id: str,
        local_submission_id: str,
        message: str,
    ) -> None:
        with self._lock:
            task = self._require_task(task_id)
            submission = self._find_submission(task, local_submission_id)
            submission.status = SubmissionStatus.REJECTED
            submission.rejection_reason = message
            task.cup_feedback.append(f"Cup submission error: {message}")
            self._set_status(task, TaskStatus.READY)

    def record_score(
        self,
        question_id: str,
        cup_submission_id: str,
        correct: bool,
        points_awarded: float,
    ) -> QuestionTask | None:
        with self._lock:
            for task in self._tasks.values():
                if task.question_id != question_id:
                    continue
                for submission in task.submissions:
                    if submission.cup_submission_id != cup_submission_id:
                        continue
                    submission.status = SubmissionStatus.SCORED
                    submission.correct = correct
                    submission.points_awarded = points_awarded
                    task.cup_feedback.append(
                        f"Cup score: correct={correct}, points_awarded={points_awarded}"
                    )
                    self._set_status(task, TaskStatus.SCORED)
                    return task
            return None

    def close_round(self, round_num: int, status: str) -> None:
        cancelled: list[str] = []
        with self._lock:
            self._round.round_num = round_num
            self._round.status = status
            self._round.last_event = f"round state {status}"
            for task in self._tasks.values():
                if task.round_num == round_num and task.status not in {
                    TaskStatus.SCORED,
                    TaskStatus.CANCELLED,
                }:
                    cancelled.extend(self._cancel_interventions(task))
                    self._set_status(task, TaskStatus.CANCELLED)
        if cancelled and self._intervention_cancel_callback is not None:
            self._intervention_cancel_callback(cancelled)
        # Free any worker still blocked on this round's reasoners (outside the lock — the
        # callback cancels concurrent futures / schedules loop work).
        if self._round_close_callback is not None:
            self._round_close_callback(round_num)

    def _validate_action(
        self,
        assignment_id: str,
        worker_id: str,
        task_version: int,
    ) -> tuple[QuestionTask, HumanAssignment]:
        assignment = self._require_assignment(assignment_id)
        task = self._require_task(assignment.task_id)
        if assignment.worker_id != worker_id:
            raise TaskConflict("assignment belongs to another worker")
        if assignment.status != AssignmentStatus.ACTIVE:
            raise TaskConflict(f"assignment is not active; status={assignment.status}")
        if assignment.task_version != task_version or task.version != task_version:
            raise TaskConflict("assignment targets a stale task version")
        expected = TaskStatus(assignment.source_kind)
        if task.status != expected:
            raise TaskConflict(f"task is no longer actionable; status={task.status}")
        return task, assignment

    def _finish_assignment(self, task: QuestionTask, winner: HumanAssignment) -> None:
        now = utc_now()
        winner.status = AssignmentStatus.COMPLETED
        winner.completed_at = now
        for assignment in task.assignments:
            if (
                assignment.assignment_id != winner.assignment_id
                and assignment.task_version == winner.task_version
                and assignment.status == AssignmentStatus.ACTIVE
            ):
                assignment.status = AssignmentStatus.SUPERSEDED
                assignment.completed_at = now

    def _set_status(self, task: QuestionTask, status: TaskStatus) -> None:
        task.status = status
        task.version += 1
        task.updated_at = utc_now()

    def cancel_active_interventions(self) -> list[str]:
        with self._lock:
            cancelled: list[str] = []
            for task in self._tasks.values():
                cancelled.extend(self._cancel_interventions(task))
            return cancelled

    @staticmethod
    def _cancel_interventions(task: QuestionTask) -> list[str]:
        cancelled: list[str] = []
        now = utc_now()
        for intervention in task.human_interventions:
            if intervention.status in {
                HumanInterventionStatus.PENDING,
                HumanInterventionStatus.CLAIMED,
            }:
                intervention.status = HumanInterventionStatus.CANCELLED
                intervention.resolved_at = now
                cancelled.append(intervention.intervention_id)
        return cancelled

    def _require_task(self, task_id: str) -> QuestionTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(task_id)
        return task

    def _require_assignment(self, assignment_id: str) -> HumanAssignment:
        assignment = self._assignments.get(assignment_id)
        if assignment is None:
            raise KeyError(assignment_id)
        return assignment

    def _require_intervention(self, intervention_id: str) -> HumanIntervention:
        intervention = self._interventions.get(intervention_id)
        if intervention is None:
            raise KeyError(intervention_id)
        return intervention

    @staticmethod
    def _find_submission(task: QuestionTask, local_submission_id: str) -> SubmissionRecord:
        for submission in task.submissions:
            if submission.local_submission_id == local_submission_id:
                return submission
        raise KeyError(local_submission_id)


def _compact_event(event: dict) -> dict:
    compact: dict = {}
    for key in ("message", "kind", "op", "level", "step_idx", "t"):
        if key in event:
            compact[key] = event[key]
    if "message" in compact:
        compact["message"] = str(compact["message"])[:MAX_EVENT_MESSAGE_CHARS]
    data = event.get("data")
    if isinstance(data, dict) and data:
        compact["data"] = _compact_event_data(data)
    return to_jsonable(compact)


def _compact_event_data(data: dict) -> dict | str:
    compact = to_jsonable(data)
    rendered = str(compact)
    if len(rendered) <= MAX_EVENT_DATA_CHARS:
        return compact
    return f"{rendered[:MAX_EVENT_DATA_CHARS]}... [truncated]"
