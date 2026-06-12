"""Thread-safe authoritative task registry and transition rules."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from enum import StrEnum
from typing import Any

from skunk_server.domain import (
    AnswerCandidate,
    Attempt,
    FailureRecord,
    HumanReview,
    HumanReviewStatus,
    QuestionTask,
    RoundState,
    SubmissionRecord,
    SubmissionStatus,
    TaskStatus,
    utc_now,
)

MAX_ATTEMPT_EVENTS = 2000
MAX_EVENT_MESSAGE_CHARS = 4000
# Total budget (chars) for an event's `data` payload. Corpus observations are huge (search
# results / extracted blocks can be ~1MB), but the trace viewer truncates display anyway and
# the durable per-question dump keeps the full event. Capping here keeps the live registry /
# SSE-stream payload small so the per-event JSON work stays off the GIL the corpus tools need.
MAX_EVENT_DATA_CHARS = 16000


class TaskConflict(ValueError):
    """Raised when a command targets a stale or no-longer-actionable task."""


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, QuestionTask] = {}
        # Per-task monotonic event counter. `seq` is stamped into each stored event so the
        # SSE backfill can dedup against live events (and survive event-list trimming).
        self._event_seq: dict[str, int] = {}
        self._round = RoundState()
        self._lock = threading.RLock()
        self._round_close_callback: Callable[[int], None] | None = None

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
            if task.status != TaskStatus.QUEUED:
                return None
            attempt = Attempt(
                task_id=task_id,
                attempt_number=len(task.attempts) + 1,
                worker_id=worker_id,
                context_feedback=list(task.cup_feedback),
                previous_attempt_ids=[item.attempt_id for item in task.attempts],
            )
            task.attempts.append(attempt)
            task.current_attempt_id = attempt.attempt_id
            self._set_status(task, TaskStatus.PROCESSING)
            return attempt

    def complete_attempt(
        self, task_id: str, attempt_id: str, candidate: AnswerCandidate
    ) -> bool:
        with self._lock:
            task = self._require_task(task_id)
            if (
                task.current_attempt_id != attempt_id
                or task.status != TaskStatus.PROCESSING
            ):
                return False
            task.attempts[-1].completed_at = utc_now()
            task.answer_candidates.append(candidate)
            task.current_attempt_id = None
            self._set_status(task, TaskStatus.READY)
            return True

    def fail_attempt(
        self, task_id: str, attempt_id: str, failure: FailureRecord
    ) -> bool:
        with self._lock:
            task = self._require_task(task_id)
            if (
                task.current_attempt_id != attempt_id
                or task.status != TaskStatus.PROCESSING
            ):
                return False
            task.attempts[-1].completed_at = utc_now()
            task.failures.append(failure)
            task.current_attempt_id = None
            self._set_status(task, TaskStatus.FAILED)
            return True

    def append_attempt_events(
        self,
        task_id: str,
        attempt_id: str,
        events: list[dict],
    ) -> list[dict]:
        """Append a batch of trace events to an attempt and return the stored (compacted,
        seq-stamped) events — empty if the task/attempt is gone. The CPU-heavy compaction
        runs OUTSIDE the lock; the lock is taken once per batch only to stamp the per-task
        monotonic `seq` and extend the list. Batching keeps both the global lock and the
        main-loop publish wakeup off the per-event hot path (see `AgentWorkerPool`'s
        batcher), so the GIL stays free for the reasoner's corpus tools."""
        if not events:
            return []
        compacted = [_compact_event(event) for event in events]  # heavy walk, lock-free
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return []
            attempt = next(
                (a for a in task.attempts if a.attempt_id == attempt_id), None
            )
            if attempt is None:
                return []
            seq = self._event_seq.get(task_id, 0)
            for event in compacted:
                event["seq"] = seq
                seq += 1
            self._event_seq[task_id] = seq
            attempt.events.extend(compacted)
            if len(attempt.events) > MAX_ATTEMPT_EVENTS:
                del attempt.events[: len(attempt.events) - MAX_ATTEMPT_EVENTS]
            task.updated_at = utc_now()
            return compacted

    def append_attempt_event(
        self, task_id: str, attempt_id: str, event: dict
    ) -> dict | None:
        stored = self.append_attempt_events(task_id, attempt_id, [event])
        return stored[0] if stored else None

    def snapshot_task_events(
        self,
        task_id: str,
    ) -> tuple[list[tuple[int, str, dict]], int]:
        """Return [(seq, attempt_id, event), ...] across all attempts in order plus the
        next seq, read atomically. The SSE event stream replays this then forwards live
        events with a higher seq."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return [], 0
            out: list[tuple[int, str, dict]] = []
            for attempt in task.attempts:
                for event in attempt.events:
                    out.append((event.get("seq", 0), attempt.attempt_id, event))
            return out, self._event_seq.get(task_id, 0)

    def begin_candidate_submission(
        self,
        task_id: str,
    ) -> tuple[QuestionTask, AnswerCandidate, SubmissionRecord]:
        with self._lock:
            task = self._require_task(task_id)
            if task.status != TaskStatus.READY:
                raise TaskConflict(
                    f"task is not ready for submission; status={task.status}"
                )
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

    # ---- Optimistic human reviews ------------------------------------------------------

    def create_review(
        self,
        task_id: str,
        attempt_id: str,
        kind: str,
        instructions: str,
        context: str | None,
        source_docs: list[str],
        guidance: dict[str, Any] | None,
    ) -> HumanReview:
        """Open a human review on a running task (the optimistic path keeps the LLM answer; this
        just records something a human MAY correct later). Returns the review."""
        with self._lock:
            task = self._require_task(task_id)
            review = HumanReview(
                task_id=task_id,
                attempt_id=attempt_id,
                kind=kind,
                instructions=instructions,
                context=context,
                source_docs=list(source_docs),
                guidance=dict(guidance or {}),
            )
            task.reviews.append(review)
            task.updated_at = utc_now()
            return review

    def set_recompute_state(self, task_id: str, state: dict[str, Any]) -> None:
        """Stash the snapshot (orchestrator.RecomputeState JSON) that produced the latest answer,
        so a resolved review can revise it without re-planning. No-op if the task is gone."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task.recompute_state = state
                task.updated_at = utc_now()

    def set_revising(self, task_id: str, revising: bool) -> None:
        """Flag/unflag a task as having an in-flight recompute (the answer is being revised),
        so the UI can show a 'Revising…' indicator. No-op if the task is gone."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None and task.revising != revising:
                task.revising = revising
                self._set_status(
                    task, task.status
                )  # bump version/updated_at, status unchanged

    def resolve_review(
        self,
        review_id: str,
        response: str,
        source_docs: list[str],
    ) -> tuple[QuestionTask, HumanReview]:
        """Record a human's correction (or accept-as-is when `response` is empty). Idempotent
        guard: re-resolving a non-open review raises `TaskConflict`."""
        with self._lock:
            task, review = self._find_review(review_id)
            if review.status != HumanReviewStatus.OPEN:
                raise TaskConflict(f"review is not open; status={review.status}")
            review.status = HumanReviewStatus.RESOLVED
            review.response = response
            review.response_source_docs = list(source_docs)
            review.resolved_at = utc_now()
            task.updated_at = utc_now()
            return task, review

    def resolved_overrides(self, task_id: str) -> dict[int, str]:
        """Accumulated human corrections for a task: `branch_id -> raw response JSON`, across all
        RESOLVED reviews with a non-empty response (accept-as-is reviews contribute nothing). A
        recompute applies all of them onto the cached snapshot, so later edits never lose earlier
        ones."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return {}
            out: dict[int, str] = {}
            for review in task.reviews:
                if review.status != HumanReviewStatus.RESOLVED:
                    continue
                if not (review.response or "").strip():
                    continue
                bid = review.guidance.get("branch_id")
                if isinstance(bid, int):
                    out[bid] = review.response  # later reviews on a branch win
            return out

    def add_revised_candidate(
        self, task_id: str, candidate: AnswerCandidate
    ) -> QuestionTask | None:
        """Append a recompute's revised answer and make it (re)submittable, unless the task is
        already terminal for this round (SCORED/CANCELLED → record only, best-effort no-op)."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.answer_candidates.append(candidate)
            if task.status in {
                TaskStatus.READY,
                TaskStatus.SUBMITTED,
                TaskStatus.SUBMITTING,
                TaskStatus.FAILED,
            }:
                self._set_status(task, TaskStatus.READY)
            else:
                task.version += 1
                task.updated_at = utc_now()
            return task

    def cancel_active_reviews(self, round_num: int | None = None) -> list[str]:
        """Close every OPEN review (optionally limited to one round) — called on round close /
        shutdown so the UI clears its overlay. Returns the cancelled review ids."""
        cancelled: list[str] = []
        with self._lock:
            for task in self._tasks.values():
                if round_num is not None and task.round_num != round_num:
                    continue
                for review in task.reviews:
                    if review.status == HumanReviewStatus.OPEN:
                        review.status = HumanReviewStatus.CANCELLED
                        review.resolved_at = utc_now()
                        cancelled.append(review.review_id)
                if cancelled:
                    task.updated_at = utc_now()
        return cancelled

    def _find_review(self, review_id: str) -> tuple[QuestionTask, HumanReview]:
        for task in self._tasks.values():
            for review in task.reviews:
                if review.review_id == review_id:
                    return task, review
        raise KeyError(review_id)

    def close_round(self, round_num: int, status: str) -> None:
        with self._lock:
            self._round.round_num = round_num
            self._round.status = status
            self._round.last_event = f"round state {status}"
            for task in self._tasks.values():
                # CANCELLED marks tasks that never got an answer submitted before the round
                # closed. Tasks that were already submitted/scored keep their terminal state
                # (and their points) so the monitor can still see what each one earned.
                if task.round_num == round_num and task.status not in {
                    TaskStatus.SUBMITTED,
                    TaskStatus.SCORED,
                    TaskStatus.CANCELLED,
                }:
                    self._set_status(task, TaskStatus.CANCELLED)
                # Close any open reviews for the round so the UI clears its overlay.
                if task.round_num == round_num:
                    for review in task.reviews:
                        if review.status == HumanReviewStatus.OPEN:
                            review.status = HumanReviewStatus.CANCELLED
                            review.resolved_at = utc_now()
        # Free any worker still blocked on this round's reasoners (outside the lock — the
        # callback schedules cancellation on each worker's own loop).
        if self._round_close_callback is not None:
            self._round_close_callback(round_num)

    def _set_status(self, task: QuestionTask, status: TaskStatus) -> None:
        task.status = status
        task.version += 1
        task.updated_at = utc_now()

    def _require_task(self, task_id: str) -> QuestionTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(task_id)
        return task

    @staticmethod
    def _find_submission(
        task: QuestionTask, local_submission_id: str
    ) -> SubmissionRecord:
        for submission in task.submissions:
            if submission.local_submission_id == local_submission_id:
                return submission
        raise KeyError(local_submission_id)


def _compact_event(event: dict) -> dict:
    """Compact one raw event for the registry/stream. `seq` is stamped later under the lock
    (see `append_attempt_events`). The `data` payload is converted to JSON-friendly form and
    capped to `MAX_EVENT_DATA_CHARS` total in a single short-circuiting walk."""
    compact: dict = {}
    for key in ("message", "kind", "op", "level", "step_idx", "t", "branch_id"):
        if key in event:
            compact[key] = event[key]
    if "message" in compact:
        compact["message"] = str(compact["message"])[:MAX_EVENT_MESSAGE_CHARS]
    data = event.get("data")
    if isinstance(data, dict) and data:
        compact["data"] = _clip_jsonable(data, [MAX_EVENT_DATA_CHARS])
    return compact


def _clip_jsonable(value: Any, budget: list[int]) -> Any:
    """JSON-ify and total-budget-cap in one pass. `budget` is a 1-element mutable counter of
    remaining chars; strings are clipped to fit and, once exhausted, list tails are dropped
    (with a marker) so a huge corpus observation costs ~budget to walk, not its full size."""
    if isinstance(value, str):
        if budget[0] <= 0:
            return ""
        if len(value) > budget[0]:
            clipped = value[: budget[0]] + "… [truncated]"
            budget[0] = 0
            return clipped
        budget[0] -= len(value)
        return value
    if isinstance(value, dict):
        return {str(key): _clip_jsonable(item, budget) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for index, item in enumerate(value):
            if budget[0] <= 0:
                out.append(f"… [{len(value) - index} more truncated]")
                break
            out.append(_clip_jsonable(item, budget))
        return out
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return _clip_jsonable(asdict(value), budget)
    return value
