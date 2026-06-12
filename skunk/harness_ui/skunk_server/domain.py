"""Domain records shared by the Skunk server modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return str(uuid4())


class TaskStatus(StrEnum):
    RECEIVED = "RECEIVED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    SCORED = "SCORED"
    CANCELLED = "CANCELLED"


class SubmissionStatus(StrEnum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    SCORED = "SCORED"


class HumanReviewStatus(StrEnum):
    OPEN = (
        "OPEN"  # awaiting a human (the task ran optimistically and is still reviewable)
    )
    RESOLVED = "RESOLVED"  # a human submitted a correction (or accept-as-is)
    CANCELLED = "CANCELLED"  # round closed / superseded before a human got to it


@dataclass
class HumanReview:
    """One open, optimistic human review of an agent result (a table/vector extract, a figure
    read, or an external lookup). Registered while the task runs WITHOUT blocking it; a resolve
    carries the human's corrected value(s) back as a JSON `response`, which drives a recompute.
    `guidance` carries the review payload the UI needs: `task` (verify_extract/figure/lookup),
    `branch_id` (the recompute target), `branch` identity, `candidates` (the model's values), and
    `fields` (the editable field set)."""

    task_id: str
    attempt_id: str
    kind: str
    instructions: str
    context: str | None = None
    source_docs: list[str] = field(default_factory=list)
    guidance: dict[str, Any] = field(default_factory=dict)
    review_id: str = field(default_factory=new_id)
    status: HumanReviewStatus = HumanReviewStatus.OPEN
    response: str | None = None
    response_source_docs: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)
    resolved_at: datetime | None = None


@dataclass
class Attempt:
    task_id: str
    attempt_number: int
    worker_id: str
    context_feedback: list[str] = field(default_factory=list)
    previous_attempt_ids: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    attempt_id: str = field(default_factory=new_id)
    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None


@dataclass
class AnswerCandidate:
    attempt_id: str
    answer_text: str
    reasoning: str
    source_docs: list[str] = field(default_factory=list)
    submission_type: str = "agent"
    candidate_id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class FailureRecord:
    attempt_id: str
    error_type: str
    error_message: str
    traceback: str | None = None
    failure_id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class SubmissionRecord:
    task_id: str
    candidate_id: str
    submission_type: str
    local_submission_id: str = field(default_factory=new_id)
    status: SubmissionStatus = SubmissionStatus.PENDING
    cup_submission_id: str | None = None
    rejection_reason: str | None = None
    correct: bool | None = None
    points_awarded: float | None = None
    submitted_at: datetime = field(default_factory=utc_now)


@dataclass
class QuestionTask:
    task_id: str
    round_num: int
    question_id: str
    prompt: str
    status: TaskStatus = TaskStatus.RECEIVED
    current_attempt_id: str | None = None
    attempts: list[Attempt] = field(default_factory=list)
    answer_candidates: list[AnswerCandidate] = field(default_factory=list)
    failures: list[FailureRecord] = field(default_factory=list)
    submissions: list[SubmissionRecord] = field(default_factory=list)
    cup_feedback: list[str] = field(default_factory=list)
    reviews: list[HumanReview] = field(default_factory=list)
    # The snapshot (JSON form of orchestrator.RecomputeState) that produced the latest answer;
    # a resolved review recomputes from it. None until the first attempt completes a compute.
    recompute_state: dict[str, Any] | None = None
    round_ends_at: datetime | None = None
    version: int = 0
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    @property
    def latest_candidate(self) -> AnswerCandidate | None:
        return self.answer_candidates[-1] if self.answer_candidates else None

    @property
    def open_reviews(self) -> list[HumanReview]:
        return [r for r in self.reviews if r.status == HumanReviewStatus.OPEN]


@dataclass
class RoundState:
    round_num: int | None = None
    status: str = "DISCONNECTED"
    ends_at: datetime | None = None
    opens_at: datetime | None = None
    resubmits_left: int | None = None
    connection_status: str = "starting"
    last_event: str = ""


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    return value
