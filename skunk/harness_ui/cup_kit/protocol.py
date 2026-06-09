"""Wire protocol for the OfficeQA Cup Competitor API.

Pydantic v2 models + enums for the four agent-facing surfaces:

- ``GET  /v1/round/current``    → :class:`RoundCurrentResponse`
- ``GET  /v1/team/status``      → :class:`TeamStatusResponse`
- ``POST /v1/submit``           ← :class:`SubmitRequest`,
                                 → :class:`SubmitAcceptedResponse` |
                                   :class:`SubmitRejectedResponse`
- ``WS   /v1/team/events``      → :class:`RoundStartedEvent` |
                                   :class:`RoundStateEvent` |
                                   :class:`SubmissionScoredEvent`

These classes are kept structurally identical to the canonical
server-side classes; a contract test in research CI guards against
drift. If you're editing the kit and the contract test fails, follow
``CONTRACT_TEST.md`` at the kit root.

Constants below mirror the server's per-payload caps; the practice
harness enforces them so the agent gets the same rejection reasons it
would against the live cup.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator, Field

# ---------------------------------------------------------------------------
# Per-payload caps and timing constants; values match the live cup so
# the harness rejects malformed work with the same reasons the live
# cup would.
# ---------------------------------------------------------------------------

MAX_QUESTION_ID_CHARS: int = 64
MAX_RESUBMITS: int = 3
MAX_ANSWER_TEXT_CHARS: int = 8_192
MAX_REASONING_CHARS: int = 200_000
AGENT_MIN_REASONING_CHARS: int = 100
MAX_SOURCE_DOCS_ENTRIES: int = 64
MAX_SOURCE_DOC_REF_CHARS: int = 512
ROUND_OPEN_DELAY_SECONDS: int = 3


# ---------------------------------------------------------------------------
# Enums and literals.
# ---------------------------------------------------------------------------


RoundType = Literal["warmup", "normal", "challenge"]


class RoundStatus(str, Enum):
    """Round lifecycle. ``TRACE_DEADLINE`` remains as a legal value for
    payload compatibility but no transitions reach it in current code."""

    PRE_ROUND = "PRE_ROUND"
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    TRACE_DEADLINE = "TRACE_DEADLINE"
    RESULTS = "RESULTS"
    SKIPPED = "SKIPPED"


class SubmissionType(str, Enum):
    AGENT = "agent"
    HUMAN = "human"


class SubmitRejectionReason(str, Enum):
    ROUND_NOT_ACTIVE = "round_not_active"
    ROUND_NOT_YET_OPEN = "round_not_yet_open"
    NO_TOKENS = "no_tokens"
    QUESTION_NOT_IN_ROUND = "question_not_in_round"
    RATE_LIMITED = "rate_limited"
    ANSWER_EMPTY = "answer_empty"
    ANSWER_TOO_LONG = "answer_too_long"
    REASONING_TOO_SHORT = "reasoning_too_short"
    REASONING_TOO_LONG = "reasoning_too_long"
    SOURCE_DOCS_REQUIRED = "source_docs_required"
    SOURCE_DOCS_TOO_MANY = "source_docs_too_many"
    SOURCE_DOC_EMPTY = "source_doc_empty"
    SOURCE_DOC_TOO_LONG = "source_doc_too_long"


# ---------------------------------------------------------------------------
# Shared fragments.
# ---------------------------------------------------------------------------


class ScoreFragment(BaseModel):
    correct: bool
    points_awarded: float = Field(ge=0.0)


class QuestionFragment(BaseModel):
    question_id: str
    round_num: int
    prompt: str


# ---------------------------------------------------------------------------
# GET /v1/round/current
# ---------------------------------------------------------------------------


class RoundCurrentResponse(BaseModel):
    round_num: int
    status: RoundStatus
    ends_at: datetime | None = None
    questions: list[QuestionFragment] = Field(default_factory=list)
    resubmits_left: int = Field(ge=0, le=MAX_RESUBMITS)
    round_type: RoundType = "normal"
    display_label: str = ""


# ---------------------------------------------------------------------------
# POST /v1/submit
# ---------------------------------------------------------------------------


class SubmitRequest(BaseModel):
    question_id: str = Field(min_length=1, max_length=MAX_QUESTION_ID_CHARS)
    answer_text: str
    reasoning: str
    source_docs: list[str]
    submission_type: SubmissionType = SubmissionType.AGENT


class SubmitAcceptedResponse(BaseModel):
    accepted: Literal[True] = True
    no_op: bool = False
    submission_id: str
    score: ScoreFragment
    tokens_remaining: int = Field(ge=0)
    superseded_submission_id: str | None = None


class SubmitRejectedResponse(BaseModel):
    accepted: Literal[False] = False
    reason: SubmitRejectionReason
    tokens_remaining: int = Field(ge=0)
    submitted_chars: int | None = None
    max_chars: int | None = None
    min_chars: int | None = None
    source_doc_index: int | None = None


# ---------------------------------------------------------------------------
# GET /v1/team/status
# ---------------------------------------------------------------------------


class SubmissionHistoryEntry(BaseModel):
    submission_id: str
    answer_text: str
    submitted_at: datetime
    correct: bool | None = None
    points_awarded: float | None = None
    superseded: bool = False


class TeamQuestionStatus(BaseModel):
    question_id: str
    submission_id: str | None = None
    answer_text: str | None = None
    correct: bool | None = None
    points_awarded: float | None = None
    speed_bonus: bool = False
    submission_history: list[SubmissionHistoryEntry] = Field(default_factory=list)


class TeamStatusResponse(BaseModel):
    team_id: str
    team_name: str = ""
    sponsor: str = ""
    round_num: int
    resubmits_left: int = Field(ge=0, le=MAX_RESUBMITS)
    per_question: list[TeamQuestionStatus] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# WS /v1/team/events
# ---------------------------------------------------------------------------


class RoundStartedEvent(BaseModel):
    type: Literal["round_started"] = "round_started"
    round_num: int
    ends_at: datetime | None = None
    opens_at: datetime | None = None
    questions: list[QuestionFragment] = Field(default_factory=list)


class RoundStateEvent(BaseModel):
    type: Literal["round_state"] = "round_state"
    round_num: int
    status: RoundStatus


class SubmissionScoredEvent(BaseModel):
    type: Literal["submission_scored"] = "submission_scored"
    submission_id: str
    question_id: str
    correct: bool
    points_awarded: float


TeamEvent = Annotated[
    RoundStartedEvent | RoundStateEvent | SubmissionScoredEvent,
    Discriminator("type"),
]
