"""Cross-module signal exceptions raised by operators, the planner, and the orchestrator."""

from __future__ import annotations

from typing import Any


class StepFailed(Exception):
    """Terminal failure from a plan step.

    `reason` is the terse, stable label (used in the exception message and the
    trace). `diagnostic` is an optional richer, multi-line explanation — what
    the step tried, anything it found, and what blocked it — produced by agents
    that can summarize their own trajectory (see `MultiTurnAgent`). It rides
    untruncated up to the replanner so it can pivot to a workable source/kind;
    operators that don't produce one (extract, plain compute) leave it None and
    the consumers degrade gracefully."""
    def __init__(
        self,
        op: str,
        reason: str,
        diagnostic: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(f"[{op}] {reason}")
        self.op = op
        self.reason = reason
        self.diagnostic = diagnostic
        self.details = dict(details or {})


class ParseError(StepFailed):
    """A `PromptedCall` parser could not parse the model's raw reply. `call()`
    re-prompts once (echoing `raw` + `detail`); a survivor is terminal.
    `retryable=False` skips the re-prompt entirely — for defects re-prompting
    cannot fix because the INPUT lacks what the reply needs (e.g. the extract
    verbatim guard on a corrupt-OCR page), so the caller fails fast to its
    fallback tier instead of burning escalating-temperature attempts."""
    def __init__(self, raw: str, detail: str, *, retryable: bool = True):
        super().__init__("parse", detail)
        self.raw = raw
        self.detail = detail
        self.retryable = retryable


class MissingData(Exception):
    """Terminal recovery failure: the orchestrator exhausted its replan budget
    and compute still reported insufficient data (`NeedsMore`). Raised only by
    `Orchestrator.execute()` — compute itself returns the structured signal
    rather than raising. `missing` names the data compute last said it needed."""
    def __init__(self, reason: str, missing: list[str] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.missing: list[str] = list(missing) if missing else []
