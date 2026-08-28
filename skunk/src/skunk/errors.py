"""Exceptions raised by the agents."""

from __future__ import annotations

# TODO: delete (?)
class StepFailed(Exception):
    """Terminal failure from a plan step.

    `reason` is the succinct error label (used in the exception message and the
    trace). `diagnostic` is an optional richer, multi-line explanation — what
    the step tried, anything it found, and what blocked it — produced by agents
    that can summarize their own trajectory (see `MultiTurnAgent`)."""
    def __init__(
        self,
        op: str,
        reason: str,
        diagnostic: str | None = None,
    ):
        super().__init__(f"[{op}] {reason}")
        self.op = op
        self.reason = reason
        self.diagnostic = diagnostic


class ParseError(StepFailed):
    """A `MultiTurnAgent` parser could not parse the model's raw reply."""
    def __init__(self, detail: str):
        super().__init__("parse", detail)
        self.detail = detail
