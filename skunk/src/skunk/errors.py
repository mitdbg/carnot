"""Cross-module signal exceptions raised by operators, the planner, and the orchestrator."""

from __future__ import annotations


class StepFailed(Exception):
    """Terminal failure from a plan step.

    `reason` is the terse, stable label (used in the exception message and the
    trace). `diagnostic` is an optional richer, multi-line explanation — what
    the step tried, anything it found, and what blocked it — produced by agents
    that can summarize their own trajectory (see `MultiTurnAgent`). It rides
    untruncated up to the replanner so it can pivot to a workable source/kind;
    operators that don't produce one (extract, plain compute) leave it None and
    the consumers degrade gracefully."""
    def __init__(self, op: str, reason: str, diagnostic: str | None = None):
        super().__init__(f"[{op}] {reason}")
        self.op = op
        self.reason = reason
        self.diagnostic = diagnostic


class ParseError(StepFailed):
    """A `PromptedCall` parser could not parse the model's raw reply. `call()`
    re-prompts once (echoing `raw` + `detail`); a survivor is terminal."""
    def __init__(self, raw: str, detail: str):
        super().__init__("parse", detail)
        self.raw = raw
        self.detail = detail


class MissingData(Exception):
    """Compute determined its input is insufficient to produce an answer.
    `missing` names the data codegen said it needed (empty unless codegen
    returned the structured missing-JSON form)."""
    def __init__(self, reason: str, missing: list[str] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.missing: list[str] = list(missing) if missing else []
