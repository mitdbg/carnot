"""Cross-module signal exceptions raised by operators, the planner, and the orchestrator."""

from __future__ import annotations


class StepFailed(Exception):
    """Terminal failure from a plan step."""
    def __init__(self, op: str, reason: str):
        super().__init__(f"[{op}] {reason}")
        self.op = op
        self.reason = reason


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
