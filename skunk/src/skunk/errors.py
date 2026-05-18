"""Cross-module signal exceptions raised by operators, the planner, and the
orchestrator.
"""

from __future__ import annotations


class StepFailed(Exception):
    """Terminal failure from a plan step."""
    def __init__(self, op: str, reason: str):
        super().__init__(f"[{op}] {reason}")
        self.op = op
        self.reason = reason


class MissingData(Exception):
    """Compute determined its input is insufficient to produce an answer.

    `missing` is an optional list of short identifier strings naming the
    data the codegen step said it would need — populated when codegen
    returned the structured missing-JSON form; empty otherwise.
    """
    def __init__(self, reason: str, missing: list[str] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.missing: list[str] = list(missing) if missing else []
