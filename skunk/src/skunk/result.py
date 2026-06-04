"""Per-question execution result + value-summary helpers.

`ExecutionResult` is the orchestrator's typed return contract — the answer and
terminal status, read programmatically (e.g. by the eval harness). It is **not**
observability: per-operator boundaries are logged through the event stream as
`step …` events, not stored here. `describe_value` summarizes an operator's
return value into that boundary event's `output=` description.

Pure data + formatting; no operator or LLM imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skunk.common import AnnotatedValue, PageRef
from skunk.plan import Plan


@dataclass
class ExecutionResult:
    """What the orchestrator produced for one question: the answer, and whether
    it terminally failed (with the reason). The step-by-step record lives in the
    event stream, not here."""

    question: str
    answer: str | None = None
    failed: bool = False
    failure_reason: str | None = None


def describe_value(v: Any) -> str:
    if v is None:
        return "(none)"
    if isinstance(v, Plan):
        return f"plan with {len(v.branches)} branch(es)"
    if isinstance(v, list) and v and isinstance(v[0], PageRef):
        return f"[{len(v)} page refs]"
    if isinstance(v, list) and v and isinstance(v[0], AnnotatedValue):
        descriptions = [e.description for e in v]
        return f"[{len(descriptions)} entries: {descriptions}]"
    if isinstance(v, str):
        return f"str: {v!r}"
    if isinstance(v, list):
        return f"list({len(v)} branches)"
    s = repr(v)
    return s[:100] + ("..." if len(s) > 100 else "")
