"""Per-question execution trace + value-summary helpers.

Pure data + formatting; no operator or LLM imports. The orchestrator
populates a `QuestionTrace` as it walks a Plan; `describe_value` /
`full_repr` are the value-summary helpers used at each step boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from skunk.models import AnnotatedValue, PageRef
from skunk.plan import Plan


@dataclass
class StepTrace:
    op: str
    args: dict
    input_desc: str
    output_desc: str
    elapsed_s: float
    error: str | None = None
    input_full: str = ""   # full repr (no truncation), for trace dump
    output_full: str = ""  # full repr (no truncation), for trace dump
    step_idx: int = 0


@dataclass
class QuestionTrace:
    question: str
    answer: str | None = None
    failed: bool = False
    failure_reason: str | None = None
    steps: list[StepTrace] = field(default_factory=list)

    def pretty(self) -> str:
        lines = [f"Question: {self.question}"]
        if self.failed:
            lines.append(f"FAILED: {self.failure_reason}")
        else:
            lines.append(f"Answer: {self.answer}")
        lines.append(f"Steps ({len(self.steps)}):")
        for s in self.steps:
            status = f"ERROR: {s.error}" if s.error else "OK"
            lines.append(f"  [{s.op}] {status} ({s.elapsed_s:.2f}s)")
            lines.append(f"    in:  {s.input_desc}")
            lines.append(f"    out: {s.output_desc}")
        return "\n".join(lines)


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


def full_repr(v: Any) -> str:
    """Untruncated repr for trace dumps."""
    if v is None:
        return "(none)"
    if isinstance(v, Plan):
        return v.model_dump_json(indent=2)
    if isinstance(v, list) and v and isinstance(v[0], PageRef):
        refs = "\n    ".join(repr(r) for r in v)
        return f"page refs ({len(v)}):\n    {refs}"
    if isinstance(v, list) and v and isinstance(v[0], AnnotatedValue):
        return repr(v)
    if isinstance(v, str):
        return f"str: {v!r}"
    if isinstance(v, list):
        parts = [f"  [{i}] {full_repr(x)}" for i, x in enumerate(v)]
        return "list:\n" + "\n".join(parts)
    return repr(v)
