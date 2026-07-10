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

from skunk.common import AnnotatedValue, Final, NeedsMore, PageRef, page_key_to_pageref
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
    if isinstance(v, Final):
        return f"str: {v.answer!r}"
    if isinstance(v, NeedsMore):
        return f"needs_more n_keep={len(v.keep)} missing={v.missing!r}"
    if isinstance(v, list):
        return f"list({len(v)} branches)"
    s = repr(v)
    return s[:100] + ("..." if len(s) > 100 else "")


def _summarize_annotated(e: AnnotatedValue) -> dict:
    """One `AnnotatedValue` as a JSON-able dict for the trace viewer's node summary."""
    return {
        "description": e.description,
        "notes": e.notes,
        "unit": e.unit,
        "value_kind": e.kind,
        "value": e.value,
        "source_stem": e.source_stem,
        "pages": list(e.pages),
    }


def summarize_value(v: Any) -> dict:
    """Structured, JSON-able summary of an operator's return value — the machine-
    readable companion to `describe_value`, attached to the `step` boundary event so
    the trace viewer can render a rich collapsed node summary (pages a retrieve
    returned, extracted values, the computed answer)."""
    if v is None:
        return {"type": "none"}
    if isinstance(v, Plan):
        return {
            "type": "plan",
            "branches": [b.model_dump(mode="json") for b in v.branches],
        }
    if isinstance(v, list) and v and isinstance(v[0], PageRef):
        return {
            "type": "pages",
            "pages": [{"stem": p.stem, "page": p.page} for p in v],
        }
    if isinstance(v, list) and v and isinstance(v[0], AnnotatedValue):
        return {"type": "values", "values": [_summarize_annotated(e) for e in v]}
    if isinstance(v, str):
        return {"type": "answer", "answer": v}
    if isinstance(v, Final):
        return {"type": "answer", "answer": v.answer}
    if isinstance(v, NeedsMore):
        return {
            "type": "needs_more",
            "missing": v.missing,
            "missing_reason": v.missing_reason,
            "keep": [_summarize_annotated(e) for e in v.keep],
        }
    if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
        # The search-agent retriever returns page keys ("YYYY_MM_pageid") rather than
        # PageRefs. Parse them back so the node summary is the same "pages" shape the
        # trace viewer renders; fall through to a plain list if any key doesn't parse.
        try:
            refs = [page_key_to_pageref(x) for x in v]
            return {"type": "pages", "pages": [{"stem": p.stem, "page": p.page} for p in refs]}
        except ValueError:
            pass
    if isinstance(v, list):
        return {"type": "list", "n": len(v)}
    dump = getattr(v, "model_dump", None)
    if callable(dump):
        try:
            return {"type": type(v).__name__, "value": dump(mode="json")}
        except Exception:
            pass
    return {"type": "scalar", "value": describe_value(v)}
