"""Eval-time helpers.

Per-question trace dumping. Lives here (not in the `skunk` package) so the
agent stays focused on the Orchestrator + operator surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from skunk.trace import truncate

if TYPE_CHECKING:
    from skunk.plan import PageRef
    from skunk.result import ExecutionResult


# Per-event message cap in trace dumps, so a single trace stays human-scannable.
# The JSONL sink keeps each message whole for post-mortems.
_TRACE_FIELD_MAX_REPR = 800


# ---------------------------------------------------------------------------
# Per-question debug trace dump
# ---------------------------------------------------------------------------

def dump_trace(
    path: str,
    *,
    uid: str | None,
    question: str,
    plan_text: str,
    golden_pages: list["PageRef"] | None,
    result: "ExecutionResult",
    events: list[dict],
    model: str,
) -> None:
    """Write a comprehensive per-question debug trace to `path`."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"UID: {uid or '(none)'}")
    lines.append("=" * 80)
    lines.append(f"Question: {question}")
    lines.append(f"Model: {model}")
    lines.append(f"Plan: {plan_text}")
    if golden_pages:
        lines.append(f"Golden pages ({len(golden_pages)}):")
        for g in golden_pages:
            lines.append(f"  - month={g.month} page={g.page}")
    else:
        lines.append("Golden pages: (none)")
    lines.append("")
    lines.append(f"Outcome: {'FAILED — ' + (result.failure_reason or '') if result.failed else 'OK'}")
    lines.append(f"Final answer: {result.answer!r}")
    lines.append("")

    # Every event carries a `(step_idx, op)` stamp. Group by step_idx; the orchestrator's
    # one `step ...` boundary event per operator call sits in its group like any other
    # line (its message holds elapsed/output/error inline). step_idx=None events
    # (out-of-step orchestrator emits) go in an "ungrouped" preamble.
    by_step: dict[int, list[dict]] = {}
    ungrouped: list[dict] = []
    for ev in events:
        idx = ev.get("step_idx")
        if idx is None:
            ungrouped.append(ev)
        else:
            by_step.setdefault(idx, []).append(ev)

    if ungrouped:
        lines.append("-" * 80)
        lines.append("Ungrouped events:")
        for ev in ungrouped:
            lines.extend(_format_event(ev))
        lines.append("")

    for idx in sorted(by_step):
        evs = by_step[idx]
        op = evs[0].get("op", "?")
        lines.append("-" * 80)
        lines.append(f"Step {idx}: {op}")
        for ev in evs:
            lines.extend(_format_event(ev))
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def _format_event(ev: dict) -> list[str]:
    """Render one captured event as an indented trace line: the event `message`
    (prefixed with the level when not "info"), capped for scannability — the JSONL
    sink keeps the full text."""
    msg = ev.get("message", "")
    level = ev.get("level", "info")
    prefix = f"{level.upper()} " if level != "info" else ""
    return [f"    {prefix}{truncate(msg, _TRACE_FIELD_MAX_REPR, '...(truncated)')}"]
