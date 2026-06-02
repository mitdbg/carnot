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


# Per-field repr cap in trace dumps. LLM input/output text is kept full for
# cost/latency post-mortems; everything else is capped so a single trace
# stays human-scannable on stderr.
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

    # Operator boundaries are `("orchestrator", "step")` events carrying
    # op / elapsed / output / error, stamped with their step_idx like every other
    # event. Group all events by step_idx; for each step the boundary event is the
    # header and the rest are its internals. step_idx=None events go in an
    # "ungrouped" preamble.
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
        boundary = next(
            (e for e in evs if e.get("source") == "orchestrator" and e.get("message") == "step"),
            None,
        )
        # Every event in the group carries the same `op` (stamped by ctx.step);
        # the boundary may be absent only on an uncaught crash, so read it off any event.
        op = evs[0].get("op", "?")
        elapsed = (boundary or {}).get("elapsed_s")
        head = f"Step {idx}: {op}"
        if isinstance(elapsed, (int, float)):
            head += f"  ({elapsed:.2f}s)"
        lines.append("-" * 80)
        lines.append(head)
        if boundary and boundary.get("error"):
            lines.append(f"  ERROR: {boundary['error']}")
        elif boundary:
            lines.append(f"  out: {boundary.get('output_full', '')}")
        internals = [e for e in evs if e is not boundary]
        if internals:
            lines.append("  events:")
            for ev in internals:
                lines.extend(_format_event(ev))
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


# Per-event meta keys rendered in the header / step grouping, not as fields.
_EVENT_META = {"source", "message", "step_idx", "op", "level"}


def _format_event(ev: dict) -> list[str]:
    """Render one captured event as indented trace lines: a `[source] message`
    header (prefixed with the level when not "info") plus its fields."""
    src = ev.get("source", "")
    msg = ev.get("message", "")
    level = ev.get("level", "info")
    prefix = f"{level.upper()} " if level != "info" else ""
    out = [f"    {prefix}[{src}] {msg}"]
    for k, v in ev.items():
        if k in _EVENT_META:
            continue
        s = repr(v)
        if k not in ("input_text", "output_text"):
            s = truncate(s, _TRACE_FIELD_MAX_REPR, "...(truncated)")
        out.append(f"      {k}: {s}")
    return out
