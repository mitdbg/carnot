"""Eval-time helpers shared across harnesses.

Plan-cache CSV read/upsert and per-question trace dumping. These are
benchmark-driven concerns (not part of the agent's public API) and live
here so the `skunk` package stays focused on the Orchestrator + operator
surface.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skunk.plan import PageRef
    from skunk.trace import QuestionTrace


# ---------------------------------------------------------------------------
# Plan cache
# ---------------------------------------------------------------------------

def load_plan_cache(plan_cache_csv: str) -> dict[str, str]:
    """Return {uid: plan_json_string} from the plan cache CSV."""
    p = Path(plan_cache_csv)
    if not p.exists():
        return {}
    with p.open(newline="", encoding="utf-8") as f:
        return {row["uid"]: row["plan_json"] for row in csv.DictReader(f) if row.get("plan_json")}


def save_plan_to_cache(uid: str, question: str, plan_json: str, plan_cache_csv: str) -> None:
    """Upsert (uid, plan_json) into the plan cache CSV."""
    p = Path(plan_cache_csv)
    p.parent.mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict] = {}
    if p.exists():
        with p.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row["uid"]] = row

    rows[uid] = {"uid": uid, "question": question, "plan_json": plan_json}

    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "question", "plan_json"])
        writer.writeheader()
        writer.writerows(rows.values())


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
    trace: "QuestionTrace",
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
    lines.append(f"Outcome: {'FAILED — ' + (trace.failure_reason or '') if trace.failed else 'OK'}")
    lines.append(f"Final answer: {trace.answer!r}")
    lines.append("")

    events_per_step: dict[int, list[dict]] = {s.step_idx: [] for s in trace.steps}
    cur_idx: int | None = None
    for ev in events:
        if ev.get("source") == "_step" and ev.get("message") == "begin":
            cur_idx = int(ev.get("step_idx", 0))
            continue
        if cur_idx is not None and cur_idx in events_per_step:
            events_per_step[cur_idx].append(ev)

    for step in trace.steps:
        lines.append("-" * 80)
        lines.append(f"Step {step.step_idx}: {step.op}  args={step.args}  ({step.elapsed_s:.2f}s)")
        lines.append(f"  in:  {step.input_full}")
        if step.error:
            lines.append(f"  ERROR: {step.error}")
        else:
            lines.append(f"  out: {step.output_full}")
        evs = events_per_step.get(step.step_idx, [])
        if evs:
            lines.append("  events:")
            for ev in evs:
                src = ev.get("source", "")
                msg = ev.get("message", "")
                extras = {k: v for k, v in ev.items() if k not in {"source", "message"}}
                lines.append(f"    [{src}] {msg}")
                for k, v in extras.items():
                    s = repr(v)
                    # Keep full LLM I/O for performance/cost analysis; truncate everything else.
                    if not (src == "llm" and k in ("input_text", "output_text")) and len(s) > 800:
                        s = s[:800] + "...(truncated)"
                    lines.append(f"      {k}: {s}")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
