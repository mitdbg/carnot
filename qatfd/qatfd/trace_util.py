"""Per-question debug trace dump (qatfd flavor).

A thin local version of skunk's eval/util.py:dump_trace — that one is OfficeQA/plan
shaped (requires plan_text / golden_pages / ExecutionResult). We only need the
question, the outcome, and the grouped event stream.
"""

from __future__ import annotations

import json
from pathlib import Path

from skunk.trace import truncate

# Cap for the scannable one-liner message; the structured `data` payload (full prompts,
# the assistant turn, retrieved context) gets a much larger cap so the .txt is a
# self-contained per-question record (the `.jsonl` carries the full structured events;
# this `.txt` is the human-readable counterpart).
_TRACE_FIELD_MAX_REPR = 800
_TRACE_DATA_MAX = 20000


def dump_trace(
    path: str,
    *,
    qid: str,
    benchmark: str,
    system: str,
    question: str,
    predicted: str,
    gold: str,
    score: float,
    failed: bool,
    reason: str,
    events: list[dict],
    model: str,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"QID: {qid}  |  benchmark={benchmark}  system={system}")
    lines.append("=" * 80)
    lines.append(f"Question: {question}")
    lines.append(f"Model: {model}")
    lines.append(f"Outcome: {'FAILED — ' + (reason or '') if failed else f'OK (score={score:.3f})'}")
    lines.append(f"Gold answer:      {gold!r}")
    lines.append(f"Predicted answer: {predicted!r}")
    lines.append("")

    # Group events by (step_idx, op), like eval/util.py.
    by_step: dict[int, list[dict]] = {}
    ungrouped: list[dict] = []
    for ev in events:
        idx = ev.get("step_idx")
        (ungrouped if idx is None else by_step.setdefault(idx, [])).append(ev)

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
    msg = ev.get("message", "")
    level = ev.get("level", "info")
    prefix = f"{level.upper()} " if level != "info" else ""
    lines = [f"    {prefix}{truncate(msg, _TRACE_FIELD_MAX_REPR, '...(truncated)')}"]
    # Render the structured `data` payload (excluded from the console / .log one-liners
    # by design) so the per-question .txt carries the full prompts / answer / retrieved
    # context — long or multi-line strings as indented blocks, everything else inline.
    data = ev.get("data")
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, str) and ("\n" in val or len(val) > 120):
                lines.append(f"        {key}:")
                block = truncate(val, _TRACE_DATA_MAX, "\n...(truncated)")
                lines.extend(f"        | {ln}" for ln in block.splitlines())
            else:
                rendered = val if isinstance(val, str) else json.dumps(val, default=str)
                lines.append(f"        {key}={truncate(rendered, _TRACE_DATA_MAX, '...(truncated)')}")
    return lines
