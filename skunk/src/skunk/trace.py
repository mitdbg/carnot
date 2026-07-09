"""Rendering helpers for the per-question event stream.

Skunk has two kinds of diagnostic output:

1. **Per-question events** — the callee-owned internal detail an operator emits
   about its own decisions (`ExecutionContext.emit`). Each event is *captured*
   into a per-question list (`ctx.events`, consumed by the trace dump), streamed
   as a JSON line to the question's own `.jsonl` file when one is open (the
   durable machine-readable record), and *echoed* to the console when the
   question runs verbose.
2. **Everything else** — offline corpus prep and build pipelines own their own
   logging (stdlib `logging`/`print`); skunk does not configure it.

This module holds the pure rendering helpers for (1): `render_line` formats one
event as a scannable console one-liner, and `truncate` is the shared
field-capping rule (also used by the trace dumps). No state, no configuration.
"""

from __future__ import annotations

from datetime import datetime

# Field reprs longer than this are capped on the rendered line (the per-question
# `.jsonl` file keeps them full). Keeps a single log line scannable.
_CONSOLE_FIELD_CAP = 200


def truncate(s: str, cap: int, suffix: str = "…") -> str:
    """Cap an over-long field rendering with a suffix. Shared by `render_line`
    (console echo) and the trace dump so the truncation rule lives in one place."""
    return s if len(s) <= cap else s[:cap] + suffix


def render_line(evt: dict) -> str:
    """Render one event dict as a single readable log line:
    `HH:MM:SS [level  ] message  key=val  key=val`. The `message` carries the
    event's data inline (interpolated at the emit site); the trailing `key=val`s
    are the event metadata (`step_idx`, `op`, and `uid` when present).
    Long string metadata is capped for readability; the per-question `.jsonl`
    file keeps it whole. Used for the verbose console echo in
    `ExecutionContext.emit`.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    level = evt.get("level", "info")
    parts = [f"{ts} [{level:<7}] {evt.get('message', '')}"]
    for key, value in evt.items():
        # `kind` (a viewer color-coding role), `data` (a structured payload, often a
        # large blob), and `t` (the viewer's seconds-since-query-start offset; the
        # line already carries a wall-clock HH:MM:SS) are for the captured record /
        # per-question `.jsonl` and the trace viewer, not the scannable one-liner —
        # keep them off the line.
        if key in ("message", "level", "kind", "data", "t"):
            continue
        s = (
            truncate(value, _CONSOLE_FIELD_CAP)
            if isinstance(value, str)
            else repr(value)
        )
        parts.append(f"{key}={s}")
    return "  ".join(parts)
