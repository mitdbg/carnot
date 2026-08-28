"""This module holds the class which captures agent traces.

Skunk has two kinds of diagnostic output:

1. **Agent events** — Agents emit events throughout their execution to record inputs,
   outputs, decisions, and various execution metrics. Each event is a JSON serializable
   dictionary which the Tracer writes in streaming, append-only fashion to a `.jsonl`
   file (unique to an agent's execution) and echoes to the console when configured.
2. **Everything else** — offline corpus prep and build pipelines own their own
   logging (stdlib `logging`/`print`); skunk does not configure it.
"""

from __future__ import annotations

import json
import time

from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
from typing import Literal, get_args

# define types for categoricals
Level = Literal["info", "warning", "error"]
LEVEL_VALUES = get_args(Level)

Kind = Literal["system", "user", "assistant", "call", "tool_call", "background", "observation", "error", "lifecycle"] # `plan` / `summary` / `step` / `note`
KIND_VALUES = get_args(Kind)

# define schema for every event
class TraceEvent(BaseModel):
    # a short, unique identifier for the call site that generated this event
    id: str
    # wall-clock timestamp for the event (epoch seconds, from time.time())
    ts: float
    # time elapsed since the agent started executing
    t: float
    # one of "info", "warning", or "error" (default: "info")
    level: Level = "info"
    # an optional categorical string which is used by trace viewers to better visualize the trace.
    kind: Kind | None = None
    # the step in the agent trace where this event was generated (optional b/c not every event is neatly tied to a step)
    step: int | None = None
    # the turn within the agent step when this event was generated (for e.g. events during a retry)
    turn: int | None = None
    # an optional human-readable string for events which don't have large data payloads.
    message: str | None = None
    # an optional JSON serializable dictionary for event payloads; it is removed from rendered lines.
    data: dict | None = None


# # Map a message's leading snake_case event key to a semantic `kind` (the role used
# # by the trace viewer for color-coding). Call sites that pass `kind=` explicitly win;
# # this only classifies the legacy one-liner emits that don't. Unknown keys → "note"
# # (a neutral, informational event).
# _KIND_BY_PREFIX: dict[str, str] = {
#     "question": "user",
#     "observation": "observation",
#     "error": "error",
#     "call": "call",
#     "step": "step",
#     "validation_failed": "error",
#     # Harness lifecycle events — messages the loop injected into the agent's context
#     # (low-steps warning) or actions it took after exhausting the step budget (the
#     # forced terminal turn). Grouped under one `lifecycle` kind so the viewer can
#     # color them distinctly and gate them behind a single toggle.
#     "steps_low_warning": "lifecycle",
#     "terminal_prompt": "lifecycle", 
#     "terminal_reply": "lifecycle",
#     "terminal_commit": "lifecycle",
#     "terminal_giveup": "lifecycle",
#     "terminal_turn_failed": "lifecycle",
#     "parallel_branch_failed": "error",
#     "compute_needs_more": "note",
#     "compute_partial_malformed": "note",
#     "pool_update": "note",
#     "compute_short_circuit": "note",
#     "recovery_exhausted": "error",
# }

# def infer_kind(message: str) -> str:
#     """Classify a one-liner `emit` message into a semantic `kind` from its leading
#     event key (see `_KIND_BY_PREFIX`). Used when a call site doesn't pass `kind=`."""
#     token = message.split(" ", 1)[0] if message else ""
#     return _KIND_BY_PREFIX.get(token, "note")


class Tracer:
    """
    Serializes agent trajectories into event streams which are written to structured .jsonl files.
    Each event in the event stream is a JSON serializable dictionary with a custom schema defined
    by the caller. This provides us with flexibility in determining what data to capture from our
    agent traces.

    Precondition:
      - log_path is a path to a location on local disk.
    """
    # character limit on the field length when rendering the event stream in the console
    _CONSOLE_FIELD_CAP = 200

    def __init__(self, log_path: str, start_time: float | None = None, verbose: bool = False):
        self._start_time = start_time or time.monotonic()
        self._verbose = verbose
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._logfile = open(log_path, "w", encoding="utf-8")

    def close(self) -> None:
        """Close the per-question log file, if one was opened."""
        if self._logfile is not None:
            self._logfile.close()
            self._logfile = None

    @staticmethod
    def truncate(s: str, cap: int, suffix: str = "…") -> str:
        """Cap an over-long field rendering with a suffix. Shared by `render_line`
        (console echo) and the trace dump so the truncation rule lives in one place."""
        return s if len(s) <= cap else s[:cap] + suffix

    def render_line(self, event: TraceEvent) -> str:
        """Render one event as a single readable log line:
        `HH:MM:SS [level  ] id  message  key=val  key=val`. The `message` carries the
        event's data inline (interpolated at the emit site); the trailing `key=val`s
        are the event metadata (`step`, `turn`). Long string metadata is capped for
        readability; the `.jsonl` file keeps it whole. Used for the verbose console
        echo in `emit`.
        """
        ts = datetime.fromtimestamp(event.ts).strftime("%H:%M:%S")
        parts = [f"{ts} [{event.level:<7}] {event.id}"]
        if event.message is not None:
            parts.append(event.message)

        # exclude fields which are already in the log-line or should not be rendered (e.g. `data`)
        metadata = event.model_dump(
            exclude={"id", "ts", "t", "level", "kind", "message", "data"}, exclude_none=True
        )
        for key, value in metadata.items():
            s = Tracer.truncate(value, self._CONSOLE_FIELD_CAP) if isinstance(value, str) else repr(value)
            parts.append(f"{key}={s}")

        return "  ".join(parts)

    def emit(
        self,
        id: str,
        level: Level = "info",
        *,
        kind: Kind | None = None,
        step: int | None = None,
        turn: int | None = None,
        message: str | None = None,
        data: dict | None = None,
    ) -> None:
        """
        Write a TraceEvent to the event stream for this agent.

        Args:
            id: a short, unique identifier for the call site that generated this event
            level: one of "info", "warning", or "error" (default: "info")
            kind: an optional categorical string which is used by trace viewers to better visualize the trace.
            step: the step in the agent trace where this event was generated (optional b/c not every event is neatly tied to a step)
            turn: the turn within the agent step when this event was generated
            message: an optional human-readable string for events which don't have large data payloads.
            data: an optional JSON serializable dictionary for event payloads; it is removed from rendered lines.
        
        Returns:
            None
        """
        # construct the TraceEvent (`ts` is wall-clock for rendering; `t` is elapsed-since-start)
        event = TraceEvent(
            id=id,
            ts=time.time(),
            t=round(time.monotonic() - self._start_time, 3),
            level=level,
            kind=kind,
            step=step,
            turn=turn,
            message=message,
            data=data,
        )

        # write the event to the logfile
        if self._logfile is not None:
            self._logfile.write(json.dumps(event.model_dump(), default=str, ensure_ascii=False) + "\n")
            self._logfile.flush()

        # render and print to console if verbose=True
        if self._verbose:
            print(self.render_line(event))
