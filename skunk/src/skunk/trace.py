"""Unified observability spine — one place to configure logging for the whole repo.

Skunk has three kinds of diagnostic output that used to be wired separately:

1. **Per-question events** — the callee-owned internal detail an operator emits
   about its own decisions (`ExecutionContext.emit`). These are *captured* into a
   per-question list (`ctx.events`, consumed by the trace dump), streamed to a
   per-question log file when one is open, and *echoed* to the console when the
   question runs verbose.
2. **Process-scoped logs** — app build pipelines (e.g. grc-officeqa's page
   index), offline corpus prep (the benchmarks' embedding/vector-db scripts),
   and library warnings (`LLMClient`
   retries) that have no per-question `ctx`. These go through stdlib
   `logging.getLogger(__name__)`.
3. **A durable machine-readable record** — one JSON line per event.

Everything renders through one function, `render_line`: the per-question console
echo and the per-question `.log` file call it directly, and process-scoped stdlib
logs reach it via `LineFormatter` on the root handler. So all output shares one
format and carries a severity level — with no external dependency (capture, the
JSONL sink, and per-question files are all per-question, which a process-global
logging framework cannot route correctly when the eval runs questions
concurrently — so they live in `ExecutionContext.emit`, not here).

Call `configure_obs()` once at process start (the eval harness does; library use
triggers it lazily on first `ExecutionContext` construction). It is idempotent.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import sys
import threading
from datetime import datetime
from typing import TextIO

# Field reprs longer than this are capped on the rendered line (the JSONL sink
# keeps them full). Keeps a single log line scannable.
_CONSOLE_FIELD_CAP = 200

_configured = False
_config_lock = threading.Lock()

# JSONL sink: a process-global file handle guarded by a lock (the eval runs ~32
# questions concurrently in one process, all writing to one run-wide file; each
# line carries its own `uid`/`step_idx`, so a single file is unambiguous).
#
# The handle is large-buffered and `write_jsonl` does NOT flush per event — across ~32
# concurrent question threads a per-event flush is thousands of write syscalls, each
# releasing/re-acquiring the GIL and contending on `_jsonl_lock`. Instead a single
# background thread flushes every ~250 ms (and `atexit` flushes the tail), trading ≤250 ms
# of durability for collapsing those syscalls into a handful. Mirrors the console's FileSink.
_jsonl_fh: TextIO | None = None
_jsonl_lock = threading.Lock()
_JSONL_BUFFER_BYTES = 1 << 20  # large buffer so a big event rarely auto-flushes mid-write
# Flush cadence; the env override is read when the flusher STARTS (`_ensure_jsonl_flusher`),
# not at import — importing skunk must not read configuration from the environment.
_jsonl_flush_interval_s = 0.25
_jsonl_stop = threading.Event()
_jsonl_flusher: threading.Thread | None = None


def truncate(s: str, cap: int, suffix: str = "…") -> str:
    """Cap an over-long field rendering with a suffix. Shared by `render_line`
    (console/`.log`) and the trace dump so the truncation rule lives in one place."""
    return s if len(s) <= cap else s[:cap] + suffix


def render_line(evt: dict) -> str:
    """Render one event dict as a single readable log line:
    `HH:MM:SS [level  ] message  key=val  key=val`. The `message` carries the
    event's data inline (interpolated at the emit site); the trailing `key=val`s
    are the event metadata (`step_idx`, `op`, and `uid`/`logger` when present).
    Long string metadata is capped for readability; the JSONL sink keeps it whole.
    Used for the console echo, the per-question `.log` file, and (via
    `LineFormatter`) stdlib records.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    level = evt.get("level", "info")
    parts = [f"{ts} [{level:<7}] {evt.get('message', '')}"]
    for key, value in evt.items():
        # `kind` (a viewer color-coding role), `data` (a structured payload, often a
        # large blob), and `t` (the viewer's seconds-since-query-start offset; the
        # line already carries a wall-clock HH:MM:SS) are for the captured/JSONL record
        # and the trace viewer, not the scannable one-liner — keep them off the line.
        if key in ("message", "level", "kind", "data", "t"):
            continue
        s = (
            truncate(value, _CONSOLE_FIELD_CAP)
            if isinstance(value, str)
            else repr(value)
        )
        parts.append(f"{key}={s}")
    return "  ".join(parts)


class LineFormatter(logging.Formatter):
    """Renders a stdlib `LogRecord` through `render_line`, so process-scoped logs
    (the `logging.getLogger(__name__)` sites, the retry warning, build pipelines)
    share the exact format of the per-question event stream."""

    def format(self, record: logging.LogRecord) -> str:
        return render_line(
            {
                "message": record.getMessage(),
                "level": record.levelname.lower(),
                "logger": record.name,
            }
        )


def configure_obs(*, jsonl_path: str | None = None) -> None:
    """Configure the process-wide logging pipeline. Idempotent.

    Installs `LineFormatter` on the root handler so every stdlib logger renders
    like the event stream. The root stays quiet (third-party libs at WARNING)
    while the whole `skunk.*` tree is allowed through at INFO. `jsonl_path`, when
    given, opens the durable JSON-line sink fed by `write_jsonl`.
    """
    global _configured, _jsonl_fh
    with _config_lock:
        if _configured:
            # Allow a later call to attach a JSONL sink even if rendering is set.
            if jsonl_path and _jsonl_fh is None:
                _jsonl_fh = open(  # noqa: SIM115
                    jsonl_path, "a", buffering=_JSONL_BUFFER_BYTES, encoding="utf-8"
                )
                _ensure_jsonl_flusher()
            return

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(LineFormatter())
        root = logging.getLogger()
        root.handlers = [handler]
        # Keep the root quiet (third-party libs stay at WARNING) but let the whole
        # `skunk.*` tree — the process-scoped `getLogger(__name__)` sites — through
        # at INFO. (Per-question events do not use the root handler; they render
        # via `render_line` in `ExecutionContext.emit`.)
        root.setLevel(logging.WARNING)
        logging.getLogger("skunk").setLevel(logging.INFO)

        if jsonl_path:
            _jsonl_fh = open(  # noqa: SIM115
                jsonl_path, "a", buffering=_JSONL_BUFFER_BYTES, encoding="utf-8"
            )
            _ensure_jsonl_flusher()

        _configured = True


def write_jsonl(event: dict) -> None:
    """Append one full (untruncated) event as a JSON line to the durable sink. No-op when no
    sink is configured. Thread-safe across concurrent questions. Buffered — NOT flushed here;
    the background flusher (or `atexit`) pushes it to disk within `_jsonl_flush_interval_s`."""
    if _jsonl_fh is None:
        return
    line = json.dumps(event, default=str, ensure_ascii=False)
    with _jsonl_lock:
        _jsonl_fh.write(line + "\n")


def flush_jsonl() -> None:
    """Flush the buffered JSONL sink to disk. Called by the background flusher and at exit."""
    if _jsonl_fh is None:
        return
    with _jsonl_lock:
        try:
            _jsonl_fh.flush()
        except (OSError, ValueError):  # ValueError if the handle was closed
            pass


def _jsonl_flush_loop() -> None:
    # Wake every interval; Event.wait returns True only when stop is set → exit.
    while not _jsonl_stop.wait(_jsonl_flush_interval_s):
        flush_jsonl()


def _ensure_jsonl_flusher() -> None:
    """Start the single background flusher (once) and register a final flush at exit. Called
    under `_config_lock` from `configure_obs` right after the sink is opened."""
    global _jsonl_flusher, _jsonl_flush_interval_s
    if _jsonl_flusher is None:
        _jsonl_flush_interval_s = float(
            os.environ.get("SKUNK_TRACE_FLUSH_S", str(_jsonl_flush_interval_s))
        )
        _jsonl_flusher = threading.Thread(
            target=_jsonl_flush_loop, name="skunk-jsonl-flush", daemon=True
        )
        _jsonl_flusher.start()
        atexit.register(flush_jsonl)  # write the buffered tail on a clean exit
