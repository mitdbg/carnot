"""Unified observability spine — one place to configure logging for the whole repo.

Skunk has three kinds of diagnostic output that used to be wired separately:

1. **Per-question events** — the callee-owned internal detail an operator emits
   about its own decisions (`HarnessContext.emit`). These are *captured* into a
   per-question list (`ctx.events`, consumed by the trace dump), streamed to a
   per-question log file when one is open, and *echoed* to the console when the
   question runs verbose.
2. **Process-scoped logs** — build pipelines (`page_index/`), offline prep
   (`search_agent/`), and library warnings (`LLMClient` retries) that have no
   per-question `ctx`. These go through stdlib `logging.getLogger(__name__)`.
3. **A durable machine-readable record** — one JSON line per event.

Everything renders through one function, `render_line`: the per-question console
echo and the per-question `.log` file call it directly, and process-scoped stdlib
logs reach it via `_LineFormatter` on the root handler. So all output shares one
format and carries a severity level — with no external dependency (capture, the
JSONL sink, and per-question files are all per-question, which a process-global
logging framework cannot route correctly when the eval runs questions
concurrently — so they live in `HarnessContext.emit`, not here).

Call `configure_obs()` once at process start (the eval harness does; library use
triggers it lazily on first `HarnessContext` construction). It is idempotent.
"""

from __future__ import annotations

import json
import logging
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
_jsonl_fh: TextIO | None = None
_jsonl_lock = threading.Lock()


def truncate(s: str, cap: int, suffix: str = "…") -> str:
    """Cap an over-long field rendering with a suffix. Shared by `render_line`
    (console/`.log`) and the trace dump so the truncation rule lives in one place."""
    return s if len(s) <= cap else s[:cap] + suffix


def render_line(evt: dict) -> str:
    """Render one event dict as a single readable log line:
    `HH:MM:SS [level  ] message  key=val  key=val`. Long string fields are capped
    for readability; the JSONL sink keeps them whole. Used for the console echo,
    the per-question `.log` file, and (via `_LineFormatter`) stdlib records.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    level = evt.get("level", "info")
    parts = [f"{ts} [{level:<7}] {evt.get('message', '')}"]
    for key, value in evt.items():
        if key in ("message", "level"):
            continue
        s = truncate(value, _CONSOLE_FIELD_CAP) if isinstance(value, str) else repr(value)
        parts.append(f"{key}={s}")
    return "  ".join(parts)


class _LineFormatter(logging.Formatter):
    """Renders a stdlib `LogRecord` through `render_line`, so process-scoped logs
    (the `logging.getLogger(__name__)` sites, the retry warning, build pipelines)
    share the exact format of the per-question event stream."""

    def format(self, record: logging.LogRecord) -> str:
        return render_line({
            "message": record.getMessage(),
            "level": record.levelname.lower(),
            "logger": record.name,
        })


def configure_obs(*, jsonl_path: str | None = None) -> None:
    """Configure the process-wide logging pipeline. Idempotent.

    Installs `_LineFormatter` on the root handler so every stdlib logger renders
    like the event stream. The root stays quiet (third-party libs at WARNING)
    while the whole `skunk.*` tree is allowed through at INFO. `jsonl_path`, when
    given, opens the durable JSON-line sink fed by `write_jsonl`.
    """
    global _configured, _jsonl_fh
    with _config_lock:
        if _configured:
            # Allow a later call to attach a JSONL sink even if rendering is set.
            if jsonl_path and _jsonl_fh is None:
                _jsonl_fh = open(jsonl_path, "a", encoding="utf-8")  # noqa: SIM115
            return

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_LineFormatter())
        root = logging.getLogger()
        root.handlers = [handler]
        # Keep the root quiet (third-party libs stay at WARNING) but let the whole
        # `skunk.*` tree — the process-scoped `getLogger(__name__)` sites — through
        # at INFO. (Per-question events do not use the root handler; they render
        # via `render_line` in `HarnessContext.emit`.)
        root.setLevel(logging.WARNING)
        logging.getLogger("skunk").setLevel(logging.INFO)

        if jsonl_path:
            _jsonl_fh = open(jsonl_path, "a", encoding="utf-8")  # noqa: SIM115

        _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Process-scoped logger for code with no per-question `ctx` (build pipelines,
    offline prep). Per-question events go through `ctx.emit`. Defaults to the
    `skunk` logger so output sits under the one tree configured at INFO."""
    if not _configured:
        configure_obs()
    return logging.getLogger(name or "skunk")


def write_jsonl(event: dict) -> None:
    """Append one full (untruncated) event as a JSON line to the durable sink.
    No-op when no sink is configured. Thread-safe across concurrent questions."""
    if _jsonl_fh is None:
        return
    line = json.dumps(event, default=str, ensure_ascii=False)
    with _jsonl_lock:
        _jsonl_fh.write(line + "\n")
        _jsonl_fh.flush()
