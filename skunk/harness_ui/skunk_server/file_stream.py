"""Filesystem stream contract shared by the split web/backend processes.

The backend writes the browser-facing firehose to a per-run directory; the web process
tails it back into a local StreamHub. This module is the single source of truth for the
on-disk layout and is deliberately skunk-free (stdlib + json only) so the web process can
import it without pulling in the reasoner.

Layout under ``$SKUNK_STREAM_DIR`` (see ARCHITECTURE / the split plan):

    status.json                 whole compact {round, tasks[]} snapshot
    status.json.<pid>.tmp       transient; write-then-os.replace (atomic, same FS)
    events.<round>.jsonl        append-only; one obj/line {"task_id","attempt_id","event"}

A new round rolls to a fresh ``events.<round>.jsonl`` (``FileSink.roll_round``) so each
round's log is self-contained and bounded; the web tailer derives the same filename from
the round it reads in ``status.json``. Within a round, all tasks interleave into the one
file and the tailer demuxes per ``task_id``.

Well-formed-JSON invariant: every record is written with ``json.dumps(..., separators``
``=(",",":"))`` — compact, never ``indent=`` — so the only literal 0x0A byte in an events
file is the ``\n`` delimiter we append (json escapes any in-string newline as the two
chars ``\\n``). Every newline is therefore a true record boundary, which is what lets the
tailer split on bytes and parse complete lines without a lock.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STREAM_DIR_ENV = "SKUNK_STREAM_DIR"

_COMPACT = (",", ":")


def _json_default(value: Any) -> Any:
    """`json.dumps` fallback for the non-JSON-native leaves a raw trace event may carry
    (datetime/enum/dataclass). Invoked only for those leaves — no separate pre-walk — so the
    agent's write stays minimal. The last-resort `str()` guarantees a write never raises."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return str(value)


def stream_dir(base: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the per-run stream directory. Explicit arg wins, else $SKUNK_STREAM_DIR."""
    if base is None:
        base = os.environ.get(STREAM_DIR_ENV)
    if not base:
        raise ValueError(f"{STREAM_DIR_ENV} is not set and no base was provided")
    return Path(base)


def status_path(base: str | os.PathLike[str] | None = None) -> Path:
    return stream_dir(base) / "status.json"


def events_path(
    base: str | os.PathLike[str] | None = None, round_num: int | None = None
) -> Path:
    """The per-round event log — one round's tasks interleaved, one obj/line
    ``{"task_id","attempt_id","event"}``. A new round rolls to a fresh
    ``events.<round_num>.jsonl`` (``FileSink.roll_round``), so each round's log is
    self-contained and bounded and the web derives the same name from the round it reads.
    The `task_id` field lets the web tailer demux per task; no per-task files, no path
    encoding. ``round_num=None`` is the pre-round/legacy fallback (``events.jsonl``)."""
    name = "events.jsonl" if round_num is None else f"events.{round_num}.jsonl"
    return stream_dir(base) / name


class FileSink:
    """The agent's byte writer for the file stream. ``write_event`` is called directly on the
    reasoner worker thread(s); its only per-event job is to stamp a cheap `seq` and append one
    raw line to the active round's ``events.<round>.jsonl`` (one shared file per round; ``roll_round``
    rolls it). All heavyweight interpretation (compaction, SSE shaping) is the web process's job
    (see ``FileTailer``). ``publish_status`` is the lifecycle-driven status write, called on the
    backend main loop by the adapter/coordinator/broker.

    The shared handle is **buffered** (NOT line-buffered). The worker's ``write_event`` only
    stamps the seq and appends to the userspace buffer under one lock — no flush, no write
    syscall on the hot path — so a burst of trace events costs ~a memcpy each. A single
    background flusher thread flushes the handle to the page cache every ~250 ms
    (``SKUNK_STREAM_FLUSH_S``), so the separate tailer process sees records within that window
    (no ``fsync`` — only same-machine visibility is needed, not durability). This trades ≤250 ms
    of trace latency for collapsing thousands of per-event flush syscalls (and their GIL
    handoffs across the reasoner threads) into a handful. The seq is a per-round monotonic counter
    (reset on roll_round); it stays monotonic within each task (all the web's per-task dedup needs)."""

    _BUFFER_BYTES = 1 << 20  # large buffer so a big event rarely auto-flushes mid-write

    def __init__(
        self,
        status_provider: Callable[[], dict[str, Any]],
        base: str | os.PathLike[str] | None = None,
    ) -> None:
        self._status_provider = status_provider
        self._dir = stream_dir(base)
        self._status_path = self._dir / "status.json"
        # The active round's event log; rolls to events.<round>.jsonl on roll_round() (None
        # until the first round → pre-round writes, if any, land in the legacy events.jsonl).
        self._round_num: int | None = None
        self._events_path = events_path(self._dir)
        self._tmp_suffix = f".{os.getpid()}.tmp"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # serializes seq + append + flush on the shared file
        self._handle: Any | None = None  # lazily opened buffered append handle
        self._seq = 0  # per-round monotonic event seq (reset on roll_round)
        self._flush_interval_s = float(os.environ.get("SKUNK_STREAM_FLUSH_S", "0.25"))
        self._stop = threading.Event()
        self._flusher: threading.Thread | None = None  # lazily started on first write

    def publish_status(self) -> None:
        """Rewrite the whole compact status snapshot via temp + os.replace, so a concurrent
        reader always sees a complete document. Always writes (the file IS the channel —
        no 'no subscribers' short-circuit like the in-process hub had)."""
        try:
            payload = json.dumps(self._status_provider(), separators=_COMPACT, default=_json_default)
        except Exception:  # noqa: BLE001 - never let a bad snapshot kill the backend loop
            logger.exception("status snapshot serialization failed")
            return
        tmp = self._status_path.with_name(self._status_path.name + self._tmp_suffix)
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp, self._status_path)  # atomic on the same filesystem
        except Exception:  # noqa: BLE001
            logger.exception("status.json write failed")
            try:
                tmp.unlink()
            except OSError:
                pass

    def roll_round(self, round_num: int) -> None:
        """Switch the event log to a fresh per-round file (``events.<round_num>.jsonl``) at the
        start of a new round, so each round's stream is self-contained and bounded and the web
        (which derives the same name from ``status.json``'s round) reads only the live round.
        Called on the backend main loop by the adapter when a round goes ACTIVE — BEFORE the new
        round's tasks are enqueued and BEFORE the status that announces the round, so the file
        exists by the time the web reacts. Idempotent: re-applying the current round (a reconnect
        replaying the same ACTIVE round) is a no-op, so it never truncates or re-seqs the live
        round. Flushes + closes the previous handle and resets the seq; the new handle opens
        lazily (append) on the next ``write_event``."""
        with self._lock:
            if round_num == self._round_num:
                return
            if self._handle is not None:
                try:
                    self._handle.flush()
                    self._handle.close()
                except (OSError, ValueError):  # ValueError if already closed
                    pass
                self._handle = None
            self._round_num = round_num
            self._events_path = events_path(self._dir, round_num)
            self._events_path.touch(exist_ok=True)
            self._seq = 0

    def write_event(self, task_id: str, attempt_id: str, event: dict) -> None:
        """Stamp a cheap per-round monotonic `seq` and append ONE raw line
        (``{"task_id","attempt_id","event"}``) to the active round's events.jsonl buffer. Called on the
        reasoner worker thread; the lock makes the seq + append atomic. NO flush — the
        background flusher pushes it to the page cache within ``_flush_interval_s``. The event is
        written RAW (uncompacted); the web tailer compacts it for display."""
        self._ensure_flusher()
        with self._lock:
            event_out = dict(event)
            event_out["seq"] = self._seq
            self._seq += 1
            line = json.dumps(
                {"task_id": task_id, "attempt_id": attempt_id, "event": event_out},
                separators=_COMPACT,
                default=_json_default,
            )
            try:
                self._writer().write(line + "\n")  # buffered; flushed periodically
            except Exception:  # noqa: BLE001
                logger.exception("event append failed for task %s", task_id)

    def flush(self) -> None:
        """Flush the shared handle to the page cache so the tailer can read it. Called by the
        background flusher on a fixed cadence; safe to call directly (e.g. tests)."""
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.flush()
                except (OSError, ValueError):  # ValueError if the handle was closed
                    pass

    def _writer(self):
        if self._handle is None:
            # Buffered (NOT line-buffered): writes stay in userspace until the periodic flush.
            self._handle = open(
                self._events_path, "a", buffering=self._BUFFER_BYTES, encoding="utf-8"
            )
        return self._handle

    def _ensure_flusher(self) -> None:
        if self._flusher is None:
            with self._lock:
                if self._flusher is None and not self._stop.is_set():
                    self._flusher = threading.Thread(
                        target=self._flush_loop, name="skunk-filesink-flush", daemon=True
                    )
                    self._flusher.start()

    def _flush_loop(self) -> None:
        # Wake every interval; Event.wait returns True only when stop is set → exit.
        while not self._stop.wait(self._flush_interval_s):
            self.flush()

    def close(self) -> None:
        """Stop the flusher and close the shared handle. Call at backend teardown."""
        self._stop.set()
        flusher = self._flusher
        if flusher is not None:
            flusher.join(timeout=2)
            self._flusher = None
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.close()
                except OSError:
                    pass
                self._handle = None
