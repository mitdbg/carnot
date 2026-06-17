"""Web-process background replay of the filesystem stream into a local StreamHub.

The backend (`FileSink`) writes `status.json` + a per-round `events.<round>.jsonl` (one obj/line
`{"task_id","attempt_id","event"}`); this tailer polls both and re-publishes into an in-process
`StreamHub`, demuxing each event line to its task's subscribers — so `hub.py`, the SSE
generators, and the whole frontend are reused verbatim and the web process never imports skunk.
It is the mirror image of the sink and shares the same FS contract (`file_stream`). The current
round comes from `status.json`; when it changes, the tailer switches to the new round's events
file (re-tailing from the top) — the same `round_num` the backend rolled its sink to.

Reader invariants (the well-formed-JSON guarantee, see `file_stream` docstring):
  1. Resume at a boundary — the stored offset only ever advances past complete `\n`-terminated
     lines, so every read starts at a record boundary.
  2. Bytes in, decode per line — read raw bytes, split on `b"\n"`, UTF-8-decode + json.loads
     each complete segment; never decode a tail-inclusive chunk (a partial trailing multibyte
     char would throw).
  3. Detect recreation — if `st_size < offset` or `st_ino` changed, reset the offset to 0 and
     re-tail; the existing seq-dedup in `event_frames` prevents double-delivery.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from skunk_server.file_stream import events_path, status_path
from skunk_server.hub import StreamHub

logger = logging.getLogger(__name__)

POLL_INTERVAL_ENV = "SKUNK_STREAM_POLL_S"
DEFAULT_POLL_S = 0.15

# Display caps applied HERE, on the web process — the agent writes events raw/uncompacted.
MAX_EVENT_MESSAGE_CHARS = 4000
MAX_EVENT_DATA_CHARS = 16000

_EMPTY_STATUS: dict[str, Any] = {"round": {}, "tasks": []}


class FileTailer:
    """Polls the stream dir and drives a local hub. Construct it, build the hub off its
    `status_provider`, then assign `tailer.hub = hub` before calling `run()`."""

    def __init__(self, base: str | os.PathLike[str], poll_s: float | None = None) -> None:
        self._dir = Path(base)
        self._status_file = status_path(self._dir)
        # The live round's events file, derived from status.json's round_num (None → the legacy
        # events.jsonl until the first round is seen); switched by _maybe_switch_round.
        self._round_num: int | None = None
        self._events_file = events_path(self._dir)
        if poll_s is None:
            poll_s = float(os.environ.get(POLL_INTERVAL_ENV, DEFAULT_POLL_S))
        self._poll_s = poll_s
        self.hub: StreamHub | None = None
        self._status_cache: dict[str, Any] = dict(_EMPTY_STATUS)
        self._status_sig: tuple[int, int] | None = None  # (st_mtime_ns, st_size)
        # (byte_offset, st_ino) for the current round's events.jsonl live tail.
        self._offset: tuple[int, int] = (0, -1)

    # ── hub provider ─────────────────────────────────────────────────────────
    def status_provider(self) -> dict[str, Any]:
        """The web hub's status provider — the last-read snapshot (empty until first read)."""
        return self._status_cache

    # ── backfill (request path) ──────────────────────────────────────────────
    def backfill(self, task_id: str) -> tuple[list[tuple[int, str, dict]], int]:
        """Scan the current round's events file and return THIS task's events:
        ([(seq, attempt_id, event), ...], next_seq). Missing file → ([], 0). Same contract
        `event_frames` expects; the events for one task are sparse in the round's interleaved
        stream but still strictly increasing, which is all the seq-dedup needs."""
        out: list[tuple[int, str, dict]] = []
        max_seq = -1
        visible_attempt_ids = self._visible_attempt_ids(task_id)
        try:
            with open(self._events_file, "rb") as handle:
                for line in handle:
                    if not line.endswith(b"\n"):
                        continue
                    obj = _parse_line(line)
                    if obj is None or obj.get("task_id") != task_id:
                        continue
                    attempt_id = obj.get("attempt_id", "")
                    if (
                        visible_attempt_ids is not None
                        and attempt_id not in visible_attempt_ids
                    ):
                        continue
                    event = _compact_event(obj.get("event") or {})
                    seq = event.get("seq", 0)
                    out.append((seq, attempt_id, event))
                    max_seq = max(max_seq, seq)
        except FileNotFoundError:
            return [], 0
        return out, max_seq + 1

    def _visible_attempt_ids(self, task_id: str) -> set[str] | None:
        for task in self._status_cache.get("tasks") or []:
            if task.get("task_id") != task_id:
                continue
            attempts = task.get("attempt_ids")
            if isinstance(attempts, list):
                return {str(attempt_id) for attempt_id in attempts}
            return None
        return None

    # ── poll loop (web main loop) ────────────────────────────────────────────
    async def run(self) -> None:
        try:
            while True:
                try:
                    self._poll_status()
                    self._poll_events()
                except Exception:  # noqa: BLE001 - a transient FS hiccup must not kill the loop
                    logger.exception("file tailer poll failed")
                await asyncio.sleep(self._poll_s)
        except asyncio.CancelledError:
            pass

    def _poll_status(self) -> None:
        try:
            st = os.stat(self._status_file)
        except FileNotFoundError:
            return
        sig = (st.st_mtime_ns, st.st_size)
        if sig == self._status_sig:
            return
        try:
            with open(self._status_file, "rb") as handle:
                data = handle.read()
            snapshot = json.loads(data.decode("utf-8"))
        except (OSError, ValueError):
            # Mid-write or a partial read — leave the cache, retry next poll. (The sink writes
            # via os.replace so this should be rare, but stay defensive.)
            return
        self._status_sig = sig
        self._status_cache = snapshot
        self._maybe_switch_round(snapshot)
        if self.hub is not None:
            self.hub.publish_status()

    def _maybe_switch_round(self, snapshot: dict[str, Any]) -> None:
        """On a new round, point the live tail at that round's events file and re-tail it from the
        top. The backend rolled its sink to `events.<round>.jsonl` (and created the file) BEFORE
        writing this status, so the name is guaranteed present once we react. Resetting the offset
        re-publishes the new file from the start; the per-task seq-dedup in `event_frames` drops
        anything a fresh subscriber already backfilled."""
        round_num = (snapshot.get("round") or {}).get("round_num")
        if not isinstance(round_num, int) or round_num == self._round_num:
            return
        self._round_num = round_num
        self._events_file = events_path(self._dir, round_num)
        self._offset = (0, -1)

    def _poll_events(self) -> None:
        """Tail the single shared events.jsonl once and demux each new line to its task's
        subscribers. Skip entirely when nobody is watching (a late subscriber backfills the
        whole file on connect, and the seq-dedup drops anything this live tail replays)."""
        if self.hub is None:
            return
        active_ids = set(self.hub.active_event_task_ids())
        if not active_ids:
            return
        try:
            st = os.stat(self._events_file)
        except FileNotFoundError:
            return
        offset, inode = self._offset
        if st.st_ino != inode or st.st_size < offset:
            offset, inode = 0, st.st_ino  # recreated/truncated → re-tail from the top
        if st.st_size == offset:
            self._offset = (offset, inode)
            return
        with open(self._events_file, "rb") as handle:
            handle.seek(offset)
            chunk = handle.read()
        last_nl = chunk.rfind(b"\n")
        if last_nl == -1:
            # Only a partial trailing record so far — don't advance; re-read next poll.
            self._offset = (offset, inode)
            return
        complete = chunk[: last_nl + 1]
        for line in complete.split(b"\n"):
            obj = _parse_line(line)
            if obj is None:
                continue
            task_id = obj.get("task_id", "")
            if task_id not in active_ids:
                continue
            event = _compact_event(obj.get("event") or {})  # clip for display (web-side work)
            self.hub.publish_event(task_id, obj.get("attempt_id", ""), [event])
        self._offset = (offset + len(complete), inode)


def _parse_line(line: bytes) -> dict | None:
    if not line.strip():
        return None
    try:
        return json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        logger.warning("skipping unparseable stream line (%d bytes)", len(line))
        return None


def _compact_event(event: dict) -> dict:
    """Select the display fields and clip `data` to `MAX_EVENT_DATA_CHARS` — the heavyweight
    interpretation the agent deliberately does NOT do. `seq` is preserved (the agent stamped it
    and `event_frames` dedups on it). The event is already JSON-native (parsed from the line),
    so no datetime/enum/dataclass conversion is needed here."""
    compact: dict = {}
    for key in ("seq", "message", "kind", "op", "level", "step_idx", "t", "branch_id"):
        if key in event:
            compact[key] = event[key]
    if "message" in compact:
        compact["message"] = str(compact["message"])[:MAX_EVENT_MESSAGE_CHARS]
    data = event.get("data")
    if isinstance(data, dict) and data:
        compact["data"] = _clip_jsonable(data, [MAX_EVENT_DATA_CHARS])
    return compact


def _clip_jsonable(value: Any, budget: list[int]) -> Any:
    """Total-budget-cap a JSON-native value in one pass. `budget` is a 1-element mutable counter
    of remaining chars; strings are clipped to fit and, once exhausted, list tails are dropped
    (with a marker) so a huge corpus observation costs ~budget to walk, not its full size."""
    if isinstance(value, str):
        if budget[0] <= 0:
            return ""
        if len(value) > budget[0]:
            clipped = value[: budget[0]] + "… [truncated]"
            budget[0] = 0
            return clipped
        budget[0] -= len(value)
        return value
    if isinstance(value, dict):
        return {str(key): _clip_jsonable(item, budget) for key, item in value.items()}
    if isinstance(value, list):
        out: list[Any] = []
        for index, item in enumerate(value):
            if budget[0] <= 0:
                out.append(f"… [{len(value) - index} more truncated]")
                break
            out.append(_clip_jsonable(item, budget))
        return out
    return value
