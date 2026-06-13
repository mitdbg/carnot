"""In-process pub/sub that fans server state to SSE subscribers.

Replaces the old full-snapshot-per-event WebSocket broadcast. Two channels:
  - status: a tiny `{round, tasks:[summary]}` snapshot, re-sent whole on any change;
  - events: per-task trace events, streamed one at a time to that task's subscribers.

All publish/subscribe calls happen on the single main event loop (the worker-thread event
hot path hops over via `call_soon_threadsafe`), so a plain set/dict is safe without a lock.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

HEARTBEAT_INTERVAL_S = 30


def sse_frame(payload: dict) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def offer(queue: asyncio.Queue, item: Any) -> None:
    """Enqueue without ever blocking the publisher: on a full queue (slow client), drop the
    oldest item and retry. The freshest state matters most for a live trace."""
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            pass


class StreamHub:
    def __init__(self, status_provider: Callable[[], dict[str, Any]], maxsize: int = 256) -> None:
        self._status_provider = status_provider
        self._status_subs: set[asyncio.Queue] = set()
        self._event_subs: dict[str, set[asyncio.Queue]] = {}
        self._maxsize = maxsize

    # ── publish (main loop) ──────────────────────────────────────────────────
    def publish_status(self) -> None:
        if not self._status_subs:
            return
        frame = sse_frame(self._status_provider())  # serialize once, fan out
        for queue in self._status_subs:
            offer(queue, frame)

    def publish_event(self, task_id: str, attempt_id: str, events: list[dict]) -> None:
        # A batch of events (the worker flushes coalesced batches, not one per event); each is
        # enqueued as (attempt_id, event) and serialized per-subscriber so the endpoint dedups by seq.
        subs = self._event_subs.get(task_id)
        if not subs:
            return
        for queue in subs:
            for event in events:
                offer(queue, (attempt_id, event))

    # ── subscribe / unsubscribe (SSE endpoints, main loop) ───────────────────
    def add_status_sub(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(self._maxsize)
        self._status_subs.add(queue)
        return queue

    def remove_status_sub(self, queue: asyncio.Queue) -> None:
        self._status_subs.discard(queue)

    def add_event_sub(self, task_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(self._maxsize)
        self._event_subs.setdefault(task_id, set()).add(queue)
        return queue

    def remove_event_sub(self, task_id: str, queue: asyncio.Queue) -> None:
        subs = self._event_subs.get(task_id)
        if subs:
            subs.discard(queue)
            if not subs:
                del self._event_subs[task_id]

    def status_snapshot(self) -> dict[str, Any]:
        return self._status_provider()
