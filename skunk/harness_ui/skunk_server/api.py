"""FastAPI routes: the static monitoring UI and the two SSE streams."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from skunk_server.hub import HEARTBEAT_INTERVAL_S, StreamHub, sse_frame
from skunk_server.task_registry import TaskRegistry

WEB_DIR = Path(__file__).resolve().parent / "web"
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # disable proxy buffering so frames flush immediately
    "Connection": "keep-alive",
}


async def status_frames(hub: StreamHub, heartbeat: float = HEARTBEAT_INTERVAL_S):
    """SSE frames for the always-on status stream: the compact snapshot on connect, then
    whatever `publish_status` fans out. A `: ping` comment keeps idle connections alive."""
    queue = hub.add_status_sub()
    try:
        yield sse_frame(hub.status_snapshot())
        while True:
            try:
                yield await asyncio.wait_for(queue.get(), heartbeat)
            except TimeoutError:
                yield ": ping\n\n"
    finally:
        hub.remove_status_sub(queue)


async def event_frames(
    registry: TaskRegistry,
    hub: StreamHub,
    task_id: str,
    heartbeat: float = HEARTBEAT_INTERVAL_S,
):
    """SSE frames for one task's trace: backfill existing events, then forward live ones.
    The subscriber is registered BEFORE the backfill snapshot so nothing is missed; any live
    event whose seq was already in the backfill is skipped so nothing is duplicated."""
    queue = hub.add_event_sub(task_id)
    try:
        backfill, _next_seq = registry.snapshot_task_events(task_id)
        seen_through = -1
        for seq, attempt_id, event in backfill:
            seen_through = max(seen_through, seq)
            yield sse_frame({"attempt_id": attempt_id, "event": event})
        while True:
            try:
                attempt_id, event = await asyncio.wait_for(queue.get(), heartbeat)
            except TimeoutError:
                yield ": ping\n\n"
                continue
            if event.get("seq", -1) <= seen_through:
                continue  # already delivered during backfill
            yield sse_frame({"attempt_id": attempt_id, "event": event})
    finally:
        hub.remove_event_sub(task_id, queue)


def install_routes(app: FastAPI, registry: TaskRegistry, hub: StreamHub) -> None:
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/stream")
    async def status_stream() -> StreamingResponse:
        return StreamingResponse(status_frames(hub), media_type="text/event-stream", headers=SSE_HEADERS)

    @app.get("/api/stream/{task_id:path}")
    async def task_event_stream(task_id: str) -> StreamingResponse:
        return StreamingResponse(
            event_frames(registry, hub, task_id), media_type="text/event-stream", headers=SSE_HEADERS
        )
