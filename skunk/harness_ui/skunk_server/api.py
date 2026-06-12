"""FastAPI routes: the static monitoring UI and the two SSE streams."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from skunk_server.domain import to_jsonable
from skunk_server.hub import HEARTBEAT_INTERVAL_S, StreamHub, sse_frame
from skunk_server.task_registry import TaskConflict, TaskRegistry

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # disable proxy buffering so frames flush immediately
    "Connection": "keep-alive",
}
_MONTH_RE = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])$")


class ReviewResolveBody(BaseModel):
    response: str = (
        ""  # human's corrected AnnotatedValue JSON (empty = accept the model as-is)
    )
    source_docs: list[str] = Field(default_factory=list)


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
        return StreamingResponse(
            status_frames(hub), media_type="text/event-stream", headers=SSE_HEADERS
        )

    @app.get("/api/stream/{task_id:path}")
    async def task_event_stream(task_id: str) -> StreamingResponse:
        return StreamingResponse(
            event_frames(registry, hub, task_id),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @app.get("/api/source/{month}/page/{page}.png")
    async def source_page(month: str, page: int) -> Response:
        # Rendered source page for the review viewer. Reads from PageStore (a 200-DPI PNG cache
        # that renders on first miss; once pages are precomputed this is a pure cache read).
        if not _MONTH_RE.match(month) or page < 1:
            return Response(status_code=404)
        try:
            import skunk_reasoner
            from skunk import SkunkConfig
            from skunk.common import PageRef
            from skunk.page_index.store import get_page_store

            skunk_reasoner.SkunkReasoner.set_default_env()
            store = get_page_store(str(SkunkConfig.from_env().pdf_dir))
            img = await asyncio.to_thread(store.image, PageRef(month=month, page=page))
        except Exception:
            logger.exception("source page render failed for %s p%s", month, page)
            return Response(status_code=500)
        if img is None:
            return Response(status_code=404)
        return Response(
            content=base64.b64decode(img.data),
            media_type=img.mime,
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.post("/api/reviews/{review_id}/resolve")
    async def resolve_review(review_id: str, body: ReviewResolveBody) -> JSONResponse:
        # Record a human's correction (or accept-as-is) and trigger a background recompute.
        broker = app.state.broker
        if broker is None:
            return JSONResponse(
                {"ok": False, "error": "reviews disabled"}, status_code=404
            )
        try:
            task, review = broker.resolve_review(
                review_id, body.response, body.source_docs
            )
        except KeyError:
            return JSONResponse(
                {"ok": False, "error": "review not found"}, status_code=404
            )
        except TaskConflict as error:  # already resolved/cancelled
            return JSONResponse({"ok": False, "error": str(error)}, status_code=409)
        return JSONResponse(
            {"ok": True, "review": to_jsonable(review), "task_id": task.task_id}
        )

    @app.post("/api/submit/{task_id:path}")
    async def submit_task(task_id: str) -> JSONResponse:
        # Manually submit a READY task's latest answer to Cup. The coordinator is assigned to
        # app.state after install_routes runs, so read it lazily here at request time.
        coordinator = app.state.coordinator
        try:
            await coordinator.submit_ready(task_id)
        except KeyError:
            return JSONResponse(
                {"ok": False, "error": "task not found or has no answer"},
                status_code=404,
            )
        except TaskConflict as error:  # not READY (already submitting/submitted/scored) — must precede ValueError
            return JSONResponse({"ok": False, "error": str(error)}, status_code=409)
        except ValueError as error:  # candidate failed Cup validation
            return JSONResponse({"ok": False, "error": str(error)}, status_code=422)
        except Exception as error:  # noqa: BLE001 - surface any Cup/transport failure to the UI
            logger.exception("manual submit failed for %s", task_id)
            return JSONResponse({"ok": False, "error": str(error)}, status_code=502)
        return JSONResponse({"ok": True, "task_id": task_id})
