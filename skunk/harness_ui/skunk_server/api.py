"""FastAPI routes: the static monitoring UI and the two SSE streams."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from skunk_server.domain import to_jsonable
from skunk_server.hub import HEARTBEAT_INTERVAL_S, StreamHub, sse_frame
from skunk_server.task_registry import TaskConflict

logger = logging.getLogger(__name__)

# (events, next_seq) for one task — `registry.snapshot_task_events` on the backend,
# the file tailer's `backfill` on the split web process. Same contract either way.
BackfillFn = Callable[[str], tuple[list[tuple[int, str, dict]], int]]

WEB_DIR = Path(__file__).resolve().parent / "web"
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # disable proxy buffering so frames flush immediately
    "Connection": "keep-alive",
}
_MONTH_RE = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])$")
_DOC_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ReviewResolveBody(BaseModel):
    response: str = (
        ""  # human's corrected AnnotatedValue JSON (empty = accept the model as-is)
    )
    source_docs: list[str] = Field(default_factory=list)
    client_id: str = Field(..., min_length=1)


class ReviewRefineBody(BaseModel):
    feedback: str  # the reviewer's natural-language instruction for revising the extraction
    candidates: list[dict] = Field(
        default_factory=list
    )  # the JSONs currently displayed (hand-edits already overlaid)
    client_id: str = Field(..., min_length=1)


class ReviewLockBody(BaseModel):
    client_id: str = Field(
        ..., min_length=1
    )  # the browser's stable per-session id holding/releasing the review lock


class SubmitTaskBody(BaseModel):
    client_id: str = Field(..., min_length=1)


class RestartTaskBody(BaseModel):
    client_id: str = Field(..., min_length=1)
    feedback: str = ""


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
    backfill_fn: BackfillFn,
    hub: StreamHub,
    task_id: str,
    heartbeat: float = HEARTBEAT_INTERVAL_S,
):
    """SSE frames for one task's trace: backfill existing events, then forward live ones.
    The subscriber is registered BEFORE the backfill snapshot so nothing is missed; any live
    event whose seq was already in the backfill is skipped so nothing is duplicated.

    `backfill_fn(task_id) -> (events, next_seq)` is pluggable so this serves both processes:
    the backend passes `registry.snapshot_task_events`; the split web process passes the
    file tailer's `backfill` (same contract, sourced from the jsonl on disk)."""
    queue = hub.add_event_sub(task_id)
    try:
        backfill, _next_seq = backfill_fn(task_id)
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


def install_stream_routes(
    app: FastAPI, backfill_fn: BackfillFn, hub: StreamHub
) -> None:
    """Static UI + the two SSE streams. Used by the split web process (with the file
    tailer's backfill + a file-fed hub); the backend in the split installs none of these."""
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

    @app.get("/")
    async def index() -> HTMLResponse:
        # Cache-bust the static refs by the newest static-file mtime so a browser always loads the
        # CURRENT js/css, not a stale cached copy (the console UI is iterated frequently). The HTML
        # itself is tiny and served no-cache, so a plain reload always picks up new assets.
        html = (WEB_DIR / "index.html").read_text()
        try:
            version = str(int(max(p.stat().st_mtime for p in (WEB_DIR / "static").glob("*"))))
        except ValueError:
            version = "0"
        html = re.sub(r"(/static/[\w./-]+\.(?:js|css))", r"\1?v=" + version, html)
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

    @app.get("/api/stream")
    async def status_stream() -> StreamingResponse:
        return StreamingResponse(
            status_frames(hub), media_type="text/event-stream", headers=SSE_HEADERS
        )

    @app.get("/api/stream/{task_id:path}")
    async def task_event_stream(task_id: str) -> StreamingResponse:
        return StreamingResponse(
            event_frames(backfill_fn, hub, task_id),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )


def install_command_routes(app: FastAPI) -> None:
    """Mutating + skunk-backed routes that need the registry/coordinator/broker (read lazily
    off `app.state`) and the corpus for page rendering. Stay on the backend; the web process
    reaches them via an httpx proxy."""

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
            img = await asyncio.to_thread(store.image, PageRef(stem=month, page=page))
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

    @app.get("/api/source-doc/{doc_id}/page/{page}.png")
    async def source_doc_page(doc_id: str, page: int) -> Response:
        # `page` is 1-based (the PDF viewer's page number, as shown in the review label and as
        # passed by review_overlay.js) — matching the sibling /api/source endpoint and the rest
        # of the corpus code. PyMuPDF indexes pages 0-based, so subtract 1 when loading.
        if not _DOC_ID_RE.match(doc_id) or page < 1:
            return Response(status_code=404)
        try:
            import fitz

            pdf_dir = Path(os.environ.get("OFFICEQA_PDF_DIR", ""))
            pdf_path = pdf_dir / f"{doc_id}.pdf"
            if not pdf_path.exists():
                pdf_path = (
                    Path(__file__).resolve().parents[2]
                    / "data"
                    / "dais"
                    / "pdfs"
                    / f"{doc_id}.pdf"
                )
            if not pdf_path.exists():
                return Response(status_code=404)
            with fitz.open(pdf_path) as pdf:
                if page > len(pdf):
                    return Response(status_code=404)
                pix = pdf[page - 1].get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72))
                data = pix.tobytes("png")
        except Exception:
            logger.exception("source doc page render failed for %s p%s", doc_id, page)
            return Response(status_code=500)
        return Response(
            content=data,
            media_type="image/png",
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
                review_id, body.response, body.source_docs, body.client_id
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

    @app.post("/api/reviews/{review_id}/refine")
    async def refine_review(review_id: str, body: ReviewRefineBody) -> JSONResponse:
        # Kick off a background LLM revision of the review's extracted candidates from the
        # reviewer's natural-language feedback. Returns immediately; the refined candidates
        # arrive via the status snapshot when the background job completes.
        broker = app.state.broker
        if broker is None:
            return JSONResponse(
                {"ok": False, "error": "reviews disabled"}, status_code=404
            )
        try:
            _task, review = broker.refine_review(
                review_id, body.feedback, body.candidates, body.client_id
            )
        except KeyError:
            return JSONResponse(
                {"ok": False, "error": "review not found"}, status_code=404
            )
        except TaskConflict as error:  # not open / already refining / refine disabled
            return JSONResponse({"ok": False, "error": str(error)}, status_code=409)
        return JSONResponse({"ok": True, "review": to_jsonable(review)})

    @app.post("/api/reviews/lock/{task_id:path}")
    async def acquire_review_lock(task_id: str, body: ReviewLockBody) -> JSONResponse:
        # Take (or heartbeat-refresh) the per-UID review lock. `ok=False` means someone else
        # holds it; `locked_by` is the current holder either way.
        broker = app.state.broker
        if broker is None:
            return JSONResponse(
                {"ok": False, "error": "reviews disabled"}, status_code=404
            )
        ok, holder = broker.acquire_review_lock(task_id, body.client_id)
        return JSONResponse({"ok": ok, "locked_by": holder})

    @app.post("/api/reviews/unlock/{task_id:path}")
    async def release_review_lock(task_id: str, body: ReviewLockBody) -> JSONResponse:
        # Release the lock when the operator exits the review overlay (idempotent).
        broker = app.state.broker
        if broker is None:
            return JSONResponse(
                {"ok": False, "error": "reviews disabled"}, status_code=404
            )
        broker.release_review_lock(task_id, body.client_id)
        return JSONResponse({"ok": True})

    @app.post("/api/submit/{task_id:path}")
    async def submit_task(task_id: str, body: SubmitTaskBody) -> JSONResponse:
        # Manually submit a READY task's latest answer to Cup. The coordinator is assigned to
        # app.state after install_routes runs, so read it lazily here at request time.
        coordinator = app.state.coordinator
        broker = app.state.broker
        try:
            if broker is not None:
                holder = broker.get_lock_holder(task_id)
                if holder is not None and holder != body.client_id:
                    return JSONResponse(
                        {"ok": False, "error": "Task is locked by another user"},
                        status_code=409,
                    )
                # NOTE: open reviews intentionally do NOT block manual submission. A reviewer must
                # always be able to submit the current optimistic answer early (e.g. to bank points
                # before the round deadline) without first clearing every review. Any reviews left
                # open stay open; resolving one still recomputes and resubmits as usual. This also
                # matches the deadline auto-submit sweep, which calls submit_ready() directly and
                # never consulted this guard.
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

    @app.post("/api/restart/{task_id:path}")
    async def restart_task(task_id: str, body: RestartTaskBody) -> JSONResponse:
        registry = app.state.registry
        queues = app.state.queues
        publish_status = app.state.publish_status
        broker = app.state.broker
        try:
            if broker is not None:
                holder = broker.get_lock_holder(task_id)
                if holder is not None and holder != body.client_id:
                    return JSONResponse(
                        {"ok": False, "error": "Task is locked by another user"},
                        status_code=409,
                    )
            task = registry.restart_failed_no_answer(task_id, body.feedback)
            queues.enqueue_agent(task_id)
            publish_status()
        except KeyError:
            return JSONResponse(
                {"ok": False, "error": "task not found"}, status_code=404
            )
        except TaskConflict as error:
            return JSONResponse({"ok": False, "error": str(error)}, status_code=409)
        return JSONResponse({"ok": True, "task_id": task.task_id})
