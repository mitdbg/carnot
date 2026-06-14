"""Web (browser-facing) process of the split Skunk console.

Serves the static UI and the two SSE streams from the filesystem firehose (via `FileTailer`
+ a local `StreamHub`), and proxies the three mutating/skunk-backed command routes to the
backend over localhost httpx. It never imports skunk — SSE is plain JSON file reads and the
corpus-heavy `/api/source` render is forwarded, not reimplemented.

    python -m skunk_server.web_app --port 8788 \
        --backend-url http://127.0.0.1:8787 --stream-dir "$SKUNK_STREAM_DIR"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from skunk_server.api import install_stream_routes
from skunk_server.file_tailer import FileTailer
from skunk_server.hub import StreamHub

logger = logging.getLogger(__name__)

# Browser-facing timeout for the backend proxy. The two POSTs (submit/resolve) await Cup /
# a recompute on the backend loop, so allow a generous read window; a connect failure
# (backend down) surfaces fast as 502.
_PROXY_TIMEOUT = httpx.Timeout(60.0, connect=5.0)
# Hop-by-hop / length headers we must not echo back from the backend response verbatim.
_DROP_RESPONSE_HEADERS = {"content-length", "transfer-encoding", "connection"}


def create_app(stream_dir: str, backend_url: str) -> FastAPI:
    tailer = FileTailer(stream_dir)
    hub = StreamHub(tailer.status_provider)
    tailer.hub = hub

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(base_url=backend_url, timeout=_PROXY_TIMEOUT)
        tail_task = asyncio.create_task(tailer.run())
        try:
            yield
        finally:
            tail_task.cancel()
            try:
                await tail_task
            except asyncio.CancelledError:
                pass
            await app.state.client.aclose()

    app = FastAPI(title="Skunk Console (web)", lifespan=lifespan)
    install_stream_routes(app, tailer.backfill, hub)

    async def _proxy(request: Request, method: str, path: str) -> Response:
        client: httpx.AsyncClient = app.state.client
        body = await request.body()
        headers = {}
        content_type = request.headers.get("content-type")
        if content_type:
            headers["content-type"] = content_type
        try:
            upstream = await client.request(method, path, content=body, headers=headers)
        except httpx.ConnectError:
            return JSONResponse({"ok": False, "error": "backend unavailable"}, status_code=502)
        except httpx.TimeoutException:
            return JSONResponse({"ok": False, "error": "backend timeout"}, status_code=504)
        except httpx.HTTPError as error:
            logger.exception("proxy to backend failed: %s %s", method, path)
            return JSONResponse({"ok": False, "error": str(error)}, status_code=502)
        passthrough = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in _DROP_RESPONSE_HEADERS
        }
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=passthrough,
            media_type=upstream.headers.get("content-type"),
        )

    @app.post("/api/submit/{task_id:path}")
    async def submit(task_id: str, request: Request) -> Response:
        return await _proxy(request, "POST", f"/api/submit/{task_id}")

    @app.post("/api/reviews/{review_id}/resolve")
    async def resolve_review(review_id: str, request: Request) -> Response:
        return await _proxy(request, "POST", f"/api/reviews/{review_id}/resolve")

    @app.post("/api/reviews/lock/{task_id:path}")
    async def acquire_review_lock(task_id: str, request: Request) -> Response:
        return await _proxy(request, "POST", f"/api/reviews/lock/{task_id}")

    @app.post("/api/reviews/unlock/{task_id:path}")
    async def release_review_lock(task_id: str, request: Request) -> Response:
        return await _proxy(request, "POST", f"/api/reviews/unlock/{task_id}")

    @app.get("/api/source/{month}/page/{page}.png")
    async def source_page(month: str, page: int, request: Request) -> Response:
        return await _proxy(request, "GET", f"/api/source/{month}/page/{page}.png")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Skunk console web/browser-facing process")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("SKUNK_SERVER_URL", "http://127.0.0.1:8787"),
    )
    parser.add_argument("--stream-dir", default=os.environ.get("SKUNK_STREAM_DIR", ""))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    if not args.stream_dir:
        raise SystemExit("SKUNK_STREAM_DIR or --stream-dir is required")
    app = create_app(args.stream_dir, args.backend_url)
    # Same SSE-never-closes reason as the backend: cap graceful shutdown so Ctrl-C doesn't
    # wait forever on open EventSource responses.
    uvicorn.run(app, host=args.host, port=args.port, timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
