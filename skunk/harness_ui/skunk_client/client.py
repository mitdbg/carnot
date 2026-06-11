"""Standalone browser UI process for the Skunk server."""

from __future__ import annotations

import argparse
import calendar
import os
import re
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))
DEFAULT_PDF_DIR = Path(__file__).resolve().parents[3] / "data/officeqa/treasury_bulletin_pdfs"
DEFAULT_PAGE_CACHE_ROOT = Path(__file__).resolve().parents[2] / "cache/page_images"
PAGE_RENDER_DPI = 200


def create_app(
    server_url: str,
    pdf_dir: Path | None = None,
    page_cache_root: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Skunk Client")
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    source_pdf_dir = (pdf_dir or Path(os.environ.get("OFFICEQA_PDF_DIR", DEFAULT_PDF_DIR))).resolve()
    cache_root = (
        page_cache_root
        or Path(os.environ.get("SKUNK_PAGE_INDEX_DIR", DEFAULT_PAGE_CACHE_ROOT))
    ).resolve()
    render_dir = cache_root / "renders"
    render_locks_guard = threading.Lock()
    render_locks: dict[str, threading.Lock] = {}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Response:
        return TEMPLATES.TemplateResponse(request, "index.html", {"server_url": server_url.rstrip("/")})  # type: ignore[call-arg]

    @app.get("/api/corpus-documents")
    async def api_corpus_documents() -> list[dict[str, str]]:
        if not source_pdf_dir.is_dir():
            return []
        documents = []
        for pdf_path in source_pdf_dir.glob("treasury_bulletin_????_??.pdf"):
            match = re.fullmatch(
                r"treasury_bulletin_(\d{4})_(0[1-9]|1[0-2])\.pdf",
                pdf_path.name,
            )
            if match is None:
                continue
            year, month = match.groups()
            document_id = f"{year}-{month}"
            documents.append(
                {
                    "id": document_id,
                    "title": f"Treasury Bulletin, {calendar.month_name[int(month)]} {year}",
                    "filename": pdf_path.name,
                    "reference": f"Treasury Bulletin {document_id} PDF",
                }
            )
        return sorted(documents, key=lambda document: document["id"], reverse=True)

    @app.get("/api/source/{month}")
    async def api_source(month: str) -> FileResponse:
        pdf_path = _source_pdf_path(source_pdf_dir, month)
        if not pdf_path.is_file():
            raise HTTPException(status_code=404, detail=f"bulletin PDF not found for {month}")
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=pdf_path.name,
            content_disposition_type="inline",
        )

    @app.get("/api/source/{month}/page/{page}.png")
    async def api_source_page(month: str, page: int) -> Response:
        pdf_path = _source_pdf_path(source_pdf_dir, month)
        if page < 1:
            raise HTTPException(status_code=400, detail="page must be a positive integer")
        image_path = render_dir / month / f"{page}.png"
        with render_locks_guard:
            render_lock = render_locks.setdefault(f"{month}/{page}", threading.Lock())
        with render_lock:
            if not image_path.is_file():
                if not pdf_path.is_file():
                    raise HTTPException(
                        status_code=404,
                        detail=f"bulletin PDF not found for {month}",
                    )
                import fitz

                try:
                    with fitz.open(pdf_path) as document:
                        if page > document.page_count:
                            raise HTTPException(
                                status_code=404,
                                detail=f"PDF page not found for {month} page {page}",
                            )
                        matrix = fitz.Matrix(PAGE_RENDER_DPI / 72, PAGE_RENDER_DPI / 72)
                        image_bytes = document[page - 1].get_pixmap(
                            matrix=matrix,
                            alpha=False,
                        ).tobytes("png")
                except HTTPException:
                    raise
                except Exception as exc:
                    raise HTTPException(
                        status_code=422,
                        detail=f"could not render PDF page for {month} page {page}",
                    ) from exc
                image_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = image_path.with_name(
                    f".{image_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                )
                tmp_path.write_bytes(image_bytes)
                tmp_path.replace(image_path)
        return Response(image_path.read_bytes(), media_type="image/png")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Skunk human-worker browser client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument(
        "--server-url",
        default=os.environ.get("SKUNK_SERVER_URL", "http://127.0.0.1:8787"),
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path(os.environ.get("OFFICEQA_PDF_DIR", DEFAULT_PDF_DIR)),
        help="Treasury Bulletin PDF directory used by source-document links",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(create_app(args.server_url, args.pdf_dir), host=args.host, port=args.port)


def _source_pdf_path(pdf_dir: Path, month: str) -> Path:
    if re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month) is None:
        raise HTTPException(status_code=400, detail="month must use YYYY-MM")
    year, mon = month.split("-")
    return pdf_dir / f"treasury_bulletin_{year}_{mon}.pdf"


if __name__ == "__main__":
    main()
