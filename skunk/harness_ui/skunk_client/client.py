"""Standalone browser UI process for the Skunk server."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))
DEFAULT_PDF_DIR = Path(__file__).resolve().parents[2] / "data/officeqa/treasury_bulletin_pdfs"


def create_app(server_url: str, pdf_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Skunk Client")
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    source_pdf_dir = (pdf_dir or Path(os.environ.get("OFFICEQA_PDF_DIR", DEFAULT_PDF_DIR))).resolve()

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Response:
        return TEMPLATES.TemplateResponse(request, "index.html", {"server_url": server_url.rstrip("/")})  # type: ignore[call-arg]

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