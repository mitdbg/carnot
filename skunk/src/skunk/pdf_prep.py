"""PDF + parsed-JSON page accessors (stateless I/O glue).

`PageRef` → text / image bytes / printed-page string. Three tiers:
parsed-JSON (preferred when the bulletin has been parsed offline),
PyMuPDF text extraction (OCR fallback), PyMuPDF page rendering (vision
fallback). No LLM, no `AnnotatedValue`, no orchestrator state — pure
file-system + `pymupdf`.
"""

from __future__ import annotations

import base64
import json
import os
from functools import lru_cache
from pathlib import Path

import fitz

from skunk.errors import StepFailed
from skunk.models import HarnessContext, PageRef


_PARSED_JSON_DEFAULT_DIR = Path.home() / "Desktop/officeqa/treasury_bulletins_parsed/jsons"
_DPI_SCALE = 300 / 72  # PyMuPDF base is 72 DPI; render pages at 300 DPI for vision.


def _parsed_json_dir() -> Path:
    d = os.environ.get("OFFICEQA_PARSED_JSON_DIR")
    return Path(d) if d else _PARSED_JSON_DEFAULT_DIR


@lru_cache(maxsize=64)
def _load_parsed_doc(month_str: str) -> dict:
    year, mon = month_str.split("-")
    p = _parsed_json_dir() / f"treasury_bulletin_{year}_{mon}.json"
    if not p.exists():
        raise StepFailed("extract", f"parsed-JSON source not found: {p}")
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise StepFailed("extract", f"corrupt parsed-JSON for {month_str}: {e}") from e


@lru_cache(maxsize=64)
def _parsed_page_index(month_str: str) -> dict[int, list[dict]]:
    doc = _load_parsed_doc(month_str)
    by_page: dict[int, list[dict]] = {}
    for el in doc.get("document", {}).get("elements", []):
        bbox = el.get("bbox") or []
        if not bbox:
            continue
        pid = bbox[0].get("page_id")
        if pid is None:
            continue
        by_page.setdefault(int(pid), []).append(el)
    return by_page


def get_text_for_pdf_page(ref: PageRef, ctx: HarnessContext) -> str | None:
    """Concatenated content for ref's PDF page. HTML tables pass through verbatim.

    Raises StepFailed if the parsed-JSON source is missing or corrupt. Returns
    None when the source is healthy but this PDF page has no parsed elements.
    """
    if ref.month is None or ref.page is None:
        return None
    idx = _parsed_page_index(ref.month)
    elements = idx.get(int(ref.page))
    if not elements:
        return None
    parts = [el["content"] for el in elements if el.get("content") is not None]
    return "\n\n".join(parts) if parts else None


def get_printed_page(ref: PageRef, ctx: HarnessContext) -> str | None:
    """Reverse lookup: bulletin printed-page footer text on ref's PDF page, or None.

    Raises StepFailed if the parsed-JSON source is missing or corrupt.
    """
    if ref.month is None or ref.page is None:
        return None
    idx = _parsed_page_index(ref.month)
    for el in idx.get(int(ref.page), []):
        if el.get("type") == "page_number" and el.get("content"):
            return str(el["content"])
    return None


def _pdf_path_for_ref(ref: PageRef) -> Path | None:
    """Resolve ref → PDF path via $OFFICEQA_PDF_DIR (skunk.page_index.pdf)."""
    if ref.month is None or ref.page is None or ref.page <= 0:
        return None
    from skunk.page_index.pdf import pdf_path_for
    return pdf_path_for(ref.month)


def extract_pdf_text(ref: PageRef, ctx: HarnessContext) -> str | None:
    """PyMuPDF text for ref's PDF page, or None when unavailable. No disk cache."""
    pdf_path = _pdf_path_for_ref(ref)
    if pdf_path is None:
        return None
    try:
        with fitz.open(pdf_path) as doc:
            return doc[ref.page - 1].get_text()
    except Exception as e:  # noqa: BLE001 — tier-fallback path; any fitz error → next tier
        ctx.emit("extract", "tier=ocr extraction failed", page=str(ref), error=str(e))
        return None


def render_pdf_page_b64(ref: PageRef, ctx: HarnessContext) -> tuple[str, str] | None:
    """Render ref's PDF page to in-memory PNG bytes and return (mime, base64). No disk cache."""
    pdf_path = _pdf_path_for_ref(ref)
    if pdf_path is None:
        return None
    try:
        with fitz.open(pdf_path) as doc:
            pix = doc[ref.page - 1].get_pixmap(matrix=fitz.Matrix(_DPI_SCALE, _DPI_SCALE))
        return "image/png", base64.standard_b64encode(pix.tobytes("png")).decode()
    except Exception as e:  # noqa: BLE001 — tier-fallback path; any fitz error → next tier
        ctx.emit("extract", "tier=vision render failed", page=str(ref), error=str(e))
        return None
