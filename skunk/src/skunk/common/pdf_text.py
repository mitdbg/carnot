"""On-demand OCR text extraction from bulletin PDFs, with page-map side effects."""

from __future__ import annotations

from pathlib import Path

import fitz

from skunk.common.context import HarnessContext
from skunk.common.page_map import build_page_map_from_ocr, load_page_map, pdf_page_for_ref
from skunk.dsl import PageRef


def ensure_page_map(ref: PageRef, ctx: HarnessContext) -> None:
    """Build the page map for ref.month by scanning the OCR PDF, if not yet built.

    As a side effect, writes per-page OCR text to cache/pages/{month}/p{NNN}.txt,
    so subsequent get_ocr_text_for_pdf_page calls hit the cache.
    """
    if ref.month is None or not ref.file_path:
        return

    existing = load_page_map(ref.month, ctx.cache_dir)
    if existing.get("bulletin_to_pdf"):
        return

    file_path = Path(ref.file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {ref.file_path} (month={ref.month})")

    pages_dir = Path(ctx.cache_dir) / "pages" / ref.month
    pages_dir.mkdir(parents=True, exist_ok=True)
    with fitz.open(ref.file_path) as doc:
        n_pages = len(doc)
        for pdf_idx in range(1, n_pages + 1):
            txt_path = pages_dir / f"p{pdf_idx:03d}.txt"
            if not txt_path.exists():
                txt_path.write_text(doc[pdf_idx - 1].get_text())
    build_page_map_from_ocr(ref.month, ctx.cache_dir, n_pages)


def get_ocr_text_for_pdf_page(ref: PageRef, ctx: HarnessContext) -> str | None:
    """Return OCR text for the bulletin page referenced by ref.

    Resolves the PDF page index via the page map (building it on demand if needed),
    then reads from cache or extracts directly with PyMuPDF.
    """
    ensure_page_map(ref, ctx)
    pdf_idx = pdf_page_for_ref(ref, ctx.cache_dir)
    if pdf_idx is None or ref.month is None:
        return None

    txt_path = Path(ctx.cache_dir) / "pages" / ref.month / f"p{pdf_idx:03d}.txt"
    if txt_path.exists():
        return txt_path.read_text(errors="replace")

    if not ref.file_path:
        return None

    try:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with fitz.open(ref.file_path) as doc:
            text = doc[pdf_idx - 1].get_text()
        txt_path.write_text(text)
        return text
    except Exception as e:
        print(f"[pdf_text] OCR extraction failed for {ref}: {e}")
        return None
