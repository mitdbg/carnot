"""On-demand PyMuPDF text extraction for a PageRef's PDF page (Tier 2).

PageRef.page is the 1-based PDF page index. We read directly from the bulletin's
PDF, caching per-page text at cache/pages/{month}/p{NNN}.txt.
"""

from __future__ import annotations

from pathlib import Path

import fitz

from skunk.common.context import HarnessContext
from skunk.dsl import PageRef


def get_ocr_text_for_pdf_page(ref: PageRef, ctx: HarnessContext) -> str | None:
    """Return PyMuPDF-extracted text for ref's PDF page; cache by PDF index."""
    if ref.month is None or ref.page is None:
        return None

    txt_path = Path(ctx.cache_dir) / "pages" / ref.month / f"p{ref.page:03d}.txt"
    if txt_path.exists():
        return txt_path.read_text(errors="replace")

    if not ref.file_path:
        return None

    try:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with fitz.open(ref.file_path) as doc:
            text = doc[ref.page - 1].get_text()
        txt_path.write_text(text)
        return text
    except Exception as e:
        print(f"[pdf_text] OCR extraction failed for {ref}: {e}")
        return None
