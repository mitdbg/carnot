"""On-demand PNG rendering of bulletin pages, shared by extract and read_visual."""

from __future__ import annotations

from pathlib import Path

import fitz

from skunk.common.context import HarnessContext
from skunk.common.page_map import pdf_page_for_ref
from skunk.dsl import PageRef

# 300 DPI from PyMuPDF's 72 DPI base: scale = 300/72 ≈ 4.17
_dpi_scale = 300 / 72


def get_png(ref: PageRef, ctx: HarnessContext) -> str | None:
    """Render a bulletin PDF page to PNG, caching at cache/pages/{month}/p{NNN}.png."""
    pdf_idx = pdf_page_for_ref(ref, ctx.cache_dir)
    if pdf_idx is None or ref.month is None:
        return None

    png_path = Path(ctx.cache_dir) / "pages" / ref.month / f"p{pdf_idx:03d}.png"
    if png_path.exists():
        return str(png_path)

    if not ref.file_path:
        return None

    file_path = Path(ref.file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {ref.file_path} (month={ref.month})")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(ref.file_path) as doc:
        pix = doc[pdf_idx - 1].get_pixmap(matrix=fitz.Matrix(_dpi_scale, _dpi_scale))
    pix.save(str(png_path))
    return str(png_path)
