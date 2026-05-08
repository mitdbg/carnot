"""Tier 1 — page renderer and OCR text aligner (lazy).

Cache files are keyed by **PDF page index** (1-based), not bulletin page number:
  cache/pages/{YYYY-MM}/p{NNN}.png   (NNN = PDF page index)
  cache/pages/{YYYY-MM}/p{NNN}.txt

The translation from bulletin printed page number (canonical) to PDF page index
lives in prep/page_map.py. Use pdf_page_for_ref(ref, cache_dir) to resolve a
PageRef.page (bulletin) to the correct PDF page index before accessing these files.

Usage (full pass over manifest):
    python -m skunk.prep.pages \
        --manifest cache/manifest.csv \
        --cache-dir cache

Or lazy render for specific bulletin:
    from skunk.prep.pages import render_bulletin
    render_bulletin(pdf_path, ocr_txt_path, month_str, cache_dir)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def render_bulletin(
    pdf_path: str,
    ocr_txt_path: str | None,
    month_str: str,
    cache_dir: str,
    dpi: int = 300,
    overwrite: bool = False,
) -> list[str]:
    """Render all pages of a PDF bulletin; return list of PNG paths."""
    try:
        import fitz
    except ImportError as e:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf") from e

    out_dir = Path(cache_dir) / "pages" / month_str
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    png_paths: list[str] = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)

    for i, page in enumerate(doc):
        page_num = i + 1
        png_path = out_dir / f"p{page_num:03d}.png"
        if not overwrite and png_path.exists():
            png_paths.append(str(png_path))
            continue
        pix = page.get_pixmap(matrix=mat)
        pix.save(str(png_path))
        png_paths.append(str(png_path))

    doc.close()

    # Align OCR text to pages if available
    if ocr_txt_path and Path(ocr_txt_path).exists():
        _align_ocr_text(ocr_txt_path, len(png_paths), out_dir, overwrite)

    return png_paths


def _align_ocr_text(
    ocr_txt_path: str,
    n_pages: int,
    out_dir: Path,
    overwrite: bool,
) -> None:
    """Split OCR text by page markers and write per-page .txt files.

    Many OCR outputs insert form-feed (\x0c) characters between pages,
    or have 'PAGE N' headers. We split on those.
    """
    text = Path(ocr_txt_path).read_text(errors="replace")

    # Try form-feed split (most OCR tools)
    pages = text.split("\x0c")
    if len(pages) == n_pages:
        for i, page_text in enumerate(pages):
            txt_path = out_dir / f"p{i + 1:03d}.txt"
            if not overwrite and txt_path.exists():
                continue
            txt_path.write_text(page_text.strip(), encoding="utf-8")
        return

    # If page count doesn't match, write the whole text as p001.txt
    # (extract subagent will use it as a fallback)
    p001 = out_dir / "p001.txt"
    if not overwrite and p001.exists():
        return
    p001.write_text(text.strip(), encoding="utf-8")


def render_all_from_manifest(manifest_path: str, cache_dir: str, dpi: int = 300) -> None:
    import pandas as pd
    df = pd.read_csv(manifest_path)
    total = len(df)
    for i, row in df.iterrows():
        print(f"[pages] {i + 1}/{total} {row['month']}...")
        try:
            render_bulletin(
                pdf_path=row["pdf_path"],
                ocr_txt_path=row["ocr_txt_path"] or None,
                month_str=row["month"],
                cache_dir=cache_dir,
                dpi=dpi,
                overwrite=False,
            )
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    render_all_from_manifest(args.manifest, args.cache_dir, args.dpi)
