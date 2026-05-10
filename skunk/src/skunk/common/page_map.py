"""Bidirectional map between bulletin printed page numbers and PDF page indices.

Page number convention (canonical throughout this codebase):
  PageRef.page     = bulletin printed page number (what appears on the physical page)
  PageRef.pdf_page = 1-based PDF page index (for PyMuPDF / cache file access)

The map for each bulletin is stored at:
  cache/page_maps/{YYYY-MM}.json
  {
    "bulletin_to_pdf": {"1": 3, "2": 4, ...},
    "pdf_to_bulletin": {"1": null, "2": null, "3": 1, "4": 2, ...}
  }
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from skunk.dsl import PageRef

# regex: one or more digits as the only content on a line (common footer format)
_PAGE_RE = re.compile(r"^\s*(\d{1,4})\s*$", re.MULTILINE)


def _map_path(month_str: str, cache_dir: str) -> Path:
    return Path(cache_dir) / "page_maps" / f"{month_str}.json"


def load_page_map(month_str: str, cache_dir: str) -> dict:
    """Load the page map for a bulletin month. Returns empty dicts if not built yet."""
    p = _map_path(month_str, cache_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"bulletin_to_pdf": {}, "pdf_to_bulletin": {}}


def build_page_map_from_ocr(
    month_str: str,
    cache_dir: str,
    n_pdf_pages: int,
    overwrite: bool = False,
) -> dict:
    """Build a page map by scanning per-page OCR text files for printed page numbers.

    Looks for files at cache/pages/{month_str}/p{NNN}.txt where NNN is the PDF page index.
    Returns the map dict (also writes it to disk).
    """
    p = _map_path(month_str, cache_dir)
    if p.exists() and not overwrite:
        return load_page_map(month_str, cache_dir)

    pages_dir = Path(cache_dir) / "pages" / month_str
    pdf_to_bulletin: dict[str, int | None] = {}

    for pdf_idx in range(1, n_pdf_pages + 1):
        txt = pages_dir / f"p{pdf_idx:03d}.txt"
        bnum: int | None = None
        if txt.exists():
            text = txt.read_text(errors="replace")
            # Look for a bare page number in footer (last ~200 chars) or header (first ~200 chars)
            footer = text[-200:] if len(text) > 200 else text
            header = text[:200]
            for region in (footer, header):
                m = _PAGE_RE.search(region)
                if m:
                    candidate = int(m.group(1))
                    # Sanity: bulletin pages are typically 1–300 range
                    if 1 <= candidate <= 500:
                        bnum = candidate
                        break
        pdf_to_bulletin[str(pdf_idx)] = bnum

    bulletin_to_pdf: dict[str, int] = {}
    for pdf_str, bnum in pdf_to_bulletin.items():
        if bnum is not None:
            # If multiple PDF pages claim the same bulletin page, keep the first
            bkey = str(bnum)
            if bkey not in bulletin_to_pdf:
                bulletin_to_pdf[bkey] = int(pdf_str)

    result = {"bulletin_to_pdf": bulletin_to_pdf, "pdf_to_bulletin": pdf_to_bulletin}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2))
    return result


def pdf_page_for_ref(ref: PageRef, cache_dir: str) -> int | None:
    """Return the PDF page index for a PageRef.

    Uses ref.pdf_page directly if set; otherwise looks up ref.page (bulletin page)
    in the page map for ref.month.
    """
    if ref.pdf_page is not None:
        return ref.pdf_page
    if ref.page is not None:
        if ref.month is None:
            raise ValueError(
                f"PageRef.page={ref.page} requires PageRef.month to be set for page-map lookup"
            )
        mapping = load_page_map(ref.month, cache_dir)
        pdf_idx = mapping.get("bulletin_to_pdf", {}).get(str(ref.page))
        if pdf_idx is not None:
            return int(pdf_idx)
    return None


