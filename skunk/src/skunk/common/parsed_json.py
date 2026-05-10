"""JSON-backed per-page index over the parsed Treasury Bulletin corpus.

Source files: treasury_bulletin_{YYYY}_{MM}.json under OFFICEQA_PARSED_JSON_DIR
(default ~/Desktop/officeqa/treasury_bulletins_parsed/jsons/).

Each JSON ships document.elements[], where each element carries
bbox[0].page_id (1-based PDF page index), type, and content (HTML for tables).
We bucket by page_id; a query for PageRef(page=N) returns the elements that
sit on PDF page N.

A separate `page_number`-typed element preserves the bulletin's printed-page
footer text per PDF page — see get_printed_page() for the reverse-lookup.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from skunk.common.context import HarnessContext
from skunk.dsl import PageRef

_DEFAULT_DIR = Path.home() / "Desktop/officeqa/treasury_bulletins_parsed/jsons"


def parsed_json_dir() -> Path:
    d = os.environ.get("OFFICEQA_PARSED_JSON_DIR")
    return Path(d) if d else _DEFAULT_DIR


@lru_cache(maxsize=64)
def _load_doc(month_str: str) -> dict | None:
    year, mon = month_str.split("-")
    p = parsed_json_dir() / f"treasury_bulletin_{year}_{mon}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


@lru_cache(maxsize=64)
def _page_index(month_str: str) -> dict[int, list[dict]] | None:
    doc = _load_doc(month_str)
    if doc is None:
        return None
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
    """Concatenated content for ref's PDF page. HTML tables pass through verbatim."""
    if ref.month is None or ref.page is None:
        return None
    idx = _page_index(ref.month)
    if idx is None:
        return None
    elements = idx.get(int(ref.page))
    if not elements:
        return None
    parts = [el["content"] for el in elements if el.get("content") is not None]
    return "\n\n".join(parts) if parts else None


def get_printed_page(ref: PageRef, ctx: HarnessContext) -> str | None:
    """Reverse lookup: bulletin printed-page footer text on ref's PDF page, or None."""
    if ref.month is None or ref.page is None:
        return None
    idx = _page_index(ref.month)
    if idx is None:
        return None
    for el in idx.get(int(ref.page), []):
        if el.get("type") == "page_number" and el.get("content"):
            return str(el["content"])
    return None
