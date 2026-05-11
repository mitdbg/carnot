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
def _load_doc(month_str: str) -> dict:
    # Lazy import: skunk.subagents.base triggers subagents/__init__.py, which
    # imports extract.py, which imports this module — module-level import cycles.
    from skunk.subagents.base import StepFailed
    year, mon = month_str.split("-")
    p = parsed_json_dir() / f"treasury_bulletin_{year}_{mon}.json"
    if not p.exists():
        raise StepFailed("extract", f"parsed-JSON source not found: {p}")
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise StepFailed("extract", f"corrupt parsed-JSON for {month_str}: {e}") from e


@lru_cache(maxsize=64)
def _page_index(month_str: str) -> dict[int, list[dict]]:
    doc = _load_doc(month_str)
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
    idx = _page_index(ref.month)
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
    idx = _page_index(ref.month)
    for el in idx.get(int(ref.page), []):
        if el.get("type") == "page_number" and el.get("content"):
            return str(el["content"])
    return None
