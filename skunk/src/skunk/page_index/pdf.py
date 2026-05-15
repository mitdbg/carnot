"""Per-page text reader.

Source: parsed-JSON cache at `~/Desktop/officeqa/treasury_bulletins_parsed/jsons/`
(override via `OFFICEQA_PARSED_JSON_DIR`). Same source the extract subagent uses
as Tier 1 — see `skunk/src/skunk/subagents/extract.py:81-94`. The parsed JSON
preserves verbatim section headers, titles, and HTML tables, where PyMuPDF on
the scanned 1940s-50s pages produces noisy text that the LLM extractor
faithfully transcribes (e.g. misreading "1949" as "1940" on UID0221's golden
page).

The public surface — `read_pdf_pages(path) -> {1-based PDF page index: text}` —
is unchanged from the prior PyMuPDF implementation, so the build pipeline
doesn't care which source feeds it.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

_FILENAME_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})\.pdf$")

_PARSED_JSON_DEFAULT_DIR = Path.home() / "Desktop/officeqa/treasury_bulletins_parsed/jsons"


def parse_bulletin_filename(path: str | Path) -> str:
    """treasury_bulletin_1953_06.pdf  ->  '1953-06'."""
    m = _FILENAME_RE.search(str(path))
    if not m:
        raise ValueError(f"not a treasury bulletin filename: {path}")
    return f"{m.group(1)}-{m.group(2)}"


def parsed_json_dir() -> Path:
    """Resolve the parsed-JSON directory (env override, then default)."""
    d = os.environ.get("OFFICEQA_PARSED_JSON_DIR")
    return Path(d) if d else _PARSED_JSON_DEFAULT_DIR


@lru_cache(maxsize=64)
def _load_parsed_doc(month_str: str, base_dir: str) -> dict:
    year, mon = month_str.split("-")
    p = Path(base_dir) / f"treasury_bulletin_{year}_{mon}.json"
    if not p.exists():
        raise FileNotFoundError(f"parsed-JSON not found: {p}")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Per-page section banner extraction (raw, no heuristics)
# ---------------------------------------------------------------------------
# Treasury bulletins print a section banner at the top of every page (e.g.
# "DEBT OUTSTANDING") that the parsed JSON exposes as either a [title]
# element (new-section page) or a [page_header] element (every page in the
# section). We surface the verbatim string with no normalization — banner
# canonicalization (em-dash/hyphen variants, ", con" suffixes, table-title
# strings that should bucket under a real ToC section) is the concept-tree
# stage's job via an LLM dedup step modeled on `dedup_labels`.


def read_page_sections(pdf_path: str | Path,
                       parsed_dir: str | Path | None = None) -> dict[int, str | None]:
    """{1-based PDF page index → raw section banner string or None}.

    Per page: first [title] element wins; else first [page_header]; else None.
    """
    base_dir = str(parsed_dir) if parsed_dir is not None else str(parsed_json_dir())
    month = parse_bulletin_filename(pdf_path)
    doc = _load_parsed_doc(month, base_dir)

    by_page: dict[int, list[dict]] = {}
    max_page = 0
    for el in doc.get("document", {}).get("elements", []):
        bbox = el.get("bbox") or []
        if not bbox:
            continue
        pid = bbox[0].get("page_id")
        if pid is None:
            continue
        pid = int(pid)
        by_page.setdefault(pid, []).append(el)
        if pid > max_page:
            max_page = pid

    out: dict[int, str | None] = {}
    for pdf_page in range(1, max_page + 1):
        elements = by_page.get(pdf_page, [])
        section: str | None = None
        for el in elements:
            if el.get("type") == "title":
                content = (el.get("content") or "").strip()
                if content:
                    section = content
                    break
        if section is None:
            for el in elements:
                if el.get("type") == "page_header":
                    content = (el.get("content") or "").strip()
                    if content:
                        section = content
                        break
        out[pdf_page] = section
    return out


def read_page_elements(pdf_path: str | Path,
                       parsed_dir: str | Path | None = None) -> dict[int, list[dict]]:
    """{1-based PDF page index → list of parsed-JSON element dicts}.

    Each element has at least `type` and `content` keys; this is the raw
    structural view the build pipeline uses to detect tables/figures cheaply
    (see `classify.has_visual_elements`).
    """
    base_dir = str(parsed_dir) if parsed_dir is not None else str(parsed_json_dir())
    month = parse_bulletin_filename(pdf_path)
    doc = _load_parsed_doc(month, base_dir)

    by_page: dict[int, list[dict]] = {}
    max_page = 0
    for el in doc.get("document", {}).get("elements", []):
        bbox = el.get("bbox") or []
        if not bbox:
            continue
        pid = bbox[0].get("page_id")
        if pid is None:
            continue
        pid = int(pid)
        by_page.setdefault(pid, []).append(el)
        if pid > max_page:
            max_page = pid

    return {pdf_page: by_page.get(pdf_page, []) for pdf_page in range(1, max_page + 1)}


def read_pdf_pages(pdf_path: str | Path,
                   parsed_dir: str | Path | None = None) -> dict[int, str]:
    """Return {1-based PDF page index: concatenated page text} from parsed JSON.

    Each page's text is the concatenation of all parsed elements' `content`
    on that page (joined with "\\n\\n"), preserving the order they appear in
    the parsed doc. HTML tables flow through verbatim — that's why the
    extractor sees clean column headers and labeled period strings.

    If a PDF page has no parsed elements (rare: blank pages, divider pages),
    an empty string is returned so the page still appears in the dict and
    the build pipeline can mark it as `blank`.

    Raises FileNotFoundError if the parsed-JSON file for this bulletin is
    missing — we explicitly do NOT fall back to PyMuPDF, because the v0.1
    decision is to trust the cleaner source uniformly.
    """
    base_dir = str(parsed_dir) if parsed_dir is not None else str(parsed_json_dir())
    month = parse_bulletin_filename(pdf_path)
    doc = _load_parsed_doc(month, base_dir)

    # Discover total PDF page count from the parsed doc's bbox page_ids.
    max_page = 0
    by_page: dict[int, list[dict]] = {}
    for el in doc.get("document", {}).get("elements", []):
        bbox = el.get("bbox") or []
        if not bbox:
            continue
        pid = bbox[0].get("page_id")
        if pid is None:
            continue
        pid = int(pid)
        by_page.setdefault(pid, []).append(el)
        if pid > max_page:
            max_page = pid

    # Tag every element with its parsed type so the LLM sees structural signal
    # (which block is a section_header vs a body paragraph vs a table). Drop
    # bare page_number elements — they're just the printed-footer digit and
    # tend to contaminate keyword lifts as stray numerals.
    out: dict[int, str] = {}
    for pdf_page in range(1, max_page + 1):
        elements = by_page.get(pdf_page, [])
        parts: list[str] = []
        for el in elements:
            content = el.get("content")
            if content is None:
                continue
            t = el.get("type") or "text"
            if t == "page_number":
                continue
            parts.append(f"[{t}] {content}")
        out[pdf_page] = "\n\n".join(parts) if parts else ""
    return out
