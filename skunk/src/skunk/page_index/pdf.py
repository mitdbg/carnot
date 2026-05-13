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
# Per-page section banner extraction
# ---------------------------------------------------------------------------
# The Treasury bulletins print a section banner at the top of every page
# (e.g. "DEBT OUTSTANDING", "CAPITAL MOVEMENTS"). In the parsed JSON this
# shows up as either:
#   - a [title] element on the page where a new section begins
#     (e.g. p30 of 1950-02 has [title]="STATUTORY DEBT LIMITATION"), or
#   - a [page_header] element on every page of the section
#     (e.g. p27-p29 have [page_header]=["February 1950", "DEBT OUTSTANDING"]).
# Both are verbatim from the page itself — far more reliable than TOC-inferred
# section spans, which mislabel pages when the printed page numbers are OCR'd
# badly.

import re as _re

_NOISE_PATTERNS = (
    _re.compile(r"^Treasury Bulletin\s*$", _re.IGNORECASE),
    _re.compile(r"^Bulletin\b.*", _re.IGNORECASE),
    _re.compile(r"^(January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s+\d{4}\s*$",
                _re.IGNORECASE),
    _re.compile(r"^\d{4}\s*$"),                  # year alone
    _re.compile(r"^[\s\d.,/]+$"),                 # digit/punct soup
)

# Treasury bulletins put table titles as [title] elements too. Reject anything
# that looks like a table title — section banners never start with "Table N.-"
# or "Chart N.-" or contain a 4-digit year.
_NOT_A_BANNER_PATTERNS = (
    _re.compile(r"^\s*(Table|Chart|Figure|Schedule)\s*\d", _re.IGNORECASE),
    _re.compile(r"\b\d{4}\b"),                    # any 4-digit year — banners don't have these
)


def _is_section_banner(text: str) -> bool:
    s = text.strip()
    if len(s) < 3 or len(s) > 80:
        return False
    if any(p.match(s) for p in _NOISE_PATTERNS):
        return False
    if any(p.search(s) for p in _NOT_A_BANNER_PATTERNS):
        return False
    return True


def _canonicalize_section(s: str) -> str:
    """Strip trailing punctuation and collapse whitespace. Keep casing intact —
    Title Case and ALL CAPS both show up across bulletins; we'll canonicalize
    the case after."""
    s = s.strip()
    s = _re.sub(r"\s+", " ", s)
    s = s.rstrip(".,;:- ")
    # Normalize casing: prefer Title Case, but uppercase Roman acronyms stay.
    if s.isupper():
        s = s.title()
    return s


def read_page_sections(pdf_path: str | Path,
                       parsed_dir: str | Path | None = None) -> dict[int, str | None]:
    """{1-based PDF page index → canonical section banner} from parsed JSON.

    Algorithm per page:
      1. If any [title] element has section-banner-like content, use it.
      2. Else, examine [page_header] elements; filter out noise (bulletin
         name, month-year, year alone, digit-only); use the first survivor.
      3. Else return None — caller falls back to TOC-inferred section.
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

        # Prefer [title] (section start)
        for el in elements:
            if el.get("type") != "title":
                continue
            content = (el.get("content") or "").strip()
            if _is_section_banner(content):
                section = _canonicalize_section(content)
                break

        # Else look at [page_header]
        if section is None:
            for el in elements:
                if el.get("type") != "page_header":
                    continue
                content = (el.get("content") or "").strip()
                if _is_section_banner(content):
                    section = _canonicalize_section(content)
                    break

        out[pdf_page] = section
    return out


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
