"""Per-page deterministic field extractor for Treasury Bulletin pages.

Consumes the parsed-JSON element list for one page and returns:
    content_blocks, keywords, min_year, max_year

`content_blocks` is one `ContentBlock` per detected table / chart / prose
block on the page — multi-content pages emit multiple entries. A page
with no blocks is non-retrievable.

`min_year` / `max_year` is the canonical year envelope — see
`_canonicalize_year_envelope` for the bulletin-fallback and the +1
FY/CY-straddle rule.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Any

from ...schema import ContentBlock
from ._classify import _BOILERPLATE_HEADER_RE, has_prose_content

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dateless-keyword filter — keywords are conceptual, dates live elsewhere.
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(
    r"\b("
    r"(19|20)\d{2}"
    r"|January|February|March|April|May|June|July|August|September|October"
    r"|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
    r"|Calendar Year|Fiscal Year|FY|CY"
    r"|Q[1-4]"
    r")\b",
    re.IGNORECASE,
)


def _is_dateful(s: str) -> bool:
    return bool(_DATE_RE.search(s))


_NOISE_KEYWORD_PATTERNS = (
    re.compile(r"^\s*\(?in\s+(thousands|millions|billions)\b", re.IGNORECASE),
    re.compile(r"^\s*\(in\s+", re.IGNORECASE),
    re.compile(r"^total$", re.IGNORECASE),
    re.compile(r"^class of security$", re.IGNORECASE),
    re.compile(r"^\W+$"),
    re.compile(r"^[\d\s.,$%/-]+$"),
    re.compile(r"^\(?con(?:t|tinued)?\)?$", re.IGNORECASE),
    re.compile(r"^section\s+(?:i{1,4}|iv|v|vi)\b$", re.IGNORECASE),
    re.compile(r"^source$", re.IGNORECASE),
    re.compile(r"^dollars?\)?$", re.IGNORECASE),
    re.compile(r"^[ivx]{1,4}-\d+$", re.IGNORECASE),
    re.compile(r"^fiscal\s+years$", re.IGNORECASE),
    re.compile(r"^by\s+type\s+and\s+country$", re.IGNORECASE),
    re.compile(r"^europe$", re.IGNORECASE),
)


def _is_meaningless_keyword(s: str) -> bool:
    if len(s) < 3:
        return True
    return any(p.search(s) for p in _NOISE_KEYWORD_PATTERNS)


_TRAILING_LEADER_RE = re.compile(r"[\s.·•:;,–—-]+$")


def _clean_keyword(s: str) -> str:
    return _TRAILING_LEADER_RE.sub("", s).strip()


def _accept_keyword(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if _is_dateful(s):
        return False
    if _is_meaningless_keyword(s):
        return False
    return True


# ---------------------------------------------------------------------------
# Year envelope.
# ---------------------------------------------------------------------------

# Plausible bulletin-era years. Treasury Bulletin runs 1939–present; widened
# slightly to admit retrospective references without letting random 4-digit
# numerics (table IDs, footnote codes) leak in.
_YEAR_RE = re.compile(r"\b(?:1[89]\d{2}|20\d{2}|21\d{2})\b")


def _extract_year_envelope(text: str) -> tuple[int | None, int | None]:
    years = [int(m.group(0)) for m in _YEAR_RE.finditer(text)]
    if not years:
        return None, None
    return min(years), max(years)


def _canonicalize_year_envelope(
    raw: tuple[int | None, int | None],
    bulletin: str,
) -> tuple[int, int]:
    """Turn the raw `(min, max)` envelope into the on-disk form.

    1. **Bulletin fallback.** No years on the page → inherit the parent
       bulletin's calendar year (most year-less pages are TOC, snapshots,
       or continuation pages that logically belong to the bulletin's CY).
    2. **+1 to end year.** A page that only prints `1991` may be CY1991,
       FY1991 (Oct 1990–Sep 1991), or FY1992 (Oct 1991–Sep 1992). Adding
       one year of forward slack lets a query for any of those three
       windows match a page that only prints `1991`. Costs precision; in
       this corpus, FY/CY ambiguity is the dominant year-filter
       false-negative source.
    """
    min_year, max_year = raw
    if min_year is None or max_year is None:
        bulletin_year = int(bulletin[:4])
        min_year = max_year = bulletin_year
    return min_year, max_year + 1


# ---------------------------------------------------------------------------
# Tiny HTML parser for `[table]` elements.
# ---------------------------------------------------------------------------

class _TableParser(HTMLParser):
    """Collects column headers, first-cell of each row, and captions."""

    def __init__(self) -> None:
        super().__init__()
        self.column_headers: list[str] = []
        self.row_first_cells: list[str] = []
        self.captions: list[str] = []

        self._in_th = False
        self._in_td = False
        self._in_caption = False
        self._row_cell_idx = 0
        self._row_has_th = False
        self._cur_text: list[str] = []
        self._first_row_cells: list[str] = []
        self._row_idx = -1

    def handle_starttag(self, tag: str, attrs):  # type: ignore[override]
        t = tag.lower()
        if t == "tr":
            self._row_idx += 1
            self._row_cell_idx = 0
            self._row_has_th = False
        elif t == "th":
            self._in_th = True
            self._cur_text = []
        elif t == "td":
            self._in_td = True
            self._cur_text = []
        elif t == "caption":
            self._in_caption = True
            self._cur_text = []

    def handle_endtag(self, tag: str):  # type: ignore[override]
        t = tag.lower()
        text = "".join(self._cur_text).strip()
        if t == "th":
            if text:
                self.column_headers.append(text)
                if self._row_cell_idx == 0:
                    self.row_first_cells.append(text)
            self._row_has_th = True
            self._row_cell_idx += 1
            self._in_th = False
            self._cur_text = []
        elif t == "td":
            if self._row_cell_idx == 0 and not self._row_has_th and text:
                self.row_first_cells.append(text)
            if self._row_idx == 0 and text:
                self._first_row_cells.append(text)
            self._row_cell_idx += 1
            self._in_td = False
            self._cur_text = []
        elif t == "caption":
            if text:
                self.captions.append(text)
            self._in_caption = False
            self._cur_text = []

    def handle_data(self, data: str):  # type: ignore[override]
        if self._in_th or self._in_td or self._in_caption:
            self._cur_text.append(data)

    def finalize(self) -> None:
        if not self.column_headers and self._first_row_cells:
            self.column_headers = list(self._first_row_cells)


def _parse_table_html(html: str) -> tuple[list[str], list[str], list[str]]:
    p = _TableParser()
    try:
        p.feed(html)
        p.close()
    except Exception as e:  # noqa: BLE001
        log.warning("table HTML parse failed: %s", e)
    p.finalize()
    return p.column_headers, p.row_first_cells, p.captions


# ---------------------------------------------------------------------------
# Caption resolution.
# ---------------------------------------------------------------------------

_TABLE_LIKE_TYPES = frozenset({"table", "figure", "image", "chart", "plot", "diagram"})
_CAPTION_TYPES = frozenset({"section_header", "title"})

_UNIT_NOTE_RE = re.compile(r"^\s*\(?in\s+(thousands|millions|billions)\b",
                           re.IGNORECASE)


# ---------------------------------------------------------------------------
# Keyword harvest.
# ---------------------------------------------------------------------------

_NOUN_SPLIT_RE = re.compile(
    r"[—–,;:.]+|\s{2,}|\s-\s|"
    r"\s+(?:as\s+of|for\s+the\s+period|"
    r"for\s+the\s+(?:fiscal\s+|calendar\s+)?(?:year|quarter|month|twelve\s+months?)\s+"
    r"(?:ended|ending)|"
    r"ended|ending)\s+",
    re.IGNORECASE,
)
_CAPTION_CONT_SUFFIX_RE = re.compile(
    r"[,\s\-–—]+(?:con|cont|continued)\s*\.?\s*\)?\s*$", re.IGNORECASE,
)
_CAPTION_PARENS_RE = re.compile(r"\([^)]*\)?")
_CAPTION_BRACKETS_RE = re.compile(r"\[[^\]]*\]?")
_CAPTION_ORPHAN_BRACKET_RE = re.compile(r"[\[\]\(\)]")


def _split_caption(caption: str) -> list[str]:
    s = re.sub(r"^\s*(?:Table|Chart|Figure|Schedule)\s+[A-Za-z0-9.\-]+\.?\s*[-–—]?\s*",
               "", caption, flags=re.IGNORECASE).strip()
    s = _CAPTION_CONT_SUFFIX_RE.sub("", s).strip()
    s = _CAPTION_PARENS_RE.sub(" ", s)
    s = _CAPTION_BRACKETS_RE.sub(" ", s)
    s = _CAPTION_ORPHAN_BRACKET_RE.sub(" ", s)
    s = re.sub(r"\s+\d+/\s*$", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if not s:
        return []
    return [p.strip() for p in _NOUN_SPLIT_RE.split(s) if p.strip()]


def _harvest_keywords(caption: str | None,
                      column_headers: list[str],
                      row_headers: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def _push(items: list[str]) -> None:
        for s in items:
            s = _clean_keyword(s)
            if not _accept_keyword(s):
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)

    if caption:
        _push(_split_caption(caption))
    return out


# ---------------------------------------------------------------------------
# Page text reconstruction.
# ---------------------------------------------------------------------------

def _page_plain_text(elements: list[dict]) -> str:
    parts: list[str] = []
    for el in elements:
        t = (el.get("type") or "").lower()
        if t == "page_number":
            continue
        content = el.get("content")
        if not content:
            continue
        if t == "table":
            content = re.sub(r"<[^>]+>", " ", str(content))
        parts.append(str(content))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prose helpers.
# ---------------------------------------------------------------------------

def _resolve_prose_title(elements: list[dict]) -> str | None:
    """First [title], else first non-boilerplate [section_header]."""
    title: str | None = None
    first_section_header: str | None = None
    for el in elements:
        t = (el.get("type") or "").lower()
        content = (el.get("content") or "").strip()
        if not content:
            continue
        if t == "title" and title is None:
            title = content
        if (t == "section_header" and first_section_header is None
                and not _BOILERPLATE_HEADER_RE.match(content)
                and len(content) >= 4):
            first_section_header = content
    return title or first_section_header


def _harvest_prose_keywords(elements: list[dict]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for el in elements:
        t = (el.get("type") or "").lower()
        if t not in ("section_header", "title"):
            continue
        s = (el.get("content") or "").strip()
        s = _clean_keyword(s)
        if not s or _BOILERPLATE_HEADER_RE.match(s):
            continue
        if not _accept_keyword(s):
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out[:15]


# ---------------------------------------------------------------------------
# Content-block extraction.
# ---------------------------------------------------------------------------

def _extract_content_blocks(
    elements: list[dict],
) -> tuple[list[ContentBlock], list[str], list[str]]:
    """Walk elements; emit one ContentBlock per table/chart, or one prose
    block if no visual but substantive headers exist.
    """
    blocks: list[ContentBlock] = []
    all_columns: list[str] = []
    all_rows: list[str] = []

    pieces: list[str] = []
    in_window = False

    def _flush_caption() -> str | None:
        if not pieces:
            return None
        joined = " ".join(pieces)
        joined = re.sub(r"\s+", " ", joined).strip().rstrip(". -")
        return joined or None

    for el in elements:
        t = (el.get("type") or "").lower()
        content = el.get("content")
        content_str = (content or "").strip() if isinstance(content, str) else ""

        if t in _CAPTION_TYPES:
            if content_str:
                pieces = [content_str]
                in_window = True
            continue

        if t == "text" and in_window:
            if not content_str:
                continue
            if _UNIT_NOTE_RE.search(content_str):
                continue
            if len(content_str) > 200:
                continue
            pieces.append(content_str)
            continue

        if t == "table" and content:
            caption = _flush_caption()
            cols, rows, caps = _parse_table_html(str(content))
            if not caption and caps:
                caption = caps[0]
            blocks.append(ContentBlock(
                kind="table",
                title=caption,
                column_headers=cols[:50],
                row_headers_sample=rows[:12],
            ))
            all_columns.extend(cols)
            all_rows.extend(rows)
            pieces = []
            in_window = False
            continue

        if t in ("figure", "image", "chart", "plot", "diagram"):
            caption = _flush_caption()
            blocks.append(ContentBlock(kind="chart", title=caption))
            pieces = []
            in_window = False
            continue

    return blocks, all_columns, all_rows


def parse_page_fields(elements: list[dict], *, bulletin: str) -> dict[str, Any]:
    """Deterministic per-page field extraction. No LLM.

    Returns dict with keys: content_blocks, keywords, min_year, max_year.
    `bulletin` is the page's parent bulletin in "YYYY-MM"; feeds the
    year-envelope fallback for pages with no 4-digit year.
    """
    blocks, all_columns, all_rows = _extract_content_blocks(elements)

    plain = _page_plain_text(elements)
    raw_envelope = _extract_year_envelope(plain)

    if blocks:
        primary_caption = next((b.title for b in blocks if b.title), None)
        keywords = _harvest_keywords(primary_caption, all_columns, all_rows[:8])
        min_year, max_year = _canonicalize_year_envelope(raw_envelope, bulletin)
        return {
            "content_blocks": blocks,
            "keywords": keywords[:20],
            "min_year": min_year,
            "max_year": max_year,
        }

    if has_prose_content(elements):
        prose_kw = _harvest_prose_keywords(elements)
        if prose_kw:
            min_year, max_year = _canonicalize_year_envelope(raw_envelope, bulletin)
            return {
                "content_blocks": [ContentBlock(
                    kind="prose",
                    title=_resolve_prose_title(elements),
                )],
                "keywords": prose_kw,
                "min_year": min_year,
                "max_year": max_year,
            }

    return {"content_blocks": [], "keywords": [],
            "min_year": None, "max_year": None}
