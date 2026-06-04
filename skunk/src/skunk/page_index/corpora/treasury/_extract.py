"""Per-page deterministic field extractor for Treasury Bulletin pages.

Consumes the parsed-JSON element list for one page and returns:
    content_blocks, keywords, dates

`content_blocks` is one `ContentBlock` per detected table / chart / prose
block on the page — multi-content pages emit multiple entries. A page
with no blocks is non-retrievable.

`dates` is the page's verbatim date strings in document order
("December 31, 1949", "Fiscal Year 1991", "1932-1939", "1940").
Retrieve parses each back via `TreasuryPeriodParser.verbatim_date_to_intervals`
and overlap-checks against the planner-emitted period. Pages with no
printed date inherit a single bare-year date matching the bulletin's CY.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Any

from skunk.corpus import page_plain_text
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
# Structured date extraction (catalog-build side).
#
# Lifts verbatim date STRINGS from page text in priority order: longer
# patterns first so they consume their span before the bare-year fallback
# sees it. The resulting `list[str]` is stored on the row as `row.dates`;
# retrieve-time logic parses each string back to ISO intervals via
# `verbatim_date_to_intervals` and overlap-checks against the period.
#
# Compared to a flat 4-digit-year regex this preserves range semantics
# ("1932-1939" stays one closed interval, not min=1932 / max=1939) and
# month/day precision ("December 31, 1949" is a one-day interval, not a
# full-year envelope). Both matter when filtering against narrow periods.
# ---------------------------------------------------------------------------

_MONTH_FULL = (r"(?:January|February|March|April|May|June|July|August|"
               r"September|October|November|December)")
_MONTH_ABBR = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
_MONTH_ANY = f"(?:{_MONTH_FULL}|{_MONTH_ABBR})"

# Slash-date pattern: matches MM/DD/YYYY and MM/DD/YY. Heavy in PDO /
# auction tables where every row carries an auction date in M/D/YY form
# and TIPS security identifiers carry maturity tags like "07/15/16-D".
# Captures month / day / year groups for normalization (2-digit years
# are resolved against the bulletin's own year — see `_extract_dates`).
_SLASH_DATE_RE = re.compile(
    r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(\d{4}|\d{2})\b"
)

_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "December 31, 1949" / "Dec. 31, 1949"
    re.compile(rf"\b{_MONTH_ANY}\s+\d{{1,2}},\s*\d{{4}}\b", re.IGNORECASE),
    # "February 1952" / "Feb. 1952"
    re.compile(rf"\b{_MONTH_ANY}\s+\d{{4}}\b", re.IGNORECASE),
    # "Calendar Year 1940" / "Fiscal Year 1948" / "CY 1940" / "FY 1948"
    re.compile(r"\b(?:Calendar Year|Fiscal Year|CY|FY)\s*\d{4}\b",
               re.IGNORECASE),
    # "Q3 1953"
    re.compile(r"\bQ[1-4]\s*\d{4}\b", re.IGNORECASE),
    # "10/12/06" / "07/15/2016" — slash dates (handled with special-case
    # normalization below so 2-digit years become 4-digit before storing).
    _SLASH_DATE_RE,
    # "1932 through Mar 1939" / "1932 to 1939" / "1932-1939"
    re.compile(rf"\b\d{{4}}\s*(?:through|to|-)\s*(?:{_MONTH_ANY}\s*)?\d{{4}}\b",
               re.IGNORECASE),
    # Bare 4-digit year — runs LAST so the longer patterns win first.
    # Restricted to plausible bulletin-era years to keep table IDs /
    # footnote codes out of the envelope.
    re.compile(r"\b(?:1[89]\d{2}|20\d{2}|21\d{2})\b"),
)


def _resolve_two_digit_year(yy: int, bulletin_year: int) -> int:
    """Map a 2-digit year to the century that puts it closest to
    `bulletin_year`. Treasury data uses 2-digit years for both auction
    dates (a year or two before publication) and maturities (decades
    after publication), so a fixed pivot doesn't fit; bulletin-centered
    resolution does. Ties (50-year distance) prefer the same century as
    bulletin_year to avoid future-year overreach."""
    cand_a = (bulletin_year // 100) * 100 + yy
    cand_b = cand_a + 100 if cand_a <= bulletin_year else cand_a - 100
    if abs(cand_a - bulletin_year) <= abs(cand_b - bulletin_year):
        return cand_a
    return cand_b


def _normalize_slash_match(match_text: str, bulletin_year: int) -> str:
    """Rewrite a `M/D/YY[YY]` match to a canonical `M/D/YYYY` form so the
    retrieve-time verbatim parser can handle one shape only."""
    m = _SLASH_DATE_RE.match(match_text)
    if not m:
        return match_text
    mon, day, year = m.group(1), m.group(2), m.group(3)
    if len(year) == 2:
        year = f"{_resolve_two_digit_year(int(year), bulletin_year):04d}"
    return f"{mon}/{day}/{year}"


def _extract_dates(text: str, *, bulletin: str | None = None) -> list[str]:
    """Verbatim date strings on the page, document order, deduped
    case-insensitively. Longer patterns claim their span first; shorter
    patterns skip anything that overlaps an earlier match.

    `bulletin` (YYYY-MM) is used to resolve 2-digit years in slash dates
    (e.g. "10/12/06" in a 2007-12 bulletin → "10/12/2006"). Without it
    we keep 2-digit years verbatim and the retrieve parser will reject
    them, which costs recall — pass bulletin context whenever available.
    """
    bulletin_year: int | None = None
    if bulletin:
        try:
            bulletin_year = int(bulletin[:4])
        except ValueError:
            bulletin_year = None

    spans: list[tuple[int, int, str]] = []
    for pat in _DATE_PATTERNS:
        is_slash = pat is _SLASH_DATE_RE
        for m in pat.finditer(text):
            if any(not (m.end() <= s or m.start() >= e) for s, e, _ in spans):
                continue
            verbatim = m.group(0)
            # Normalize slash dates so all entries in `dates` share a
            # parser-friendly shape. 2-digit years require a bulletin
            # context; without it we drop the match (returning unresolved
            # "06" would mislead the retrieve filter).
            if is_slash:
                if len(m.group(3)) == 2:
                    if bulletin_year is None:
                        continue
                    verbatim = _normalize_slash_match(verbatim, bulletin_year)
                else:
                    verbatim = _normalize_slash_match(verbatim, bulletin_year or 2000)
            spans.append((m.start(), m.end(), verbatim))
    spans.sort()
    seen: set[str] = set()
    out: list[str] = []
    for _, _, s in spans:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


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

    Returns dict with keys: content_blocks, keywords, dates.
    `bulletin` (YYYY-MM) is the page's parent bulletin; pages with no
    explicit date strings inherit a single bare-year date matching the
    bulletin's calendar year so retrieve has *some* signal to overlap-
    check (TOC / snapshot / continuation pages).
    """
    blocks, all_columns, all_rows = _extract_content_blocks(elements)
    plain = page_plain_text(elements)
    dates = _extract_dates(plain, bulletin=bulletin)
    # Always seed `dates` with the bulletin's publication month
    # (`YYYY-MM`) AND its calendar year (`YYYY`):
    #
    #   - The YM signal catches retrospective tables whose printed dates
    #     don't include the period but whose publication month equals it
    #     (e.g. a Sep 1991 issue whose tables only print 1988–1990 data
    #     still answers a Sep 1991 question via "1991-09").
    #   - The bare-year signal is the recall safety net for pages with
    #     no printed dates at all (TOC, snapshot, continuation, or
    #     month-without-year prose like "during September") — without it
    #     we'd drop these for any narrow-period query inside the
    #     bulletin's own year.
    #
    # Dedup below absorbs the case where the text extractor already
    # surfaced one of these signals.
    seen = {d.lower() for d in dates}
    seeds = []
    if bulletin.lower() not in seen:
        seeds.append(bulletin)
        seen.add(bulletin.lower())
    yr = bulletin[:4]
    if yr.lower() not in seen:
        seeds.append(yr)
    dates = seeds + dates

    if blocks:
        primary_caption = next((b.title for b in blocks if b.title), None)
        keywords = _harvest_keywords(primary_caption, all_columns, all_rows[:8])
        return {
            "content_blocks": blocks,
            "keywords": keywords[:20],
            "dates": dates,
        }

    if has_prose_content(elements):
        prose_kw = _harvest_prose_keywords(elements)
        if prose_kw:
            return {
                "content_blocks": [ContentBlock(
                    kind="prose",
                    title=_resolve_prose_title(elements),
                )],
                "keywords": prose_kw,
                "dates": dates,
            }

    return {"content_blocks": [], "keywords": [], "dates": []}
