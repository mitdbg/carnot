"""Per-page deterministic field extractor — NO LLM.

Consumes the parsed-JSON element list for one page and returns the
fields a `PageCatalogRow` carries:
    content_blocks, keywords, min_year, max_year

`content_blocks` is one `ContentBlock` per detected table / chart /
prose block on the page — multi-content pages (table at top, chart at
bottom) emit multiple entries. A page with no blocks is non-retrievable.

`min_year` / `max_year` is the canonical year envelope for the page —
see `_canonicalize_year_envelope` for the bulletin-fallback and the
+1-to-end-year fiscal-vs-calendar straddle rule.

The parsed-JSON elements come from a layout parser and look like:
    {"type": "section_header", "content": "Table 1.- Status under Limitation, December 31, 1949", "bbox": [...]}
    {"type": "table",          "content": "<table>...</table>",                                   "bbox": [...]}
    {"type": "title",          "content": "STATUTORY DEBT LIMITATION",                            "bbox": [...]}
    ...

The HTML inside `[table]` elements is the verbatim layout parser output.
We pull column headers, the first column of each row (as row-headers sample),
and any caption-like rows from it using a small stdlib HTML parser.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

from .schema import ContentBlock


# ---------------------------------------------------------------------------
# Dateless-keyword filter — keywords are conceptual, dates live in `dates`.
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(
    r"\b("
    r"(19|20)\d{2}"                                 # 4-digit year
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


# Unit notes and other low-signal phrases we drop from keywords.
_NOISE_KEYWORD_PATTERNS = (
    re.compile(r"^\s*\(?in\s+(thousands|millions|billions)\b", re.IGNORECASE),
    re.compile(r"^\s*\(in\s+", re.IGNORECASE),
    re.compile(r"^total$", re.IGNORECASE),
    re.compile(r"^class of security$", re.IGNORECASE),
    re.compile(r"^\W+$"),                # pure punctuation/whitespace
    re.compile(r"^[\d\s.,$%/-]+$"),      # numeric / units only
    # Residual caption noise (the continuation suffix should already be stripped
    # by _split_caption; these catch the leftovers and standalone fragments).
    re.compile(r"^\(?con(?:t|tinued)?\)?$", re.IGNORECASE),
    re.compile(r"^section\s+(?:i{1,4}|iv|v|vi)\b$", re.IGNORECASE),
    re.compile(r"^source$", re.IGNORECASE),
    re.compile(r"^dollars?\)?$", re.IGNORECASE),
    # Page-ID artifacts like "I-1", "II-2", "III-1", "IV-3".
    re.compile(r"^[ivx]{1,4}-\d+$", re.IGNORECASE),
    re.compile(r"^fiscal\s+years$", re.IGNORECASE),
    re.compile(r"^by\s+type\s+and\s+country$", re.IGNORECASE),
    re.compile(r"^europe$", re.IGNORECASE),                # caption leakage
)


def _is_meaningless_keyword(s: str) -> bool:
    if len(s) < 3:
        return True
    return any(p.search(s) for p in _NOISE_KEYWORD_PATTERNS)


_TRAILING_LEADER_RE = re.compile(r"[\s.·•:;,–—-]+$")


def _clean_keyword(s: str) -> str:
    """Strip trailing leader-line dots, punctuation, and whitespace."""
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
# Year envelope extraction — every 4-digit year on the page, reduced to
# (min, max). Year granularity is intentional: month/day precision on the
# page side is noisy (footnotes, sequence numbers) and the lookup-side
# period already collapses to year boundaries for FY/CY/Q.
# ---------------------------------------------------------------------------

# Plausible bulletin-era years. Treasury Bulletin runs 1939–present;
# we widen the band slightly to admit retrospective references without
# letting random 4-digit numerics (table IDs, footnote codes) leak in.
_YEAR_RE = re.compile(r"\b(?:1[89]\d{2}|20\d{2}|21\d{2})\b")


def _extract_year_envelope(text: str) -> tuple[int | None, int | None]:
    """Return `(min_year, max_year)` across every 4-digit year occurrence in
    `text`, or `(None, None)` when no year is found. RAW envelope — no
    fiscal-year slack and no bulletin fallback; both are applied higher up
    in `_canonicalize_year_envelope`.
    """
    years = [int(m.group(0)) for m in _YEAR_RE.finditer(text)]
    if not years:
        return None, None
    return min(years), max(years)


def _canonicalize_year_envelope(
    raw: tuple[int | None, int | None],
    bulletin: str,
) -> tuple[int, int]:
    """Turn the raw `(min, max)` year envelope into the on-disk canonical form.

    Two adjustments applied in order:

    1. **Bulletin fallback.** If the page has no 4-digit years on it
       (raw == (None, None)), the page inherits the calendar year of its
       parent bulletin. A page in the `1991-08` bulletin with no year
       text gets `(1991, 1991)` before step 2. Rationale: most year-less
       pages are TOC, snapshots without explicit year labels, or
       continuation pages whose dated header sits on a sibling page —
       all of which logically belong to the bulletin's CY. Dropping
       them entirely was the previous behavior and cost recall on a
       non-trivial slice.

    2. **+1 to the end year.** After step 1 the envelope's upper bound
       is bumped by one calendar year. Rationale: a page whose only
       printed year is `1991` may legitimately be reporting CY1991
       (Jan–Dec 1991), FY1991 (Oct 1990–Sep 1991), or FY1992 (Oct 1991–
       Sep 1992). The two fiscal interpretations both straddle the
       calendar boundary into the next year, so admitting one year of
       forward slack on every page lets a query for any of those three
       windows match a page that only prints `1991`. It costs precision
       (a 1990–1992 page also matches CY1993 queries after +1) but the
       trade-off is intentional: in this corpus, FY/CY ambiguity is the
       dominant source of year-filter false negatives.
    """
    min_year, max_year = raw
    if min_year is None or max_year is None:
        bulletin_year = int(bulletin[:4])
        min_year = max_year = bulletin_year
    return min_year, max_year + 1


# ---------------------------------------------------------------------------
# Tiny HTML parser for `[table]` elements. Pulls column headers, first-cell
# of each row (row-headers sample), and any caption rows.
# ---------------------------------------------------------------------------

class _TableParser(HTMLParser):
    """Collects:
       column_headers: text of every <th> cell (in document order), and as a
         fallback the cells of the FIRST <tr> if no <th> exists.
       row_first_cells: text of the first <td>/<th> in each <tr> (skipping
         rows that are header-only).
       captions: text inside <caption> elements.
    """

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
        """If no <th> was seen, treat the first <tr>'s cells as column headers."""
        if not self.column_headers and self._first_row_cells:
            self.column_headers = list(self._first_row_cells)


def _parse_table_html(html: str) -> tuple[list[str], list[str], list[str]]:
    """Return (column_headers, row_first_cells, captions) from one table's HTML."""
    p = _TableParser()
    try:
        p.feed(html)
        p.close()
    except Exception:
        # Parser is tolerant; in the worst case we get nothing back.
        pass
    p.finalize()
    return p.column_headers, p.row_first_cells, p.captions


# ---------------------------------------------------------------------------
# Caption resolution — pick the [section_header] (or [title]) most relevant
# to the page's primary table/figure.
# ---------------------------------------------------------------------------

_TABLE_LIKE_TYPES = frozenset({"table", "figure", "image", "chart", "plot", "diagram"})
_CAPTION_TYPES = frozenset({"section_header", "title"})


# Reject "(In millions of dollars)"-style unit notes when joining caption text.
_UNIT_NOTE_RE = re.compile(r"^\s*\(?in\s+(thousands|millions|billions)\b",
                           re.IGNORECASE)


def _resolve_caption(elements: list[dict]) -> str | None:
    """Resolve the caption of the page's first table/figure.

    Walk elements in order. A `[section_header]` or `[title]` opens a caption
    window. Inside the window, every `[text]` element is appended to the
    caption (filtering unit-notes and very long paragraphs). The window
    closes when we hit a `[table]`/`[figure]` (caption complete) or a new
    `[section_header]`/`[title]` before any table (caption was for something
    else; restart). The first completed caption wins.
    """
    in_window = False
    pieces: list[str] = []
    result: str | None = None

    for el in elements:
        t = (el.get("type") or "").lower()
        content = (el.get("content") or "").strip()
        if not content:
            continue
        if t in _CAPTION_TYPES:
            # New caption block — drop any half-built one and restart.
            pieces = [content]
            in_window = True
        elif t == "text" and in_window:
            if _UNIT_NOTE_RE.search(content):
                continue
            if len(content) > 200:
                continue
            pieces.append(content)
        elif t in _TABLE_LIKE_TYPES and in_window:
            if result is None and pieces:
                joined = " ".join(pieces)
                joined = re.sub(r"\s+", " ", joined).strip().rstrip(". -")
                if joined:
                    result = joined
            in_window = False
            pieces = []
    return result


# ---------------------------------------------------------------------------
# Keyword harvest — caption noun phrases + headers, dedupe + filter.
# ---------------------------------------------------------------------------

# Split on punctuation / wide whitespace / inline " - " AND on date-introducer
# phrases like "As of <date>", "For the Period <date>", "Ending <date>" — these
# words attach a date tail to an otherwise-dateless concept phrase, and without
# splitting there the conceptual prefix gets rejected as "dateful" downstream.
# `through` and `covering` are intentionally NOT in this list: they often join
# two concept parts (e.g. "New Money Financing through Regular Weekly Treasury
# Bills") rather than introduce a date.
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
# Strip ALL parenthesized and bracketed content. Treasury caption parens almost
# always carry annotations (unit notes "(in millions of dollars)", acronyms
# "(OASI)", method notes "(Price decimals are 32nds)", source markers
# "[Source: Treasury Foreign Currency Reporting]") rather than concept content
# that retrieval needs. Removing them preserves meaning and eliminates the
# paren-fragment leakage that survives splitting.
# Both regexes tolerate unclosed brackets: the `\)?` / `\]?` makes the closing
# bracket optional so unbalanced "(in millions of dollars" or "foo]" tails
# also disappear.
_CAPTION_PARENS_RE = re.compile(r"\([^)]*\)?")
_CAPTION_BRACKETS_RE = re.compile(r"\[[^\]]*\]?")
_CAPTION_ORPHAN_BRACKET_RE = re.compile(r"[\[\]\(\)]")


def _split_caption(caption: str) -> list[str]:
    """Break a caption into rough noun-phrase candidates.
    Treasury captions look like 'Table 1.- Status under Limitation, December 31, 1949'.
    We strip the leading 'Table N.-' prefix, any trailing '- Continued' / ', con'
    suffix (continuation markers are page-merge signals, not keywords), ALL
    parenthesized / bracketed annotations (units, acronyms, source markers —
    these carry no concept content), any trailing footnote markers ('2/'), and
    then split on punctuation and date-introducers."""
    # "Table 1.- ...", "Table MQ-3. - ...", "Table FFO-2.—..." — the label
    # may contain dashes (PDO-3, MQ-3, FFO-2, TSO-3), so allow `-` inside.
    s = re.sub(r"^\s*(?:Table|Chart|Figure|Schedule)\s+[A-Za-z0-9.\-]+\.?\s*[-–—]?\s*",
               "", caption, flags=re.IGNORECASE).strip()
    s = _CAPTION_CONT_SUFFIX_RE.sub("", s).strip()
    # Strip parens / brackets uniformly (balanced or unbalanced). The orphan
    # pass cleans up stray bracket characters left by a tail that ran off the
    # end of the string with no closing partner that the previous regexes
    # couldn't pair (e.g. a stray "]" at end after the matching "[" was
    # already consumed earlier).
    s = _CAPTION_PARENS_RE.sub(" ", s)
    s = _CAPTION_BRACKETS_RE.sub(" ", s)
    s = _CAPTION_ORPHAN_BRACKET_RE.sub(" ", s)
    # Strip trailing footnote markers like "2/" or "1/" at end of caption.
    s = re.sub(r"\s+\d+/\s*$", "", s)
    # Collapse whitespace left by the bracket stripping.
    s = re.sub(r"\s{2,}", " ", s).strip()
    if not s:
        return []
    return [p.strip() for p in _NOUN_SPLIT_RE.split(s) if p.strip()]


def _harvest_keywords(caption: str | None,
                      column_headers: list[str],
                      row_headers: list[str]) -> list[str]:
    """Dateless concept phrases from the caption only, deduped order-preserving.

    column_headers and row_headers are intentionally NOT pushed in. They live
    on PageCatalogRow as their own fields. Pushing them in was the source of
    vocab pollution like 'Country', 'Europe', individual country names,
    'Issue date', 'Maturity date', etc.
    """
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
# Page text reconstruction (for date regex) — concat element contents.
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
        # For tables, strip HTML tags so the date regex sees clean text.
        if t == "table":
            content = re.sub(r"<[^>]+>", " ", str(content))
        parts.append(str(content))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------

def _resolve_prose_title(elements: list[dict]) -> str | None:
    """For prose pages: pick the [title] (if present) else the first
    non-boilerplate [section_header] as the page's headline title."""
    title: str | None = None
    first_section_header: str | None = None
    from .classify import _BOILERPLATE_HEADER_RE
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
    """For prose pages: lift each non-boilerplate [section_header] (and the
    [title], if any) as a candidate keyword phrase, then apply the standard
    dateless + meaningless filter."""
    from .classify import _BOILERPLATE_HEADER_RE
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


def _extract_content_blocks(
    elements: list[dict],
) -> tuple[list[ContentBlock], list[str], list[str]]:
    """Walk elements in document order; emit a ContentBlock per
    table / chart element, or one prose ContentBlock if neither is
    present but the page has substantive headers.

    Returns (content_blocks, all_column_headers, all_row_headers) — the
    flattened header lists feed the page-level keyword harvest as a
    backup signal when no caption is found.
    """
    blocks: list[ContentBlock] = []
    all_columns: list[str] = []
    all_rows: list[str] = []

    # Caption window state — accumulates [section_header]/[title]+text
    # pieces until the next visual element closes it.
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

    Returns a dict with keys: content_blocks, keywords, min_year, max_year.
    `bulletin` is the page's parent bulletin in "YYYY-MM" form; it feeds
    the year-envelope fallback for pages with no 4-digit year on them
    (see `_canonicalize_year_envelope`). A page with no blocks is
    non-retrievable; blank / toc are decided upstream by `cheap_classify`
    and never reach this function.
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

    # No table / chart: try prose. Emit one prose block iff at least
    # one substantive section_header survives boilerplate filtering.
    from .classify import has_prose_content
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

    # Non-retrievable page: skip the year-envelope canonicalization
    # entirely. There is nothing to retrieve, so a year window is moot.
    return {"content_blocks": [], "keywords": [],
            "min_year": None, "max_year": None}
