"""Per-page deterministic field extractor — NO LLM.

Consumes the parsed-JSON element list for one page and returns the same
schema fields the v0.3 LLM extractor used to:
    page_kind, table_title, column_headers, row_headers_sample, keywords, dates

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
# Date extraction — verbatim strings, in order of appearance, deduped.
# ---------------------------------------------------------------------------

_MONTH_FULL = (r"(?:January|February|March|April|May|June|July|August|"
               r"September|October|November|December)")
_MONTH_ABBR = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
_MONTH_ANY = f"(?:{_MONTH_FULL}|{_MONTH_ABBR})"

_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "December 31, 1949" / "Dec. 31, 1949"
    re.compile(rf"\b{_MONTH_ANY}\s+\d{{1,2}},\s*\d{{4}}\b", re.IGNORECASE),
    # "February 1952" / "Feb. 1952"
    re.compile(rf"\b{_MONTH_ANY}\s+\d{{4}}\b", re.IGNORECASE),
    # "Calendar Year 1940" / "Fiscal Year 1948" / "CY 1940" / "FY 1948"
    re.compile(r"\b(?:Calendar Year|Fiscal Year|CY|FY)\s*\d{4}\b", re.IGNORECASE),
    # "Q3 1953"
    re.compile(r"\bQ[1-4]\s*\d{4}\b", re.IGNORECASE),
    # "1932 through Mar 1939" / "1932-1939" range
    re.compile(rf"\b\d{{4}}\s*(?:through|to|-)\s*(?:{_MONTH_ANY}\s*)?\d{{4}}\b",
               re.IGNORECASE),
    # bare 4-digit year — runs LAST so the longer patterns win first
    re.compile(r"\b(19|20)\d{2}\b"),
)


def _extract_dates(text: str) -> list[str]:
    """Lift verbatim date strings from `text`, in order, deduped (case-insensitive)."""
    spans: list[tuple[int, int, str]] = []  # (start, end, verbatim)
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            # Skip if this span overlaps an earlier (longer-pattern) match.
            if any(not (m.end() <= s or m.start() >= e) for s, e, _ in spans):
                continue
            spans.append((m.start(), m.end(), m.group(0)))
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
    on PageCatalogRow as their own fields and remain usable as ranking context
    at L3 leaf_rank, but they are not part of the conceptual index vocabulary.
    Pushing them in was the source of vocab pollution like 'Country', 'Europe',
    individual country names, 'Issue date', 'Maturity date', etc.
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


def parse_page_fields(elements: list[dict]) -> dict[str, Any]:
    """Deterministic per-page field extraction. No LLM.

    Returns a dict with keys: page_kind, table_title, column_headers,
    row_headers_sample, keywords, dates. `page_kind` is:
      - "table" if any [table] element is present
      - "chart" if any [figure]/[image]/etc.
      - "prose" if there's a substantive [section_header] or [title]
        (narrative page with extractable concepts in its headers)
      - "text" otherwise (skipped from index)
    """
    has_table = False
    has_chart = False
    column_headers: list[str] = []
    row_headers: list[str] = []
    extra_captions: list[str] = []

    for el in elements:
        t = (el.get("type") or "").lower()
        content = el.get("content")
        if t == "table" and content:
            has_table = True
            cols, rows, caps = _parse_table_html(str(content))
            if not column_headers:
                column_headers = cols
            if not row_headers:
                row_headers = rows
            extra_captions.extend(caps)
        elif t in ("figure", "image", "chart", "plot", "diagram"):
            has_chart = True

    plain = _page_plain_text(elements)
    dates = _extract_dates(plain)

    if has_table or has_chart:
        caption = _resolve_caption(elements)
        if not caption and extra_captions:
            caption = extra_captions[0]
        keywords = _harvest_keywords(caption, column_headers, row_headers[:8])
        return {
            "page_kind": "table" if has_table else "chart",
            "table_title": caption,
            "column_headers": column_headers[:50],
            "row_headers_sample": row_headers[:12],
            "keywords": keywords[:20],
            "dates": dates,
        }

    # No table / chart: try prose. Index iff at least one substantive
    # section_header survives boilerplate filtering.
    from .classify import has_prose_content
    if has_prose_content(elements):
        prose_kw = _harvest_prose_keywords(elements)
        if prose_kw:
            return {
                "page_kind": "prose",
                "table_title": _resolve_prose_title(elements),
                "column_headers": [],
                "row_headers_sample": [],
                "keywords": prose_kw,
                "dates": dates,
            }

    return {
        "page_kind": "text",
        "table_title": None,
        "column_headers": [],
        "row_headers_sample": [],
        "keywords": [],
        "dates": [],
    }
