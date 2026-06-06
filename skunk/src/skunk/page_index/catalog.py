"""Per-page catalog build: parsed-JSON pages -> BuildPage.

Deterministic (no LLM): the structural scaffold the `summarize` LLM pass
later cleans and enriches. Two cohesive layers: page-shape probes
(visual/prose detection, plus a ToC probe shared with the L1 stage) and
per-page structural extraction — tables and charts with verbatim captions +
column/row labels (via a small HTML parser), prose blocks, running-head
banners, and printed-page labels — driven per bulletin by
TreasuryCatalogBuilder.

Keyword/date harvesting and content-vs-blank classification are intentionally
left to `summarize`: the LLM overwrites both, so doing them here would be
redundant. This stage only produces what the LLM is barred from inventing —
faithful table structure and layout facts."""

from __future__ import annotations

import re
import logging
from html.parser import HTMLParser
from skunk.corpus import page_tagged_text

from .data_model import BuildPage, ContentBlock

log = logging.getLogger(__name__)


_TOC_HEAD_PATTERNS = ("table of contents", "contents", "tableofcontents")

# A line that ends with a short integer (the printed-page reference of a ToC
# entry, e.g. "Federal fiscal operations ........  9").
_TOC_LINE_RE = re.compile(r"\S.+?\s+\d{1,3}\s*$")

# Boilerplate section-header content that doesn't, on its own, make a page
# prose-indexable (every prose page has "Note", "Source", etc.).
_BOILERPLATE_HEADER_RE = re.compile(
    r"^\s*(note|notes|source|sources|footnote|footnotes|legend|key|"
    r"introduction|disclaimer|preface|index|table of contents|contents|"
    r"references?)\s*[:.\-]?\s*$",
    re.IGNORECASE,
)

_VISUAL_ELEMENT_TYPES = frozenset({
    "table", "figure", "image", "chart", "plot", "diagram",
})

_HEADER_ELEMENT_TYPES = frozenset({"section_header", "title"})


# A page with fewer than this many non-whitespace-stripped chars carries no
# retrievable prose. Read by the deterministic blank-skip in `summarize`.
BLANK_CHAR_THRESHOLD = 50


def looks_like_toc(text: str, page_index: int) -> bool:
    """ToC pages live near the front of a bulletin and either advertise
    themselves ("contents" near the top) OR have many lines ending in a
    printed-page reference (the 1949–1964 bulletins). Shared with the L1 stage.
    """
    if page_index > 25:
        return False
    head = text[:400].lower().replace(" ", "")
    if any(p.replace(" ", "") in head for p in _TOC_HEAD_PATTERNS):
        return True
    n_entry_lines = 0
    for line in text.splitlines():
        if _TOC_LINE_RE.match(line.strip()):
            n_entry_lines += 1
            if n_entry_lines >= 6:
                return True
    return False


def has_visual_elements(elements: list[dict]) -> bool:
    for el in elements:
        if (el.get("type") or "").lower() in _VISUAL_ELEMENT_TYPES:
            return True
    return False


def has_prose_content(elements: list[dict]) -> bool:
    """True if the page has at least one substantive [section_header] or
    [title] element — i.e. a narrative page with structural anchors we
    can index. Pure-boilerplate pages return False.
    """
    for el in elements:
        if (el.get("type") or "").lower() not in _HEADER_ELEMENT_TYPES:
            continue
        content = (el.get("content") or "").strip()
        if not content or len(content) < 4:
            continue
        if _BOILERPLATE_HEADER_RE.match(content):
            continue
        return True
    return False


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
# Content-block extraction.
# ---------------------------------------------------------------------------

_CAPTION_TYPES = frozenset({"section_header", "title"})

_UNIT_NOTE_RE = re.compile(r"^\s*\(?in\s+(thousands|millions|billions)\b",
                           re.IGNORECASE)

# Continuation-page marker on the first table block's title.
_CONT_SUFFIX_RE = re.compile(
    r"[,\s\-–—]+(?:con|cont|continued)\s*\.?\s*\)?\s*$", re.IGNORECASE,
)


def _extract_content_blocks(elements: list[dict]) -> list[ContentBlock]:
    """Walk elements; emit one ContentBlock per table/chart, or one prose
    block if no visual but substantive headers exist.
    """
    blocks: list[ContentBlock] = []

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
                column_headers=cols,
                row_headers=rows,
            ))
            pieces = []
            in_window = False
            continue

        if t in ("figure", "image", "chart", "plot", "diagram"):
            caption = _flush_caption()
            blocks.append(ContentBlock(kind="chart", title=caption))
            pieces = []
            in_window = False
            continue

    return blocks


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


def _extract_page_blocks(elements: list[dict]) -> list[ContentBlock]:
    """Deterministic per-page structural extraction (no LLM): one ContentBlock
    per table/chart (verbatim caption + column/row labels), else a single prose
    block for a narrative page. Block summaries and page keywords are added
    later by `summarize`. Returns [] for a page with no indexable structure
    (blank / front-matter / ToC)."""
    blocks = _extract_content_blocks(elements)
    if blocks:
        return blocks
    if has_prose_content(elements):
        return [ContentBlock(kind="prose", title=_resolve_prose_title(elements))]
    return []


# ---------------------------------------------------------------------------
# Banners + printed-page labels (placement/merge inputs).
# ---------------------------------------------------------------------------

# Pages whose [title] / [page_header] is one of these would pollute the
# section-banner pool. Drops month-year cover headers ("May 1972"), bare
# years, and the recurring publication masthead.
_NOISE_BANNER_RE = re.compile(
    r"^\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec|"
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{4}\s*\.?\s*$",
    re.IGNORECASE,
)
_BARE_YEAR_RE = re.compile(r"^\s*\d{4}\s*\.?\s*$")
_MASTHEAD_RE = re.compile(
    r"^\s*(?:treasury\s+bulletin|u\.?\s*s\.?\s+treasury(?:\s+department)?|"
    r"department\s+of\s+the\s+treasury)\s*\.?\s*$",
    re.IGNORECASE,
)


def _is_section_banner(content: str) -> bool:
    s = (content or "").strip()
    if len(s) < 6:
        return False
    if _NOISE_BANNER_RE.match(s):
        return False
    if _BARE_YEAR_RE.match(s):
        return False
    if _MASTHEAD_RE.match(s):
        return False
    return True


def _extract_printed_page(elements: list[dict]) -> str | None:
    for el in elements:
        if el.get("type") == "page_number":
            content = el.get("content")
            if content is not None:
                txt = str(content).strip()
                if txt:
                    return txt
    return None


def _extract_page_banner(elements: list[dict]) -> str | None:
    """First `[title]` or `[page_header]` element passing the banner-shape
    filter. Treasury bulletins place the section name either as a title
    (section-start pages) or as a second page_header below the masthead."""
    for el in elements:
        if el.get("type") not in ("title", "page_header"):
            continue
        content = (el.get("content") or "").strip()
        if content and _is_section_banner(content):
            return content
    return None


def _first_titled_block(r: BuildPage):
    for b in r.content_blocks:
        if b.kind == "prose":
            continue
        if b.title:
            return b
    return None


def _merge_continuation_pages(rows: list[BuildPage]) -> int:
    """Forward-fill table identity across continuation pages.

    A content page is a continuation when EITHER:
      (a) its first table/chart block's title ends with `con / cont / continued`, OR
      (b) the page has table/chart blocks with no title AND the previous
          content page has a non-empty title (caption-less continuation,
          ~9% of table pages — the layout parser drops the caption).
    """
    rows.sort(key=lambda r: r.page)
    parent: BuildPage | None = None
    merged = 0
    for r in rows:
        has_visual = any(b.kind != "prose" for b in r.content_blocks)
        if not has_visual:
            parent = None
            continue

        first_block = next((b for b in r.content_blocks if b.kind != "prose"), None)
        is_explicit_cont = bool(
            first_block and first_block.title
            and _CONT_SUFFIX_RE.search(first_block.title)
        )
        parent_titled = _first_titled_block(parent) if parent else None
        is_implicit_cont = (
            first_block is not None and not first_block.title
            and parent_titled is not None
        )
        if (is_explicit_cont or is_implicit_cont) and parent is not None and parent_titled is not None:
            if first_block is not None:
                first_block.title = parent_titled.title
            merged += 1
        else:
            if _first_titled_block(r):
                parent = r
    return merged


class TreasuryCatalogBuilder:
    """`CatalogBuilder` impl for U.S. Treasury Bulletin pages."""

    def parse_bulletin(
        self,
        bulletin: str,
        pages: dict[int, list[dict]],
    ) -> list[BuildPage]:
        rows: list[BuildPage] = []
        for pdf_idx in sorted(pages.keys()):
            elements = pages[pdf_idx]
            text = page_tagged_text(elements)
            row = BuildPage(
                bulletin=bulletin,
                page=pdf_idx,
                char_count=len(text),
            )
            row.printed_page = _extract_printed_page(elements)
            row.banner_self = _extract_page_banner(elements)
            row.content_blocks = _extract_page_blocks(elements)
            rows.append(row)

        _merge_continuation_pages(rows)
        return rows
