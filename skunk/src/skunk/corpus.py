"""Unified corpus access for the Treasury Bulletin corpus: path resolution, parsed-JSON
reading (LRU-cached, indexed by 1-based PDF page), PyMuPDF page rendering, and text
cleaning. Deliberately free of `ExecutionContext`/orchestrator coupling so the offline
prep scripts can import it.

Text accessors read parsed JSON only (cleaner than PyMuPDF text on scanned 1940s-50s
pages); PyMuPDF is used solely for rendering page images."""

from __future__ import annotations

import html as _html
import json
import re
from functools import lru_cache
from pathlib import Path

# Re-exported for `skunk.search_agent.prep.page_cleaner`, which imports it from here; the
# rasterizer itself now lives in `common` (the single page-render primitive).
from skunk.common import pdf_path_for, render_page_b64  # noqa: F401
from skunk.errors import StepFailed

_FILENAME_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})\.pdf$")


def parse_bulletin_filename(path: str | Path) -> str:
    """treasury_bulletin_1953_06.pdf  ->  '1953-06'."""
    m = _FILENAME_RE.search(str(path))
    if not m:
        raise ValueError(f"not a treasury bulletin filename: {path}")
    return f"{m.group(1)}-{m.group(2)}"


# Directory args below are required, not defaulted: the corpus lives wherever the
# caller's single `SkunkConfig` says (`config.parsed_json_dir` / `config.pdf_dir`).
# These functions never resolve a directory on their own — that would reintroduce a
# second, possibly-divergent source of truth alongside the entrypoint's config.


@lru_cache(maxsize=64)
def load_parsed_doc(month: str, base_dir: str) -> dict:
    """Load a bulletin's parsed-JSON document from `base_dir` (LRU-cached on both args).
    Raises FileNotFoundError if missing; json errors propagate."""
    year, mon = month.split("-")
    p = Path(base_dir) / f"treasury_bulletin_{year}_{mon}.json"
    if not p.exists():
        raise FileNotFoundError(f"parsed-JSON not found: {p}")
    return json.loads(p.read_text())


def page_elements(
    month: str,
    *,
    base_dir: str | Path,
    fill_gaps: bool = False,
) -> dict[int, list[dict]]:
    """`{1-based PDF page index → list of parsed-JSON element dicts}`. `fill_gaps=False`
    (default) includes only pages carrying elements; `fill_gaps=True` includes every
    page `1..max_page` (empty list for blanks) — the dense view the build pipeline wants."""
    doc = load_parsed_doc(month, str(base_dir))

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

    if fill_gaps:
        return {p: by_page.get(p, []) for p in range(1, max_page + 1)}
    return by_page


def get_page_text(month: str | None, page: int | None, *, base_dir: str | Path) -> str | None:
    """Concatenated parsed-JSON `content` for a PDF page (extract Tier 1; HTML tables
    verbatim). Raises `StepFailed` if the source is missing/corrupt; returns None when
    the source is healthy but this page has no content."""
    if month is None or page is None:
        return None
    try:
        idx = page_elements(month, base_dir=base_dir)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
        raise StepFailed("extract", f"parsed-JSON unavailable for {month}: {e}") from e
    elements = idx.get(int(page))
    if not elements:
        return None
    parts = [el["content"] for el in elements if el.get("content") is not None]
    return "\n\n".join(parts) if parts else None


def page_tagged_text(elements: list[dict]) -> str:
    """Concatenate one page's parsed-JSON elements, each prefixed with its parsed
    `[type]` for structural signal. `page_number` elements are dropped; an empty
    page yields "". This is the per-page form of `page_text_tagged`."""
    parts: list[str] = []
    for el in elements:
        content = el.get("content")
        if content is None:
            continue
        t = el.get("type") or "text"
        if t == "page_number":
            continue
        parts.append(f"[{t}] {content}")
    return "\n\n".join(parts) if parts else ""


def page_text_tagged(month: str, *, base_dir: str | Path | None = None) -> dict[int, str]:
    """`{1-based PDF page index: concatenated page text}` for the build pipeline. Each
    element is prefixed with its parsed `[type]` for structural signal; `page_number`
    elements are dropped, blank pages map to "". Raises FileNotFoundError if missing."""
    return {
        pdf_page: page_tagged_text(elements)
        for pdf_page, elements in page_elements(month, base_dir=base_dir, fill_gaps=True).items()
    }


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TD_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL)
_MULTI_NL_RE = re.compile(r"\n{3,}")
_LONG_DOTS_RE = re.compile(r"\.{4,}")
# Four-digit years 1776–2026 as whole tokens.
_YEAR_RE = re.compile(r"\b(177[6-9]|17[89]\d|1[89]\d\d|200\d|201\d|202[0-6])\b")
# Cells with no semantic content (pure numbers, money/percent, ranges, placeholders).
_PLACEHOLDER_CELL_RE = re.compile(
    r"[+\-]?\$\s*[\d,]+(\.\d+)?%?"   # $1,234.56 / $ 194.3
    r"|[+\-]?[\d,]+(\.\d+)?%?"        # 1,234 / 3.5 / 50%
    r"|\d[\d,]*[\/\-]\d[\d,]*"         # 4-5 / 283/444
    r"|\$\s*-+"                         # $ -- / $ -
    r"|\$\s*\.\d+"                      # $ .6 / $.1
    r"|\.\d+"                           # .6 / .2 (bare decimals)
    r"|\*+"                             # * / ** (footnote markers)
    r"|-{2,}"                           # -- / --- (dash placeholders)
)
# Single whitespace-delimited *token* with no semantic content. Same shapes as
# _PLACEHOLDER_CELL_RE but matched per token (+ footnote refs like "8/"), so a cell is
# recognised as a pure value when EVERY token is a placeholder — catching multi-token
# numerics that _PLACEHOLDER_CELL_RE misses, e.g. dollars+cents "$216,370,286 77" or
# value+footnote "9,198 2/". Used only when `drop_numeric_cells=True` (see preprocess_text).
_PLACEHOLDER_TOKEN_RE = re.compile(
    r"[+\-]?\$?[\d,]+(\.\d+)?%?"   # 1,234 / $1,234.56 / 3.5 / 50% / 77
    r"|[+\-]?\$?\.\d+%?"            # .6 / $.1
    r"|[\d,]+/"                      # footnote refs: 8/ 13/ 2/
    r"|\d[\d,]*[\/\-]\d[\d,]*"       # 4-5 / 283/444
    r"|\$?-+"                        # - / -- / $- / $--
    r"|\$+"                          # lone $ (e.g. the "$" of a spaced "$ --" blank value)
    r"|\*+"                          # * / ** (footnote markers)
)


def _cell_is_all_placeholder(cell: str) -> bool:
    """True if every whitespace-delimited token in `cell` is a numeric/placeholder token, so
    the whole cell is a pure value (e.g. "$216,370,286 77", "9,198 2/") carrying no text."""
    tokens = cell.split()
    return bool(tokens) and all(_PLACEHOLDER_TOKEN_RE.fullmatch(t) for t in tokens)


def page_plain_text(elements: list[dict]) -> str:
    """Reduce one page's parsed-JSON elements to plain text: drops `page_number`
    elements, strips HTML tags from `table` content, joins the rest with newlines."""
    parts: list[str] = []
    for el in elements:
        t = (el.get("type") or "").lower()
        if t == "page_number":
            continue
        content = el.get("content")
        if not content:
            continue
        if t == "table":
            content = _HTML_TAG_RE.sub(" ", str(content))  # strip HTML tags from table markup
        parts.append(str(content))
    return "\n".join(parts)


# Table elements carry the page's numeric grid — the bulk of its tokens. For uses
# that only need a page's *topic* (not its values), truncate each table to a short
# identifying snippet. Logic originally from `search_agent.prep.page_cleaner`
# (table-truncation for the page-reorder cleaner); homed here so the request path
# can reuse it without importing the prep module.
TRUNCATE_TABLE_CHARS = 50


def _table_display_texts(page_elements: list[dict]) -> dict:
    """Map element id → display string. Non-table elements pass through whole; each
    table is shortened to whichever is shorter of its first row or its first
    `TRUNCATE_TABLE_CHARS` chars, extended to the shortest prefix that stays unique
    among the page's tables, with '...(truncated)' appended when shortened."""
    def _short_cand(content):
        row = content.split('\n')[0]
        return row if len(row) <= TRUNCATE_TABLE_CHARS else content[:TRUNCATE_TABLE_CHARS]

    display_texts = {elt['id']: elt['content'] for elt in page_elements if elt['type'] != 'table'}

    table_elts = [elt for elt in page_elements if elt['type'] == 'table']
    candidates = {e['id']: _short_cand(e['content']) for e in table_elts}

    if len(table_elts) > 1:
        cand_values = list(candidates.values())
        for elt in table_elts:
            if cand_values.count(candidates[elt['id']]) > 1:
                content = elt['content']
                other_contents = [e['content'] for e in table_elts if e['id'] != elt['id']]
                for length in range(len(candidates[elt['id']]) + 1, len(content) + 1):
                    prefix = content[:length]
                    if not any(other.startswith(prefix) for other in other_contents):
                        candidates[elt['id']] = prefix
                        break

    for elt in table_elts:
        cand = candidates[elt['id']]
        display_texts[elt['id']] = cand + '...(truncated)' if len(cand) < len(elt['content']) else elt['content']

    return display_texts


def page_sanitized_text(elements: list[dict]) -> str:
    """Like `page_plain_text`, but each TABLE is truncated to a short identifying
    snippet — its title / first row (via `_table_display_texts`) — instead of its
    full grid. Keeps titles/prose/column-headers; drops per-row detail and the
    token-heavy numeric body."""
    if not elements:
        return ""
    display = _table_display_texts(elements)
    parts: list[str] = []
    for el in elements:
        t = (el.get("type") or "").lower()
        if t == "page_number":
            continue
        txt = display.get(el.get("id"), el.get("content"))
        if not txt:
            continue
        if t == "table":
            txt = _HTML_TAG_RE.sub(" ", str(txt))  # strip HTML from the (truncated) snippet
        parts.append(str(txt))
    return "\n".join(parts)

def preprocess_text(text: str, elt_type: str, strip_years: bool = False,
                    drop_numeric_cells: bool = False) -> str:
    """Preprocess element text for embedding: tables → tag-stripped non-placeholder
    cells joined by spaces; others → collapsed newline runs. Always drops dot-leader
    runs; `strip_years` also removes 1776–2026 year tokens.

    `drop_numeric_cells` (default False preserves the original single-token behaviour)
    additionally drops empty cells and any cell whose every whitespace token is numeric/
    placeholder — so multi-token values like "$216,370,286 77" (dollars+cents) or "9,198 2/"
    (value+footnote) are removed too, leaving mostly text (headers, row/column names,
    footnotes). Enabled for the DAIS corpus; off for the OfficeQA `qwen-v2` recipe."""
    if elt_type == "table":
        # inner text of each <td>/<th> cell (HTML-unescaped + tag-stripped),
        # dropping pure-number/placeholder cells (page numbers, dot leaders, …)
        cells = []
        for m in _TD_RE.finditer(text):
            cell = _HTML_TAG_RE.sub("", _html.unescape(m.group(1))).strip()
            if drop_numeric_cells:
                if not cell or _cell_is_all_placeholder(cell):
                    continue
            elif _PLACEHOLDER_CELL_RE.fullmatch(cell):
                continue
            cells.append(cell)
        text = " ".join(cells)
    else:
        text = _MULTI_NL_RE.sub("\n\n", text)

    text = _LONG_DOTS_RE.sub("", text)

    if strip_years:
        text = _YEAR_RE.sub("", text)

    return text.strip()
