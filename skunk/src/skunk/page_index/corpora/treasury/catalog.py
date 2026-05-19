"""Treasury Bulletin catalog builder — `CatalogBuilder` impl.

Deterministic per-bulletin parse: parsed-JSON elements → per-page
PageCatalogRow with content_blocks, keywords, year envelope, banner,
printed-page label. No LLM at this stage.

Treasury-specific signal heuristics live in `_classify` (page-shape
heuristics) and `_extract` (table/caption/keyword extraction).
"""

from __future__ import annotations

import logging
import re

from ...schema import PageCatalogRow
from ._classify import (
    char_metrics, cheap_classify, has_prose_content, has_visual_elements,
)
from ._extract import parse_page_fields

log = logging.getLogger(__name__)


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

# Continuation-page marker on the first table block's title.
_CONT_SUFFIX_RE = re.compile(
    r"[,\s\-–—]+(?:con|cont|continued)\s*\.?\s*\)?\s*$", re.IGNORECASE,
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


def _first_titled_block(r: PageCatalogRow):
    for b in r.content_blocks:
        if b.kind == "prose":
            continue
        if b.title:
            return b
    return None


def _reconstruct_text(elements: list[dict]) -> str:
    """Concat element contents the same way the parsed-JSON reader does
    (`[type] content` per element, dropping bare page_number elements).
    Used so the catalog builder can re-derive page text without a
    separate I/O pass.
    """
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


def _merge_continuation_pages(rows: list[PageCatalogRow]) -> int:
    """Forward-fill table identity across continuation pages.

    A content page is a continuation when EITHER:
      (a) its first table/chart block's title ends with `con / cont / continued`, OR
      (b) the page has table/chart blocks with no title AND the previous
          content page has a non-empty title (caption-less continuation,
          ~9% of table pages — the layout parser drops the caption).
    """
    rows.sort(key=lambda r: r.page)
    parent: PageCatalogRow | None = None
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
            seen = {k.lower() for k in r.keywords}
            for k in parent.keywords:
                if k.lower() not in seen:
                    r.keywords.append(k)
                    seen.add(k.lower())
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
    ) -> list[PageCatalogRow]:
        rows: list[PageCatalogRow] = []
        for pdf_idx in sorted(pages.keys()):
            elements = pages[pdf_idx]
            text = _reconstruct_text(elements)
            cc, dr = char_metrics(text)
            row = PageCatalogRow(
                bulletin=bulletin,
                page=pdf_idx,
                char_count=cc,
                digit_ratio=round(dr, 3),
            )
            row.printed_page = _extract_printed_page(elements)
            row.banner_self = _extract_page_banner(elements)

            cheap_kind = cheap_classify(text, pdf_idx)
            if cheap_kind is not None:
                # blank or toc — skip the parser, leave content_blocks empty.
                rows.append(row)
                continue

            if not (has_visual_elements(elements) or has_prose_content(elements)):
                rows.append(row)
                continue

            fields = parse_page_fields(elements, bulletin=bulletin)
            row.content_blocks = fields["content_blocks"]
            row.keywords = fields["keywords"]
            row.dates = fields["dates"]
            rows.append(row)

        _merge_continuation_pages(rows)
        return rows
