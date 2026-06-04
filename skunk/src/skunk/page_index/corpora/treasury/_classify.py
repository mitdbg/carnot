"""Heuristic page classifier — runs before the field extractor.

Cheap pass that filters pages we can confidently skip (blank) or flag
as obvious TOC pages so the deterministic parser doesn't mistake a ToC
entry list for a real table. Everything else falls through to the
parser in `_extract`.
"""

from __future__ import annotations

import re


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


def char_metrics(text: str) -> tuple[int, float]:
    """`(char_count, digit_ratio)` for `text`; digit_ratio over non-whitespace."""
    n = len(text)
    if n == 0:
        return 0, 0.0
    non_ws = sum(1 for c in text if not c.isspace())
    if non_ws == 0:
        return n, 0.0
    digits = sum(1 for c in text if c.isdigit())
    return n, digits / non_ws


def looks_blank(text: str) -> bool:
    return len(text.strip()) < 50


def looks_like_toc(text: str, page_index: int) -> bool:
    """ToC pages live near the front of a bulletin and either advertise
    themselves ("contents" near the top) OR have many lines ending in a
    printed-page reference (the 1949–1964 bulletins).
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


def cheap_classify(text: str, page_index: int) -> str | None:
    """Return `"blank"` or `"toc"` iff the heuristic is confident; else None."""
    if looks_blank(text):
        return "blank"
    if looks_like_toc(text, page_index):
        return "toc"
    return None


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
