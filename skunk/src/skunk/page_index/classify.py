"""Heuristic page classifier — runs before any LLM call.

Cheap pass that filters out pages we can confidently skip (blank) or
flag as obvious TOC pages so the deterministic per-page parser doesn't
mistake a ToC entry list for a real table. Everything else falls
through to the parser in `extract_fields.py`.
"""

from __future__ import annotations

import re


_TOC_HEAD_PATTERNS = ("table of contents", "contents", "tableofcontents")

# A line that ends with a short integer (the printed-page reference of a ToC
# entry, e.g. "Federal fiscal operations ........  9"). We count these per
# page; pages with many such lines look like ToC pages even when their first
# 200 chars don't include the literal word "contents" — the 1949–1964
# bulletins routinely fall into this category.
_TOC_LINE_RE = re.compile(r"\S.+?\s+\d{1,3}\s*$")


def char_metrics(text: str) -> tuple[int, float]:
    """Return (char_count, digit_ratio) for `text`. digit_ratio is over non-whitespace chars."""
    n = len(text)
    if n == 0:
        return 0, 0.0
    non_ws = sum(1 for c in text if not c.isspace())
    if non_ws == 0:
        return n, 0.0
    digits = sum(1 for c in text if c.isdigit())
    return n, digits / non_ws


def looks_blank(text: str) -> bool:
    """Page has effectively no extractable text."""
    return len(text.strip()) < 50


def looks_like_toc(text: str, page_index: int) -> bool:
    """Heuristic: a ToC page lives near the front of a bulletin and either
    advertises itself ("contents" near the top) OR has the structural shape
    of a ToC — many lines ending in a printed-page reference.

    The keyword check catches the modern era (1965+) where the ToC is a
    titled page. The shape check catches mid-century bulletins (1949–1964)
    where the ToC is buried inside front matter without a "contents"
    header — those pages still look like a ToC because each entry ends in
    a page number. The LLM harvest pass then confirms.
    """
    if page_index > 25:
        return False
    head = text[:400].lower().replace(" ", "")
    if any(p.replace(" ", "") in head for p in _TOC_HEAD_PATTERNS):
        return True
    # Shape fallback: at least 6 lines of "<entry> ... <printed page>" form.
    n_entry_lines = 0
    for line in text.splitlines():
        if _TOC_LINE_RE.match(line.strip()):
            n_entry_lines += 1
            if n_entry_lines >= 6:
                return True
    return False


def cheap_classify(text: str, page_index: int) -> str | None:
    """Return "blank" or "toc" iff the heuristic is confident (caller
    skips the parser for those); otherwise None.
    """
    if looks_blank(text):
        return "blank"
    if looks_like_toc(text, page_index):
        return "toc"
    return None


_VISUAL_ELEMENT_TYPES = frozenset({
    "table", "figure", "image", "chart", "plot", "diagram",
})

_HEADER_ELEMENT_TYPES = frozenset({"section_header", "title"})


def has_visual_elements(elements: list[dict]) -> bool:
    """True if the page's parsed-JSON element list contains any table/figure/image."""
    for el in elements:
        if (el.get("type") or "").lower() in _VISUAL_ELEMENT_TYPES:
            return True
    return False


# Boilerplate section_header content that should NOT trigger prose indexing on
# its own (every prose page has "Note", "Source", etc.). A page needs at least
# one substantive header.
_BOILERPLATE_HEADER_RE = __import__("re").compile(
    r"^\s*(note|notes|source|sources|footnote|footnotes|legend|key|"
    r"introduction|disclaimer|preface|index|table of contents|contents|"
    r"references?)\s*[:.\-]?\s*$",
    __import__("re").IGNORECASE,
)


def has_prose_content(elements: list[dict]) -> bool:
    """True if the page has at least one substantive [section_header] or [title]
    element — i.e. it's a narrative page with structural anchors we can index
    (Treasury Financing Operations write-ups, Profile of the Economy notes,
    etc.). Pure-boilerplate pages whose only headers are 'Note' / 'Source'
    return False.
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
