"""Heuristic page classifier — runs before any LLM call.

Cheap pass that filters out pages we can confidently skip (blank) or mark
without an LLM (obvious TOC pages). Everything else falls through to the
per-page LLM extractor in `extract_fields.py`, which assigns the final
page_kind among {table, text, chart}.
"""

from __future__ import annotations

from .schema import PageKind


_TOC_HEAD_PATTERNS = ("table of contents", "contents", "tableofcontents")


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
    """Heuristic: TOC pages live near the front of a bulletin and have the
    string 'contents' near the top. We let the LLM confirm in the harvest pass.
    """
    if page_index > 15:
        return False
    head = text[:200].lower().replace(" ", "")
    return any(p.replace(" ", "") in head for p in _TOC_HEAD_PATTERNS)


def cheap_classify(text: str, page_index: int) -> PageKind | None:
    """Return a page_kind iff the heuristic is confident; otherwise None
    (caller falls back to the LLM extractor).
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
