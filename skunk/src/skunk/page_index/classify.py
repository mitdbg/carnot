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
