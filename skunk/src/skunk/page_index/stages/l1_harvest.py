"""Stage 2: L1 harvester — per-bulletin chapter-span discovery.

Extracts the bulletin's own top-level chapter list from its Table of
Contents (or an equivalent corpus-specific source). Output is one
`SectionSpan` per chapter covering the bulletin body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from skunk.common import LLMClient

from ..schema import PageCatalogRow


@dataclass(frozen=True)
class SectionSpan:
    """A chapter span within a single bulletin.

    `start_page_printed` / `end_page_printed` are the *printed* page
    labels as they appear in the bulletin footer (e.g. "9", "A-1") —
    NOT PDF page indices. They're verbatim strings because some corpora
    use non-numeric page labels.
    """
    section: str
    start_page_printed: str
    end_page_printed: str


_DIGIT_RE = re.compile(r"^\s*0*(\d+)\s*$")


def _printed_to_int(s: str | None) -> int | None:
    if not s:
        return None
    m = _DIGIT_RE.match(s)
    return int(m.group(1)) if m else None


def section_for_printed_page(
    spans: list[SectionSpan], printed_page: str | None,
) -> str | None:
    """Look up the section label covering `printed_page`. Returns None
    when the page falls outside every span, when `printed_page` is None,
    or when neither side parses numerically (no fallback)."""
    n = _printed_to_int(printed_page)
    if n is None:
        return None
    for sp in spans:
        lo = _printed_to_int(sp.start_page_printed)
        hi = _printed_to_int(sp.end_page_printed)
        if lo is None or hi is None:
            continue
        if lo <= n <= hi:
            return sp.section
    return None


class L1Harvester(Protocol):
    """Discover one bulletin's top-level chapter spans.

    Implementations typically read the bulletin's ToC pages, but the
    contract is just "return the list of chapter spans for this
    bulletin". Empty list is acceptable when no ToC exists.

    `pages_text` is `{1-based PDF page: concatenated page text}` —
    handed to the implementation so it can run heuristics or LLM calls
    over the bulletin's prose.
    """

    def harvest_bulletin(
        self,
        bulletin: str,
        rows: list[PageCatalogRow],
        pages_text: dict[int, str],
        llm: LLMClient,
    ) -> list[SectionSpan]: ...
