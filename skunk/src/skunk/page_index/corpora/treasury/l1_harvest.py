"""Treasury Bulletin L1 harvester — `L1Harvester` impl.

Two-path discovery of a bulletin's top-level chapter list:

  1. **ToC harvest (primary).** Heuristic-flag ToC candidate pages
     (`_classify.looks_like_toc`), then one LLM call extracts chapter
     spans with their printed-page labels.

  2. **Banner fallback.** When the ToC is missing or unreadable (notably
     the 1949–1964 bulletins where the layout parser drops the ToC
     table), tally the per-page section banners and treat any banner
     appearing on ≥ `min_pages` pages as an L1 entry. SectionSpans
     emitted via this path carry empty printed-page labels; the placer
     handles them through banner-exact match (Method B) instead of
     printed-page span lookup (Method A).
"""

from __future__ import annotations

import logging
from collections import Counter

from skunk.llm_client import LLMClient

from ...schema import PageCatalogRow
from ...stages.l1_harvest import SectionSpan
from ...util import safe_json_loads
from ._classify import looks_like_toc
from .prompts import TOC_HARVEST_SYSTEM

log = logging.getLogger(__name__)


# Banner-fallback knobs (empirically tuned for the 1949–1964 broken-ToC era).
_BANNER_MIN_PAGES = 3       # ignore one-off banners
_BANNER_MAX_L1 = 25         # cap distinct L1 entries per bulletin


def _harvest_toc_llm(
    bulletin: str,
    toc_candidate_pages: list[tuple[int, str]],
    total_pdf_pages: int,
    llm: LLMClient,
) -> list[SectionSpan]:
    if not toc_candidate_pages:
        return []
    user_parts: list[str] = [f"Bulletin month: {bulletin}",
                             f"Total PDF pages: {total_pdf_pages}",
                             ""]
    for pdf_idx, text in toc_candidate_pages:
        user_parts.append(f"--- TOC page (PDF page {pdf_idx}) ---")
        user_parts.append(text.strip())
        user_parts.append("")
    resp = llm.call(system=TOC_HARVEST_SYSTEM, user="\n".join(user_parts),
                    temperature=0.0)
    obj = safe_json_loads(resp.text, context=f"toc/{bulletin}")
    if not isinstance(obj, dict):
        return []

    spans: list[SectionSpan] = []
    for raw in obj.get("sections", []) or []:
        if not isinstance(raw, dict):
            continue
        try:
            label = str(raw["section"]).strip()
            start = str(raw["start_page_printed"]).strip()
            end = str(raw["end_page_printed"]).strip()
        except (KeyError, ValueError, TypeError):
            continue
        if not label or not start or not end:
            continue
        spans.append(SectionSpan(section=label,
                                 start_page_printed=start,
                                 end_page_printed=end))
    return _sort_spans(spans)


def _sort_spans(spans: list[SectionSpan]) -> list[SectionSpan]:
    """Sort by numeric printed page when possible; lexicographic fallback."""
    def _key(sp: SectionSpan) -> tuple[int, int, str]:
        # Bucket non-numeric labels after numeric ones.
        n = _printed_to_int(sp.start_page_printed)
        if n is None:
            return (1, 0, sp.start_page_printed)
        return (0, n, "")
    return sorted(spans, key=_key)


def _printed_to_int(s: str | None) -> int | None:
    if not s:
        return None
    s = s.strip()
    if s.isdigit() or (s.startswith("0") and s.lstrip("0").isdigit()):
        return int(s)
    return None


def _l1_from_banners(rows: list[PageCatalogRow]) -> list[SectionSpan]:
    """Derive L1 vocabulary from per-page banner_self frequency.

    Used as a fallback when ToC harvest returns nothing. SectionSpans
    carry empty `start_page_printed` / `end_page_printed` because we
    don't know where each chapter starts/ends — only that it exists.
    The placer's Method B (banner-exact match) handles this case.
    """
    counts: Counter[str] = Counter()
    for r in rows:
        if not r.content_blocks:
            continue
        b = (r.banner_self or "").strip()
        if b:
            counts[b] += 1
    spans: list[SectionSpan] = []
    for banner, n in counts.most_common(_BANNER_MAX_L1):
        if n < _BANNER_MIN_PAGES:
            break
        spans.append(SectionSpan(
            section=banner, start_page_printed="", end_page_printed="",
        ))
    return spans


class TreasuryL1Harvester:
    """`L1Harvester` impl for Treasury Bulletin."""

    def harvest_bulletin(
        self,
        bulletin: str,
        rows: list[PageCatalogRow],
        pages_text: dict[int, str],
        llm: LLMClient,
    ) -> list[SectionSpan]:
        toc_candidates = [
            (idx, text) for idx, text in pages_text.items()
            if looks_like_toc(text, idx)
        ]
        spans = _harvest_toc_llm(
            bulletin=bulletin,
            toc_candidate_pages=toc_candidates,
            total_pdf_pages=len(pages_text),
            llm=llm,
        )
        if spans:
            return spans
        return _l1_from_banners(rows)
