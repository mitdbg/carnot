"""Build phase 3a — L1 chapter-span harvest.

Per bulletin: map the Table of Contents (or, as a fallback, page banners) to its
top-level (L1) chapter spans (`SectionSpan`). Consumed by the placer."""

from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel, ConfigDict

from skunk.common import LLMClient, parse_json_response

from .catalog import looks_like_toc
from .data_model import BuildPage

class SectionSpan(BaseModel):
    """A chapter span within a single bulletin.

    `start_page_printed` / `end_page_printed` are the *printed* page
    labels as they appear in the bulletin footer (e.g. "9", "A-1") —
    NOT PDF page indices. They're verbatim strings because some corpora
    use non-numeric page labels.
    """
    model_config = ConfigDict(frozen=True)

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
    obj = parse_json_response(resp.text)
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


def _l1_from_banners(rows: list[BuildPage]) -> list[SectionSpan]:
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
        rows: list[BuildPage],
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


TOC_HARVEST_SYSTEM = """You map a U.S. Treasury Bulletin's Table of Contents to its TOP-LEVEL
(L1) chapter spans. Return chapter-level headings only — never sub-sections.

You receive the publication month (YYYY-MM), the raw text of each candidate TOC page, and the
bulletin's total PDF page count. TOC headings are followed by a printed page number — the
label shown in the body-page footer (e.g. "9", "27", "A-1"). Return those printed labels
VERBATIM; never convert them to PDF indices.

COUNT as L1: unindented, usually ALL-CAPS/bold headings naming a recurring body chapter —
e.g. FEDERAL FISCAL OPERATIONS, FEDERAL DEBT, PUBLIC DEBT OPERATIONS, CAPITAL MOVEMENTS,
FOREIGN CURRENCY POSITIONS, INTERNATIONAL FINANCIAL STATISTICS, TRUST FUNDS, GOVERNMENT
CORPORATIONS AND OTHER BUSINESS-TYPE ACTIVITIES, PROFILE OF THE ECONOMY. In 1940s-50s
bulletins the analogue may be roman-numeral prefixed ("I. PUBLIC DEBT AND GUARANTEED
OBLIGATIONS") or longer-phrased — still L1.

SKIP:
  - Indented sub-sections ("Budget Receipts and Expenditures", "Ownership of Federal
    Securities", "Treasury Survey of Ownership") — these are L2.
  - Table/figure titles — anything starting "Table N." or "Figure N." is not a chapter.
  - Front matter that isn't a body chapter: "Cover", "Contents", "Treasury staff",
    "Subscription information", "Glossary".
  - Single specific reports/articles ("The Role of Saving in a Dynamic U.S. Economy") —
    emit only their parent chapter.

Output a SINGLE JSON object, no prose or fences:
  {"sections": [{"section": "<verbatim heading>",
                 "start_page_printed": "<verbatim printed page>",
                 "end_page_printed": "<verbatim printed page>"}]}

  - `section` and page labels are VERBATIM from the TOC (preserve casing, hyphens, ampersands,
    hyphenation like "A-1"/"F-12", leading zeros).
  - Sort by body order. `end_page_printed` of chapter i is the printed page just before
    chapter i+1's start (for the last chapter, the last printed page in the TOC).
  - No TOC at all → {"sections": []}.

A typical bulletin yields 5-12 chapters. Emitting >20 means you're including sub-sections —
recheck and drop the L2 entries.
"""


