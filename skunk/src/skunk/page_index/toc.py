"""Native TOC harvest — one Gemini call per bulletin.

Input: concatenated text of the bulletin's TOC-candidate pages.
Output: list of (section_label, start_page, end_page) covering the bulletin.

Page numbers in the TOC are typically *printed* page numbers (a different
convention from `PageRef.page` = 1-based PDF page). We pass the LLM both
flavours so it can disambiguate; we ask it to return 1-based PDF pages
that we can use to label `PageCatalogRow.section` directly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from skunk.common import LLMClient


@dataclass(frozen=True)
class SectionSpan:
    section: str        # verbatim section label as printed
    start_page: int     # 1-based PDF page index where the section begins
    end_page: int       # 1-based PDF page index where the section ends (inclusive)


_TOC_SYSTEM = """You map a U.S. Treasury Bulletin's Table of Contents into a list of section spans.

You will receive:
  - The bulletin's publication month (YYYY-MM)
  - Each candidate TOC page, with both its 1-based PDF page index and its raw text.
  - The total number of PDF pages in the bulletin.

The bulletin's TOC pages list section headings and a page number per heading.
These printed page numbers are NOT the same as the PDF page index — printed page 1
often appears around PDF page 11 or 13 due to front matter (covers, copies, TOC
itself). You must infer the printed-to-PDF offset by aligning a couple of section
headings against the actual text on those PDF pages.

Output a SINGLE JSON object with this shape (no prose, no markdown fences):

{
  "printed_to_pdf_offset": <int>,
  "sections": [
    {"section": "<verbatim heading>", "start_page_pdf": <int>, "end_page_pdf": <int>}
  ]
}

Rules:
  - Section labels are VERBATIM as printed in the TOC (preserve casing, hyphens).
  - The sections must cover the body of the bulletin without overlap; sort by start_page_pdf.
  - end_page_pdf for section i is start_page_pdf of section i+1 minus 1; the last
    section's end_page_pdf is the total number of PDF pages.
  - If the TOC lists only top-level sections (not sub-sections), return just those.
  - If the bulletin appears to have no TOC at all, return {"printed_to_pdf_offset": 0, "sections": []}.
"""


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


def harvest_toc(
    bulletin_month: str,
    toc_candidate_pages: list[tuple[int, str]],
    total_pdf_pages: int,
    llm: LLMClient,
) -> list[SectionSpan]:
    """One Gemini call → list of SectionSpan covering the bulletin body.

    `toc_candidate_pages` is [(pdf_page_index, text)] for each page flagged
    as TOC-like by the heuristic in `classify.looks_like_toc`. If empty,
    we still ask the LLM in case TOC detection missed (returns empty list).
    """
    if not toc_candidate_pages:
        return []

    user_parts: list[str] = [f"Bulletin month: {bulletin_month}",
                             f"Total PDF pages: {total_pdf_pages}",
                             ""]
    for pdf_idx, text in toc_candidate_pages:
        user_parts.append(f"--- PDF page {pdf_idx} ---")
        user_parts.append(text.strip())
        user_parts.append("")

    resp = llm.call(system=_TOC_SYSTEM, user="\n".join(user_parts), temperature=0.0)
    try:
        obj = json.loads(_strip_code_fence(resp.text))
    except json.JSONDecodeError:
        return []

    spans: list[SectionSpan] = []
    for raw in obj.get("sections", []):
        try:
            label = str(raw["section"]).strip()
            start = int(raw["start_page_pdf"])
            end = int(raw["end_page_pdf"])
        except (KeyError, ValueError, TypeError):
            continue
        if not label or start < 1 or end < start or end > total_pdf_pages:
            continue
        spans.append(SectionSpan(section=label, start_page=start, end_page=end))

    spans.sort(key=lambda s: s.start_page)
    return spans


def section_for_page(spans: list[SectionSpan], pdf_page: int) -> str | None:
    """Look up the section label covering `pdf_page`, or None if outside any span."""
    for sp in spans:
        if sp.start_page <= pdf_page <= sp.end_page:
            return sp.section
    return None
