"""Native TOC harvest — one Gemini call per bulletin.

Input: concatenated text of the bulletin's TOC-candidate pages.
Output: list of (section_label, start_page_printed, end_page_printed)
spans covering the bulletin's printed-page range.

The TOC lists section headings against the bulletin's *printed* page
numbers (the footer text like "9"). Each row of `PageCatalogRow` also
carries its `printed_page` field extracted from the page's parsed-JSON
`page_number` element. We match them directly — no PDF-page offset
inference needed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from skunk.common import LLMClient


@dataclass(frozen=True)
class SectionSpan:
    section: str               # verbatim section label as printed in the ToC
    start_page_printed: str    # printed page label where the section begins
    end_page_printed: str      # printed page label where the section ends (inclusive)


_TOC_SYSTEM = """You map a U.S. Treasury Bulletin's Table of Contents into a list of
TOP-LEVEL chapter spans. Return ONLY chapter-level (L1) headings — never
sub-sections.

You will receive:
  - The bulletin's publication month (YYYY-MM)
  - Each candidate TOC page, with the raw text of that TOC page.
  - The total number of PDF pages in the bulletin.

The bulletin's TOC pages list section headings at multiple nesting depths,
each followed by a printed page number — the same number you would see
in the page footer of the body pages (e.g. "9", "27", "A-1"). Return those
PRINTED page labels verbatim. Do NOT translate them to PDF page indices.

What COUNTS as a top-level chapter heading:
  - Unindented, typically ALL-CAPS or bolded headings that name a recurring
    body chapter. Treasury bulletins consistently use names like:
      FEDERAL FISCAL OPERATIONS
      FEDERAL DEBT
      PUBLIC DEBT OPERATIONS
      CAPITAL MOVEMENTS
      FOREIGN CURRENCY POSITIONS
      INTERNATIONAL FINANCIAL STATISTICS
      TRUST FUNDS
      GOVERNMENT CORPORATIONS AND OTHER BUSINESS-TYPE ACTIVITIES
      PROFILE OF THE ECONOMY
    In older bulletins (1940s-50s) the analogous chapter may be
    roman-numeral prefixed ("I. PUBLIC DEBT AND GUARANTEED OBLIGATIONS")
    or use longer phrasing — still count as L1.

What you MUST SKIP:
  - Indented sub-sections (e.g. "Budget Receipts and Expenditures",
    "Ownership of Federal Securities", "Treasury Survey of Ownership") —
    these are L2 under their parent chapter.
  - Table or figure titles (e.g. "Table FFO-2. — Budget Receipts by
    Principal Sources", "Table PDO-5. — Unmatured Marketable
    Securities"). Anything starting with "Table N." or "Figure N." is NOT
    a chapter.
  - Front-matter entries that aren't body chapters: "Cover", "Contents",
    "Treasury staff", "Subscription information", "Glossary".
  - Any heading describing a single specific report or article (e.g. "The
    Role of Saving in a Dynamic U.S. Economy") — those live under
    PROFILE OF THE ECONOMY or similar; emit only the parent chapter.

Output a SINGLE JSON object with this shape (no prose, no markdown fences):

{
  "sections": [
    {"section": "<verbatim chapter heading>",
     "start_page_printed": "<verbatim printed page>",
     "end_page_printed":   "<verbatim printed page>"}
  ]
}

Rules:
  - Section labels are VERBATIM as printed in the TOC (preserve casing,
    hyphens, ampersands).
  - Page labels are also VERBATIM as printed in the TOC (preserve
    hyphenation like "A-1", "F-12"; preserve leading zeros if present).
  - Sort the array by the order the chapters appear in the bulletin body.
  - end_page_printed for chapter i should be the printed page immediately
    before the start of chapter i+1 (or, for the last chapter, the last
    printed page listed in the TOC).
  - If the bulletin appears to have no TOC at all, return {"sections": []}.

A typical bulletin yields 5–12 chapters. If you find yourself emitting
more than 20, you are almost certainly including sub-sections — recheck
and drop the L2 entries.
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
    as TOC-like by the heuristic in `classify.looks_like_toc`. The PDF
    page index is shown to the LLM only for provenance; the LLM emits
    printed page labels in its response.
    """
    if not toc_candidate_pages:
        return []

    user_parts: list[str] = [f"Bulletin month: {bulletin_month}",
                             f"Total PDF pages: {total_pdf_pages}",
                             ""]
    for pdf_idx, text in toc_candidate_pages:
        user_parts.append(f"--- TOC page (PDF page {pdf_idx}) ---")
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
            start = str(raw["start_page_printed"]).strip()
            end = str(raw["end_page_printed"]).strip()
        except (KeyError, ValueError, TypeError):
            continue
        if not label or not start or not end:
            continue
        spans.append(SectionSpan(section=label,
                                 start_page_printed=start,
                                 end_page_printed=end))

    # Sort by numeric printed page when possible — handles plain integers
    # which is the dominant case in this corpus. Non-numeric labels fall
    # back to lexicographic order.
    def _sort_key(sp: SectionSpan) -> tuple[int, str]:
        n = _printed_to_int(sp.start_page_printed)
        return (0 if n is None else 1, sp.start_page_printed if n is None else "") \
            if n is None else (1, f"{n:08d}")

    spans.sort(key=_sort_key)
    return spans


_DIGIT_RE = re.compile(r"^\s*0*(\d+)\s*$")


def _printed_to_int(s: str | None) -> int | None:
    """Parse a printed page label to an int when it's a plain integer
    (optionally with leading zeros). Returns None for hyphenated labels
    like 'A-1' or any non-integer string."""
    if not s:
        return None
    m = _DIGIT_RE.match(s)
    return int(m.group(1)) if m else None


def section_for_printed_page(
    spans: list[SectionSpan], printed_page: str | None,
) -> str | None:
    """Look up the section label covering `printed_page`. Returns None
    when the page falls outside every span, when `printed_page` is None,
    or when neither side can be parsed numerically (no fallback)."""
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
