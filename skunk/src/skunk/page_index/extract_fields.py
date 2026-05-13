"""Per-page LLM field extractor — one Gemini call per indexable PDF page.

The call returns a JSON object that fills in the optional fields on a
`PageCatalogRow`: page_kind, table_title, headers, keywords, periods_covered,
granularity. Pages with `page_kind=text` short-circuit (no extracted fields)
to keep narrative pages cheap to index.

Verbatim-phrase rule: `table_title`, `column_headers`, `row_headers_sample`
and `keywords` MUST be lifted character-for-character from the page text.
We never ask the model to summarize or paraphrase — paraphrase reintroduces
the embedding-smear problem this whole index is designed to avoid.
"""

from __future__ import annotations

import json
import re
from typing import Any

from skunk.common import LLMClient

from .schema import Granularity, PageKind, PeriodSpec


_VALID_GRANULARITY = {"monthly", "quarterly", "annual", "point", "mixed", "unknown"}
_VALID_PAGE_KIND = {"table", "text", "chart"}
_VALID_PERIOD_KIND = {"CY", "FY", "Q", "month", "day", "year", "range"}

_PAGE_SYSTEM = """You extract structured retrieval-index fields from ONE page of a
U.S. Treasury Bulletin (1939–2025). The output is consumed by a retrieval system,
not a summarization system — your job is to lift VERBATIM phrases from the page
text and to classify the periods the page reports on.

You will receive: the bulletin's publication month (YYYY-MM), the 1-based PDF
page index, and the parsed text of that page. The text comes from a layout
parser and is broken into blocks, each prefixed with a single-token tag in
square brackets indicating the block type:
  [title]          — page-level title.
  [section_header] — table heading or sub-section heading. Usually contains
                     the table number ("Table 1.-") and the period it covers
                     ("December 31, 1949"). These are the highest-value source
                     for `table_title` and for date extraction.
  [text]           — paragraph or footnote prose.
  [footnote]       — footnote prose.
  [page_header]    — repeating section banner ("DEBT OUTSTANDING").
  [table]          — full HTML table preserved verbatim. Read column/row
                     headers and any caption rows directly from the HTML.
The [type] tags are metadata — never lift them into table_title, headers,
or keywords; lift only the content after the tag.

Step 1. Classify `page_kind`:
  - "table"  — the page contains one or more data tables (rows × columns of numbers).
  - "chart"  — the page is a chart, figure, plot, or scanned image with axis labels
               but little tabular data.
  - "text"   — the page is narrative prose (introductions, footnotes, glossaries),
               a divider page, a section title page, or otherwise has no tabular data
               useful for retrieval. (For "text" pages, output ONLY page_kind and STOP.)

Step 2. (only for page_kind in {"table","chart"}) Lift VERBATIM:
  - `table_title`        — main heading of the (largest) table on the page.
                           Concatenate multi-line titles with ' — '. NEVER paraphrase.
  - `column_headers`     — the column header strings on the (largest) table, in order.
  - `row_headers_sample` — first ~8 row labels on the (largest) table, in order.
  - `keywords`           — 5–15 short noun phrases lifted verbatim from titles, headers,
                           and captions that uniquely identify what this page reports
                           on. Include line-item names ("National defense", "Public debt
                           retirement"), series names, basis notes ("daily Treasury
                           statements"), unit notes ("In millions of dollars"). Lift
                           AS-IS — do NOT normalize casing, hyphens, or word order.

Step 3. (only for page_kind in {"table","chart"}) Classify `periods_covered`.

Emit ONE entry per distinct named period the page reports on. Use the smallest grain
that the page actually labels. Two patterns to capture:

  (a) The OVERALL span of the page's data, as ONE entry with kind="range", with
      `start` = earliest date and `end` = latest date. Example: a page with a
      monthly series from Jan-1932 to Mar-1939 produces
        {"kind":"range","start":"1932-01-01","end":"1939-03-31","raw":"1932 through Mar 1939"}.

  (b) Any EXPLICITLY NAMED Calendar Year, Fiscal Year, Quarter, or single-date total
      that has its own row/column on the page (e.g. a "Calendar Year 1940 total" row).
      Emit each as its own entry:
        - CY:   {"kind":"CY","start":"YYYY-01-01","end":"YYYY-12-31","raw":"<verbatim>"}
        - FY:   {"kind":"FY","start":"(YYYY-1)-07-01","end":"YYYY-06-30","raw":"<verbatim>"}
                (US federal FY ends June 30 through 1976; FY1977 onward ends Sep 30.
                 If unclear from the page, use the post-1976 convention.)
        - Q1:   {"kind":"Q","start":"YYYY-01-01","end":"YYYY-03-31","raw":"<verbatim>"}
                (Q2: Apr-Jun; Q3: Jul-Sep; Q4: Oct-Dec.)
        - month:{"kind":"month","start":"YYYY-MM-01","end":"YYYY-MM-<lastday>","raw":"<verbatim>"}
        - day:  {"kind":"day","start":"YYYY-MM-DD","end":"YYYY-MM-DD","raw":"<verbatim>"}
        - year: {"kind":"year","start":"YYYY-01-01","end":"YYYY-12-31","raw":"<verbatim>"}

Do NOT enumerate every individual data point as a period — if the page has 50 monthly
columns from 1932-Jan to 1936-Feb, emit ONE range entry covering the span. Reserve
explicit named-period entries for prominently-labeled totals.

Step 4. Classify `granularity`: "monthly" | "quarterly" | "annual" | "point" | "mixed".

Step 5. Output a SINGLE JSON object (no prose, no markdown fences):

For page_kind="text":
  {"page_kind": "text"}

For page_kind in {"table","chart"}:
  {
    "page_kind": "table",
    "table_title": "...",
    "column_headers": [...],
    "row_headers_sample": [...],
    "keywords": [...],
    "periods_covered": [
      {"kind":"range","start":"1932-01-01","end":"1939-03-31","raw":"..."},
      {"kind":"CY","start":"1940-01-01","end":"1940-12-31","raw":"Calendar Year 1940"}
    ],
    "granularity": "monthly"
  }
"""


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


def _coerce_period(p: dict[str, Any]) -> PeriodSpec | None:
    """Validate and normalize a period dict. Returns None if invalid."""
    try:
        kind = str(p["kind"])
        start = str(p["start"])
        end = str(p["end"])
    except (KeyError, TypeError):
        return None
    if kind not in _VALID_PERIOD_KIND:
        return None
    if not (re.fullmatch(r"\d{4}-\d{2}-\d{2}", start)
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}", end)):
        return None
    if end < start:
        return None
    return PeriodSpec(kind=kind, start=start, end=end, raw=str(p.get("raw", "")))


def extract_page_fields(
    bulletin_month: str,
    pdf_page: int,
    page_text: str,
    llm: LLMClient,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run one Gemini call; return a dict of the extracted fields.

    Always returns a dict with at least `page_kind`. Caller is responsible for
    merging these fields onto its `PageCatalogRow`.

    On any parse failure we fall back to {"page_kind": "text"} rather than
    raising — the row still lands in the index, just without structured fields.
    """
    user = (
        f"Bulletin month: {bulletin_month}\n"
        f"PDF page index (1-based): {pdf_page}\n\n"
        f"--- PAGE TEXT ---\n{page_text.strip()}\n--- END PAGE TEXT ---\n"
    )
    resp = llm.call(system=_PAGE_SYSTEM, user=user, temperature=temperature)
    try:
        obj = json.loads(_strip_code_fence(resp.text))
    except json.JSONDecodeError:
        return {"page_kind": "text"}

    page_kind = obj.get("page_kind")
    if page_kind not in _VALID_PAGE_KIND:
        return {"page_kind": "text"}

    if page_kind == "text":
        return {"page_kind": "text"}

    granularity: Granularity = obj.get("granularity", "unknown")
    if granularity not in _VALID_GRANULARITY:
        granularity = "unknown"

    periods: list[PeriodSpec] = []
    for p in obj.get("periods_covered", []) or []:
        spec = _coerce_period(p)
        if spec is not None:
            periods.append(spec)

    def _str_list(key: str, cap: int) -> list[str]:
        v = obj.get(key, []) or []
        if not isinstance(v, list):
            return []
        out = [str(x).strip() for x in v if str(x).strip()]
        return out[:cap]

    return {
        "page_kind": page_kind,
        "table_title": (str(obj.get("table_title") or "").strip() or None),
        "column_headers": _str_list("column_headers", 50),
        "row_headers_sample": _str_list("row_headers_sample", 12),
        "keywords": _str_list("keywords", 20),
        "periods_covered": periods,
        "granularity": granularity,
    }
