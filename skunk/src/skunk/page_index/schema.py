"""Catalog schema — what one indexed page looks like on disk.

Dataclasses stay consistent with the rest of the codebase (no pydantic).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

PageKind = Literal["table", "text", "chart", "toc", "blank"]
Granularity = Literal["monthly", "quarterly", "annual", "point", "mixed", "unknown"]


@dataclass
class PeriodSpec:
    """One period that a page reports on, normalized to (start, end) ISO dates.

    `kind` retains the original form for human inspection and tiebreaker rules
    (e.g. prefer pages whose period kind matches the query's period kind).
    """
    kind: Literal["CY", "FY", "Q", "month", "day", "year", "range"]
    start: str   # ISO date "YYYY-MM-DD" (first day of the span)
    end: str     # ISO date "YYYY-MM-DD" (last day of the span, inclusive)
    raw: str = ""  # original string as written by the page (e.g. "Calendar Year 1940")

    def overlaps(self, other_start: str, other_end: str) -> bool:
        return not (self.end < other_start or self.start > other_end)


@dataclass
class PageCatalogRow:
    bulletin: str                                   # "YYYY-MM"
    page: int                                       # 1-based PDF page index
    file_path: str                                  # absolute path to the PDF
    page_kind: PageKind = "blank"

    # Populated only for page_kind ∈ {table, chart}:
    section: str | None = None                      # from native TOC harvest
    table_title: str | None = None                  # verbatim
    column_headers: list[str] = field(default_factory=list)
    row_headers_sample: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)  # verbatim noun phrases
    periods_covered: list[PeriodSpec] = field(default_factory=list)
    granularity: Granularity = "unknown"
    is_retrospective: bool = False                  # max(periods.end) < bulletin month

    # Filled in by Stage 4 clustering pass (later); None until then.
    keyword_cluster_id: str | None = None

    # Diagnostic — useful for debugging extraction quality without re-reading the PDF.
    char_count: int = 0
    digit_ratio: float = 0.0

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False, sort_keys=False)

    @classmethod
    def from_json(cls, line: str) -> "PageCatalogRow":
        d: dict[str, Any] = json.loads(line)
        periods = [PeriodSpec(**p) for p in d.pop("periods_covered", [])]
        return cls(periods_covered=periods, **d)
