"""Catalog schema — what one indexed page looks like on disk.

Dataclasses stay consistent with the rest of the codebase (no pydantic).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

PageKind = Literal["table", "text", "chart", "prose", "toc", "blank"]


@dataclass
class PageCatalogRow:
    bulletin: str                                   # "YYYY-MM"
    page: int                                       # 1-based PDF page index
    file_path: str                                  # absolute path to the PDF
    page_kind: PageKind = "blank"

    # Populated only for page_kind ∈ {table, chart}:
    section: str | None = None                      # ToC section label (None ⇒ Unsectioned)
    printed_page: str | None = None                 # page-footer label (e.g. "9", "A-1")
    table_title: str | None = None                  # verbatim caption/title
    column_headers: list[str] = field(default_factory=list)
    row_headers_sample: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)  # dateless concept phrases
    dates: list[str] = field(default_factory=list)     # verbatim date strings on the page

    # Diagnostic — useful for debugging extraction quality without re-reading the PDF.
    char_count: int = 0
    digit_ratio: float = 0.0

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False, sort_keys=False)

    @classmethod
    def from_json(cls, line: str) -> "PageCatalogRow":
        d: dict[str, Any] = json.loads(line)
        # Tolerate legacy fields written by v0.2/v0.3 catalogs.
        for legacy in ("periods_covered", "granularity", "is_retrospective",
                       "keyword_cluster_id", "page_summary"):
            d.pop(legacy, None)
        return cls(**d)
