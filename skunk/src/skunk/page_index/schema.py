"""Catalog schema — what one indexed page looks like on disk.

Dataclasses stay consistent with the rest of the codebase (no pydantic).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal


@dataclass
class ContentBlock:
    """One detected content block on a page: a table, a chart, or the
    page's prose. A page may carry multiple blocks (a table at the top
    plus a chart at the bottom is common in the bulletin)."""
    kind: Literal["table", "chart", "prose"]
    title: str | None = None
    # Tables only — empty for chart/prose.
    column_headers: list[str] = field(default_factory=list)
    row_headers_sample: list[str] = field(default_factory=list)


@dataclass
class PageCatalogRow:
    """One catalog row per PDF page. A page is retrievable iff
    `content_blocks` is non-empty; pure-text / blank / ToC pages have
    no blocks and only carry diagnostic fields (char_count, keywords)."""
    bulletin: str                                   # "YYYY-MM"
    page: int                                       # 1-based PDF page index

    # Phase 2 placement (set on rows with content_blocks).
    l1_local: str | None = None
    printed_page: str | None = None                 # page-footer label (e.g. "9", "A-1")

    # One entry per detected table / chart / prose block on the page.
    content_blocks: list[ContentBlock] = field(default_factory=list)

    # Page-level signals.
    keywords: list[str] = field(default_factory=list)
    # Verbatim date strings detected on the page ("December 31, 1949",
    # "Fiscal Year 1991", "1932-1939", "1940"). Retrieve parses each
    # back into ISO intervals and overlap-checks against the query
    # period — see TreasuryPeriodParser.verbatim_date_to_intervals.
    dates: list[str] = field(default_factory=list)
    char_count: int = 0
    digit_ratio: float = 0.0

    # Internal-only — populated by build, consumed by extract_l1 + place_pages,
    # stripped by _persist_catalog_by_bulletin at the end of Phase 2 so it
    # doesn't leak into the final on-disk catalog.
    banner_self: str | None = None

    @property
    def primary_title(self) -> str | None:
        """First non-empty title across content_blocks, or None."""
        return next((b.title for b in self.content_blocks if b.title), None)

    def all_column_headers(self) -> list[str]:
        out: list[str] = []
        for b in self.content_blocks:
            out.extend(b.column_headers)
        return out

    def to_json(self, *, drop_banner_self: bool = False) -> str:
        """Serialize to one JSON line. `drop_banner_self=True` is used by
        the Phase-2 final persist to strip the internal-only banner from
        the on-disk catalog; intermediate persists (Phase 0) keep it so
        downstream stages can read it back when re-running."""
        d = asdict(self)
        if drop_banner_self:
            d.pop("banner_self", None)
        return json.dumps(d, ensure_ascii=False, sort_keys=False)

    @classmethod
    def from_json(cls, line: str) -> "PageCatalogRow":
        d: dict[str, Any] = json.loads(line)
        valid = {f.name for f in fields(cls)}
        blocks_raw = d.get("content_blocks") or []
        d = {k: v for k, v in d.items() if k in valid and k != "content_blocks"}
        row = cls(**d)
        for b in blocks_raw:
            if not isinstance(b, dict):
                continue
            row.content_blocks.append(ContentBlock(
                kind=b.get("kind", "prose"),
                title=b.get("title"),
                column_headers=list(b.get("column_headers") or []),
                row_headers_sample=list(b.get("row_headers_sample") or []),
            ))
        return row
