"""Stage 3: Page placer — assign each content page to a chapter.

Given a bulletin's catalog rows and its L1 spans (from `L1Harvester`),
mutate each `PageCatalogRow.l1_local` in place to point at one of the
bulletin's chapter names. Pages that can't be confidently placed land
in the `UNFILED` bucket.
"""

from __future__ import annotations

from typing import Protocol

from skunk.llm_client import LLMClient

from ..schema import PageCatalogRow
from .l1_harvest import SectionSpan


UNFILED = "_Unfiled"


class PagePlacer(Protocol):
    """Place every content page in `rows` under one of `spans`' chapters.

    Mutates `rows` in place (sets `l1_local`). Returns a per-method
    counter of placements (for visibility into which fallback each page
    used). Caller persists `rows` after this returns.
    """

    def place_bulletin(
        self,
        rows: list[PageCatalogRow],
        spans: list[SectionSpan],
        llm: LLMClient | None,
    ) -> dict[str, int]: ...
