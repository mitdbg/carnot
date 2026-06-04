"""Stage 1: Catalog builder — parse a bulletin's pages into PageCatalogRow.

Deterministic per-page extraction: no LLM, no I/O. The driver reads the
parsed-JSON elements off disk and hands them to the builder one
bulletin at a time.
"""

from __future__ import annotations

from typing import Protocol

from ..schema import PageCatalogRow


class CatalogBuilder(Protocol):
    """Parse one bulletin's pages into PageCatalogRow list.

    `pages` is `{1-based PDF page index: list of parsed-JSON element dicts}`
    as produced by `skunk.corpus.page_elements`. The implementation is
    expected to be deterministic — same input, same output.
    """

    def parse_bulletin(
        self,
        bulletin: str,
        pages: dict[int, list[dict]],
    ) -> list[PageCatalogRow]: ...
