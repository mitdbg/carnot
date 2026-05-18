"""Stage 4: Chapter merger — cross-bulletin canonicalization.

After per-bulletin placement, every content page has an `l1_local`
naming its own bulletin's chapter. This stage merges those per-bulletin
labels into a flat global canonical chapter set and produces the
shipped `concept_tree.json`.
"""

from __future__ import annotations

from typing import Any, Protocol

from skunk.common import LLMClient

from ..schema import PageCatalogRow


class ChapterMerger(Protocol):
    """Merge per-bulletin chapter labels into a global canonical tree.

    Returns a dict matching the shipped `concept_tree.json` shape:

        {"chapters": {
            "<canonical>": {
                "n_pages": int,
                "description": str,
                "examples": list[str],
                "pages": [{"bulletin", "page", "key_phrases"}, ...]
            }
        }}
    """

    def build_tree(
        self,
        catalog: list[PageCatalogRow],
        llm: LLMClient,
        *,
        drop_unfiled: bool = True,
        verbose: bool = True,
    ) -> dict[str, Any]: ...
