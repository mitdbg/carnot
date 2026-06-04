"""Treasury Bulletin corpus profile.

Concrete implementations of each stage Protocol live in sibling modules
here. `treasury_profile()` composes them into a `CorpusProfile`.
"""

from __future__ import annotations

import re

from ...profile import CorpusProfile


_TREASURY_FILENAME_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})\.pdf$")
_TREASURY_EXPECTED_CHAPTER_COUNT = 8


def treasury_profile() -> CorpusProfile:
    """Build the Treasury Bulletin CorpusProfile.

    Heavy imports are deferred to this function so importing the
    `corpora` package doesn't drag every Treasury module in unless a
    profile is actually requested.
    """
    from .catalog import TreasuryCatalogBuilder
    from .l1_harvest import TreasuryL1Harvester
    from .merger import TreasuryChapterMerger
    from .period import TreasuryPeriodParser
    from .placer import TreasuryPagePlacer

    return CorpusProfile(
        name="treasury",
        catalog_builder=TreasuryCatalogBuilder(),
        l1_harvester=TreasuryL1Harvester(),
        page_placer=TreasuryPagePlacer(),
        chapter_merger=TreasuryChapterMerger(),
        period_parser=TreasuryPeriodParser(),
        bulletin_filename_re=_TREASURY_FILENAME_RE,
        expected_chapter_count=_TREASURY_EXPECTED_CHAPTER_COUNT,
    )
