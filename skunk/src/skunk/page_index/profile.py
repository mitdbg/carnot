"""CorpusProfile — bundle of stage implementations for one corpus.

Each profile supplies:
  - one CatalogBuilder
  - one L1Harvester
  - one PagePlacer
  - one ChapterMerger
  - one PeriodParser (runtime)
  - corpus-discovery knobs (filename regex, expected chapter count)

The pipeline driver loads a profile by name (`--profile treasury`) and
stays corpus-agnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .stages import (
    CatalogBuilder, ChapterMerger, L1Harvester, PagePlacer, PeriodParser,
)


class StageError(RuntimeError):
    """Raised by the pipeline driver when a stage's prerequisites are
    missing on disk (e.g. running `place_pages` before `extract_l1`).
    Carries a remediation hint in the message.
    """


@dataclass(frozen=True)
class CorpusProfile:
    """All corpus-specific behavior in one place.

    A new corpus = a new instance of this dataclass + concrete stage
    implementations. The pipeline driver consumes it through stage
    Protocols only.
    """
    name: str

    # Stages
    catalog_builder: CatalogBuilder
    l1_harvester: L1Harvester
    page_placer: PagePlacer
    chapter_merger: ChapterMerger
    period_parser: PeriodParser

    # Corpus-discovery knobs that don't fit a stage interface.
    bulletin_filename_re: re.Pattern[str]
    expected_chapter_count: int  # rough target for sanity-checking merger output
