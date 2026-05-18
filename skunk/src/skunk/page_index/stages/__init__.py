"""Abstract stage interfaces for the page-index build pipeline.

A `CorpusProfile` (see `..profile`) supplies one concrete implementation
of each Protocol here. The pipeline driver in `..pipeline` composes
them and stays corpus-agnostic.

Stage order: catalog → l1_harvest → placer → merger.
"""

from .catalog import CatalogBuilder
from .l1_harvest import L1Harvester, SectionSpan, section_for_printed_page
from .merger import ChapterMerger
from .period import PeriodParser
from .placer import UNFILED, PagePlacer

__all__ = [
    "CatalogBuilder",
    "L1Harvester",
    "SectionSpan",
    "section_for_printed_page",
    "ChapterMerger",
    "PeriodParser",
    "PagePlacer",
    "UNFILED",
]
