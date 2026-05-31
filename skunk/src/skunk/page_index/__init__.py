"""Page-index: an offline-built catalog + the retriever that queries it.

The package separates cleanly into two halves plus a shared base:

  Build pipeline (offline) — `pipeline.py`, `stages/`, `corpora/`.
    Produces a per-page catalog + flat concept tree over a configurable
    corpus. Output lives under `cache/page_index/` (`catalog/{bulletin}.jsonl`
    + `concept_tree.json`).

  Query path (per-query) — `query.py` (`PageIndexRetriever`, the entry
    point), `query_toc.py` (ToC chapter pick), `query_semfilter.py`
    (two-stage parallel semantic filter). The pipeline is:
    ToC pick → year filter → semantic filter → candidate set.

  Shared — `schema.py` (catalog row, written by build / read by query),
    `pdf.py` (parsed-JSON page reader), `util.py`, `profile.py`.

Runtime callers fetch their profile via `default_profile()`; the active
profile is selected by `SKUNK_CORPUS_PROFILE` (default: treasury).
"""

from __future__ import annotations

import os

from .corpora import load_profile
from .profile import CorpusProfile, StageError


def default_profile() -> CorpusProfile:
    """Load the runtime corpus profile.

    Env override: `SKUNK_CORPUS_PROFILE` (default: `"treasury"`).
    """
    return load_profile(os.environ.get("SKUNK_CORPUS_PROFILE", "treasury"))


__all__ = ["CorpusProfile", "StageError", "default_profile"]
