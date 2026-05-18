"""Page-index build and load utilities.

Offline pipeline produces a per-page catalog over a configurable corpus
(see `corpora/` for available profiles). Output lives at
`cache/page_index/{bulletin}.jsonl`, one row per page.

Runtime callers should fetch their profile via `default_profile()`; the
active profile is selected by `SKUNK_CORPUS_PROFILE` (default: treasury).
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
