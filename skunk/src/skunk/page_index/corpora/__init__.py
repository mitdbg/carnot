"""Registry of available corpus profiles.

Add a new corpus by:
  1. Creating a sibling package (e.g. `corpora/fedreserve/`) with concrete
     implementations of each stage Protocol from `..stages`.
  2. Exposing a `<name>_profile()` factory returning a `CorpusProfile`.
  3. Registering it in `PROFILES` below.
"""

from __future__ import annotations

from collections.abc import Callable

from ..profile import CorpusProfile
from .treasury import treasury_profile


PROFILES: dict[str, Callable[[], CorpusProfile]] = {
    "treasury": treasury_profile,
}


def load_profile(name: str) -> CorpusProfile:
    """Look up a profile factory by name and call it."""
    try:
        factory = PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(PROFILES)) or "(none)"
        raise ValueError(
            f"unknown corpus profile: {name!r}. Known profiles: {known}"
        ) from None
    return factory()
