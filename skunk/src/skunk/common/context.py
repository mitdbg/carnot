from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from skunk.dsl import DocHandle


@dataclass
class HarnessContext:
    question: str
    manifest_path: str | None = None
    cache_dir: str = "cache"
    golden_handle: DocHandle | None = None
    cache_only: bool = False  # blocks live external API calls; use frozen cache for eval re-runs

    def __post_init__(self) -> None:
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
