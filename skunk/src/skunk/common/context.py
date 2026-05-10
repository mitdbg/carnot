from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skunk.dsl import DocHandle


@dataclass
class HarnessContext:
    question: str
    manifest_path: str | None = None
    cache_dir: str = "cache"
    golden_handle: DocHandle | None = None
    cache_only: bool = False  # blocks live external API calls; use frozen cache for eval re-runs
    verbose: bool = False     # live-print orchestrator + subagent events to stdout
    events: list[dict] = field(default_factory=list)  # per-question diagnostic events

    def __post_init__(self) -> None:
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def emit(self, source: str, message: str, **fields: Any) -> None:
        """Record a diagnostic event. Subagents call this with their op name as `source`."""
        evt = {"source": source, "message": message, **fields}
        self.events.append(evt)
        if self.verbose:
            extra = ""
            if fields:
                bits = []
                for k, v in fields.items():
                    s = repr(v)
                    if len(s) > 200:
                        s = s[:200] + "..."
                    bits.append(f"{k}={s}")
                extra = " | " + ", ".join(bits)
            print(f"  [{source}] {message}{extra}")
