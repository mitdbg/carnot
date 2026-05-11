from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from skunk.config import SkunkConfig

if TYPE_CHECKING:
    from skunk.common.llm import LLMClient


@dataclass
class HarnessContext:
    question: str
    verbose: bool = False     # live-print orchestrator + subagent events to stdout
    events: list[dict] = field(default_factory=list)  # per-question diagnostic events
    config: SkunkConfig = field(default_factory=SkunkConfig.from_env)
    llm_client: LLMClient | None = None  # inject a mock for tests; auto-created otherwise

    def __post_init__(self) -> None:
        if self.llm_client is None:
            from skunk.common.llm import LLMClient
            self.llm_client = LLMClient(self.config)

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
