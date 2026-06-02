"""SearchAgent — iterative code-execution retriever.

Inherits the multi-turn loop from `MultiTurnAgent`; supplies the
chroma + page-map-backed tool set, gets its prompt from the
`PromptedCall` built by `make_search_agent_prompt()` (so prompt-overrides
routing stays in one place), and exposes the `Retriever` protocol entry point.

TODO: `self._observations` (inherited) and other per-call state on
this reused-across-questions instance are not thread-safe; address
with `threading.local` or per-call agent instantiation next time
this is touched.
"""

from __future__ import annotations

from chromadb.api.models.Collection import Collection

from skunk.common import _make_genai_client
from skunk.config import SkunkConfig
from skunk.common import HarnessContext
from skunk.multi_turn_agent import MultiTurnAgent
from skunk.search_agent.base import Retriever
from skunk.search_agent.prompted_call import make_search_agent_prompt
from skunk.search_agent.search_tools import (
    RetrievePageInfoTool, RunGrepTool, VectorSearchTool,
)


class SearchAgent(MultiTurnAgent, Retriever):
    # Larger than MultiTurnAgent default — search chains have many
    # page-content observations and a 20-step ceiling.
    context_budget_chars: int = 500_000 * 4  # ~500K tokens × 4 char/token

    def __init__(self, config: SkunkConfig, clean_page_map: dict, chroma_collection: Collection):
        self.config = config
        self.client = _make_genai_client()
        self.chroma_collection = chroma_collection
        self.clean_page_map = clean_page_map
        self.emb_model_id = config.emb_model_id.removeprefix("google/")
        # Tool instances capture their deps; the prompt's `## Tools` section is
        # generated from their `doc`s, so tools and docs can't drift.
        tools = [
            RetrievePageInfoTool(self.clean_page_map),
            VectorSearchTool(self.chroma_collection, self.emb_model_id, self.client),
            RunGrepTool(),
        ]
        super().__init__(make_search_agent_prompt(tools), tools)
        self.max_steps = config.agent_max_steps

    def retrieve(
        self,
        ctx: HarnessContext,
        question: str,
        *,
        branch_key: str | None = None,
        branch_period: str | None = None,
        branch_as_of: str | None = None,
    ) -> list[str]:
        parts = [f"Question: {question}"]
        if branch_key:
            parts.append(f"Search focus: {branch_key}")
        if branch_period:
            parts.append(f"Time period (of the data): {branch_period}")
        if branch_as_of:
            parts.append(f"Reported in / as of: {branch_as_of}")
        payload = self.call(ctx, "\n".join(parts))
        keys = payload.get("page_keys") or []
        if isinstance(keys, str):
            return [keys]
        return [str(k) for k in keys]
