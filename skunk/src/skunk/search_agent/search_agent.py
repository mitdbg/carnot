"""SearchAgent — iterative code-execution retriever.

Inherits the multi-turn loop from `MultiTurnAgent`; supplies the
chroma + page-map-backed tool set, delegates prompt assembly to
`SearchAgentPromptedCall` (so prompt-overrides routing stays in one
place), and exposes the `Retriever` protocol entry point.

TODO: `self._observations` (inherited) and other per-call state on
this reused-across-questions instance are not thread-safe; address
with `threading.local` or per-call agent instantiation next time
this is touched.
"""

from __future__ import annotations

from chromadb.api.models.Collection import Collection

from skunk.common import _make_genai_client
from skunk.config import SkunkConfig
from skunk.models import HarnessContext
from skunk.multi_turn_agent import MultiTurnAgent, final_answer
from skunk.search_agent.base import Retriever
from skunk.search_agent.prompted_call import SearchAgentPromptedCall
from skunk.search_agent.search_tools import (
    _make_retrieve_page_info, _make_vector_search, run_grep,
)

# Token-budget trim — SearchAgent-specific (lookup chains stay short).
_EFFECTIVE_CONTEXT_WINDOW_CHARS = 500_000 * 4  # ~500K tokens × 4 char/token


def _trim(messages: list[dict], budget: int) -> list[dict]:
    """Keep system + question + as many of the most-recent messages as
    fit in `budget` chars; drop the middle behind a placeholder."""
    if sum(len(m["content"]) for m in messages) <= budget:
        return messages
    head = [messages[0], messages[1],
            {"role": "user", "content": "...(earlier steps truncated)..."}]
    remaining = budget - sum(len(m["content"]) for m in head)
    tail: list[dict] = []
    for m in reversed(messages[2:]):
        if remaining - len(m["content"]) < 0:
            break
        tail.append(m)
        remaining -= len(m["content"])
    return head + tail[::-1]


class SearchAgent(MultiTurnAgent, Retriever):
    name: str = "search_agent"

    def __init__(self, config: SkunkConfig, clean_page_map: dict, chroma_collection: Collection):
        self.config = config
        self.client = _make_genai_client()
        self.chroma_collection = chroma_collection
        self.clean_page_map = clean_page_map
        self.emb_model_id = config.emb_model_id.removeprefix("google/")
        self.max_steps = config.agent_max_steps
        self._prompted_call = SearchAgentPromptedCall()

    # Delegate prompt + template vars to the existing PromptedCall sibling.
    def assemble_system_prompt(self, ctx: HarnessContext) -> str:
        return self._prompted_call.assemble_system_prompt(ctx)

    def template_vars(self, ctx: HarnessContext) -> dict:
        return self._prompted_call.template_vars(ctx)

    def tools(self) -> dict:
        return {
            "retrieve_page_info": _make_retrieve_page_info(self.clean_page_map),
            "vector_search": _make_vector_search(
                self.chroma_collection, self.emb_model_id, self.client,
            ),
            "run_grep": run_grep,
            "final_answer": final_answer,
        }

    def _generate(self, ctx: HarnessContext, messages: list[dict]) -> str:
        return super()._generate(ctx, _trim(messages, _EFFECTIVE_CONTEXT_WINDOW_CHARS))

    def retrieve(
        self,
        ctx: HarnessContext,
        question: str,
        *,
        branch_key: str | None = None,
        branch_period: str | None = None,
    ) -> list[str]:
        parts = [f"Question: {question}"]
        if branch_key:
            parts.append(f"Search focus: {branch_key}")
        if branch_period:
            parts.append(f"Time period: {branch_period}")
        payload = self.call(ctx, "\n".join(parts))
        keys = payload.get("page_keys") or []
        if isinstance(keys, str):
            return [keys]
        return [str(k) for k in keys]
