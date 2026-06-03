"""SearchAgent — iterative code-execution retriever.

Inherits the multi-turn loop from `MultiTurnAgent`, and follows its
content-fragment contract: this class supplies `name`, `briefing`,
`final_answer_doc`, and the chroma/page-map-backed tool set; the base owns
`_SYSTEM_TEMPLATE` and assembles the `PromptedCall` (splicing `{{ tools_doc }}`,
wiring the Jinja vars). `{{ max_steps }}` is the base's universal var;
`{{ max_pages }}` (referenced inside a tool `doc`) comes from this class's `_prompt_vars`.
Corpus / few-shot / lessons overrides addressed to `search_agent` (e.g. the
dataset blurb rendered under `## Dataset`) are appended by the `PromptedCall`
— same mechanism as every other call-site. Exposes the `Retriever` entry point.

This instance is safe to reuse concurrently across questions: `MultiTurnAgent.call()`
keeps all per-question state in locals (the deps captured here — config, client,
chroma collection, page map — are read-only), so a single `SearchAgent` can serve
many questions in parallel.
"""

from __future__ import annotations

from chromadb.api.models.Collection import Collection

from skunk.common import _make_genai_client
from skunk.config import SkunkConfig
from skunk.common import ExecutionContext
from skunk.multi_turn_agent import MultiTurnAgent
from skunk.search_agent.base import Retriever
from skunk.search_agent.search_tools import (
    RetrievePageInfoTool, RunGrepTool, VectorSearchTool,
)


class SearchAgent(MultiTurnAgent, Retriever):
    name = "search_agent"
    # Larger than MultiTurnAgent default — search chains have many
    # page-content observations and a 20-step ceiling.
    context_budget_chars: int = 500_000 * 4  # ~500K tokens × 4 char/token
    warn_steps_remaining = 2

    briefing = (
        "You are a helpful assistant for retrieving relevant information from a large "
        "collection of files. You will be given a question, and your task is to identify "
        "which files are relevant to answering the question. Some questions require "
        "looking up information that is not contained in the files; this will be handled "
        "by a separate agent, so you should only focus on retrieving relevant files for "
        "the remainder of the question. "
        "You do not need to retrieve everything in a single tool call: use early steps to "
        "explore files of potential relevance, then refine your searches in later steps "
        "based on what you find."
    )

    final_answer_doc = """\
A JSON object with the page keys you identified as relevant, in the format
"year_month_page_id". Use the page_id that corresponds to the index of the page in
the bulletin, not the page number printed on the page itself.
```json
{"page_keys": ["2002_06_1", "2002_12_26"]}
```"""

    def __init__(self, config: SkunkConfig, clean_page_map: dict, chroma_collection: Collection):
        self.config = config
        self.client = _make_genai_client()
        self.chroma_collection = chroma_collection
        self.clean_page_map = clean_page_map
        self.emb_model_id = config.emb_model_id.removeprefix("google/")
        # Tool instances capture their deps; the prompt's `## Tools` section is
        # generated from their `doc`s by the base, so tools and docs can't drift.
        tools = [
            RetrievePageInfoTool(self.clean_page_map, config.agent_max_pages_per_tool_call),
            VectorSearchTool(self.chroma_collection, self.emb_model_id, self.client),
            RunGrepTool(),
        ]
        super().__init__(tools, max_steps=config.agent_max_steps)

    async def retrieve(
        self,
        ctx: ExecutionContext,
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
        payload = await self.call(ctx, "\n".join(parts))
        keys = payload.get("page_keys") or []
        if isinstance(keys, str):
            return [keys]
        return [str(k) for k in keys]
