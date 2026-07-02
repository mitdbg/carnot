"""System #2: the SearchAgent.

Wraps skunk's `SearchAgent` (grep / vector-search / read-document / prune) in the
qatfd `retrieve()`/`compute()` shape. Two agent modes:
  - "retrieve": the agent returns relevant doc_ids; a downstream LLM answers (compute()).
  - "answer":  the agent answers the question directly (different prompt); compute()
    passes that answer through.

`_build_agent` constructs skunk's `SearchAgent` directly, using its hooks to (a) pass
this question's `LLMClient` (which owns embedding backend dispatch + usage accounting),
(b) add extra tools (system #3), and (c) override the briefing / final-answer prompt per mode.
"""

from __future__ import annotations

from skunk.common import ExecutionContext
from skunk.config import SearchAgentConfig
from skunk.search_agent.search_agent import SearchAgent

from qatfd.benchmarks.base import BenchmarkResources
from qatfd.systems.base import RetrieveComputeSystem
from qatfd.types import Question, Retrieved

# ---- direct-answer ("answer" mode) prompt overrides --------------------------

_ANSWER_BRIEFING = (
    "You are a research assistant answering a question using a large collection of "
    "documents. The corpus is organized as documents (each identified by a `doc_id`), "
    "each split into chunks (text spans, tables, titles, ...) identified by a `chunk_id`; "
    "every chunk carries metadata you can filter on. Use the tools to search, grep, and "
    "read documents until you can answer the question. Use `prune(...)` aggressively on "
    "chunks/docs you have ruled out, to keep your context focused. When you are confident, "
    "produce your final answer."
)
_ANSWER_FINAL_DOC = """\
A JSON object with your final answer to the question under "answer", and the
`doc_id`s you relied on under "doc_ids":
```json
{"answer": "<your final answer>", "doc_ids": ["2002_06_1", "2002_12_26"]}
```
Use each `doc_id` exactly as it appears in the search / grep results."""


class SearchAgentSystem(RetrieveComputeSystem):
    name = "search_agent"

    def __init__(self, config: SearchAgentConfig) -> None:
        self.config = config

    # Hooks for subclasses ------------------------------------------------------

    def _extra_tools(self, ctx: ExecutionContext, resources: BenchmarkResources) -> tuple:
        return ()

    def _include_search_corpus(self) -> bool:
        """Whether the agent gets the `search_corpus` (vector-search) tool. Subclasses can
        drop it to force the agent onto other retrieval tools."""
        return True

    def _prompts(self) -> tuple[str | None, str | None]:
        """(briefing, final_answer_doc) overrides; None => skunk SearchAgent defaults."""
        if self.config.agent_mode == "answer":
            return _ANSWER_BRIEFING, _ANSWER_FINAL_DOC
        return None, None

    def _build_agent(self, ctx: ExecutionContext, resources: BenchmarkResources) -> SearchAgent:
        briefing, final_answer_doc = self._prompts()
        return SearchAgent(
            config=self.config,
            document_map=resources.document_map,
            chroma_collection=resources.chroma_collection,
            # Reuse this question's LLMClient so the agent's search-tool embeddings are
            # billed onto the same usage tracker as its LLM calls; backend dispatch
            # (openrouter / local) is read from config.emb_provider by the client.
            llm_client=ctx.llm_client,
            emb_model_id=self.config.emb_model_id,
            extra_tools=self._extra_tools(ctx, resources),
            include_search_corpus=self._include_search_corpus(),
            briefing=briefing,
            final_answer_doc=final_answer_doc,
            pdf_dir=resources.pdf_dir,
        )

    # Pipeline ------------------------------------------------------------------

    async def retrieve(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext) -> Retrieved:
        agent = self._build_agent(ctx, resources)
        payload = await agent.call(ctx, q.text)
        doc_ids = _coerce_doc_ids(payload)
        if self.config.agent_mode == "answer":
            answer = payload.get("answer") if isinstance(payload, dict) else None
            return Retrieved(doc_ids=doc_ids, direct_answer=None if answer is None else str(answer))
        return Retrieved(doc_ids=doc_ids)


def _coerce_doc_ids(payload) -> list[str]:
    if not isinstance(payload, dict):
        return []
    keys = payload.get("doc_ids") or []
    if isinstance(keys, str):
        return [keys]
    return [str(k) for k in keys]
