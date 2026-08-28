"""System #2: the SearchAgent.

Wraps skunk's `SearchAgent` (grep / vector-search / read-document / prune) in the
qatfd `retrieve()`/`compute()` shape. Two agent modes:
  - "retrieve": the agent returns relevant doc_ids; a downstream LLM answers (compute()).
  - "answer":  the agent answers the question directly (different prompt); compute()
    passes that answer through.

`_build_retrieve_agent` constructs skunk's `SearchAgent` directly.
"""

from __future__ import annotations

import uuid

from jinja2 import Environment, StrictUndefined

from skunk.common import ExecutionContext
from skunk.config import SearchAgentConfig
from skunk.agents.search_agent.search_agent import SearchAgent
from skunk.search_state.working_set_registry import WorkingSetRegistry

from qatfd.benchmarks.base import BenchmarkResources
from qatfd.prompts import load_qatfd_prompts
from qatfd.systems.base import RetrieveComputeSystem
from qatfd.types import Question, Retrieved

_ENV = Environment(
    autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined
)
_PROMPTS = load_qatfd_prompts("search_agent")


class SearchAgentSystem(RetrieveComputeSystem):
    name = "search_agent"

    def _build_retrieve_agent(self, ctx: ExecutionContext, resources: BenchmarkResources, q: Question) -> SearchAgent:
        self.retrieve_config: SearchAgentConfig
        additional_notes_template = _PROMPTS["additional_notes"]
        additional_notes = _ENV.from_string(additional_notes_template).render(
            compute_objective=resources.compute_objective,
            corpus_details=resources.corpus_details,
        )
        return SearchAgent(
            config=self.retrieve_config,
            document_map=resources.document_map,
            chroma_collection=resources.chroma_collection,
            llm_client=ctx.llm_client,
            storage_config=ctx.config.storage,
            agent_id=self.retrieve_usage_key,
            registry=WorkingSetRegistry(
                chroma_host=ctx.config.storage.chroma_server_host,
                chroma_port=ctx.config.storage.chroma_server_port,
            ),
            working_set_name=f"{q.qid}_{str(uuid.uuid4())}",
            additional_notes=additional_notes,
        )

    async def retrieve(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext) -> Retrieved:
        agent = self._build_retrieve_agent(ctx, resources, q)
        # Validate the returned doc_ids against the corpus and let the agent correct any that
        # name no real document, so recall reflects reality and (in retrieve mode) the answerer
        # gets real page text rather than an empty stub for a mis-cited id.
        _, doc_ids = await agent.call(ctx, q.text)
        # Surface why the agent's loop stopped (budget/steps) so the runner can record it; the
        # doc-id correction loop shares the agent instance, so this reflects the final state.
        ts = agent._terminate_state

        return Retrieved(doc_ids=doc_ids, terminate_state=ts)


