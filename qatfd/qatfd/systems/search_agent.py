"""System #2: the SearchAgent.

Wraps skunk's `SearchAgent` (grep / vector-search / read-document / prune) in the
qatfd `retrieve()`/`compute()` shape. Two agent modes:
  - "retrieve": the agent returns relevant doc_ids; a downstream LLM answers (compute()).
  - "answer":  the agent answers the question directly (different prompt); compute()
    passes that answer through.

`_build_search_agent` constructs skunk's `SearchAgent` directly.
"""

from __future__ import annotations

import time
import uuid
from collections import deque

from jinja2 import Environment, StrictUndefined
from threading import Lock

from skunk.common import ExecutionContext
from skunk.config import AgentConfig, InferenceConfig
# from skunk.agents.search_agent.search_agent import SearchAgent
# from skunk.search_state.working_set_registry import WorkingSetRegistry

from qatfd.agents.bootstrap_agent import BootstrapAgent
from qatfd.agents.enrich_agent import EnrichAgent
from qatfd.agents.search_agent import SearchAgent
from qatfd.benchmarks.base import BenchmarkResources
from qatfd.config import QATFDSearchAgentConfig
from qatfd.prompts import load_qatfd_prompts
from qatfd.systems.base import RetrieveComputeSystem
from qatfd.tools import DeleteCollectionTool, ListCollectionsTool
from qatfd.types import AnswerOutput, Question, Retrieved

_ENV = Environment(
    autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined
)
_SA_PROMPTS = load_qatfd_prompts("search_agent")
_BS_PROMPTS = load_qatfd_prompts("bootstrap_agent")
_EN_PROMPTS = load_qatfd_prompts("enrich_agent")
_WS_PROMPTS = load_qatfd_prompts("working_set")


class SearchAgentSystem(RetrieveComputeSystem):
    name = "search_agent"

    def __init__(self, retrieve_config: AgentConfig, compute_config: AgentConfig, inference_cfg: InferenceConfig) -> None:
        super().__init__(retrieve_config, compute_config, inference_cfg)
        self._question_lock = Lock()
        self._question_num = None


        assert isinstance(retrieve_config, QATFDSearchAgentConfig)
        max_previous_queries = retrieve_config.enrich_config.max_previous_queries
        self._question_history: deque[str] = deque(maxlen=max(0, max_previous_queries))

    @property
    def extra_usage_keys(self) -> dict[str, str]:
        self.retrieve_config: QATFDSearchAgentConfig
        return {
            "precompute": self.retrieve_config.bootstrap_config.agent_id,
            "enrich": self.retrieve_config.enrich_config.agent_id,
        }

    def _build_search_agent(self, ctx: ExecutionContext, resources: BenchmarkResources, q: Question) -> SearchAgent:
        self.retrieve_config: QATFDSearchAgentConfig
        additional_notes_template = _SA_PROMPTS["additional_notes"]
        additional_notes = _ENV.from_string(additional_notes_template).render(
            compute_objective=resources.compute_objective,
            corpus_details=resources.corpus_details,
        )
        with self._question_lock:
            q_num = self._question_num

        return SearchAgent(
            config=self.retrieve_config,
            document_map=resources.document_map,
            chroma_client=resources.chroma_client,
            llm_client=ctx.llm_client,
            storage_config=ctx.config.storage,
            working_set_collection_on=not self.retrieve_config.working_set_collection_off,
            collection_name=f"docs_for_q{q_num}_{str(uuid.uuid4())}",
            additional_notes=additional_notes,
            page_locator=resources.page_locator,
        )

    def _build_bootstrap_agent(self, ctx: ExecutionContext, resources: BenchmarkResources) -> BootstrapAgent:
        self.retrieve_config: QATFDSearchAgentConfig
        additional_notes = None
        if resources.corpus_details:
            additional_notes_template = _BS_PROMPTS["additional_notes"]
            additional_notes = _ENV.from_string(additional_notes_template).render(
                corpus_details=resources.corpus_details,
            )
        return BootstrapAgent(
            config=self.retrieve_config.bootstrap_config,
            document_map=resources.document_map,
            chroma_client=resources.chroma_client,
            llm_client=ctx.llm_client,
            storage_config=ctx.config.storage,
            additional_notes=additional_notes,
            page_locator=resources.page_locator,
        )

    def _build_enrich_agent(self, ctx: ExecutionContext, resources: BenchmarkResources) -> EnrichAgent:
        self.retrieve_config: QATFDSearchAgentConfig
        additional_notes = None
        if resources.corpus_details:
            additional_notes_template = _EN_PROMPTS["additional_notes"]
            additional_notes = _ENV.from_string(additional_notes_template).render(
                corpus_details=resources.corpus_details,
            )
        return EnrichAgent(
            config=self.retrieve_config.enrich_config,
            document_map=resources.document_map,
            chroma_client=resources.chroma_client,
            llm_client=ctx.llm_client,
            storage_config=ctx.config.storage,
            additional_notes=additional_notes,
            page_locator=resources.page_locator,
        )

    @staticmethod
    def _base_collection_summary(ctx: ExecutionContext, resources: BenchmarkResources) -> str:
        """Name / size / metadata fields of the base collection (the Bootstrap agent's whole input)."""
        base_collection_name = ctx.config.storage.collection_name
        c = resources.chroma_client.get_collection(base_collection_name)
        return _ENV.from_string(_BS_PROMPTS["base_collection_summary"]).render(
            base_collection_name=base_collection_name,
            total_num_chunks=c.count(),
            metadata_fields=(c.metadata or {}).get("fields"),
        )

    @staticmethod
    def _collection_summaries(ctx: ExecutionContext, resources: BenchmarkResources) -> str:
        """One `working_set_summary` per agent-created collection (description, size, fields, actions)."""
        base_collection_name = ctx.config.storage.collection_name
        listing = ListCollectionsTool(resources.chroma_client, base_collection_name)()["collections"]
        template = _ENV.from_string(_WS_PROMPTS["working_set_summary"])
        summaries = [
            template.render(
                name=s["name"],
                description=s["description"],
                total_num_chunks=s["num_chunks"],
                metadata_fields=s["fields"],
                actions="\n".join(s["actions"]),
            )
            for s in listing
            if not s["is_base"]
        ]
        return "\n\n".join(summaries)

    async def retrieve(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext) -> Retrieved:
        agent = self._build_search_agent(ctx, resources, q)
        # Validate the returned doc_ids against the corpus and let the agent correct any that
        # name no real document, so recall reflects reality and (in retrieve mode) the answerer
        # gets real page text rather than an empty stub for a mis-cited id.
        doc_ids = await agent.call(ctx, q.text)
        # Surface why the agent's loop stopped (budget/steps) so the runner can record it; the
        # doc-id correction loop shares the agent instance, so this reflects the final state.
        ts = agent._terminate_state

        return Retrieved(doc_ids=doc_ids, terminate_state=ts)


    async def _precompute_working_sets(self, ctx: ExecutionContext, resources: BenchmarkResources):
        # construct the bootstrap agent
        agent = self._build_bootstrap_agent(ctx, resources)

        # have the agent create an initial set of working sets from a description of the base collection
        _ = await agent.call(ctx, self._base_collection_summary(ctx, resources))


    async def _enrich(self, ctx: ExecutionContext, resources: BenchmarkResources, history: list[str]) -> None:
        """Have the EnrichAgent curate the collections given the base collection, the existing
        (agent-created) collections, and the most recent questions handled by this system."""
        # construct the enrich agent
        agent = self._build_enrich_agent(ctx, resources)

        # describe the base collection, the existing collections, and the recent query workload
        enrich_summary = _ENV.from_string(_EN_PROMPTS["enrich_summary"]).render(
            base_collection_summary=self._base_collection_summary(ctx, resources),
            collection_summaries=self._collection_summaries(ctx, resources),
            previous_queries=history,
        )

        # have the agent curate the collections
        _ = await agent.call(ctx, enrich_summary)

        # clear the SearchAgent working set collections
        if self.retrieve_config.hide_and_clear_working_sets:
            base_collection_name = ctx.config.storage.collection_name
            delete_collection = DeleteCollectionTool(resources.chroma_client, base_collection_name)
            collections = ListCollectionsTool(resources.chroma_client, base_collection_name)()["collections"]
            for collection in collections:
                if collection["is_working_set"] and collection["created_by_agent_type"] == "SearchAgent":
                    delete_collection(collection["name"])

    async def answer(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext, analytics_id: str) -> AnswerOutput:
        t0 = time.monotonic()
        mode = self.retrieve_config.enrich_working_sets
        with self._question_lock:
            first_question = self._question_num is None
            if first_question:
                self._question_num = 0
            if mode in ("before", "both") and first_question:
                await self._precompute_working_sets(ctx, resources)
        t1 = time.monotonic()

        try:
            answer_output = await super().answer(q, resources, ctx, analytics_id)
        except Exception as e:
            answer_output = AnswerOutput("", error=str(e))

        t2 = time.monotonic()
        with self._question_lock:
            assert isinstance(self._question_num, int)
            self._question_num += 1
            self._question_history.append(q.text)

            if mode in ("after", "both"):
                assert self.retrieve_config.enrich_query_batch_size is not None
                if self._question_num % self.retrieve_config.enrich_query_batch_size == 0:
                    await self._enrich(ctx, resources, list(self._question_history))
        t3 = time.monotonic()

        # update answer_output with timing info
        answer_output.precompute_wall_s = t1 - t0 if mode in ("before", "both") else 0.0
        answer_output.enrich_wall_s = t3 - t2 if mode in ("after", "both") else 0.0

        return answer_output
