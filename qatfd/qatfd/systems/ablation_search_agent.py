"""Tool-ablation SearchAgent: the retrieval tool set is chosen by config.

Same retrieve()/compute() machinery as the vanilla SearchAgent (system #2), but which of the
three discovery/narrowing tools the agent gets — `search_corpus` (vector), `grep_corpus`
(lexical), `semantic_filter` — is driven by three booleans on `AblationSearchAgentConfig`.
`read_document` and `prune` are always present. This lets one system express every point in the
tool lattice (grep-only, vector-only, sem-only, grep+vector = vanilla SearchAgent, grep+sem =
vanilla QATFD, all three) so each tool's usefulness can be ablated on a fixed model/benchmark.

The semantic-filter tool is constructed exactly as QATFDSearchAgentSystem builds it.
"""

from __future__ import annotations

from skunk.common import ExecutionContext
from skunk.search_agent.search_tools import SemanticFilterTool

from qatfd.benchmarks.base import BenchmarkResources
from qatfd.config import AblationSearchAgentConfig
from qatfd.systems.search_agent import SearchAgentSystem


class AblationSearchAgentSystem(SearchAgentSystem):
    name = "ablation_search_agent"
    config: AblationSearchAgentConfig

    def _include_search_corpus(self) -> bool:
        return self.config.tool_vector

    def _include_grep_corpus(self) -> bool:
        return self.config.tool_grep

    def _extra_tools(self, ctx: ExecutionContext, resources: BenchmarkResources) -> tuple:
        if not self.config.tool_semantic_filter:
            return ()
        # The judge (per-candidate keep/drop) calls run on `semantic_filter_model` when set, else
        # the agent's `llm_model`; the search agent itself always stays on `llm_model`. Filtered-out
        # candidates only reach the judge, so a cheaper judge model cuts cost without touching the
        # agent's reasoning. ctx.llm_client is the usage-tracking wrapper, so judge calls + top_k
        # query embeddings are both counted.
        filter_model = self.config.semantic_filter_model or self.config.llm_model
        return (SemanticFilterTool(
            ctx.llm_client, resources.document_map, filter_model,
            chroma_collection=resources.chroma_collection,
            emb_model_id=self.config.emb_model_id,
            max_output_tokens=self.config.grep_max_output_tokens,
            ctx=ctx,
            provider_order=self.config.semantic_filter_provider_order,
        ),)
