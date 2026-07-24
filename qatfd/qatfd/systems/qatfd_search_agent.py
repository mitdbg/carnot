"""System #3: SearchAgent with the semantic-filter TOOL *instead of* vector search.

Same as the SearchAgent (#2), but the agent's `search_corpus` (vector search) tool is
replaced by skunk's `semantic_filter` tool, which it can call mid-loop to filter the
corpus (by metadata and/or a top_k vector prefilter — the tool embeds internally when
`top_k` is used) or to narrow doc_ids it already collected, down to the documents that
satisfy a predicate. `grep_corpus` remains as the lexical discovery path.

Rationale: in a prior experiment where the agent had BOTH tools, it always chose vector
search and never the sem-filter tool. Dropping `search_corpus` as a standalone tool
forces the agent onto semantic filtering, so we can measure that variant in isolation.

Inherits the agent_mode machinery, so both retrieve-only and direct-answer variants
are available.
"""

from __future__ import annotations

from skunk.common import ExecutionContext
from skunk.search_agent.search_tools import SemanticFilterTool

from qatfd.benchmarks.base import BenchmarkResources
from qatfd.systems.search_agent import SearchAgentSystem


class QATFDSearchAgentSystem(SearchAgentSystem):
    name = "qatfd_search_agent"

    def _extra_tools(self, ctx: ExecutionContext, resources: BenchmarkResources) -> tuple:
        model = self.inference_cfg.llm_model
        # ctx.llm_client is the usage-tracking wrapper, so the tool's LLM (judge) calls and
        # its top_k query embeddings are both counted.
        return (SemanticFilterTool(
            ctx.llm_client, resources.document_map, model,
            chroma_collection=resources.chroma_collection,
            emb_model_id=self.inference_cfg.emb_model_id,
            max_output_tokens=self.config.grep_max_output_tokens,
            ctx=ctx,
            context_limits=self.inference_cfg.llm_context_limits,
            judge_max_output_tokens=self.config.semantic_filter_max_output_tokens,
        ),)

    def _include_search_corpus(self) -> bool:
        # Drop the `search_corpus` (vector-search) tool so the agent is forced to use the
        # `semantic_filter` tool for semantic narrowing. See the module docstring.
        return False
