"""System #3: SearchAgent with the semantic-filter TOOL *instead of* vector search.

Same as the SearchAgent (#2), but the agent's `search_corpus` (vector search) tool is
replaced by a `semantic_filter` tool it can call mid-loop to narrow a candidate doc set
down to those that satisfy a predicate. `grep_corpus` remains as the lexical discovery
path that surfaces the candidate `doc_id`s to feed into `semantic_filter`.

Rationale: in a prior experiment where the agent had BOTH tools, it always chose vector
search and never the sem-filter tool. Dropping vector search forces it to use semantic
filtering, so we can measure that variant in isolation.

Inherits the agent_mode machinery, so both retrieve-only and direct-answer variants
are available.
"""

from __future__ import annotations

from skunk.common import ExecutionContext

from qatfd.benchmarks.base import BenchmarkResources
from qatfd.systems.search_agent import SearchAgentSystem
from qatfd.systems.semfilter import SemanticFilterTool


class QATFDSearchAgentSystem(SearchAgentSystem):
    name = "qatfd_search_agent"

    def _extra_tools(self, ctx: ExecutionContext, resources: BenchmarkResources) -> tuple:
        model = self.config.agent_model_id
        # ctx.llm_client is the usage-tracking wrapper, so the tool's LLM calls are counted.
        return (SemanticFilterTool(ctx.llm_client, resources.document_map, model, ctx=ctx),)

    def _include_search_corpus(self) -> bool:
        # Drop the `search_corpus` (vector-search) tool so the agent is forced to use the
        # `semantic_filter` tool for semantic narrowing. See the module docstring.
        return False
