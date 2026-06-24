"""System #3: SearchAgent + a semantic-filter TOOL.

Same as the SearchAgent (#2), but the agent gets an extra `semantic_filter` tool it
can call mid-loop to narrow a candidate doc set down to those that satisfy a
predicate. Inherits the agent_mode machinery, so both retrieve-only and
direct-answer variants are available.
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
