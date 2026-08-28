"""skunk — library for agentic retrieval and computation over document corpora.

Declarative QA pipeline: question → Planner → Plan (branches) → orchestrator →
{retrieve, extract, lookup_external, compute}, where `compute` is the chain terminator
and subsumes formatting. Public API is re-exported below; see ARCHITECTURE.md for intent."""

from skunk.common import ExecutionContext, PageRef
from skunk.llm_client import LLMClient
from skunk.config import SkunkConfig
from skunk.errors import StepFailed

__all__ = [
    "ExecutionContext",
    "LLMClient",
    "SkunkConfig",
    "PageRef",
    "StepFailed",
]
