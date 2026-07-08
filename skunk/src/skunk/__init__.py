"""skunk — library for agentic retrieval and computation over document corpora.

Declarative QA pipeline: question → Planner → Plan (branches) → orchestrator →
{retrieve, extract, lookup_external, compute}, where `compute` is the chain terminator
and subsumes formatting. Public API is re-exported below; see ARCHITECTURE.md for intent."""

from skunk.common import AnnotatedValue, ExecutionContext, PageRef
from skunk.llm_client import LLMClient
from skunk.config import PipelineConfig
from skunk.errors import MissingData, StepFailed
from skunk.orchestrator import Orchestrator
from skunk.plan import Plan
from skunk.prompted_call import load_prompt_overrides
from skunk.question_explainer import ConceptExplanation
from skunk.result import ExecutionResult

__all__ = [
    "AnnotatedValue",
    "ExecutionResult",
    "ExecutionContext",
    "LLMClient",
    "MissingData",
    "Orchestrator",
    "PageRef",
    "Plan",
    "ConceptExplanation",
    "PipelineConfig",
    "StepFailed",
    "load_prompt_overrides",
]
