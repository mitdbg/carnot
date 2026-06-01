"""OfficeQA — declarative QA pipeline over the U.S. Treasury Bulletin corpus.

question → Planner → Plan (branches + computation + presentation) → orchestrator →
{retrieve, extract, lookup_external, compute}, where `compute` is the chain terminator
and subsumes formatting. Public API is re-exported below; see ARCHITECTURE.md for intent."""

from skunk.common import AnnotatedValue, HarnessContext, LLMClient, PageRef
from skunk.config import SkunkConfig
from skunk.errors import MissingData, StepFailed
from skunk.orchestrator import Orchestrator
from skunk.plan import Plan
from skunk.prompted_call import load_prompt_overrides
from skunk.question_explainer import ConceptExplanation
from skunk.trace import QuestionTrace

__all__ = [
    "AnnotatedValue",
    "HarnessContext",
    "LLMClient",
    "MissingData",
    "Orchestrator",
    "PageRef",
    "Plan",
    "ConceptExplanation",
    "QuestionTrace",
    "SkunkConfig",
    "StepFailed",
    "load_prompt_overrides",
]
