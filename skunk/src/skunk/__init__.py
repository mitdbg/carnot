"""OfficeQA — declarative QA pipeline over the U.S. Treasury Bulletin corpus.

Standalone Python package. No Palimpzest dependency.

Architecture:
  question -> PlannerExecutor (LLM, single shot; src/skunk/plan.py)
            -> Plan (one branches list + computation + presentation)
            -> orchestrator
            -> {retrieve, extract, lookup_external, compute}

`compute` is the chain terminator and subsumes formatting (it self-plans, codegens,
execs, then self-critiques the result against the question with full context).

Public API:
  from skunk import (
      Orchestrator, HarnessContext, SkunkConfig,
      Plan, PageRef, AnnotatedValue,
      LLMClient, QuestionTrace,
      MissingData, StepFailed,
      load_prompt_overrides,
  )

See ARCHITECTURE.md for design intent + the canonical Plan shape.
"""

from skunk.common import LLMClient
from skunk.config import SkunkConfig
from skunk.errors import MissingData, StepFailed
from skunk.models import AnnotatedValue, HarnessContext, PageRef
from skunk.orchestrator import Orchestrator
from skunk.plan import Plan
from skunk.prompt_overrides import load_prompt_overrides
from skunk.trace import QuestionTrace

__all__ = [
    "AnnotatedValue",
    "HarnessContext",
    "LLMClient",
    "MissingData",
    "Orchestrator",
    "PageRef",
    "Plan",
    "QuestionTrace",
    "SkunkConfig",
    "StepFailed",
    "load_prompt_overrides",
]
