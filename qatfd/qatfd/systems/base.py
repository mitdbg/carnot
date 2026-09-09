"""The `System` abstraction.

Public contract is a single `answer()`. Most systems decompose into `retrieve()`
-> `compute()` (`RetrieveComputeSystem`), which cleanly spans both retrieve-only
agents (a downstream LLM answers from the retrieved docs) and direct-answer agents
(the agent produces the answer itself; `compute()` passes it through).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from jinja2 import Environment, StrictUndefined

from skunk.agents.multi_turn_agent import MultiTurnAgent, StepOutput, parse_step
from skunk.common import ExecutionContext
from skunk.config import AgentConfig, InferenceConfig
from skunk.errors import ParseError
from skunk.sandbox.local_python_executor import BASE_BUILTIN_MODULES
from skunk.storage.document_map import DocumentMap

from qatfd.benchmarks.base import BenchmarkResources
from qatfd.prompts import load_qatfd_prompts
from qatfd.types import AnswerOutput, Question, Retrieved

_ENV = Environment(
    autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined
)
_PROMPTS = load_qatfd_prompts("base")

def _docs_to_context(doc_ids: list[str], document_map: DocumentMap) -> str:
    return "\n\n".join(f"=== doc_id={d} ===\n{document_map.get(d, '') or ''}" for d in doc_ids)


class CodeAnswerAgent(MultiTurnAgent):
    """Tools-free `MultiTurnAgent` that answers from the provided context, with Python
    code execution as its only action. The retrieved chunks are packed into the first
    user message (no corpus tools); the agent reads them directly, transcribes the
    relevant values into Python steps to compute, then emits a ```json``` final answer
    ``{"answer": "..."}``.

    Shared across systems as the single, fixed compute step so a run's score reflects only
    the retrieval method that produced the context (see `RetrieveComputeSystem.compute`).
    """

    name = "compute_answer_agent"
    authorized_imports = [
        "math", "statistics", "numpy", "scipy", "statsmodels",
        "decimal", "fractions", "itertools", "collections", "json", "re",
    ]

    @staticmethod
    def parse_step(text: str) -> StepOutput:
        """Apply the basic answer parser and validate that a final answer has an `answer` field."""
        step_output = parse_step(text)

        if step_output.is_final and "answer" not in step_output.result:
            raise ParseError(
                detail='Final answer must be a JSON object with a single "answer" key, e.g. {"answer": "..."}.'
            )

        return step_output

    def __init__(self, config: AgentConfig, answer_format_hint: str) -> None:
        # override the config name to match the agent name
        config.name = self.name

        # construct the system prompt
        system_prompt_template = _PROMPTS["compute_agent_system_prompt"]
        system_prompt = _ENV.from_string(system_prompt_template).render(
            authorized_imports=list(set(BASE_BUILTIN_MODULES) | set(self.authorized_imports)),
            max_steps=config.max_steps,
            answer_format_hint=answer_format_hint,
        )

        # construct the terminal prompt
        terminal_prompt_template = _PROMPTS["compute_agent_terminal_prompt"]
        terminal_prompt = _ENV.from_string(terminal_prompt_template).render(
            answer_format_hint=answer_format_hint,
        )

        # construct the agent instance
        super().__init__(config, tools=[], system_prompt=system_prompt, terminal_prompt=terminal_prompt, parse=self.parse_step)


class System(ABC):
    name: str  # set to the registry key on the instance by build_system
    inference_cfg: InferenceConfig

    @property
    def system_usage_key(self) -> str:
        ...

    @abstractmethod
    async def answer(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext, session_id: str) -> AnswerOutput:
        ...


class RetrieveComputeSystem(System):
    """retrieve() -> compute() pipeline behind answer(). Subclasses implement retrieve();
    compute() runs the shared `CodeAnswerAgent` over the retrieved context/docs. Holding the
    compute step fixed across systems isolates the contribution of each retrieval method."""

    # step budget for the code-execution answer agent; we use a fixed value across all systems
    # so that compute is identical and only retrieval varies
    ANSWER_MAX_STEPS: int = 5

    def __init__(self, retrieve_config: AgentConfig, compute_config: AgentConfig, inference_cfg: InferenceConfig) -> None:
        self.retrieve_config = retrieve_config
        self.compute_config = compute_config
        self.inference_cfg = inference_cfg

        # enforce that retrieve and compute have agent ids which are distinct so that we can
        # separate out the cost of retrieval from the cost of computing the final answer
        assert self.retrieve_config.agent_id and self.compute_config.agent_id and self.retrieve_config.agent_id != self.compute_config.agent_id

    # stable per-phase usage-attribution keys
    @property
    def retrieve_usage_key(self) -> str:
        return self.retrieve_config.agent_id

    @property
    def compute_usage_key(self) -> str:
        return self.compute_config.agent_id

    @abstractmethod
    async def retrieve(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext) -> Retrieved:
        ...

    async def compute(self, q: Question, r: Retrieved, resources: BenchmarkResources, ctx: ExecutionContext) -> str:
        self.compute_config.max_steps = self.ANSWER_MAX_STEPS
        agent = CodeAnswerAgent(
            config=self.compute_config,
            answer_format_hint=resources.answer_format_hint,
        )
        context = r.context if r.context is not None else _docs_to_context(r.doc_ids, resources.document_map)
        payload = await agent.call(ctx, f"Question: {q.text}\n\nDocuments:\n{context}")
        if isinstance(payload, dict) and payload.get("answer") is not None:
            return str(payload["answer"])

        return ""

    async def answer(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext, session_id: str) -> AnswerOutput:
        t0 = time.monotonic()
        r = await self.retrieve(q, resources, ctx)
        t1 = time.monotonic()
        ans = await self.compute(q, r, resources, ctx)
        t2 = time.monotonic()
        return AnswerOutput(
            answer=ans,
            retrieved_doc_ids=r.doc_ids,
            terminate_state=r.terminate_state,
            retrieve_wall_s=t1 - t0,
            compute_wall_s=t2 - t1,
        )
