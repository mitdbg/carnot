"""The `System` abstraction.

Public contract is a single `answer()`. Most systems decompose into `retrieve()`
-> `compute()` (`RetrieveComputeSystem`), which cleanly spans both retrieve-only
agents (a downstream LLM answers from the retrieved docs) and direct-answer agents
(the agent produces the answer itself; `compute()` passes it through).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from string import Template

from skunk.common import ExecutionContext
from skunk.config import AgentConfig, InferenceConfig
from skunk.multi_turn_agent import MultiTurnAgent
from skunk.storage.document_map import DocumentMap

from qatfd.benchmarks.base import BenchmarkResources
from qatfd.types import AnswerOutput, Question, Retrieved


def _docs_to_context(doc_ids: list[str], document_map: DocumentMap) -> str:
    return "\n\n".join(f"=== doc_id={d} ===\n{document_map.get(d, '') or ''}" for d in doc_ids)


# A string.Template (not str.format / f-string) so the literal `{...}` JSON examples below
# pass through untouched; only the `$max_steps` / `$answer_format_hint` placeholders fill in.
_CODE_ANSWER_SYSTEM = Template("""\
You are a question-answering assistant with access to a Python interpreter. You will be
given a question and a set of reference documents. Answer the question using ONLY the
information in those documents.

The documents are included as plain text in the message below — READ THEM DIRECTLY. The
Python interpreter is a separate, empty scratchpad: it does NOT have access to the
documents, the filesystem, or the environment. Do not try to open files, list directories,
import os/sys, or inspect globals()/locals() — those are blocked and only waste steps. To
compute over the documents, copy the relevant numbers out of the document text into your
code as literals, then compute. Do not do arithmetic in your head — use code.

## Step budget
You have at most $max_steps steps. Each block you emit — a ```python``` block OR the
final ```json``` block — consumes one step. Read the documents, do any computation in as
few steps as possible, and emit your final ```json``` answer as soon as you have it; do not
spend steps exploring the environment.

## Response format
On every step, output EITHER:
  - exactly ONE fenced ```python``` block to compute intermediate results, e.g.:

```python
# values copied from the document text:
defense = [998, 1436, 1002, 808, 935]
print(sum(defense))
```

  - OR a single ```json``` block with your final answer (emit this once, when ready):

```json
{"answer": "<value>"}
```

You may write a brief "Thoughts:" line before the block, but no other text, and exactly
one block per step. The ```json``` final answer is parsed as data (not executed) — write a
plain JSON literal.

## Final answer
<value> must be formatted exactly per these rules: $answer_format_hint
If the documents do not contain what is needed, give your best inference from what is
provided.""")


class CodeAnswerAgent(MultiTurnAgent):
    """Tools-free `MultiTurnAgent` that answers from the provided context, with Python
    code execution as its only action. The retrieved chunks are packed into the first
    user message (no corpus tools); the agent reads them directly, transcribes the
    relevant values into Python steps to compute, then emits a ```json``` final answer
    ``{"answer": "..."}``.

    Shared across systems as the single, fixed compute step so a run's score reflects only
    the retrieval method that produced the context (see `RetrieveComputeSystem.compute`)."""

    name = "rag_answer"
    # Widen the per-step sandbox so the agent can do real numeric work over the tables.
    authorized_imports = [
        "math", "statistics", "numpy", "scipy", "statsmodels",
        "decimal", "fractions", "itertools", "collections", "json", "re",
    ]
    # The retrieved context is packed into the first message; keep it (and accumulated
    # step outputs) from being trimmed away.
    context_budget_chars = 600_000
    # Warn with two steps left (not one), so the model has a turn to react and commit.
    warn_steps_remaining = 2

    def __init__(self, answer_format_hint: str, max_steps: int, agent_id: str | None = None) -> None:
        system_prompt = _CODE_ANSWER_SYSTEM.substitute(
            max_steps=max_steps, answer_format_hint=answer_format_hint
        )
        super().__init__([], max_steps=max_steps, agent_id=agent_id, system_prompt_override=system_prompt)

    def validate_final_answer(self, payload: object, observations: list[str]) -> str | None:
        if not isinstance(payload, dict) or "answer" not in payload:
            return 'Emit a JSON object with a single "answer" key, e.g. {"answer": "..."}.'
        return None


class System(ABC):
    name: str  # set to the registry key on the instance by build_system

    def __init__(self, config: AgentConfig, inference_cfg: InferenceConfig) -> None:
        self.config = config
        self.inference_cfg = inference_cfg

    # Stable per-phase usage-attribution keys, derived from the system's `agent_id` (the system
    # name; see configs/systems/base.yaml). Suffixing splits the retrieval agent's spend from the
    # shared compute answerer's so the runner can record retrieve_cost and compute_cost separately.
    # None when no agent_id is configured (agents then mint their own uuid and go unattributed).
    @property
    def retrieve_usage_key(self) -> str | None:
        return f"{self.config.agent_id}_retrieve" if self.config.agent_id else None

    @property
    def compute_usage_key(self) -> str | None:
        return f"{self.config.agent_id}_compute" if self.config.agent_id else None

    @abstractmethod
    async def answer(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext) -> AnswerOutput:
        ...


class RetrieveComputeSystem(System):
    """retrieve() -> compute() pipeline behind answer(). Subclasses implement retrieve();
    compute() runs the shared `CodeAnswerAgent` over the retrieved context/docs, or passes
    through when the agent already produced an answer (direct-answer mode). Holding the
    compute step fixed across systems isolates the contribution of each retrieval method."""

    # Step budget for the shared code-execution answer agent. One value across all systems
    # so compute is identical and only retrieval varies; subclasses may override.
    answer_max_steps: int = 5

    @abstractmethod
    async def retrieve(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext) -> Retrieved:
        ...

    async def compute(self, q: Question, r: Retrieved, resources: BenchmarkResources, ctx: ExecutionContext) -> str:
        if r.direct_answer is not None:
            return r.direct_answer
        context = r.context if r.context is not None else _docs_to_context(r.doc_ids, resources.document_map)
        agent = CodeAnswerAgent(
            answer_format_hint=resources.answer_format_hint, max_steps=self.answer_max_steps,
            agent_id=self.compute_usage_key,
        )
        payload = await agent.call(ctx, f"Question: {q.text}\n\nDocuments:\n{context}")
        if isinstance(payload, dict) and payload.get("answer") is not None:
            return str(payload["answer"])
        return ""

    async def answer(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext) -> AnswerOutput:
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
