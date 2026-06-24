"""The `System` abstraction.

Public contract is a single `answer()`. Most systems decompose into `retrieve()`
-> `compute()` (`RetrieveComputeSystem`), which cleanly spans both retrieve-only
agents (a downstream LLM answers from the retrieved docs) and direct-answer agents
(the agent produces the answer itself; `compute()` passes it through).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from skunk.common import ExecutionContext
from skunk.config import SystemConfig
from skunk.prompted_call import overrides_tail

from qatfd.benchmarks.base import BenchmarkResources
from qatfd.types import AnswerOutput, Question, Retrieved


def _docs_to_context(doc_ids: list[str], document_map: dict[str, str]) -> str:
    return "\n\n".join(f"=== doc_id={d} ===\n{document_map.get(d, '') or ''}" for d in doc_ids)


async def generate_answer_from_context(
    ctx: ExecutionContext, question: str, context: str, *, answer_format_hint: str, model: str | None = None
) -> str:
    """Single LLM call: answer `question` using only `context`."""
    system = (
        "You are answering a question using ONLY the provided documents. If the answer "
        "is not in them, answer with your best inference from what is given. " + answer_format_hint
    )
    system += overrides_tail(ctx.prompt_overrides, "final_answer")
    user = f"Question: {question}\n\nDocuments:\n{context}"

    # emit the answer call's turns under an "answer" step so the trace viewer can show
    # the compute/answer LLM trace as its own card (the agent steps emit op=None).
    with ctx.step("answer"):
        ctx.emit(f"answer_system chars={len(system)}", kind="system", data={"text": system})
        ctx.emit(f"answer_user chars={len(user)}", kind="user", data={"text": user})
        resp = await ctx.llm_client.acall( # type: ignore
            system=system,
            user=user,
            temperature=0.0,
            model=model or ctx.config.llm_model,
            ctx=ctx,
            call_site="final_answer",
        )
        answer = (resp.text or "").strip()
        ctx.emit(f"answer_assistant chars={len(answer)}", kind="assistant", data={"text": answer})
    return answer


class System(ABC):
    name: str  # set to the registry key on the instance by build_system

    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    @abstractmethod
    async def answer(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext) -> AnswerOutput:
        ...


class RetrieveComputeSystem(System):
    """retrieve() -> compute() pipeline behind answer(). Subclasses implement
    retrieve(); compute() defaults to a downstream LLM answer over the retrieved
    context/docs, or pass-through when the agent already produced an answer."""

    @abstractmethod
    async def retrieve(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext) -> Retrieved:
        ...

    async def compute(self, q: Question, r: Retrieved, resources: BenchmarkResources, ctx: ExecutionContext) -> str:
        if r.direct_answer is not None:
            return r.direct_answer
        context = r.context if r.context is not None else _docs_to_context(r.doc_ids, resources.document_map)
        return await generate_answer_from_context(
            ctx, q.text, context, answer_format_hint=resources.answer_format_hint
        )

    async def answer(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext) -> AnswerOutput:
        r = await self.retrieve(q, resources, ctx)
        ans = await self.compute(q, r, resources, ctx)
        return AnswerOutput(answer=ans, retrieved_doc_ids=r.doc_ids)
