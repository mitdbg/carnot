"""System #1: RAG + LLM.

One vector search over the corpus, then a bounded code-execution agent answers from the
retrieved chunks. retrieve() returns the chunk context (and the chunks' doc_ids for
doc-recall); compute() runs a tools-free `MultiTurnAgent` whose only action is writing and
executing Python over that context, so the model can extract values and compute numeric
answers from the tables rather than doing arithmetic in its head.
"""

from __future__ import annotations

from string import Template

from skunk.common import ExecutionContext
from skunk.multi_turn_agent import MultiTurnAgent
from skunk.search_agent.search_tools import SearchCorpusTool

from qatfd.benchmarks.base import BenchmarkResources
from qatfd.config import RAGLLMConfig
from qatfd.systems.base import RetrieveComputeSystem
from qatfd.types import Question, Retrieved


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
    ``{"answer": "..."}``."""

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

    def __init__(self, answer_format_hint: str, max_steps: int) -> None:
        system_prompt = _CODE_ANSWER_SYSTEM.substitute(
            max_steps=max_steps, answer_format_hint=answer_format_hint
        )
        super().__init__([], max_steps=max_steps, system_prompt_override=system_prompt)

    def validate_final_answer(self, payload: object, observations: list[str]) -> str | None:
        if not isinstance(payload, dict) or "answer" not in payload:
            return 'Emit a JSON object with a single "answer" key, e.g. {"answer": "..."}.'
        return None


class RAGLLMSystem(RetrieveComputeSystem):
    name = "rag_llm"
    # Step budget for the code-execution answer agent (one vector search up front, then up
    # to this many extract/compute turns before it must emit the final answer).
    answer_max_steps = 5

    def __init__(self, config: RAGLLMConfig) -> None:
        self.config = config

    async def compute(
        self, q: Question, r: Retrieved, resources: BenchmarkResources, ctx: ExecutionContext
    ) -> str:
        context = r.context or ""
        agent = CodeAnswerAgent(
            answer_format_hint=resources.answer_format_hint, max_steps=self.answer_max_steps
        )
        payload = await agent.call(ctx, f"Question: {q.text}\n\nDocuments:\n{context}")
        if isinstance(payload, dict) and payload.get("answer") is not None:
            return str(payload["answer"])
        return ""

    async def retrieve(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext) -> Retrieved:
        # Embed via this question's LLMClient (ctx.llm_client) so query-embedding tokens/cost
        # land on the same usage tracker the runner reads; backend = config.emb_provider.
        tool = SearchCorpusTool(
            resources.chroma_collection, self.config.emb_model_id, ctx.llm_client,
            set(), set(), ctx=ctx,
        )

        # emit under a "retrieve" step so the per-question trace records what this vector search returned
        with ctx.step("retrieve"):
            out = tool(query=q.text, top_k=self.config.top_k)  # type: ignore
            chunks = out.get("chunks", []) if isinstance(out, dict) else []
            context = "\n\n".join(c["text"] for c in chunks)

            # unique doc_ids in retrieval order, for doc-recall scoring.
            seen: set[str] = set()
            doc_ids: list[str] = []
            for c in chunks:
                d = c["doc_id"]
                if d not in seen:
                    seen.add(d)
                    doc_ids.append(d)

            ctx.emit(
                f"retrieved top_k={self.config.top_k} chunks={len(chunks)} docs={len(doc_ids)}",
                kind="observation",
                data={
                    "doc_ids": doc_ids,
                    "chunk_ids": [c.get("chunk_id") for c in chunks],
                    "context": context,
                },
            )
        return Retrieved(doc_ids=doc_ids, context=context)
