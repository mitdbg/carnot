"""System #1: RAG + LLM.

One vector search over the corpus; retrieve() returns the top-k chunk context (and the
chunks' doc_ids for doc-recall). The answer is then produced by the shared code-execution
`CodeAnswerAgent` in `RetrieveComputeSystem.compute()` — identical across all systems, so a
run's score reflects this system's retrieval (a single vector search) and nothing else.
"""

from __future__ import annotations

from skunk.common import ExecutionContext
from skunk.search_agent.search_tools import SearchCorpusTool

from qatfd.benchmarks.base import BenchmarkResources
from qatfd.config import RAGLLMConfig
from qatfd.systems.base import RetrieveComputeSystem
from qatfd.types import Question, Retrieved


class RAGLLMSystem(RetrieveComputeSystem):
    name = "rag_llm"

    def __init__(self, config: RAGLLMConfig) -> None:
        self.config = config

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
