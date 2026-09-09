"""System #1: RAG + LLM.

One vector search over the corpus; retrieve() returns the top-k chunk context (and the
chunks' doc_ids for doc-recall). The answer is then produced by the shared code-execution
`CodeAnswerAgent` in `RetrieveComputeSystem.compute()` — identical across all systems, so a
run's score reflects this system's retrieval (a single vector search) and nothing else.
"""

from __future__ import annotations

import time

from skunk.common import ExecutionContext
from skunk.agents.search_agent.search_tools import SearchCorpusTool
from skunk.search_state.working_set import WorkingSet

from qatfd.benchmarks.base import BenchmarkResources
from qatfd.config import RAGLLMConfig
from qatfd.systems.base import RetrieveComputeSystem
from qatfd.types import Question, Retrieved


class RAGLLMSystem(RetrieveComputeSystem):
    name = "rag_llm"

    async def retrieve(self, q: Question, resources: BenchmarkResources, ctx: ExecutionContext) -> Retrieved:
        self.retrieve_config: RAGLLMConfig

        # NOTE: WorkingSet is required argument; but will not be used b/c working_set_collection_off=True
        tool = SearchCorpusTool(
            resources.chroma_collection,
            ctx.llm_client,
            WorkingSet(collection=resources.chroma_collection),
            usage_key=str(self.retrieve_usage_key),
            working_set_collection_off=True,
            id_tracking_off=True,
        )

        # perform top-k vector search query
        tool_start_time = time.monotonic()
        out = tool(query=q.text, top_k=self.retrieve_config.top_k, read=True, fetch=True)  # type: ignore
        chunks = out.get("read_chunks", []) if isinstance(out, dict) else []
        context = "\n\n".join(c["text"] for c in chunks)

        # unique doc_ids in retrieval order, for doc-recall scoring.
        seen: set[str] = set()
        doc_ids: list[str] = []
        for c in chunks:
            d = c["doc_id"]
            if d not in seen:
                seen.add(d)
                doc_ids.append(d)

        ctx.tracer.emit(
            "retrieved",
            kind="observation",
            data={
                "query": q.text,
                "top_k": self.retrieve_config.top_k,
                "latency_s": time.monotonic() - tool_start_time,
                "doc_ids": doc_ids,
                "chunk_ids": [c.get("chunk_id") for c in chunks],
                "chunks": [
                    {
                        "rank": i,
                        "chunk_id": c.get("chunk_id"),
                        "doc_id": c.get("doc_id"),
                        "type": c.get("type"),
                        "distance": c.get("distance"),
                        "text": c.get("text"),
                    }
                    for i, c in enumerate(chunks, 1)
                ],
                "context": context,
            },
        )

        return Retrieved(doc_ids=doc_ids, context=context)
