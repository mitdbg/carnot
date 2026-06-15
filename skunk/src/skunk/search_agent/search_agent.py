"""SearchAgent — iterative code-execution retriever.

Subclasses `MultiTurnAgent`: it supplies the search-specific system prompt
(`briefing` + `final_answer_doc`) and the chroma/page-map-backed tool set
(`search_corpus` / `grep_corpus` / `read_document` / `prune`), and lets the base
own the multi-turn loop, the block trajectory, and the JSON final-answer
mechanism. The final answer is a ```json``` block ({"page_keys": [...]}), not a
tool.

Two overrides specialise the base for retrieval:
  - `_blocks_from_output` turns the tools' structured (tagged-dict) returns into
    `ChunkBlock`s (one per chunk, carrying chunk_id / doc_id) so the full
    trajectory is preserved for reward computation.
  - `_block_is_visible` redacts chunks the agent has `prune(...)`d from the
    LLM-facing render, faithful to its prune commands during rollout.

The three tools that read/write prune state share the agent's per-question
`_pruned_chunk_ids` / `_pruned_doc_ids` sets. The orchestrator builds one
`SearchAgent` per question / branch, so these sets never leak across questions.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from chromadb.api.models.Collection import Collection

from skunk.common import (
    B64Image,
    ExecutionContext,
    HumanInterventionHandler,
    make_genai_client,
)
from skunk.config import SkunkConfig
from skunk.human_intervention import RequestHumanTool
from skunk.local_python_executor import CodeOutput
from skunk.multi_turn_agent import Block, ChunkBlock, ImageBlock, MultiTurnAgent, TextBlock
from skunk.search_agent.search_tools import (
    EMPTY_RESULT_MESSAGE,
    GREP_RESULT_TAG,
    PAGE_INDEX_RESULT_TAG,
    PRUNE_RESULT_TAG,
    READ_DOCUMENT_RESULT_TAG,
    SEARCH_RESULT_TAG,
    VIEW_FIGURE_RESULT_TAG,
    EmbeddingClient,
    GrepCorpusTool,
    PageIndexSearchTool,
    PruneTool,
    ReadDocumentTool,
    SearchCorpusTool,
    ViewFigureTool,
)

if TYPE_CHECKING:
    from skunk.page_index.query import PageIndexRetriever


def _make_embedding_client(emb_model_id: str) -> tuple[EmbeddingClient, str]:
    """Pick the embedding backend from the model id and return (client, model_id).

    Gemini embeddings go through genai (AI Studio); everything else (e.g. Qwen)
    goes through OpenRouter. The returned model id has any `google/` prefix
    stripped for the genai path and is passed through unchanged otherwise.
    """
    cleaned = emb_model_id.removeprefix("google/")
    if "gemini" in cleaned.lower():
        return make_genai_client(), cleaned
    from openrouter import OpenRouter

    return OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"]), emb_model_id


class SearchAgent(MultiTurnAgent):
    name = "search_agent"
    # Larger than the MultiTurnAgent default — search chains accumulate many
    # page-content observations across a 20-step ceiling. Char budget, but the model limit is
    # in TOKENS (~1.05M for gemini-3.5-flash): the Treasury tables are dense numerics at only
    # ~1.5 chars/token, so keep this conservative — 1.3M chars ≈ 870K tokens, leaving headroom
    # for the system prompt + output under the input ceiling (a higher budget 400'd requests).
    context_budget_chars: int = 1_300_000
    warn_steps_remaining = 2

    briefing = (
        "You are a helpful assistant for retrieving relevant information from a large "
        "collection of documents. You will be given a question, and your task is to "
        "identify which documents are relevant to answering it. The corpus is organized "
        "as documents (each identified by a `doc_id`), each split into chunks (text spans, "
        "tables, titles, ...) identified by a `chunk_id`; every chunk carries metadata you "
        "can filter on. Some questions require looking up information that is not contained "
        "in the corpus; that is handled by a separate agent, so you should only focus on "
        "retrieving relevant documents for the remainder of the question. "
        "You do not need to retrieve everything in a single tool call: use early steps to "
        "explore documents of potential relevance, then refine your searches in later steps "
        "based on what you find. Use `prune(...)` aggressively on chunks and docs you have "
        "ruled out, to keep later searches focused and your context window manageable."
    )

    final_answer_doc = """\
A JSON object with the `doc_id`s you identified as relevant, under the key
"page_keys":
```json
{"page_keys": ["2002_06_1", "2002_12_26"]}
```
Use each `doc_id` exactly as it appears in the search / grep results."""

    def __init__(
        self,
        config: SkunkConfig,
        document_map: dict[str, str],
        chroma_collection: Collection,
        *,
        system_prompt_override: str | None = None,
        generation_backend=None,
        sampling_params: dict | None = None,
        capture_logprobs: bool = False,
        human_intervention_handler: HumanInterventionHandler | None = None,
        required_bulletins: list[str] | None = None,
        page_index_retriever: PageIndexRetriever | None = None,
    ):
        self.config = config
        self.chroma_collection = chroma_collection
        # Set per-call (in `retrieve`) so the page_index tool — built before retrieve(ctx)
        # runs — can reach the live ExecutionContext and the agent's main event loop. The
        # tool runs in a worker thread but must drive the async retriever ON this loop (the
        # LLM client's async objects are bound to it). One SearchAgent per question/branch.
        self._ctx: ExecutionContext | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        required_doc_prefixes = {
            bulletin.replace("-", "_") + "_"
            for bulletin in required_bulletins or []
        }
        self.document_map = (
            {
                doc_id: text
                for doc_id, text in document_map.items()
                if any(doc_id.startswith(prefix) for prefix in required_doc_prefixes)
            }
            if required_doc_prefixes
            else document_map
        )
        self.emb_client, self.emb_model_id = _make_embedding_client(config.emb_model_id)
        required_filter = None
        if required_bulletins:
            clauses = [
                {
                    "$and": [
                        {"year": bulletin[:4]},
                        {"month": bulletin[5:]},
                    ]
                }
                for bulletin in required_bulletins
            ]
            required_filter = clauses[0] if len(clauses) == 1 else {"$or": clauses}

        # Bound each search-step LLM call: cap output (was uncapped → runaway
        # generations streamed to the 65535-token ceiling at 200–800s each) and
        # impose a hard per-request wall-clock timeout. See SkunkConfig for the
        # thinking/max_output_tokens interaction caveat.
        self.max_output_tokens = config.search_agent_max_output_tokens
        self.request_timeout_s = config.search_agent_request_timeout_s

        # Per-question prune state: shared by the search / grep / prune tools and
        # read by `_block_is_visible` for redaction. One SearchAgent per
        # question / branch ⇒ these sets never cross-talk between questions.
        self._pruned_chunk_ids: set[str] = set()
        self._pruned_doc_ids: set[str] = set()

        # Tool instances capture their deps; the prompt's tool docs are generated
        # from their `doc`s by the base, so tools and docs can't drift.
        tools = [
            SearchCorpusTool(
                self.chroma_collection, self.emb_model_id, self.emb_client,
                self._pruned_chunk_ids, self._pruned_doc_ids,
                required_filter,
            ),
            GrepCorpusTool(
                self.chroma_collection,
                self._pruned_chunk_ids,
                self._pruned_doc_ids,
                config.grep_max_output_tokens,
                required_filter,
            ),
            ReadDocumentTool(
                self.document_map,
                config.agent_max_pages_per_tool_call,
                config.read_document_max_output_chars,
            ),
            ViewFigureTool(self.document_map, config.pdf_dir),
            PruneTool(self._pruned_chunk_ids, self._pruned_doc_ids),
        ]
        # Optional concept-aware retrieval tool (SKUNK_SEARCH_AGENT_PAGEINDEX). Wired only
        # when the caller passes the shared PageIndexRetriever; reads ctx via `self._ctx`.
        if page_index_retriever is not None:
            tools.append(
                PageIndexSearchTool(
                    page_index_retriever,
                    config.pdf_dir,
                    lambda: self._require_ctx(),
                    lambda: self._require_loop(),
                    config.read_document_max_output_chars // PageIndexSearchTool._CHARS_PER_TOKEN,
                    required_bulletins=required_bulletins,
                )
            )
        if human_intervention_handler is not None:
            tools.append(RequestHumanTool(human_intervention_handler))
        super().__init__(
            tools, max_steps=config.agent_max_steps,
            max_misfires=config.agent_max_misfires,
            system_prompt_override=system_prompt_override,
            generation_backend=generation_backend,
            sampling_params=sampling_params,
            capture_logprobs=capture_logprobs,
        )

    def _require_ctx(self) -> ExecutionContext:
        """The live ExecutionContext, set at the top of `retrieve`. Lets the page_index
        tool (constructed in `__init__`, before ctx exists) reach it at call time."""
        if self._ctx is None:
            raise RuntimeError("page_index_search called before retrieve() set the context")
        return self._ctx

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        """The agent's main event loop, captured at the top of `retrieve`. The page_index
        tool schedules the async retriever onto it from its worker thread."""
        if self._loop is None:
            raise RuntimeError("page_index_search called before retrieve() captured the loop")
        return self._loop

    # ------------------------------------------------------------------
    # Block rendering / redaction (override the base hooks)
    # ------------------------------------------------------------------

    def _block_is_visible(self, block: Block) -> bool:
        """Redact pruned chunks from the LLM-facing render (full trajectory is kept)."""
        if isinstance(block, ChunkBlock):
            if block.chunk_id is not None and block.chunk_id in self._pruned_chunk_ids:
                return False
            if block.doc_id in self._pruned_doc_ids:
                return False
        return True

    def _blocks_from_output(self, out: CodeOutput) -> list[Block]:
        """Render a tool result into blocks. Chunk-bearing payloads become one
        `ChunkBlock` per chunk (redactable); everything else is a `TextBlock`."""
        blocks: list[Block] = []
        stdout_s = (out.logs or "").strip()
        if stdout_s:
            blocks.append(TextBlock(f"[stdout]\n{stdout_s}"))
        output = out.output

        if isinstance(output, dict) and output.get(SEARCH_RESULT_TAG):
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            elif not output["chunks"]:
                blocks.append(TextBlock(EMPTY_RESULT_MESSAGE))
            else:
                blocks.extend(
                    ChunkBlock(chunk_id=c["chunk_id"], doc_id=c["doc_id"], text=c["text"])
                    for c in output["chunks"]
                )
            return blocks

        if isinstance(output, dict) and output.get(GREP_RESULT_TAG):
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            elif not output["groups"]:
                blocks.append(TextBlock(EMPTY_RESULT_MESSAGE))
            else:
                for group in output["groups"]:
                    blocks.append(TextBlock(group["header"]))
                    blocks.extend(
                        ChunkBlock(chunk_id=c["chunk_id"], doc_id=c["doc_id"], text=c["text"])
                        for c in group["chunks"]
                    )
            # Surfaced when the output cap dropped hits (visible TextBlock, not redactable).
            if output.get("truncation_note"):
                blocks.append(TextBlock(output["truncation_note"]))
            return blocks

        if isinstance(output, dict) and output.get(READ_DOCUMENT_RESULT_TAG):
            blocks.extend(
                ChunkBlock(chunk_id=None, doc_id=d["doc_id"], text=d["text"])
                for d in output["docs"]
            )
            return blocks

        if isinstance(output, dict) and output.get(PAGE_INDEX_RESULT_TAG):
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            elif not output["results"]:
                blocks.append(TextBlock(EMPTY_RESULT_MESSAGE))
            else:
                # One ChunkBlock per surviving block (keyed by page_key so it's citeable /
                # readable / prunable like read_document output).
                blocks.extend(
                    ChunkBlock(chunk_id=None, doc_id=r["page_key"], text=r["text"])
                    for r in output["results"]
                )
            if output.get("truncation_note"):
                blocks.append(TextBlock(output["truncation_note"]))
            return blocks

        if isinstance(output, dict) and output.get(VIEW_FIGURE_RESULT_TAG):
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            else:
                caption = (
                    f"[full-page image of doc_id={output['doc_id']} "
                    f"(contains <figure id={output['figure_id']}>)]"
                )
                blocks.append(ImageBlock(
                    doc_id=output["doc_id"],
                    figure_id=output["figure_id"],
                    image=B64Image(mime=output["mime"], data=output["data"]),
                    text=caption,
                ))
            return blocks

        if isinstance(output, dict) and output.get(PRUNE_RESULT_TAG):
            blocks.append(TextBlock(
                f"[result]\nPruned {output['new_chunk_count']} chunk(s) and "
                f"{output['new_doc_count']} doc(s). {output['total_chunks']} chunk(s) and "
                f"{output['total_docs']} doc(s) are now excluded from future searches."
            ))
            return blocks

        # Non-structured output: default [result] rendering (mirrors the base).
        result_s = "" if output is None else str(output).strip()
        if result_s and result_s != stdout_s and result_s not in stdout_s:
            blocks.append(TextBlock(f"[result]\n{result_s}"))
        if not blocks:
            blocks.append(TextBlock("[no output]"))
        return blocks

    # ------------------------------------------------------------------
    # Retriever entry point
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        ctx: ExecutionContext,
        question: str,
        *,
        branch_key: str | None = None,
        branch_period: str | None = None,
        required_bulletins: list[str] | None = None,
    ) -> list[str]:
        # Expose ctx + the running loop to the page_index tool (built before either existed).
        self._ctx = ctx
        self._loop = asyncio.get_running_loop()
        parts = [f"Question: {question}"]
        if branch_key:
            parts.append(f"Search focus: {branch_key}")
        if branch_period:
            parts.append(f"Time period (of the data): {branch_period}")
        if required_bulletins:
            parts.append(
                "Human-required source bulletins (hard scope): "
                + ", ".join(required_bulletins)
            )
        payload = await self.call(ctx, "\n".join(parts))
        return self._page_keys_from_payload(payload)

    @staticmethod
    def _page_keys_from_payload(payload: Any) -> list[str]:
        keys = payload.get("page_keys") or []
        if isinstance(keys, str):
            return [keys]
        return [str(k) for k in keys]
