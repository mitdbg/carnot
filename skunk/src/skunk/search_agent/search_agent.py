"""SearchAgent — iterative code-execution retriever.

Subclasses `MultiTurnAgent`: it supplies the search-specific system prompt
(`briefing` + `final_answer_doc`) and the chroma/page-map-backed tool set
(`search_corpus` / `grep_corpus` / `read_document` / `prune`), and lets the base
own the multi-turn loop, the block trajectory, and the JSON final-answer
mechanism. The final answer is a ```json``` block ({"doc_ids": [...]}), not a
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

from typing import Any

from chromadb.api.models.Collection import Collection

from skunk.common import (
    B64Image,
    ExecutionContext,
)
from skunk.config import SearchAgentConfig
from skunk.llm_client import LLMClient
from skunk.local_python_executor import CodeOutput
from skunk.multi_turn_agent import Block, ChunkBlock, ImageBlock, MultiTurnAgent, TextBlock, Tool
from skunk.search_agent.search_tools import (
    EMPTY_RESULT_MESSAGE,
    GREP_RESULT_TAG,
    PRUNE_RESULT_TAG,
    READ_DOCUMENT_RESULT_TAG,
    SEARCH_RESULT_TAG,
    VIEW_FIGURE_RESULT_TAG,
    GrepCorpusTool,
    PruneTool,
    ReadDocumentTool,
    SearchCorpusTool,
    ViewFigureTool,
)


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
A JSON object with the `doc_id`s you identified as relevant, under the key "doc_ids". Here is an example:
```json
{"doc_ids": ["combined_statement__historical__cs-1872_12", "govinfo_receipts__1893__SERIALSET-03108_00_00-002-0256-0000_3"]}
```
Use each `doc_id` exactly as it appears in the search / grep results."""

    def __init__(
        self,
        config: SearchAgentConfig,
        document_map: dict[str, str],
        chroma_collection: Collection,
        *,
        pdf_dir: str | None = None,
        page_renders_dir: str | None = None,
        llm_client: LLMClient | None = None,
        emb_model_id: str | None = None,
        extra_tools: tuple[Tool, ...] = (),
        include_search_corpus: bool = True,
        briefing: str | None = None,
        final_answer_doc: str | None = None,
        system_prompt_override: str | None = None,
        generation_backend=None,
        sampling_params: dict | None = None,
        capture_logprobs: bool = False,
    ):
        # `briefing` / `final_answer_doc` override the class-level defaults on this
        # instance so MultiTurnAgent's template path picks them up (only consulted when
        # there is no `system_prompt_override`). Set before super().__init__.
        if briefing is not None:
            self.briefing = briefing
        if final_answer_doc is not None:
            self.final_answer_doc = final_answer_doc
        self.config = config
        self.chroma_collection = chroma_collection
        self.document_map = document_map
        # Query embedding goes through an LLMClient (it owns backend dispatch + usage
        # accounting). Callers on the per-question request path pass `ctx.llm_client` so
        # embedding spend is billed onto that question's tracker; offline / build paths
        # pass nothing and get a standalone client (embeddings work, untracked).
        self._emb_llm_client = llm_client or LLMClient(config)
        self.emb_model_id = emb_model_id or config.emb_model_id

        # Bound each search-step LLM call: cap output (was uncapped → runaway
        # generations streamed to the 65535-token ceiling at 200–800s each) and
        # impose a hard per-request wall-clock timeout. See SearchAgentConfig for the
        # thinking/max_output_tokens interaction caveat.
        self.max_output_tokens = config.search_agent_max_output_tokens
        self.request_timeout_s = config.search_agent_request_timeout_s

        # Per-question prune state: shared by the search / grep / prune tools and
        # read by `_block_is_visible` for redaction. One SearchAgent per
        # question / branch ⇒ these sets never cross-talk between questions.
        self._pruned_chunk_ids: set[str] = set()
        self._pruned_doc_ids: set[str] = set()
        # Already-fetched chunks/docs, auto-excluded from subsequent search/grep so each
        # call surfaces new material (no re-duplication in context). Distinct from the
        # pruned sets: fetched chunks STAY visible — `prune(...)` is only for ruling out
        # irrelevant material. Shared by reference with the search/grep/read tools.
        self._seen_chunk_ids: set[str] = set()
        self._seen_doc_ids: set[str] = set()

        # Tool instances capture their deps; the prompt's tool docs are generated
        # from their `doc`s by the base, so tools and docs can't drift.
        if pdf_dir is not None:
            extra_tools += (ViewFigureTool(self.document_map, pdf_dir, renders_dir=page_renders_dir),)
        tools: list[Tool] = []
        # Vector search over the corpus. A caller can drop it (`include_search_corpus=False`)
        # to force the agent onto other retrieval tools — e.g. qatfd system #3 removes it so
        # the agent must use its `semantic_filter` tool for semantic narrowing (in the prior
        # experiment the agent always chose vector search and never the sem-filter tool).
        if include_search_corpus:
            tools.append(SearchCorpusTool(
                self.chroma_collection, self.emb_model_id, self._emb_llm_client,
                self._pruned_chunk_ids, self._pruned_doc_ids,
                seen_chunk_ids=self._seen_chunk_ids, seen_doc_ids=self._seen_doc_ids,
            ))
        tools += [
            GrepCorpusTool(
                self.chroma_collection,
                self._pruned_chunk_ids,
                self._pruned_doc_ids,
                config.grep_max_output_tokens,
                seen_chunk_ids=self._seen_chunk_ids, seen_doc_ids=self._seen_doc_ids,
            ),
            ReadDocumentTool(
                self.document_map,
                config.agent_max_pages_per_tool_call,
                config.read_document_max_output_chars,
                seen_doc_ids=self._seen_doc_ids,
            ),
            PruneTool(self._pruned_chunk_ids, self._pruned_doc_ids),
            *extra_tools,
        ]
        super().__init__(
            tools, max_steps=config.agent_max_steps,
            max_misfires=config.agent_max_misfires,
            system_prompt_override=system_prompt_override,
            generation_backend=generation_backend,
            sampling_params=sampling_params,
            capture_logprobs=capture_logprobs,
        )

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
    ) -> list[str]:
        parts = [f"Question: {question}"]
        if branch_key:
            parts.append(f"Search focus: {branch_key}")
        if branch_period:
            parts.append(f"Time period (of the data): {branch_period}")
        payload = await self.call(ctx, "\n".join(parts))
        return self._doc_ids_from_payload(payload)

    @staticmethod
    def _doc_ids_from_payload(payload: Any) -> list[str]:
        keys = payload.get("doc_ids") or []
        if isinstance(keys, str):
            return [keys]
        return [str(k) for k in keys]
