"""SearchAgent — iterative code-execution retriever.

Subclasses `MultiTurnAgent`: it supplies the search-specific system prompt
(`briefing` + `final_answer_doc`) and the chroma/page-map-backed tool set
(`search_corpus` / `grep_corpus` / `semantic_filter` / `read_document` / `prune`),
and lets the base own the multi-turn loop, the block trajectory, and the JSON
final-answer mechanism. The final answer is a ```json``` block ({"doc_ids": [...]}), not a
tool.

Two overrides specialise the base for retrieval:
  - `_blocks_from_output` turns the tools' structured (tagged-dict) returns into
    `ChunkBlock`s (one per chunk, carrying chunk_id / doc_id) so the full
    trajectory is preserved for reward computation.
  - `_block_is_visible` redacts chunks the agent has `prune(...)`d from the
    LLM-facing render, faithful to its prune commands during rollout.

The tools share the agent's per-question `RetrievalState` (pruned + seen sets,
see its docstring for the contract). The orchestrator builds one `SearchAgent`
per question / branch, so state never leaks across questions.
"""

from __future__ import annotations

from typing import Any
from chromadb.api.models.Collection import Collection

from skunk.common import B64Image, ExecutionContext
from skunk.config import SearchAgentConfig
from skunk.errors import StepFailed
from skunk.llm_client import LLMClient
from skunk.sandbox.local_python_executor import CodeOutput
from skunk.multi_turn_agent import Block, ChunkBlock, ImageBlock, MultiTurnAgent, TextBlock, Tool
from skunk.prompts import load_prompts
from skunk.search_agent.retrieval_state import RetrievalState
from skunk.search_agent.search_tools import (
    EMPTY_RESULT_MESSAGE,
    GREP_RESULT_TAG,
    PRUNE_RESULT_TAG,
    READ_DOCUMENT_RESULT_TAG,
    SEARCH_RESULT_TAG,
    SEMFILTER_RESULT_TAG,
    VIEW_FIGURE_RESULT_TAG,
    GrepCorpusTool,
    PruneTool,
    ReadDocumentTool,
    SearchCorpusTool,
    SemanticFilterTool,
    ViewFigureTool,
)
from skunk.storage.document_map import DocumentMap

_PROMPTS = load_prompts("search_agent")


class SearchAgent(MultiTurnAgent):
    name = "search_agent"
    # Larger than the MultiTurnAgent default — search chains accumulate many
    # page-content observations across a 20-step ceiling. Char budget, but the model limit is
    # in TOKENS (~1.05M for gemini-3.5-flash): the Treasury tables are dense numerics at only
    # ~1.5 chars/token, so keep this conservative — 1.3M chars ≈ 870K tokens, leaving headroom
    # for the system prompt + output under the input ceiling (a higher budget 400'd requests).
    context_budget_chars: int = 1_300_000
    chunks_per_summary: int = 10
    warn_steps_remaining = 2
    briefing = _PROMPTS["briefing"]
    final_answer_doc = _PROMPTS["final_answer_doc"]

    def __init__(
        self,
        config: SearchAgentConfig,
        document_map: DocumentMap,
        chroma_collection: Collection,
        llm_client: LLMClient,
        *,
        ctx: ExecutionContext | None = None,
        pdf_dir: str | None = None,
        page_renders_dir: str | None = None,
        extra_tools: tuple[Tool, ...] = (),
        include_search_corpus: bool = True,
        include_grep_corpus: bool = True,
        include_semantic_filter: bool = False,
        briefing: str | None = None,
        final_answer_doc: str | None = None,
        agent_id: str | None = None,
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
        self._llm_client = llm_client

        # Bound each search-step LLM call: cap output (was uncapped → runaway
        # generations streamed to the 65535-token ceiling at 200–800s each) and
        # impose a hard per-request wall-clock timeout. See SearchAgentConfig for the
        # thinking/max_output_tokens interaction caveat.
        self.max_output_tokens = config.search_agent_max_output_tokens
        self.request_timeout_s = config.search_agent_request_timeout_s

        # Per-question retrieval state (pruned + read + fetched sets), shared by
        # reference with the search / grep / semantic-filter / read / prune tools
        # and read by `_block_is_visible` for redaction — see `RetrievalState`'s
        # docstring for the per-set contract. One SearchAgent per question / branch
        # ⇒ state never cross-talks between questions.
        self._state = RetrievalState()

        # Tool instances capture their deps; the prompt's tool docs are generated
        # from their `doc`s by the base, so tools and docs can't drift. `ctx` is threaded
        # into the tools that emit trace events / bill usage on it (embeds, judge calls).
        if pdf_dir is not None:
            extra_tools += (ViewFigureTool(
                self.document_map, pdf_dir, renders_dir=page_renders_dir,
            ),)
        tools: list[Tool] = []
        # Vector search over the corpus. A caller can drop it (`include_search_corpus=False`)
        # to force the agent onto other retrieval tools — e.g. qatfd system #3 removes it so
        # the agent must use the `semantic_filter` tool for semantic narrowing (in the prior
        # experiment the agent always chose vector search and never the sem-filter tool).
        if include_search_corpus:
            tools.append(SearchCorpusTool(
                self.chroma_collection, self._llm_client, self._state, ctx,
            ))
        # Grep is optional too (symmetric with `include_search_corpus`): a caller can drop it to
        # force the agent onto vector search / semantic filtering, e.g. a tool-ablation experiment.
        if include_grep_corpus:
            tools.append(GrepCorpusTool(
                self.chroma_collection,
                self._state,
            ))
        # LLM-judged predicate filter over candidate documents. Off by default; callers opt in
        # (e.g. qatfd systems #3 / ablation). The judge model, its provider pinning, and its
        # output cap come from config; its context limit is resolved from the client's config.
        if include_semantic_filter:
            tools.append(SemanticFilterTool(
                self.chroma_collection,
                self._llm_client,
                self.document_map,
                config.semantic_filter_model,
                state=self._state,
                ctx=ctx,
                provider_order=config.semantic_filter_provider_order,
                judge_max_output_tokens=config.semantic_filter_max_output_tokens,
                disable_judge_reasoning=config.semantic_filter_disable_reasoning,
            ))
        tools += [
            ReadDocumentTool(
                self.document_map,
                self._state,
            ),
            PruneTool(self._state),
            *extra_tools,
        ]
        super().__init__(
            tools, max_steps=config.agent_max_steps,
            max_misfires=config.agent_max_misfires,
            cost_budget=config.cost_budget,
            latency_budget=config.latency_budget,
            agent_id=agent_id if agent_id is not None else config.agent_id,
            system_prompt_override=system_prompt_override,
            generation_backend=generation_backend,
            sampling_params=sampling_params,
            capture_logprobs=capture_logprobs,
        )

        # overwrite default usage key for search and sem filter tools now that agent has one
        for tool in self._tools:
            if isinstance(tool, SearchCorpusTool) or isinstance(tool, SemanticFilterTool):
                tool._usage_key = str(self.agent_id)


    # ------------------------------------------------------------------
    # Block rendering / redaction (override the base hooks)
    # ------------------------------------------------------------------

    def _block_is_visible(self, block: Block) -> bool:
        """Redact pruned chunks from the LLM-facing render (full trajectory is kept)."""
        if isinstance(block, ChunkBlock):
            if block.chunk_id is not None and block.chunk_id in self._state.pruned_chunk_ids | self._state.redacted_chunk_ids:
                return False
        if isinstance(block, (ChunkBlock, ImageBlock)):
            if block.doc_id in self._state.pruned_doc_ids | self._state.redacted_doc_ids:
                return False
        return True

    def _make_block_invisible(self, doc_id: str | None = None, chunk_id: str | None = None) -> None:
        """Redact all blocks which have the doc_id or chunk_id."""
        assert doc_id is not None or chunk_id is not None
        if chunk_id:
            self._state.redacted_chunk_ids.add(chunk_id)
        if doc_id:
            self._state.redacted_doc_ids.add(doc_id)

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
            elif not output["read_chunks"] and not output["fetched_chunks"]:
                blocks.append(TextBlock(EMPTY_RESULT_MESSAGE))
            elif output["read_chunks"]:
                blocks.extend(
                    ChunkBlock(chunk_id=c["chunk_id"], doc_id=c["doc_id"], text=c["text"])
                    for c in output["read_chunks"]
                )
            elif output["fetched_chunks"]:
                total_est_num_tokens = sum(chunk["est_num_tokens"] for chunk in output["fetched_chunks"])
                fetch_summary = f"Retrieved {len(output['fetched_chunks'])} chunks with {total_est_num_tokens:,} est. tokens. Here are the top-{self.chunks_per_summary} chunks by est. token count:\n"
                chunks_desc_token_order = sorted(output["fetched_chunks"], key=lambda c: c["est_num_tokens"], reverse=True)
                for chunk in chunks_desc_token_order[:self.chunks_per_summary]:
                    fetch_summary += f" - {chunk['header']}\n"
                blocks.append(TextBlock(fetch_summary))

            return blocks

        if isinstance(output, dict) and output.get(GREP_RESULT_TAG):
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            elif not output["read_groups"] and not output["fetched_groups"]:
                blocks.append(TextBlock(EMPTY_RESULT_MESSAGE))
            elif output["read_groups"]:
                for group in output["read_groups"]:
                    blocks.append(TextBlock(group["header"]))  # TODO: would it be alright to change this to `ChunkBlock(chunk_id=None, doc_id=group["doc_id"], text=group["header"])`? This way it gets removed with prune calls; let's leave as-is though if that would in any way confuse the agent into thinking this is the entire document text
                    if group["read_chunks"]:
                        blocks.extend(
                            ChunkBlock(chunk_id=c["chunk_id"], doc_id=c["doc_id"], text=c["text"])
                            for c in group["read_chunks"]
                        )
                    else:
                        blocks.append(TextBlock("[chunks truncated...]"))
            elif output["fetched_groups"]:
                total_chunks = sum(len(g["fetched_chunks"]) for g in output["fetched_groups"])
                total_groups = len(output["fetched_groups"])
                total_est_num_tokens = sum([c["est_num_tokens"] for g in output["fetched_groups"] for c in g["fetched_chunks"]])
                fetch_summary = f"Retrieved {total_chunks} chunks from {total_groups} documents with {total_est_num_tokens:,} est. tokens. Here are the top-{self.chunks_per_summary} chunks by est. token count:\n"
                chunks_desc_token_order = sorted([c for g in output["fetched_groups"] for c in g["fetched_chunks"]], key=lambda c: c["est_num_tokens"], reverse=True)
                for chunk in chunks_desc_token_order[:self.chunks_per_summary]:
                    fetch_summary += f"{chunk['header']}\n"
                blocks.append(TextBlock(fetch_summary))

            # Surfaced when the output cap dropped hits (visible TextBlock, not redactable).
            if output.get("truncation_note"):
                blocks.append(TextBlock(output["truncation_note"]))
            return blocks

        if isinstance(output, dict) and output.get(SEMFILTER_RESULT_TAG):
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            elif not output.get("summary") and not output["read_chunks"] and not output["fetched_chunks"]:
                blocks.append(TextBlock(EMPTY_RESULT_MESSAGE))
            elif output["summary"] and not output["read_chunks"] and not output["fetched_chunks"]:
                blocks.append(TextBlock(output["summary"]))
            elif output["read_chunks"]:
                if output.get("summary"):
                    blocks.append(TextBlock(output["summary"]))
                blocks.extend(
                    ChunkBlock(chunk_id=c["chunk_id"], doc_id=c["doc_id"], text=c["text"])
                    for c in output["read_chunks"]
                )
            elif output["fetched_chunks"]:
                if output.get("summary"):
                    blocks.append(TextBlock(output["summary"]))
                total_chunks = len(output["fetched_chunks"])
                total_docs = len(set(output["kept_doc_ids"]))
                total_est_num_tokens = sum(c["est_num_tokens"] for c in output["fetched_chunks"])
                fetch_summary = f"Retrieved {total_chunks} chunks from {total_docs} documents with {total_est_num_tokens:,} est. tokens. Here are the top-{self.chunks_per_summary} chunks by est. token count:\n"
                chunks_desc_token_order = sorted(output["fetched_chunks"], key=lambda c: c["est_num_tokens"], reverse=True)
                for chunk in chunks_desc_token_order[:self.chunks_per_summary]:
                    fetch_summary += f"{chunk['header']}\n"
                blocks.append(TextBlock(fetch_summary))

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
                caption = f"[full-page image of doc_id={output['doc_id']}]"
                blocks.append(ImageBlock(
                    doc_id=output["doc_id"],
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
    # Final answer: validate + correct the returned doc_ids
    # ------------------------------------------------------------------

    async def run_with_validated_doc_ids(
        self, ctx: ExecutionContext, user: str, *, correction_steps: int | None = None,
    ) -> tuple[Any, list[str]]:
        """Run the agent, then make sure the `doc_ids` it returned name real documents.

        A returned id is valid when it is a key of `document_map`. This catches failures where
        the agent returns a hallucinated id or (more plausibly) a partial identifier. For example,
        we have observed the model use source ids (e.g. `MICROSOFT_2023_10K`) instead of the
        page-level document id it was shown (`MICROSOFT_2023_10K::p59`). If any id is invalid,
        the agent is re-prompted (resuming the same conversation, so it still sees everything it
        read) to fix its mistake(s) for up to `correction_steps` extra turns. This budget is
        separate from `max_steps`. The final (sub)set of valid `doc_ids` are returned. A failure
        is only raised if the agent returns no valid ids after all correction steps are exhausted.

        Returns `(final_payload, valid_doc_ids)`. The payload is returned untouched (callers
        in "answer" mode still read its `answer` field); only the id list is validated.
        """
        if correction_steps is None:
            correction_steps = self.config.doc_id_correction_steps
        payload = await self.call(ctx, user)
        doc_ids = doc_ids_from_payload(payload)
        for _ in range(correction_steps):
            bad = [d for d in doc_ids if d not in self.document_map]
            if not bad:
                break
            ctx.emit(f"doc_id_correction n_bad={len(bad)} bad={bad!r}", data={"bad": bad})
            payload = await self.call(
                ctx, _doc_id_correction_message(bad), resume=True, max_steps=1
            )
            doc_ids = doc_ids_from_payload(payload)
        valid = [d for d in doc_ids if d in self.document_map]
        dropped = [d for d in doc_ids if d not in self.document_map]
        ctx.emit(
            f"doc_ids_validated kept={len(valid)} dropped={len(dropped)}",
            data={"kept": valid, "dropped": dropped},
        )
        if not valid:
            raise StepFailed(
                self.name,
                "no well-formed doc_ids after correction",
                diagnostic=f"agent returned only unrecognized doc_ids: {doc_ids!r}",
            )
        return payload, valid


def _doc_id_correction_message(bad: list[str]) -> str:
    """The follow-up shown to the agent when some returned doc_ids name no real document.
    Written as plain sentences because the model reads it directly."""
    return (
        "Some of the document identifiers you listed do not match any document in the "
        f"collection: {bad}. This usually happens when an identifier is missing part of its "
        "form, such as the page it refers to. Please look back through the search, grep, and "
        "read results earlier in this conversation, find each identifier exactly as it was "
        "written there, and return your full list again under the \"doc_ids\" key. Every "
        "identifier must be copied exactly as it appeared in those results."
    )


def doc_ids_from_payload(payload: Any) -> list[str]:
    """The `doc_ids` list out of a SearchAgent final-answer payload, coerced to
    strings; [] for a malformed payload (non-dict, or a missing/empty key). A bare
    string value is treated as a single id. Each id is whitespace-trimmed, empties
    are dropped, and duplicates are removed preserving first-seen order, so callers
    can validate the ids without tripping over cosmetic differences. The one place
    the final-answer shape (`final_answer_doc`) is decoded — callers compose their
    own user message, `await agent.call(ctx, msg)`, and decode with this."""
    if not isinstance(payload, dict):
        return []
    keys = payload.get("doc_ids") or []
    if isinstance(keys, str):
        keys = [keys]
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        s = str(k).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out
