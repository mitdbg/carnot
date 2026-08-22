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

import json
import re
import time
from typing import Any
from chromadb.api.models.Collection import Collection
from jinja2 import Environment, StrictUndefined

from skunk.common import B64Image, ExecutionContext, strip_code_fence
from skunk.config import SearchAgentConfig
from skunk.errors import ParseError, StepFailed
from skunk.llm_client import LLMClient
from skunk.sandbox.local_python_executor import CodeOutput
from skunk.multi_turn_agent import Block, ChunkBlock, ImageBlock, MultiTurnAgent, TextBlock, Tool
from skunk.prompted_call import PromptedCall
from skunk.prompts import load_prompts
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
from skunk.search_state.working_set import WorkingSet
from skunk.search_state.working_set_registry import WorkingSetRegistry, WS_PREFIX, DEFAULT_CHROMA_PATH
from skunk.storage.document_map import DocumentMap

_ENV = Environment(
    autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined
)

_PROMPTS = load_prompts("search_agent")

_JSON_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_]*)\n(.*?)```", re.DOTALL)


def _parse_list(text: str, _: ExecutionContext) -> list[str]:
    """Parse the model's reply as a JSON array of strings, raising `ParseError` on
    anything else so `PromptedCall` re-prompts with the failure echoed back.

    Accepts a bare array or one wrapped in a ```json``` fence (what the selector prompts
    ask for). When the reply wraps prose around the fence, the LAST fenced block wins —
    models that think out loud tend to rehearse candidate lists before committing to the
    final one. Non-string entries are rejected rather than coerced: callers feed these
    straight into `WorkingSetRegistry.get()`, where a stringified int is a silent miss.
    """
    fences = _JSON_FENCE_RE.findall(text)
    body = fences[-1].strip() if fences else strip_code_fence(text)
    if not body:
        raise ParseError(
            raw=text,
            detail="Your reply was empty. Emit a single ```json``` block holding a JSON "
            'array of the ids you selected, e.g. ["id1", "id2"], or [] if none apply.',
        )
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as e:
        raise ParseError(
            raw=text, detail=f"the JSON array was malformed — {e}"
        ) from e
    if not isinstance(parsed, list):
        raise ParseError(
            raw=text,
            detail=f"expected a JSON array, got {type(parsed).__name__}. Emit ONE "
            '```json``` block holding a flat array of strings, e.g. ["id1", "id2"].',
        )
    non_strings = [v for v in parsed if not isinstance(v, str)]
    if non_strings:
        raise ParseError(
            raw=text,
            detail=f"every array entry must be a string; got {non_strings!r}. Emit a flat "
            'array of ids, e.g. ["id1", "id2"] — no nested objects or arrays.',
        )
    return parsed


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
        registry: WorkingSetRegistry | None = None,
        extra_tools: tuple[Tool, ...] = (),
        briefing: str | None = None,
        final_answer_doc: str | None = None,
        agent_id: str | None = None,
        system_prompt_override: str | None = None,
        working_set_name: str | None = None,
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
        self._registry = registry or WorkingSetRegistry(chroma_path=DEFAULT_CHROMA_PATH)

        # Bound each search-step LLM call: cap output (was uncapped → runaway
        # generations streamed to the 65535-token ceiling at 200–800s each) and
        # impose a hard per-request wall-clock timeout. See SearchAgentConfig for the
        # thinking/max_output_tokens interaction caveat.
        self.max_output_tokens = config.search_agent_max_output_tokens
        self.request_timeout_s = config.search_agent_request_timeout_s
        self.extra_tools = extra_tools
        super().__init__(
            tools=[],  # NOTE: we construct tools at runtime based on ExecutionContext configuration
            max_steps=config.agent_max_steps,
            max_misfires=config.agent_max_misfires,
            cost_budget=config.cost_budget,
            latency_budget=config.latency_budget,
            agent_id=agent_id if agent_id is not None else config.agent_id,
            system_prompt_override=system_prompt_override,
            generation_backend=generation_backend,
            sampling_params=sampling_params,
            capture_logprobs=capture_logprobs,
        )

        # create (or get) working set for agent
        working_set_name = f"{WS_PREFIX}{working_set_name if working_set_name else self.agent_id}"
        self._working_set = self._registry.get_or_create(name=working_set_name)

    # ------------------------------------------------------------------
    # Block rendering / redaction (override the base hooks)
    # ------------------------------------------------------------------

    def _block_is_visible(self, block: Block) -> bool:
        """Redact pruned chunks from the LLM-facing render (full trajectory is kept)."""
        if isinstance(block, ChunkBlock):
            if block.chunk_id is not None and block.chunk_id in self._working_set.pruned_chunk_ids | self._working_set.redacted_chunk_ids:
                return False
        if isinstance(block, (ChunkBlock, ImageBlock)):
            if block.doc_id in self._working_set.pruned_doc_ids | self._working_set.redacted_doc_ids:
                return False
        return True

    def _make_block_invisible(self, doc_id: str | None = None, chunk_id: str | None = None) -> None:
        """Redact all blocks which have the doc_id or chunk_id."""
        assert doc_id is not None or chunk_id is not None
        if chunk_id:
            self._working_set.redacted_chunk_ids.add(chunk_id)
        if doc_id:
            self._working_set.redacted_doc_ids.add(doc_id)

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

    def _build_tools(self, ctx: ExecutionContext) -> list[Tool]:
        """Build the set of tools used by the SearchAgent at runtime."""
        tools: list[Tool] = [
            ReadDocumentTool(
                self.document_map,
                self._working_set,
                id_tracking_off=ctx.config.search.id_tracking_off,
            ),
            PruneTool(self._working_set),
            *self.extra_tools,
        ]

        if ctx.config.search.include_search_corpus:
            tools.append(SearchCorpusTool(
                self.chroma_collection,
                self._llm_client,
                self._working_set,
                ctx,
                usage_key=self.agent_id,
                working_set_collection_off=ctx.config.search.working_set_collection_off,
                id_tracking_off=ctx.config.search.id_tracking_off,
            ))

        if ctx.config.search.include_grep_corpus:
            tools.append(GrepCorpusTool(
                self.chroma_collection,
                self._working_set,
                working_set_collection_off=ctx.config.search.working_set_collection_off,
                id_tracking_off=ctx.config.search.id_tracking_off,
            ))

        if ctx.config.search.include_semantic_filter:
            tools.append(SemanticFilterTool(
                self.chroma_collection,
                self._llm_client,
                self.document_map,
                self._working_set,
                ctx.config.search.semantic_filter_model,
                ctx=ctx,
                provider_order=ctx.config.search.semantic_filter_provider_order,
                judge_max_output_tokens=ctx.config.search.semantic_filter_max_output_tokens,
                disable_judge_reasoning=ctx.config.search.semantic_filter_disable_reasoning,
                usage_key=self.agent_id,
                working_set_collection_off=ctx.config.search.working_set_collection_off,
                id_tracking_off=ctx.config.search.id_tracking_off,
            ))

        if ctx.config.storage.pdf_dir is not None:
            tools.append(ViewFigureTool(
                self.document_map, ctx.config.storage.pdf_dir, renders_dir=ctx.config.storage.page_renders_dir,
            ))

        return tools

    async def _find_related_working_sets(self, ctx: ExecutionContext, user: str) -> list[WorkingSet]:
        """Search the registry for existing WorkingSet(s) which may be useful for answering the user query."""
        # TODO: eventually, subsample working set summaries; but for now throw them all into a prompt
        all_working_sets: list[WorkingSet] = [ws for ws in self._registry if ws.id != self._working_set.id]
        if len(all_working_sets) == 0:
            return []

        # for now, working set summaries consist of the actions used to construct them; because of the
        # incremental nature of our working sets, this requires pulling all fetch operations from ancestors
        working_set_summaries: list[dict] = []
        for ws in all_working_sets:
            ancestors = self._registry.ancestors_of(ws)
            fetch_actions = [action for action, is_fetch in ws.actions if is_fetch]
            for ancestor in ancestors:
                ancestor_fetch_actions = [action for action, is_fetch in ancestor.actions if is_fetch]
                if ancestor_fetch_actions:
                    fetch_actions = ancestor_fetch_actions + fetch_actions
            working_set_summaries.append({"id": ws.id, "actions": fetch_actions})

        # construct prompt and ask llm for list of relevant working set ids
        relevant_working_sets_system_template = _PROMPTS["relevant_working_set_selector_system_prompt"]
        relevant_working_sets_system_prompt = _ENV.from_string(relevant_working_sets_system_template).render(
            working_set_summaries=working_set_summaries,
        )
        relevant_working_sets_user_template = _PROMPTS["relevant_working_set_selector_user_prompt"]
        relevant_working_sets_user_prompt = _ENV.from_string(relevant_working_sets_user_template).render(
            user=user
        )
        _prompt = PromptedCall(
            name=self.name,
            system_prompt=relevant_working_sets_system_prompt,
            default_effort=self.default_effort,
            parse=_parse_list,
            max_parse_retries=self.max_recover_retries,
        )
        relevant_working_set_ids = await _prompt.call(ctx, user=relevant_working_sets_user_prompt, usage_key=self.agent_id)

        return [self._registry.get(id) for id in relevant_working_set_ids if self._registry.contains(id)]

    def _add_working_set_message(self) -> None:
        """Add a message to the agent's context informing it of the working set(s) at its disposal."""
        ancestors = self._registry.ancestors_of(self._working_set)
        working_set_summary = self._working_set.to_message(ancestors)
        message = f"{_PROMPTS['working_set']}\n\n{working_set_summary}\n\n"
        self.messages.append({"role": "system", "blocks": [TextBlock(message)]})

    # ------------------------------------------------------------------
    # Final answer: validate + correct the returned doc_ids
    # ------------------------------------------------------------------

    async def call(
        self, ctx: ExecutionContext, user: str, *, correction_steps: int | None = None, **_
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
        # rename booleans to make logic more readable
        working_set_collection_on = not ctx.config.search.working_set_collection_off
        id_tracking_on = not ctx.config.search.id_tracking_off

        # if configured: retrieve and union any working sets that are relevant to the query
        t0 = time.monotonic()
        if ctx.config.search.fetch_related_working_sets:
            working_sets = await self._find_related_working_sets(ctx, user)
            self._working_set.add_parents(working_sets)
            t1 = time.monotonic()
            ctx.emit(
                f"_find_related_working_set n_found={len(working_sets)} found_ids={[ws.id for ws in working_sets]} time={t1 - t0:.3f}",
                kind="_find_related_working_set",
                data={"n_found": len(working_sets), "found_ids": [ws.id for ws in working_sets], "time": t1 - t0},
            )

            # TODO: remove this and update search tools to query ancestors through working set .query() and .get()
            for ancestor in working_sets:
                # copy data
                if working_set_collection_on:
                    results = ancestor.collection.get(include=["documents", "metadatas", "embeddings"])
                    if results["ids"]:
                        self._working_set.collection.upsert(
                            ids=results["ids"],
                            embeddings=results["embeddings"],
                            metadatas=results["metadatas"],
                            documents=results["documents"],
                        )

                # union fetched chunk/doc ids
                self._working_set.fetched_chunk_ids = self._working_set.fetched_chunk_ids.union(ancestor.fetched_chunk_ids)
                self._working_set.fetched_doc_ids = self._working_set.fetched_doc_ids.union(ancestor.fetched_doc_ids)

            t2 = time.monotonic()
            ctx.emit(
                f"_copy_ancestor_working_sets time={t2 - t1:.3f}",
                kind="_copy_ancestor_working_sets",
                data={"time": t2 - t1},
            )

        # add message summarizing the state of the working set
        if working_set_collection_on or id_tracking_on:
            self._add_working_set_message()

        # construct the tools, then rebuild the system prompt (it splices in the tool docs)
        # and the local python executor from them
        self._tools = self._build_tools(ctx)
        self._rebuild_prompt()
        self._executor = self._build_executor()

        # run the agent on the user query
        payload = await super().call(ctx, user)

        # TODO: compute the working set's summary
        # NOTE: can be done off the critical path
        if working_set_collection_on:
            self._working_set.compute_summary()

        # parse the doc_ids from the agent output
        doc_ids = doc_ids_from_payload(payload)

        # reprompt the agent to correct any mistakes if there is an issue parsing the doc_ids
        if correction_steps is None:
            correction_steps = self.config.doc_id_correction_steps
        for _ in range(correction_steps):
            bad = [d for d in doc_ids if d not in self.document_map]
            if not bad:
                break
            ctx.emit(f"doc_id_correction n_bad={len(bad)} bad={bad!r}", data={"bad": bad})
            payload = await super().call(
                ctx, _doc_id_correction_message(bad), resume=True, max_steps=1
            )
            doc_ids = doc_ids_from_payload(payload)
        valid = [d for d in doc_ids if d in self.document_map]
        dropped = [d for d in doc_ids if d not in self.document_map]
        ctx.emit(
            f"doc_ids_validated kept={len(valid)} dropped={len(dropped)}",
            data={"kept": valid, "dropped": dropped},
        )

        # persist the computed working set back into the registry
        self._working_set.persist()
        self._registry.update(self._working_set.id, self._working_set)

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
