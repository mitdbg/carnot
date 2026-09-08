"""SearchAgent — iterative code-execution retriever.

Subclasses `MultiTurnAgent`: it supplies the search-specific system prompt
and the chroma-backed tool set (`search_corpus` / `grep_corpus` / `semantic_filter`
/ `read_document` / `prune`). The tools share the agent's `WorkingSet`.

`_blocks_from_output()` turns the tools' structured (tagged-dict) returns into
`ChunkBlock`s (one per chunk, carrying chunk_id / doc_id) so the full trajectory
is preserved (useful for metrics and future reward computation).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any
from chromadb.api.models.Collection import Collection
from jinja2 import Environment, StrictUndefined

from skunk.agents.multi_turn_agent import (
    Block,
    ChunkBlock,
    ImageBlock,
    Message,
    MultiTurnAgent,
    StepOutput,
    TextBlock,
    Tool,
    parse_step,
)
from skunk.agents.search_agent.search_tools import (
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
from skunk.common import B64Image, ExecutionContext, strip_code_fence
from skunk.config import SearchAgentConfig, StorageConfig
from skunk.errors import ParseError, StepFailed
from skunk.llm_client import LLMClient
from skunk.prompts import load_prompts
from skunk.sandbox.local_python_executor import CodeOutput, BASE_BUILTIN_MODULES
from skunk.search_state.working_set import WorkingSet
from skunk.search_state.working_set_registry import WorkingSetRegistry, WS_PREFIX, DEFAULT_CHROMA_PATH
from skunk.storage.document_map import DocumentMap

_ENV = Environment(
    autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined
)
_PROMPTS = load_prompts("search_agent")

_JSON_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_]*)\n(.*?)```", re.DOTALL)


def _parse_list(text: str) -> list[str]:
    """Parse the model's reply as a JSON array of strings, raising `ParseError` on
    anything else so the agent can retry.

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
            detail="Your reply was empty. Emit a single ```json``` block holding a JSON "
            'array of the ids you selected, e.g. ["id1", "id2"], or [] if none apply.',
        )
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as e:
        raise ParseError(
            detail=f"the JSON array was malformed — {e}"
        ) from e
    if not isinstance(parsed, list):
        raise ParseError(
            detail=f"expected a JSON array, got {type(parsed).__name__}. Emit ONE "
            '```json``` block holding a flat array of strings, e.g. ["id1", "id2"].',
        )
    non_strings = [v for v in parsed if not isinstance(v, str)]
    if non_strings:
        raise ParseError(
            detail=f"every array entry must be a string; got {non_strings!r}. Emit a flat "
            'array of ids, e.g. ["id1", "id2"] — no nested objects or arrays.',
        )
    return parsed


class SearchAgent(MultiTurnAgent):
    """
    Preconditions:
    - corpus is divided into documents with unique `doc_id`; documents are further split into chunks with unique `chunk_id`
    """

    @staticmethod
    def parse_step(text: str) -> StepOutput:
        """Apply the basic answer parser and validate that a final answer has a `doc_ids` field."""
        step_output = parse_step(text)

        if step_output.is_final and "doc_ids" not in step_output.result:
            raise ParseError(
                detail='Final answer must be a JSON object with a single "doc_ids" key, e.g. {"doc_ids": ["...", ...]}.'
            )

        return step_output

    def __init__(
        self,
        config: SearchAgentConfig,
        document_map: DocumentMap,
        chroma_collection: Collection,
        llm_client: LLMClient,
        storage_config: StorageConfig,
        *,
        agent_id: str | None = None,
        registry: WorkingSetRegistry | None = None,
        working_set_name: str | None = None,
        additional_notes: str | None = None,
    ):
        # override the default agent_id if one is provided
        config.agent_id = config.agent_id if agent_id is None else agent_id

        # set variables
        self.document_map = document_map
        self.chroma_collection = chroma_collection
        self._llm_client = llm_client
        self._registry = registry or WorkingSetRegistry(chroma_path=DEFAULT_CHROMA_PATH)

        # create (or get) working set for agent
        working_set_name = f"{WS_PREFIX}{working_set_name if working_set_name else config.agent_id}"
        self._working_set = self._registry.get_or_create(name=working_set_name)

        # construct the SearchAgent's tools
        # NOTE: if we ever stray from this approach 
        tools = self._build_tools(config, storage_config)

        # construct the system prompt
        system_prompt_template = _PROMPTS["system_prompt"]
        system_prompt = _ENV.from_string(system_prompt_template).render(
            max_steps=config.max_steps,
            authorized_imports=list(set(BASE_BUILTIN_MODULES) | set(config.authorized_imports)),
            tools="\n\n".join(t.doc for t in tools),
            cost_budget=config.cost_budget,
            latency_budget=config.latency_budget,
            additional_notes=additional_notes,
        )

        # construct the terminal prompt
        terminal_prompt_template = _PROMPTS["terminal_prompt"]
        terminal_prompt = _ENV.from_string(terminal_prompt_template).render()

        # initialize the agent
        super().__init__(config, tools=tools, system_prompt=system_prompt, terminal_prompt=terminal_prompt, parse=SearchAgent.parse_step)
        self.config: SearchAgentConfig

    def _resolve_semantic_filter_llm_model(self) -> str:
        """Model to use for semantic filter judgements; precedence is:
        1. SearchAgentConfig.semantic_filter_llm_model
        2. SearchAgentConfig.llm_model
        3. LLMClient.config.llm_model
        """
        if self.config.semantic_filter_llm_model:
            return self.config.semantic_filter_llm_model
        elif self.config.llm_model:
            return self.config.llm_model

        return self._llm_client.config.llm_model

    def _blocks_from_output(self, ctx: ExecutionContext, out: CodeOutput) -> list[Block]:
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
                summary_num_chunks = min(self.config.chunks_per_summary, len(output["fetched_chunks"]))
                total_est_num_tokens = sum(chunk["est_num_tokens"] for chunk in output["fetched_chunks"])
                fetch_summary = f"Retrieved {len(output['fetched_chunks'])} chunks with {total_est_num_tokens:,} est. tokens. Here are the top-{summary_num_chunks} chunks by similarity (smallest distance):\n"
                chunks_asc_distance = sorted(output["fetched_chunks"], key=lambda c: c["distance"])
                for chunk in chunks_asc_distance[:summary_num_chunks]:
                    fetch_summary += f" - {chunk['header']}\n"
                blocks.append(TextBlock(fetch_summary))

            ctx.tracer.emit(
                id="search_corpus_tool_call",
                kind="tool_call",
                step=self._step,
                turn=self._turn,
                data={
                    "tool": output["tool"],
                    "tool_kwargs": output["tool_kwargs"],
                    "read_chunks": [{k: v for k, v in chunk.items() if k not in ["header", "text"]} for chunk in output["read_chunks"]],
                    "fetched_chunks": [{k: v for k, v in chunk.items() if k not in ["header", "text"]} for chunk in output["fetched_chunks"]],
                    "error": output.get("error"),
                },
            )

        elif isinstance(output, dict) and output.get(GREP_RESULT_TAG):
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            elif not output["read_groups"] and not output["fetched_groups"]:
                blocks.append(TextBlock(EMPTY_RESULT_MESSAGE))
            elif output["read_groups"]:
                for group in output["read_groups"]:
                    blocks.append(TextBlock(group["header"]))  # TODO: would it be alright to change this to `ChunkBlock(chunk_id=None, doc_id=group["doc_id"], text=group["header"])`? This way it gets removed with prune calls; let's leave as-is though if that would in any way confuse the agent into thinking this is the entire document text
                    if group["chunks"]:
                        blocks.extend(
                            ChunkBlock(chunk_id=c["chunk_id"], doc_id=c["doc_id"], text=c["text"])
                            for c in group["chunks"]
                        )
                    else:
                        blocks.append(TextBlock("[chunks truncated...]"))
            elif output["fetched_groups"]:
                total_chunks = sum(len(g["chunks"]) for g in output["fetched_groups"])
                total_groups = len(output["fetched_groups"])
                total_est_num_tokens = sum([c["est_num_tokens"] for g in output["fetched_groups"] for c in g["chunks"]])
                summary_num_chunks = min(self.config.chunks_per_summary, total_chunks)
                fetch_summary = f"Retrieved {total_chunks} chunks from {total_groups} documents with {total_est_num_tokens:,} est. tokens. Here are the top-{summary_num_chunks} chunks by similarity (smallest distance):\n"
                chunks_asc_distance = sorted([c for g in output["fetched_groups"] for c in g["chunks"]], key=lambda c: c["distance"])
                for chunk in chunks_asc_distance[:summary_num_chunks]:
                    fetch_summary += f"{chunk['header']}\n"
                blocks.append(TextBlock(fetch_summary))

            # Surfaced when the output cap dropped hits (visible TextBlock, not redactable).
            if output.get("truncation_note"):
                blocks.append(TextBlock(output["truncation_note"]))

            ctx.tracer.emit(
                id="grep_corpus_tool_call",
                kind="tool_call",
                step=self._step,
                turn=self._turn,
                data={
                    "tool": output["tool"],
                    "tool_kwargs": output["tool_kwargs"],
                    "read_groups": [
                        {k: v for k, v in chunk.items() if k not in ["header", "text"]}
                        for group in output["read_groups"]
                        for chunk in group["chunks"]
                    ],
                    "fetched_groups": [
                        {k: v for k, v in chunk.items() if k not in ["header", "text"]}
                        for group in output["fetched_groups"]
                        for chunk in group["chunks"]
                    ],
                    "error": output.get("error"),
                    "truncation_note": output.get("truncation_note"),
                },
            )

        elif isinstance(output, dict) and output.get(SEMFILTER_RESULT_TAG):
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
                summary_num_chunks = min(self.config.chunks_per_summary, total_chunks)
                fetch_summary = f"Retrieved {total_chunks} chunks from {total_docs} documents with {total_est_num_tokens:,} est. tokens. Here are the top-{summary_num_chunks} chunks by similarity (smallest distance):\n"
                chunks_asc_distance = sorted(output["fetched_chunks"], key=lambda c: c["distance"])
                for chunk in chunks_asc_distance[:summary_num_chunks]:
                    fetch_summary += f"{chunk['header']}\n"
                blocks.append(TextBlock(fetch_summary))

            ctx.tracer.emit(
                id="sem_filter_tool_call",
                kind="tool_call",
                step=self._step,
                turn=self._turn,
                data={
                    "tool": output["tool"],
                    "tool_kwargs": output["tool_kwargs"],
                    "mode": output.get("mode"),
                    "summary": output.get("summary"),
                    "read_chunks": [{k: v for k, v in chunk.items() if k not in ["header", "text"]} for chunk in output.get("read_chunks", [])],
                    "fetched_chunks": [{k: v for k, v in chunk.items() if k not in ["header", "text"]} for chunk in output.get("fetched_chunks", [])],
                    "kept_doc_ids": output.get("kept_doc_ids"),
                    "rejected_doc_ids": output.get("rejected_doc_ids"),
                    "error": output.get("error"),
                },
            )

        elif isinstance(output, dict) and output.get(READ_DOCUMENT_RESULT_TAG):
            blocks.extend(
                ChunkBlock(chunk_id=None, doc_id=d["doc_id"], text=d["text"])
                for d in output["docs"]
            )

            ctx.tracer.emit(
                id="read_document_tool_call",
                kind="tool_call",
                step=self._step,
                turn=self._turn,
                data={
                    "tool": output["tool"],
                    "tool_kwargs": output["tool_kwargs"],
                    "doc_ids": [doc["doc_id"] for doc in output["docs"]],
                },
            )

        elif isinstance(output, dict) and output.get(VIEW_FIGURE_RESULT_TAG):
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            else:
                caption = f"[full-page image of doc_id={output['doc_id']}]"
                blocks.append(ImageBlock(
                    doc_id=output["doc_id"],
                    image=B64Image(mime=output["mime"], data=output["data"]),
                    text=caption,
                ))

            ctx.tracer.emit(
                id="view_figure_tool_call",
                kind="tool_call",
                step=self._step,
                turn=self._turn,
                data={
                    "tool": output["tool"],
                    "tool_kwargs": output["tool_kwargs"],
                    "doc_id": output.get("doc_id"),
                    "mime": output.get("mime"),
                    "error": output.get("error"),
                },
            )

        elif isinstance(output, dict) and output.get(PRUNE_RESULT_TAG):
            # modify the block visibility for any chunks / docs that were pruned
            # NOTE: we do not need to change visibility for any other tool calls because they
            #       will append new blocks with the data they read / fetched; un-redacting their
            #       previous blocks in the message history will only take up more space and mess
            #       with the prefix cache
            self._make_blocks_invisible(doc_ids=output["new_pruned_doc_ids"], chunk_ids=output["new_pruned_chunk_ids"])
            blocks.append(TextBlock(
                f"[result]\nPruned {output['new_chunk_count']} chunk(s) and "
                f"{output['new_doc_count']} doc(s). {output['total_chunks']} chunk(s) and "
                f"{output['total_docs']} doc(s) are now excluded from future searches."
            ))

            ctx.tracer.emit(
                id="prune_tool_call",
                kind="tool_call",
                step=self._step,
                turn=self._turn,
                data={
                    "tool": output["tool"],
                    "tool_kwargs": output["tool_kwargs"],
                    "new_pruned_doc_ids": output["new_pruned_doc_ids"],
                    "new_pruned_chunk_ids": output["new_pruned_chunk_ids"],
                },
            )

        else:
            # non-structured output: default [result] rendering (mirrors the base).
            result_s = "" if output is None else str(output).strip()
            if result_s and result_s != stdout_s and result_s not in stdout_s:
                blocks.append(TextBlock(f"[result]\n{result_s}"))
            if not blocks:
                blocks.append(TextBlock("[no output]"))

        return blocks

    def _build_tools(self, config: SearchAgentConfig, storage_config: StorageConfig) -> list[Tool]:
        """Build the set of tools used by the SearchAgent."""
        tools: list[Tool] = [
            ReadDocumentTool(
                self.document_map,
                self._working_set,
                id_tracking_off=config.id_tracking_off,
            ),
            PruneTool(self._working_set),
        ]

        if config.include_search_corpus:
            tools.append(SearchCorpusTool(
                self.chroma_collection,
                self._llm_client,
                self._working_set,
                usage_key=config.agent_id,
                working_set_collection_off=config.working_set_collection_off,
                id_tracking_off=config.id_tracking_off,
            ))

        if config.include_grep_corpus:
            tools.append(GrepCorpusTool(
                self.chroma_collection,
                self._working_set,
                working_set_collection_off=config.working_set_collection_off,
                id_tracking_off=config.id_tracking_off,
            ))

        if config.include_semantic_filter:
            tools.append(SemanticFilterTool(
                self.chroma_collection,
                self._llm_client,
                self.document_map,
                self._working_set,
                config,
                self._resolve_semantic_filter_llm_model(),
                usage_key=config.agent_id,
                working_set_collection_off=config.working_set_collection_off,
                id_tracking_off=config.id_tracking_off,
            ))

        if storage_config.pdf_dir is not None:
            tools.append(ViewFigureTool(
                self.document_map, storage_config.pdf_dir, renders_dir=storage_config.page_renders_dir,
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
        messages = [
            {"role": "system", "content": relevant_working_sets_system_prompt},
            {"role": "user", "content": relevant_working_sets_user_prompt},
        ]
        relevant_working_set_ids: list[str] = []
        try:
            resp = await ctx.llm_client.acall(
                messages=messages,
                model=self._resolve_model(ctx),
                temperature=self.config.temperature,
                effort=self._resolve_effort(ctx),
                call_site=self.config.name,
                max_output_tokens=self.config.max_output_tokens,
                timeout_s=self.config.request_timeout_s,
                usage_key=self.agent_id,
            )
            relevant_working_set_ids = _parse_list(resp.text)
        except ParseError as e:
            ctx.tracer.emit(id="find_related_working_sets", level="error", kind="background", message="ParseError when fetching related working sets", data={"error": e.detail})
            relevant_working_set_ids = []
        except Exception as e:
            ctx.tracer.emit(id="find_related_working_sets", level="error", kind="background", message="Unexpected error when fetching related working sets", data={"error": f"{type(e).__name__}: {e}"})
            relevant_working_set_ids = []

        return [self._registry.get(id) for id in relevant_working_set_ids if self._registry.contains(id)]

    def _add_working_set_message(self) -> None:
        """Add a message to the agent's context informing it of the working set(s) at its disposal."""
        ancestors = self._registry.ancestors_of(self._working_set)
        working_set_summary = self._working_set.to_message(ancestors)
        message = f"{_PROMPTS['working_set']}\n\n{working_set_summary}\n\n"
        self._messages.append(Message(role="system", blocks=[TextBlock(message)]))

    # ------------------------------------------------------------------
    # Final answer: validate + correct the returned doc_ids
    # ------------------------------------------------------------------

    async def call(
        self, ctx: ExecutionContext, input: str, *, correction_steps: int | None = None, **_
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
            working_sets = await self._find_related_working_sets(ctx, input)
            self._working_set.add_parents(working_sets)
            t1 = time.monotonic()
            ctx.tracer.emit(
                id="find_related_working_sets",
                kind="background",
                data={
                    "n_found": len(working_sets),
                    "found_ids": [ws.id for ws in working_sets],
                    "time": t1 - t0,
                }
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
            ctx.tracer.emit(
                id="copy_ancestor_working_sets",
                kind="background",
                data={"time": t2 - t1},
            )

        # add message summarizing the state of the working set
        if working_set_collection_on or id_tracking_on:
            self._add_working_set_message()

        # run the agent on the user query
        payload = await super().call(ctx, input)

        # TODO: compute the working set's summary
        # NOTE: can be done off the critical path
        if working_set_collection_on:
            self._working_set.compute_summary()

        # parse the doc_ids from the agent output
        doc_ids = doc_ids_from_payload(payload)

        # reprompt the agent to correct any mistakes if there is an issue parsing the doc_ids
        if correction_steps is None:
            correction_steps = self.config.doc_id_correction_steps
        for turn in range(correction_steps):
            bad = [d for d in doc_ids if d not in self.document_map]
            if not bad:
                break
            ctx.tracer.emit(id="doc_id_correction", kind="lifecycle", step=self._step, turn=self._turn + turn, data={"bad": bad})
            payload = await super().call(
                ctx, _doc_id_correction_message(bad), resume=True, max_steps=1
            )
            doc_ids = doc_ids_from_payload(payload)
        valid = [d for d in doc_ids if d in self.document_map]
        dropped = [d for d in doc_ids if d not in self.document_map]
        ctx.tracer.emit(
            id="doc_ids_validated",
            kind="lifecycle",
            data={"kept": valid, "dropped": dropped},
        )

        # persist the computed working set back into the registry
        self._working_set.persist()
        self._registry.update(self._working_set.id, self._working_set)

        if not valid:
            raise StepFailed(
                self.config.name,
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
