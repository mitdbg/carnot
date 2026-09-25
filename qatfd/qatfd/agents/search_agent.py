from __future__ import annotations

from chromadb.api import ClientAPI
from jinja2 import Environment, StrictUndefined

from skunk.agents.multi_turn_agent import (
    Block,
    ChunkBlock,
    ImageBlock,
    MultiTurnAgent,
    StepOutput,
    TextBlock,
    Tool,
    parse_step,
)
from skunk.agents.search_agent.search_agent import doc_ids_from_payload, _doc_id_correction_message
from skunk.common import B64Image, ExecutionContext, PageLocator
from skunk.config import StorageConfig
from skunk.errors import ParseError, StepFailed
from skunk.llm_client import LLMClient
from skunk.sandbox.local_python_executor import CodeOutput, BASE_BUILTIN_MODULES
from skunk.storage.document_map import DocumentMap

from qatfd.config import QATFDSearchAgentConfig
from qatfd.constants import METADATA_LIST_DELIMITER
from qatfd.prompts import load_qatfd_prompts
from qatfd.tools import (
    managed_collection_metadata,
    GrepCorpusTool,
    PruneTool,
    ReadDocumentTool,
    SearchCorpusTool,
    SearchResult,
    SemanticFilterTool,
    ViewFigureTool,
)

_ENV = Environment(
    autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined
)
_SA_PROMPTS = load_qatfd_prompts("search_agent")
_WS_PROMPTS = load_qatfd_prompts("working_set")

# message for when grep or semantic filter returns an empty result
EMPTY_RESULT_MESSAGE = "No results found."

# chromadb imposes a limit of 100 collections per-call to list_collections()
_LIST_LIMIT = 100
# limit the number of embeddings in each upsert
_UPSERT_STEP = 256

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
        config: QATFDSearchAgentConfig,
        document_map: DocumentMap,
        chroma_client: ClientAPI,
        llm_client: LLMClient,
        storage_config: StorageConfig,
        *,
        agent_id: str | None = None,
        working_set_collection_on: bool = False,
        collection_name: str | None = None,
        additional_notes: str | None = None,
        page_locator: PageLocator | None = None,
    ):
        # override the default agent_id if one is provided
        config.agent_id = config.agent_id if agent_id is None else agent_id

        # set variables
        self.document_map = document_map
        self.chroma_client = chroma_client
        self.page_locator = page_locator
        self._llm_client = llm_client

        # get or create collection for agent
        collection_name = collection_name if collection_name else config.agent_id
        # metadata is given at creation so no sibling ever sees the collection with metadata=None
        self._collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata=managed_collection_metadata(
                description=f"Working set of search agent {config.agent_id}: the chunks it searched and fetched while answering its question.",
                created_by=config.agent_id,
                created_by_agent_type="SearchAgent",
            ),
        )

        # construct the SearchAgent's tools
        tools = self._build_tools(config, storage_config)

        # construct the system prompt
        objective_prompt_template = (
            _SA_PROMPTS["objective_with_working_sets"]
            if working_set_collection_on
            else _SA_PROMPTS["objective_without_working_sets"]
        )
        objective_prompt = _ENV.from_string(objective_prompt_template).render(max_steps=config.max_steps)
        system_prompt_template = _SA_PROMPTS["system_prompt"]
        system_prompt = _ENV.from_string(system_prompt_template).render(
            objective=objective_prompt,
            max_steps=config.max_steps,
            authorized_imports=list(set(BASE_BUILTIN_MODULES) | set(config.authorized_imports)),
            tools="\n\n".join(t.doc for t in tools),
            cost_budget=config.cost_budget,
            latency_budget=config.latency_budget,
            additional_notes=additional_notes,
        )

        # construct the terminal prompt
        terminal_prompt_template = _SA_PROMPTS["terminal_prompt"]
        terminal_prompt = _ENV.from_string(terminal_prompt_template).render()

        # initialize the agent
        super().__init__(config, tools=tools, system_prompt=system_prompt, terminal_prompt=terminal_prompt, parse=SearchAgent.parse_step)
        self.config: QATFDSearchAgentConfig

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

    def _update_collection_actions(self, tool: str, tool_kwargs: dict) -> None:
        """Add the action to the collection's metadata."""
        # reconstruct string for the tool call
        kwargs_str = ""
        for key, arg in tool_kwargs.items():
            kwargs_str += f"{key}='{arg}', " if isinstance(arg, str) else f"{key}={arg}, "
        kwargs_str = kwargs_str[:-2]
        tool_call = f"{tool}({kwargs_str})"

        # get existing actions
        metadata = self._collection.metadata
        actions = metadata.get("actions", "")

        # add action to metadata
        action_items = actions.split(METADATA_LIST_DELIMITER)
        action_items.append(tool_call)
        metadata["actions"] = METADATA_LIST_DELIMITER.join(action_items)
        self._collection.modify(metadata=metadata)

    def _build_tools(self, config: QATFDSearchAgentConfig, storage_config: StorageConfig) -> list[Tool]:
            """Build the set of tools used by the SearchAgent."""
            tools: list[Tool] = [
                ReadDocumentTool(self.document_map),
                PruneTool(),
            ]
    
            if config.include_search_corpus:
                tools.append(SearchCorpusTool(
                    self.chroma_client,
                    self._llm_client,
                    usage_key=config.agent_id,
                    timeout_s=config.request_timeout_s,
                    max_parallel_chroma_queries=config.max_parallel_chroma_queries,
                    include_embeddings=True,
                ))
    
            if config.include_grep_corpus:
                tools.append(GrepCorpusTool(
                    self.chroma_client,
                    max_parallel_chroma_queries=config.max_parallel_chroma_queries,
                    include_embeddings=True,
                ))
    
            if config.include_semantic_filter:
                tools.append(SemanticFilterTool(
                    self.chroma_client,
                    self._llm_client,
                    self.document_map,
                    config,
                    self._resolve_semantic_filter_llm_model(),
                    usage_key=config.agent_id,
                    max_parallel_chroma_queries=config.max_parallel_chroma_queries,
                    include_embeddings=True,
                ))
    
            if self.page_locator is not None:
                tools.append(ViewFigureTool(
                    self.document_map, self.page_locator, renders_dir=storage_config.page_renders_dir,
                ))
    
            return tools

    def _handle_tool_call(self, ctx: ExecutionContext, out: CodeOutput) -> list[Block]:
        """Handle tool call results and render blocks."""
        blocks: list[Block] = []
        stdout_s = (out.logs or "").strip()
        if stdout_s:
            blocks.append(TextBlock(f"[stdout]\n{stdout_s}"))

        # NOTE: this is ResultSet for SearchCorpus, GrepCorpus, and SemanticFilterTool; dicts for the other tools
        output = out.output
        if isinstance(output, dict) and output["tool"] in [SearchCorpusTool.name, GrepCorpusTool.name, SemanticFilterTool.name]:
            trace_results = []
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            elif not output["results"]:
                blocks.append(TextBlock(EMPTY_RESULT_MESSAGE))
            elif output["results"]:
                final_results: dict[str, list[SearchResult]] = output["results"]
                for collection, results in final_results.items():
                    for res in results:
                        blocks.append(
                            ChunkBlock(collection=collection, chunk_id=res.chunk_id, doc_id=res.doc_id, text=f"{res.header}\n{res.text}")
                        )
                        trace_results.append({"collection": collection, "chunk_id": res.chunk_id, "doc_id": res.doc_id})
                    # chroma 1.x rejects requests over 40 MB (413 "Payload too large"); a 4096-dim
                    # float embedding is ~40 KB as JSON, so keep upserts to a few hundred chunks
                    for i in range(0, len(results), _UPSERT_STEP):
                        self._collection.upsert(
                            ids=[res.chunk_id for res in results[i:i + _UPSERT_STEP]],
                            embeddings=[res.embedding for res in results[i:i + _UPSERT_STEP]],  # type: ignore
                            metadatas=[res.metadata for res in results[i:i + _UPSERT_STEP]],
                            documents=[res.text for res in results[i:i + _UPSERT_STEP]],
                        )

                    # update the list of actions taken by this agent
                    self._update_collection_actions(output["tool"], output["tool_kwargs"])

            tracer_data = {
                "tool": output["tool"],
                "tool_kwargs": output["tool_kwargs"],
                "results": trace_results,
                "error": output.get("error"),
            }
            if output["tool"] == SemanticFilterTool.name:
                tracer_data["kept_doc_ids"] = output.get("kept_doc_ids")
                tracer_data["rejected_doc_ids"] = output.get("rejected_doc_ids")

            ctx.tracer.emit(
                id=f"{output['tool']}_tool_call",
                kind="tool_call",
                step=self._step,
                turn=self._turn,
                data=tracer_data,
            )

        elif isinstance(output, dict) and output["tool"] == ReadDocumentTool.name:
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

        elif isinstance(output, dict) and output["tool"] == ViewFigureTool.name:
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

        elif isinstance(output, dict) and output["tool"] == PruneTool.name:
            # modify the block visibility for any chunks / docs that were pruned
            # NOTE: we do not need to change visibility for any other tool calls because they
            #       will append new blocks with the data they read / fetched; un-redacting their
            #       previous blocks in the message history will only take up more space and mess
            #       with the prefix cache
            self._make_blocks_invisible(doc_ids=output["doc_ids"], chunk_ids=output["chunk_ids"])
            blocks.append(TextBlock(
                f"[result]\nPruned {len(output['chunk_ids'] or [])} chunk(s) and {len(output['doc_ids'] or [])} doc(s)."
            ))

            ctx.tracer.emit(
                id="prune_tool_call",
                kind="tool_call",
                step=self._step,
                turn=self._turn,
                data={
                    "tool": output["tool"],
                    "tool_kwargs": output["tool_kwargs"],
                    "doc_ids": output["doc_ids"],
                    "chunk_ids": output["chunk_ids"],
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

    def _create_working_set_summaries(self, base_collection_name: str):
        """Return a string with summaries of the names, size, metadata fields, and actions for all (non-base) collections."""
        working_set_summaries = ""
        ws_summary_template = _WS_PROMPTS["working_set_summary"]
        offset = 0
        hide_collection_names = [base_collection_name, self._collection.name]
        collections = self.chroma_client.list_collections(limit=_LIST_LIMIT, offset=offset)
        while len(collections) > 0:
            for collection in collections:
                if self.config.hide_and_clear_working_sets and collection.metadata.get("created_by_agent_type") == "SearchAgent":
                    continue

                if collection.name in hide_collection_names or not collection.metadata.get("is_working_set"):
                    continue

                # get information about this collection
                total_num_chunks = collection.count()
                description = collection.metadata.get("description", "")
                metadata_fields = collection.metadata.get("fields")
                actions = collection.metadata["actions"].split(METADATA_LIST_DELIMITER)

                # summarize this collection
                ws_summary = _ENV.from_string(ws_summary_template).render(
                    name=collection.name,
                    description=description,
                    total_num_chunks=total_num_chunks,
                    metadata_fields=metadata_fields,
                    actions="\n".join(actions),
                )

                # add summary
                working_set_summaries += f"{ws_summary}\n\n"

            # stop once we get an incomplete page
            if len(collections) < _LIST_LIMIT:
                break

            # otherwise, fetch next page
            offset += _LIST_LIMIT
            collections = self.chroma_client.list_collections(limit=_LIST_LIMIT, offset=offset)

        return working_set_summaries

    def _add_collections_message(self, input: str, base_collection_name: str, working_set_collection_on: bool) -> str:
        """Add a message to the agent's context informing it of the collection(s) at its disposal."""
        # get total chunks and metadata fields for the base collection
        c = self.chroma_client.get_collection(base_collection_name)
        total_num_chunks = c.count()
        metadata_fields = c.metadata.get("fields")

        collections_summary = ""
        if working_set_collection_on:
            collections_summary_template = _SA_PROMPTS["collections_summary_with_working_sets"]
            collections_summary = _ENV.from_string(collections_summary_template).render(
                base_collection_name=base_collection_name,
                total_num_chunks=total_num_chunks,
                metadata_fields=metadata_fields,
                working_set_summaries=self._create_working_set_summaries(base_collection_name),
            )
        else:
            collections_summary_template = _SA_PROMPTS["collections_summary_without_working_sets"]
            collections_summary = _ENV.from_string(collections_summary_template).render(
                base_collection_name=base_collection_name,
                total_num_chunks=total_num_chunks,
                metadata_fields=metadata_fields,
            )

        # prepend the collections summary to the input question
        return f"{collections_summary}\n\n{input}"

    # ------------------------------------------------------------------
    # Final answer: validate + correct the returned doc_ids
    # ------------------------------------------------------------------

    async def call(
        self, ctx: ExecutionContext, input: str, *, correction_steps: int | None = None, **_
    ) -> list[str]:
        """Run the agent, then make sure the `doc_ids` it returned name real documents.

        A returned id is valid when it is a key of `document_map`. This catches failures where
        the agent returns a hallucinated id or (more plausibly) a partial identifier. For example,
        we have observed the model use source ids (e.g. `MICROSOFT_2023_10K`) instead of the
        page-level document id it was shown (`MICROSOFT_2023_10K::p59`). If any id is invalid,
        the agent is re-prompted (resuming the same conversation, so it still sees everything it
        read) to fix its mistake(s) for up to `correction_steps` extra turns. This budget is
        separate from `max_steps`. The final (sub)set of valid `doc_ids` are returned. A failure
        is only raised if the agent returns no valid ids after all correction steps are exhausted.

        Returns `valid_doc_ids` which is the list of validated ids.
        """
        # rename booleans to make logic more readable
        working_set_collection_on = not ctx.config.search.working_set_collection_off

        # add message summarizing the state of the working set
        input = self._add_collections_message(input, ctx.config.storage.collection_name, working_set_collection_on)

        # run the agent on the user query
        payload = await super().call(ctx, input)

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

        if not valid:
            raise StepFailed(
                self.config.name,
                "no well-formed doc_ids after correction",
                diagnostic=f"agent returned only unrecognized doc_ids: {doc_ids!r}",
            )

        return valid
