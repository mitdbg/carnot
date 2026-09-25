"""Shared implementation of the collection-building agents (Bootstrap + Enrich).

Both agents explore the corpus with the search tools, enrich collections with the map tools, and
create / modify collections with the collection tools; they differ only in their prompts and config.
`CollectionAgent` holds the tool construction and the observation rendering; the subclasses set
`_PROMPTS`.

Rendering policy: these agents move thousands of chunks per tool call, so search / grep results are
rendered as a *summary* (counts + the first few chunks) rather than one block per chunk, and the map
and collection tools render one short status block. See `_handle_tool_call`.
"""

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
from skunk.common import B64Image, ExecutionContext, PageLocator
from skunk.config import StorageConfig
from skunk.llm_client import LLMClient
from skunk.storage.document_map import DocumentMap
from skunk.sandbox.local_python_executor import CodeOutput, BASE_BUILTIN_MODULES

from qatfd.config import BootstrapConfig, EnrichConfig
from qatfd.prompts import load_qatfd_prompts
from qatfd.tools import (
    AddToCollectionTool,
    CopyCollectionTool,
    CreateCollectionTool,
    DeleteCollectionTool,
    GrepCorpusTool,
    ListCollectionsTool,
    MapResult,
    MapTool,
    MergeCollectionsTool,
    ReadDocumentTool,
    SearchCorpusTool,
    SearchResult,
    SemanticMapTool,
    ViewFigureTool,
)

_ENV = Environment(autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined)
_WS_PROMPTS = load_qatfd_prompts("working_set")

# message for when search or grep returns an empty result
EMPTY_RESULT_MESSAGE = "No results found."

# the `tool` name a ResultSet produced by set algebra (`a | b`, `.where(fn)`, ...) carries
RESULT_SET_TOOL = "result_set"
_SEARCH_TOOLS = (SearchCorpusTool.name, GrepCorpusTool.name, RESULT_SET_TOOL)
_MAP_TOOLS = (SemanticMapTool.name, MapTool.name)
_COLLECTION_WRITE_TOOLS = (CreateCollectionTool.name, AddToCollectionTool.name, CopyCollectionTool.name, MergeCollectionsTool.name)


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit].rstrip() + f" …[{len(text) - limit} more chars]"


class CollectionAgent(MultiTurnAgent):
    """Base for agents that build and maintain collections. Subclasses set `_PROMPTS` (a prompt
    file with `system_prompt` and `terminal_prompt`)."""

    _PROMPTS: dict[str, str]
    _AGENT_TYPE: str

    @staticmethod
    def parse_step(text: str) -> StepOutput:
        """Apply the basic answer parser."""
        return parse_step(text)

    def __init__(
        self,
        config: BootstrapConfig | EnrichConfig,
        document_map: DocumentMap,
        chroma_client: ClientAPI,
        llm_client: LLMClient,
        storage_config: StorageConfig,
        *,
        agent_id: str | None = None,
        additional_notes: str | None = None,
        page_locator: PageLocator | None = None,
    ) -> None:
        # override the default agent_id if one is provided
        config.agent_id = config.agent_id if agent_id is None else agent_id

        # set variables
        self.document_map = document_map
        self.chroma_client = chroma_client
        self.page_locator = page_locator
        self._llm_client = llm_client
        self.base_collection_name = storage_config.collection_name

        # construct the agent's tools
        tools = self._build_tools(config, storage_config)

        # construct the system prompt
        system_prompt_template = self._PROMPTS["system_prompt"]
        system_prompt = _ENV.from_string(system_prompt_template).render(
            max_steps=config.max_steps,
            authorized_imports=list(set(BASE_BUILTIN_MODULES) | set(config.authorized_imports)),
            tools="\n\n".join(t.doc for t in tools),
            cost_budget=config.cost_budget,
            latency_budget=config.latency_budget,
            additional_notes=additional_notes,
        )

        # construct the terminal prompt
        terminal_prompt_template = self._PROMPTS["terminal_prompt"]
        terminal_prompt = _ENV.from_string(terminal_prompt_template).render()

        # initialize the agent
        super().__init__(config, tools=tools, system_prompt=system_prompt, terminal_prompt=terminal_prompt, parse=CollectionAgent.parse_step)
        self.config: BootstrapConfig | EnrichConfig

    def _resolve_semantic_map_llm_model(self) -> str:
        """Model to use for semantic map judgements; precedence is:
        1. config.semantic_map_llm_model
        2. config.llm_model
        3. LLMClient.config.llm_model
        """
        if self.config.semantic_map_llm_model:
            return self.config.semantic_map_llm_model
        elif self.config.llm_model:
            return self.config.llm_model

        return self._llm_client.config.llm_model

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _build_tools(self, config: BootstrapConfig | EnrichConfig, storage_config: StorageConfig) -> list[Tool]:
        """Build the set of tools used by the agent."""
        # the collection tools need `self.config` for the semantic-map model resolution below
        self.config = config

        # search / data exploration tools. Embeddings are NOT returned with the results: the collection
        # tools copy chunks server-side by id, so results only need ids / metadata / text for previews.
        tools: list[Tool] = [
            ReadDocumentTool(self.document_map),
            SearchCorpusTool(
                self.chroma_client,
                self._llm_client,
                usage_key=config.agent_id,
                timeout_s=config.request_timeout_s,
                max_parallel_chroma_queries=config.max_parallel_chroma_queries,
                include_embeddings=False,
            ),
            GrepCorpusTool(
                self.chroma_client,
                max_parallel_chroma_queries=config.max_parallel_chroma_queries,
                include_embeddings=False,
            ),
        ]

        if self.page_locator is not None:
            tools.append(ViewFigureTool(
                self.document_map, self.page_locator, renders_dir=storage_config.page_renders_dir,
            ))

        # data enrichment tools; EnrichAgent gets semantic map, but bootstrap does not
        tools.append(MapTool(self.chroma_client, self.document_map, config))
        if self._AGENT_TYPE == "EnrichAgent":
            tools.append(SemanticMapTool(
                self.chroma_client,
                self._llm_client,
                self.document_map,
                config,
                self._resolve_semantic_map_llm_model(),
                usage_key=config.agent_id,
            ))

        # collection management tools
        collection_tool_kwargs = dict(
            agent_id=config.agent_id,
            agent_type=self._AGENT_TYPE,
            max_copy_chunks=config.max_copy_chunks,
            max_workers=config.max_parallel_chroma_queries,
        )
        for tool_cls in (
            CreateCollectionTool,
            AddToCollectionTool,
            CopyCollectionTool,
            MergeCollectionsTool,
            DeleteCollectionTool,
            ListCollectionsTool,
        ):
            tools.append(tool_cls(self.chroma_client, self.base_collection_name, **collection_tool_kwargs))  # type: ignore

        return tools

    # ------------------------------------------------------------------
    # Observation rendering
    # ------------------------------------------------------------------

    def _render_search_results(self, output: dict) -> tuple[list[Block], list[dict]]:
        """Summary blocks for a ResultSet: totals per collection plus the first
        `search_preview_chunks` chunks of each (text truncated to `search_preview_chars`)."""
        blocks: list[Block] = []
        trace_results: list[dict] = []
        final_results: dict[str, list[SearchResult]] = output["results"]
        n_chunks = sum(len(hits) for hits in final_results.values())
        n_docs = len({hit.doc_id for hits in final_results.values() for hit in hits})
        header = f"[result]\n{output['tool']}: {n_chunks} chunk(s) from {n_docs} doc(s) across {len(final_results)} collection(s)."
        if n_chunks > 0:
            header += " Assign this call to a variable and pass it to create_collection / add_to_collection to index all of these chunks."
        blocks.append(TextBlock(header))

        n_preview = self.config.search_preview_chunks
        n_chars = self.config.search_preview_chars
        for collection, hits in final_results.items():
            docs = len({hit.doc_id for hit in hits})
            shown = hits[:n_preview]
            note = f" (showing the first {len(shown)})" if len(hits) > len(shown) else ""
            blocks.append(TextBlock(f"collection={collection}: {len(hits)} chunk(s) / {docs} doc(s){note}"))
            for res in shown:
                blocks.append(ChunkBlock(
                    collection=collection, chunk_id=res.chunk_id, doc_id=res.doc_id, text=f"{res.header}\n{_truncate(res.text, n_chars)}"
                ))
            trace_results.extend({"collection": collection, "chunk_id": hit.chunk_id, "doc_id": hit.doc_id} for hit in hits)
        return blocks, trace_results

    def _render_map_results(self, output: dict) -> tuple[list[Block], dict]:
        """One status line per mapped collection plus a few sample rows."""
        blocks: list[Block] = []
        trace: dict = {}
        n_samples = self.config.map_preview_samples
        results: dict[str, list[MapResult]] = output["results"]
        for collection, pages in results.items():
            # for semantic maps; MapResult.error will be set if the collection is too large
            errors = [p.error for p in pages if p.error]
            if errors:
                blocks.append(TextBlock(f"[error]\ncollection={collection}: {errors[0]}"))
                trace[collection] = {"error": errors[0]}
                continue

            # flatten the samples returned across all pages for this collection
            samples = [s for p in pages for s in p.samples]
            errors = [sample.error for sample in samples if sample.error is not None]
            line = f"collection={collection}: mapped {len(samples)} chunk(s)"
            if errors:
                line += f" ({len(errors)} failed with an error)"
            blocks.append(TextBlock(line))
            for sample in samples[:n_samples]:
                blocks.append(TextBlock(f"  chunk_id={sample.chunk_id} | doc_id={sample.doc_id} | fields={sample.fields}"))
            unique_errors_str = "Errors:\n"
            for error in set(errors):
                unique_errors_str += f" - {error}\n"
            blocks.append(TextBlock(unique_errors_str))
            trace[collection] = {"n_mapped": len(samples), "n_failed": len(errors)}
        if output.get("warning"):
            blocks.append(TextBlock(f"[warning]\n{output['warning']}"))
        return blocks, trace

    def _render_collection_summary(self, summary: dict) -> str:
        return _ENV.from_string(_WS_PROMPTS["working_set_summary"]).render(
            name=summary["name"],
            description=summary["description"],
            total_num_chunks=summary["num_chunks"],
            metadata_fields=summary["fields"],
            actions="\n".join(summary["actions"]),
        )

    def _handle_tool_call(self, ctx: ExecutionContext, out: CodeOutput) -> list[Block]:
        """Handle tool call results and render blocks."""
        blocks: list[Block] = []
        stdout_s = (out.logs or "").strip()
        if stdout_s:
            blocks.append(TextBlock(f"[stdout]\n{stdout_s}"))
        output = out.output

        if isinstance(output, dict) and output.get("tool") in _SEARCH_TOOLS:
            trace_results: list[dict] = []
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            elif not output["results"]:
                blocks.append(TextBlock(EMPTY_RESULT_MESSAGE))
            else:
                result_blocks, trace_results = self._render_search_results(output)
                blocks.extend(result_blocks)

            ctx.tracer.emit(
                id=f"{output['tool']}_tool_call",
                kind="tool_call",
                step=self._step,
                turn=self._turn,
                data={
                    "tool": output["tool"],
                    "tool_kwargs": output["tool_kwargs"],
                    "results": trace_results,
                    "error": output.get("error"),
                },
            )

        elif isinstance(output, dict) and output.get("tool") == ReadDocumentTool.name:
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

        elif isinstance(output, dict) and output.get("tool") == ViewFigureTool.name:
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

        elif isinstance(output, dict) and output.get("tool") in _MAP_TOOLS:
            trace: dict = {}
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            else:
                map_blocks, trace = self._render_map_results(output)
                blocks.extend(map_blocks)

            ctx.tracer.emit(
                id=f"{output['tool']}_tool_call",
                kind="tool_call",
                step=self._step,
                turn=self._turn,
                data={
                    "tool": output["tool"],
                    "tool_kwargs": output["tool_kwargs"],
                    "results": trace,
                    "warning": output.get("warning"),
                    "error": output.get("error"),
                },
            )

        elif isinstance(output, dict) and output.get("tool") in _COLLECTION_WRITE_TOOLS:
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            else:
                verb = "Added to" if output["tool"] == AddToCollectionTool.name else "Created"
                line = f"[result]\n{verb} collection '{output['collection']}': now {output['num_chunks']} chunk(s)."
                stats = output.get("stats")
                if stats:
                    line += (
                        f" This call: {stats['n_new']} new, {stats['n_existing']} already present, "
                        f"{stats['n_missing']} not found, {stats['n_docs']} distinct doc(s); sources={stats['sources']}."
                    )
                if output.get("fields"):
                    line += f"\nMetadata fields:\n{output['fields'].rstrip()}"
                blocks.append(TextBlock(line))

            ctx.tracer.emit(
                id=f"{output['tool']}_tool_call",
                kind="tool_call",
                step=self._step,
                turn=self._turn,
                data={
                    "tool": output["tool"],
                    "tool_kwargs": output["tool_kwargs"],
                    "collection": output.get("collection"),
                    "num_chunks": output.get("num_chunks"),
                    "stats": output.get("stats"),
                    "error": output.get("error"),
                },
            )

        elif isinstance(output, dict) and output.get("tool") == DeleteCollectionTool.name:
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            else:
                blocks.append(TextBlock(f"[result]\nDeleted collection '{output['collection']}'."))

            ctx.tracer.emit(
                id="delete_collection_tool_call",
                kind="tool_call",
                step=self._step,
                turn=self._turn,
                data={"tool": output["tool"], "tool_kwargs": output["tool_kwargs"], "collection": output.get("collection"), "error": output.get("error")},
            )

        elif isinstance(output, dict) and output.get("tool") == ListCollectionsTool.name:
            summaries = output["collections"]
            text = f"[result]\n{len(summaries)} collection(s):\n\n" + "\n\n".join(self._render_collection_summary(s) for s in summaries)
            blocks.append(TextBlock(text))

            ctx.tracer.emit(
                id="list_collections_tool_call",
                kind="tool_call",
                step=self._step,
                turn=self._turn,
                data={
                    "tool": output["tool"],
                    "tool_kwargs": output["tool_kwargs"],
                    "collections": [{"name": s["name"], "num_chunks": s["num_chunks"]} for s in summaries],
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
