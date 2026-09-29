import asyncio
import base64
import json
import os
import socket
import threading
import time

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from fastmcp.utilities.types import Image
from urllib.parse import urlsplit

from qatfd.tools import (
    GrepCorpusTool,
    ReadDocumentTool,
    SearchCorpusTool,
    SearchResult,
    ViewFigureTool,
)
from skunk.config import InferenceConfig
from skunk.llm_client import LLMClient

from qatfd.benchmarks.base import BenchmarkResources

SERVER_NAME = "CorpusSearchServer"
SERVER_INSTRUCTIONS = "This server provides access to a data corpus via search, grep, and read operations. Use the provided tools to interact with the corpus."
SESSION_HEADER = "x-session-id"
LLM_TIMEOUT_S = 120.0

SEARCH_CORPUS_DESC = """This tool performs a vector search over the specified `collection(s)` by embedding the input `query` and returning the `top_k` most relevant chunks, each labelled with its `chunk_id` and `doc_id`. You may better target the search by passing a `metadata_filter`, which is a ChromaDB-style where clause over chunk metadata.

Supported `metadata_filter` syntax:
- Equality: `{"field": value}`
- Set membership: `{"field": {"$in": [v1, v2, ...]}}`
- Negation / not-in: `{"field": {"$nin": [...]}}`
- Compound: `{"$and": [clause1, clause2, ...]}` or `{"$or": [...]}`

You will be provided separately with a list of the collections and their available metadata fields.

Args:
  collections: the names of the collections to query
  query: the search query
  top_k: the number of chunks most relevant to the `query` to return
  metadata_filter: an optional metadata filter for hybrid vector search

Returns:
  A dictionary with the keys "results" and "error". "results" contains a list of dictionaries (one per chunk) with keys:
  - "collection": the name of the collection the chunk came from
  - "chunk_id": the unique id of the chunk
  - "doc_id": the id of the document the chunk came from
  - "metadata": string containing a JSON dump of the chunk's metadata fields and values
  - "text": the text of the chunk
  
  "error" contains any error message (None on successful tool calls).
"""

GREP_CORPUS_DESC = """This tool performs a regex search over the specified `collection(s)` and returns the matching chunks grouped by their `doc_id` then `chunk_id`. The output is unlimited by default (`limit=None`). This is useful for exhaustive queries -- "find every doc that mentions X" -- but you should pass `limit=N` for narrower exploratory searches. You may better target the search by passing a `metadata_filter`, which is a ChromaDB-style where clause over chunk metadata.

Supported `metadata_filter` syntax:
- Equality: `{"field": value}`
- Set membership: `{"field": {"$in": [v1, v2, ...]}}`
- Negation / not-in: `{"field": {"$nin": [...]}}`
- Compound: `{"$and": [clause1, clause2, ...]}` or `{"$or": [...]}`

You will be provided separately with a list of the collections and their available metadata fields.

Args:
  collections: the names of the collections to query
  pattern: the pattern for the grep search
  metadata_filter: an optional metadata filter for hybrid vector search
  limit: an optional limit on the number of chunks to return

Returns:
  A dictionary with the keys "results" and "error". "results" contains a list of dictionaries (one per chunk) with keys:
  - "collection": the name of the collection the chunk came from
  - "chunk_id": the unique id of the chunk
  - "doc_id": the id of the document the chunk came from
  - "metadata": string containing a JSON dump of the chunk's metadata fields and values
  - "text": the text of the chunk

  "error" contains any error message (None on successful tool calls).
"""

READ_DOCUMENT_DESC = """This tool returns the text of one or more documents, given their `doc_ids`. You may specify `start_char_indices` and `end_char_indices` to fetch only a substring of each document. `start_char_indices` and `end_char_indices` must be lists of the same length as `doc_ids` or `None`, and each index pair will be applied to the corresponding document. If `end_char_indices` exceeds the document length, the document will be truncated to its actual length. If `start_char_indices` is negative, it counts from the end of the document. If `end_char_indices` is negative, it counts from the end of the document.

Args:
  doc_ids: the id(s) of one or more documents to read
  start_char_indices: the indic(es) at which to start reading each document
  end_char_indices: the indic(es) at which to finish reading each document

Returns:
  A dictionary with the key "docs", this contains a list of dictionaries (one per document) with keys:
  - "doc_id": the id of the document
  - "text": the text of the document
"""

VIEW_FIGURE_DESC = """When you read a document and see a `<figure id=N>` placeholder (a chart/figure that is NOT in the searchable text), call this tool to render it. It returns an image of the entire document — so you see the figure on the document in context — just like reading the document's text. Use it to judge whether a document whose answer may live in a chart is relevant. Pass the `doc_id` of the document you want to read.

Args:
  doc_id: the id of the document to render

Returns:
  An image of the page the document came from, or a dictionary with the key "error" if it cannot be rendered.
"""


FORBIDDEN_BASE_COLLECTIONS = [
    "officeqa-qwen-8b"
]

class _CodexSearchCorpusTool(SearchCorpusTool):
    """This subclass does two things for Codex:
    
    1. It forwards the caller's x-session-id (Codex sends it on every MCP request) to the
    OpenRouter embeddings call, so embedding usage is attributed to the same session
    as the codex agent's own model calls.

    2. It unpacks the tool call output to return a list of dictionaries.
    """

    def _embed_query(self, query: str) -> list[float]:
        # get_http_headers() never raises; returns {} outside a request. Keys are lowercased.
        sid = get_http_headers().get(SESSION_HEADER)
        headers = {SESSION_HEADER: sid} if sid else None
        return self._llm_client.embed_query(query, usage_key=self._usage_key, http_headers=headers, timeout_s=LLM_TIMEOUT_S)

    def __call__(
        self,
        collections: list[str],
        query: str,
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> dict:
        for forbidden_collection in FORBIDDEN_BASE_COLLECTIONS:
            assert forbidden_collection not in collections, f"You may not invoke this tool on collection: {forbidden_collection}"
        tool_output = super().__call__(collections, query, top_k, metadata_filter)
        output = {"results": [], "error": tool_output.get("error", None)}
        for collection, results in tool_output["results"].items():
            for result in results:
                result: SearchResult
                output["results"].append({
                    "collection": collection,
                    "chunk_id": result.chunk_id,
                    "doc_id": result.doc_id,
                    "metadata": json.dumps(result.metadata),
                    "text": result.text,
                })

        return output


class _CodexGrepCorpusTool(GrepCorpusTool):
    """This subclass unpacks the tool call output to return a list of dictionaries."""

    def __call__(
        self,
        collections: list[str],
        pattern: str,
        metadata_filter: dict | None = None,
        limit: int | None = None,
    ) -> dict:
        for forbidden_collection in FORBIDDEN_BASE_COLLECTIONS:
            assert forbidden_collection not in collections, f"You may not invoke this tool on collection: {forbidden_collection}"
        tool_output = super().__call__(collections, pattern, metadata_filter, limit)
        output = {"results": [], "error": tool_output.get("error", None)}
        for collection, results in tool_output["results"].items():
            for result in results:
                result: SearchResult
                output["results"].append({
                    "collection": collection,
                    "chunk_id": result.chunk_id,
                    "doc_id": result.doc_id,
                    "metadata": json.dumps(result.metadata),
                    "text": result.text,
                })

        return output


def _build_base_tools(res: BenchmarkResources, inference: InferenceConfig) -> tuple[SearchCorpusTool, GrepCorpusTool, ReadDocumentTool, ViewFigureTool | None]:
    # construct LLM client for MCP server
    llm = LLMClient(inference, openrouter_api_key=os.environ["OPENROUTER_CODEX_API_KEY"])

    # construct the skunk tools to be placed in the MCP server
    # NOTE: usage key is not needed for search corpus tool b/c Codex tracks cost via API
    search = _CodexSearchCorpusTool(res.chroma_client, llm, timeout_s=LLM_TIMEOUT_S)
    grep = _CodexGrepCorpusTool(res.chroma_client)
    read = ReadDocumentTool(res.document_map)
    view = None
    if res.page_locator is not None:
        view = ViewFigureTool(res.document_map, res.page_locator)

    return search, grep, read, view


def build_mcp_server(res: BenchmarkResources, inference: InferenceConfig) -> FastMCP:
    # build the tools for the mcp server
    search, grep, read, view = _build_base_tools(res, inference)

    # construct the MCP server; wrap the tools to hide WorkingSet interface details
    mcp = FastMCP(SERVER_NAME, instructions=SERVER_INSTRUCTIONS)

    @mcp.tool(name="search_corpus", description=SEARCH_CORPUS_DESC, annotations={"readOnlyHint": True})
    def search_corpus(collections: list[str], query: str, top_k: int, metadata_filter: dict | None = None) -> dict:
        return search(collections=collections, query=query, top_k=top_k, metadata_filter=metadata_filter)

    @mcp.tool(name="grep_corpus", description=GREP_CORPUS_DESC, annotations={"readOnlyHint": True})
    def grep_corpus(collections: list[str], pattern: str, metadata_filter: dict | None = None, limit: int | None = None) -> dict:
        return grep(collections=collections, pattern=pattern, metadata_filter=metadata_filter, limit=limit)


    @mcp.tool(name="read_document", description=READ_DOCUMENT_DESC, annotations={"readOnlyHint": True})
    def read_document(doc_ids: list[str], start_char_indices: list[int] | None = None, end_char_indices: int | None | list[int | None] = None) -> dict:
        result_dict = read(doc_ids=doc_ids, start_char_indices=start_char_indices, end_char_indices=end_char_indices)
        return {"docs": result_dict["docs"]}

    if view is not None:
        @mcp.tool(name="view_figure", description=VIEW_FIGURE_DESC, annotations={"readOnlyHint": True})
        def view_figure(doc_id: str) -> Image | dict:
            result_dict = view(doc_id=doc_id)
            if result_dict.get("error"):
                return {"error": result_dict["error"]}
            return Image(data=base64.standard_b64decode(result_dict["data"]), format=result_dict["mime"].split("/")[-1])

    return mcp


def start_mcp_server(mcp: FastMCP, url: str, ready_timeout_s: float = 30.0) -> threading.Thread:
    """Serve `mcp` over streamable-HTTP at `url` on a daemon thread with its own event loop,
    blocking until the port accepts connections. The thread dies with the process, so no
    explicit shutdown is needed; codex subprocesses connect to it over localhost."""
    parts = urlsplit(url)
    host, port, path = parts.hostname, parts.port, parts.path

    # refuse to silently reuse a stale server (e.g. a leftover scripts/codex_mcp_server.py)
    try:
        with socket.create_connection((host, port), timeout=0.5):
            raise RuntimeError(f"[qatfd] ABORT: something is already listening on {host}:{port}; stop it or change mcp_url")
    except OSError:
        pass

    failure: list[BaseException] = []
    def _serve() -> None:
        try:
            # own event loop: the worker threads each asyncio.run() their own loop, so there is no
            # shared loop to attach to. uvicorn only installs signal handlers on the main thread.
            asyncio.run(mcp.run_http_async(
                transport="http", host=host, port=port, path=path,
                show_banner=False, log_level="warning",
            ))
        except BaseException as e:  # noqa: BLE001 — uvicorn exits via SystemExit on bind failure
            failure.append(e)

    t = threading.Thread(target=_serve, name="qatfd-mcp-server", daemon=True)
    t.start()

    deadline = time.monotonic() + ready_timeout_s
    while time.monotonic() < deadline:
        if failure:
            raise RuntimeError(f"[qatfd] MCP server failed to start on {url}") from failure[0]
        try:
            with socket.create_connection((host, port), timeout=0.5):
                print(f"[qatfd] MCP server {mcp.name!r} serving at {url}")
                return t
        except OSError:
            time.sleep(0.2)

    raise RuntimeError(f"[qatfd] MCP server did not become ready at {url} within {ready_timeout_s:.0f}s")
