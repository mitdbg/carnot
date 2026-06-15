"""Tool implementations for the SearchAgent.

All retrieval tools are backed by a single ChromaDB collection in which each
row is one *element* (chunk) extracted from a *document* (e.g. a page of a
treasury bulletin, or a scraped web page). The layout — produced by
`prep/create_vector_db.py` — is:

    id:        the `chunk_id` (e.g. "1946_11_41_6")
    document:  the cleaned element text (read from the `documents` field)
    metadata:  {
        doc_id:     str   -- the document key, e.g. "1946_11_41"
        chunk_id:   str   -- duplicate of the row id, for $nin filterability
        element_id: int   -- 0-based index of the chunk within its document
        type:       str   -- "text" | "title" | "table" | ...
        ...               -- any corpus-specific extras (year, month, url, ...)
    }

These tools port the old `skunk.retrieve.search_tools` closures onto the
`MultiTurnAgent.Tool` ABC: each is an instance whose `doc` is spliced into the
system prompt, and whose `__call__` is registered with the sandbox under `name`
and invoked from model-emitted python.

Structured returns: every retrieval tool returns a *tagged dict* (never a bare
string — errors are carried in an "error" field of the tagged payload) so the
return shape is uniform. Each chunk carries its `chunk_id` / `doc_id`, which
`SearchAgent._blocks_from_output` turns into `ChunkBlock`s: this keeps the full
trajectory for reward computation while redacting pruned chunks from the
generation view (`_block_is_visible`). The tag constants below discriminate each
payload shape. The final answer is a JSON block (handled by the agent loop), not
a tool.

Prune state: `SearchCorpusTool`, `GrepCorpusTool`, and `PruneTool` share two
mutable sets (`pruned_chunk_ids` / `pruned_doc_ids`). `PruneTool` is the single
writer — it mutates them directly — and the search/grep tools read them on every
call (server-side `$nin`), so a prune takes effect immediately for subsequent
searches. `SearchAgent` also reads the same sets to redact already-emitted
chunks at render time. These sets are per-question state: the owning
`SearchAgent` creates them (and the tool instances closing over them) in its
`__init__`, and the orchestrator builds one `SearchAgent` per question / branch.
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from chromadb.api.models.Collection import Collection
from google import genai
from openrouter import OpenRouter

from skunk.common import get_rate_limiter, pageref_to_page_key
from skunk.corpus import render_page_b64
from skunk.multi_turn_agent import Tool

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop

    from skunk.common import ExecutionContext
    from skunk.page_index.query import PageIndexRetriever

# Tags identifying each tool's structured return payload to the SearchAgent.
PRUNE_RESULT_TAG = "__prune__"
SEARCH_RESULT_TAG = "__search_result__"
GREP_RESULT_TAG = "__grep_result__"
READ_DOCUMENT_RESULT_TAG = "__read_document_result__"
VIEW_FIGURE_RESULT_TAG = "__view_figure_result__"
PAGE_INDEX_RESULT_TAG = "__page_index_result__"

# Embedding clients we know how to query. Gemini embeddings go through
# `genai.Client`; Qwen (and other OpenRouter-hosted) embeddings are only
# available through `OpenRouter`. `SearchCorpusTool` dispatches on the type.
EmbeddingClient = genai.Client | OpenRouter

# Generic message surfaced when search_corpus / grep_corpus return zero hits.
# ChromaDB does not tell us whether a server-side `$nin` prune filter was what
# eliminated all candidates, so the message mentions both possibilities.
EMPTY_RESULT_MESSAGE = (
    "No results found; it is possible that exclusion filters on all "
    "previously returned chunks prevented any results from being returned."
)


def _build_metadata_where(
    *,
    metadata_filter: dict | None,
    ignore_chunk_ids: set[str] | None,
    ignore_doc_ids: set[str] | None,
) -> dict | None:
    """Construct a ChromaDB ``where`` dict from a user filter + ignore inputs.

    ``metadata_filter`` is passed through as a ChromaDB-compatible where clause
    (e.g. ``{"year": "2010"}``, ``{"page_id": {"$in": [19, 26]}}``, or a
    compound ``{"$and": [...]}`` / ``{"$or": [...]}``). It is ANDed with the
    server-side prune filters built from ``ignore_chunk_ids`` / ``ignore_doc_ids``.
    """
    clauses: list[dict] = []

    if metadata_filter:
        clauses.append(metadata_filter)
    if ignore_doc_ids:
        clauses.append({"doc_id": {"$nin": sorted(ignore_doc_ids)}})
    if ignore_chunk_ids:
        clauses.append({"chunk_id": {"$nin": sorted(ignore_chunk_ids)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class SearchCorpusTool(Tool):
    name = "search_corpus"

    def __init__(
        self,
        chroma_collection: Collection,
        emb_model_id: str,
        emb_client: EmbeddingClient,
        pruned_chunk_ids: set[str],
        pruned_doc_ids: set[str],
        required_metadata_filter: dict | None = None,
    ):
        self._chroma_collection = chroma_collection
        self._emb_model_id = emb_model_id
        self._emb_client = emb_client
        self._pruned_chunk_ids = pruned_chunk_ids
        self._pruned_doc_ids = pruned_doc_ids
        self._required_metadata_filter = required_metadata_filter

    def _embed_query(self, query: str) -> list[float]:
        """Embed `query` with the same model that produced the stored embeddings.

        Dispatches on the client type: `genai.Client` for Gemini embeddings,
        `OpenRouter` for Qwen (and other OpenRouter-hosted) embeddings. The
        process-wide "embed" bucket keeps all workers under the endpoint quota.
        """
        get_rate_limiter("embed").acquire()
        if isinstance(self._emb_client, genai.Client):
            emb_result = self._emb_client.models.embed_content(
                model=self._emb_model_id, contents=query
            )
            return list(emb_result.embeddings[0].values)  # type: ignore
        # OpenRouter (e.g. Qwen embeddings, only available via OpenRouter).
        resp = self._emb_client.embeddings.generate(input=query, model=self._emb_model_id)
        return list(resp.data[0].embedding)  # type: ignore

    def __call__(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> dict:
        query_embedding = self._embed_query(query)

        combined_filter = metadata_filter
        if self._required_metadata_filter and metadata_filter:
            combined_filter = {
                "$and": [self._required_metadata_filter, metadata_filter]
            }
        elif self._required_metadata_filter:
            combined_filter = self._required_metadata_filter
        where = _build_metadata_where(
            metadata_filter=combined_filter,
            ignore_chunk_ids=self._pruned_chunk_ids,
            ignore_doc_ids=self._pruned_doc_ids,
        )
        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["metadatas", "documents", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where

        try:
            results = self._chroma_collection.query(**query_kwargs)
        except Exception as e:
            return {SEARCH_RESULT_TAG: True, "chunks": [], "error": f"search_corpus error: {e}"}

        ids = results["ids"][0]
        documents = results["documents"][0]  # type: ignore
        metadatas = results["metadatas"][0]  # type: ignore
        distances = results["distances"][0]  # type: ignore

        chunks: list[dict] = []
        for rank, (cid, doc, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True), 1
        ):
            doc_id = meta.get("doc_id", "?")
            elt_type = meta.get("type", "?")
            chunks.append({
                "chunk_id": cid,
                "doc_id": doc_id,
                "text": (
                    f"[{rank}] chunk_id={cid} | doc_id={doc_id} | "
                    f"type={elt_type} | distance={dist:.4f}\n{doc or ''}"
                ),
            })
        return {SEARCH_RESULT_TAG: True, "chunks": chunks}

    doc = """\
### search_corpus(query: str, top_k: int, metadata_filter: dict | None = None)
This tool performs a vector search over all chunks in the corpus. The input `query` is embedded and the `top_k` most relevant chunks are returned, each labelled with its `chunk_id` and `doc_id`. You can optionally restrict the search to a subset of the corpus by passing a `metadata_filter`, which is a ChromaDB-style where clause over chunk metadata. Any chunks or docs you have previously pruned via `prune(...)` are automatically excluded from the results.

Supported `metadata_filter` syntax:
- Equality: `{"field": value}`
- Set membership: `{"field": {"$in": [v1, v2, ...]}}`
- Negation / not-in: `{"field": {"$nin": [...]}}`
- Compound: `{"$and": [clause1, clause2, ...]}` or `{"$or": [...]}`

```python
# find the 100 chunks most relevant to "topic X" anywhere in the corpus
search_corpus("topic X", top_k=100)

# find the 50 chunks most relevant to "topic Y" within a filtered subset
search_corpus("topic Y", top_k=50, metadata_filter={"$and": [{"field_a": "value_a"}, {"field_b": {"$in": [1, 2, 3]}}]})
```"""


class GrepCorpusTool(Tool):
    name = "grep_corpus"

    # Token→char factor for the output cap (mirrors `_estimate_prompt_tokens`'s ~4 chars/token).
    _CHARS_PER_TOKEN = 4
    # Display unit only: the truncation note renders the token cap as "Nk tokens" for
    # readability (200000 → "200k"). Not a model/context-limit knob — the cap itself is
    # `SkunkConfig.grep_max_output_tokens`; this is just thousands-formatting.
    _TOKENS_PER_K = 1000

    def __init__(
        self,
        chroma_collection: Collection,
        pruned_chunk_ids: set[str],
        pruned_doc_ids: set[str],
        max_output_tokens: int,
        required_metadata_filter: dict | None = None,
    ):
        self._chroma_collection = chroma_collection
        self._pruned_chunk_ids = pruned_chunk_ids
        self._pruned_doc_ids = pruned_doc_ids
        # Hard cap on the rendered observation size (chars). `limit=None` returns every
        # matching chunk, so a broad pattern can otherwise dump 100s of K of tokens into
        # the context in one shot and 400 the next request (see SkunkConfig.grep_max_output_tokens).
        self._max_output_chars = max_output_tokens * self._CHARS_PER_TOKEN
        self._required_metadata_filter = required_metadata_filter

    def __call__(
        self,
        pattern: str,
        metadata_filter: dict | None = None,
        limit: int | None = None,
    ) -> dict:
        combined_filter = metadata_filter
        if self._required_metadata_filter and metadata_filter:
            combined_filter = {
                "$and": [self._required_metadata_filter, metadata_filter]
            }
        elif self._required_metadata_filter:
            combined_filter = self._required_metadata_filter
        where = _build_metadata_where(
            metadata_filter=combined_filter,
            ignore_chunk_ids=self._pruned_chunk_ids,
            ignore_doc_ids=self._pruned_doc_ids,
        )
        get_kwargs: dict = {
            "where_document": {"$regex": pattern},
            "include": ["metadatas", "documents"],
        }
        if where is not None:
            get_kwargs["where"] = where
        if limit is not None:
            get_kwargs["limit"] = limit

        try:
            res = self._chroma_collection.get(**get_kwargs)
        except Exception as e:
            return {GREP_RESULT_TAG: True, "groups": [], "error": f"grep_corpus error: {e}"}

        ids = res["ids"]
        documents = res["documents"] or []
        metadatas = res["metadatas"] or []
        if not ids:
            return {GREP_RESULT_TAG: True, "groups": []}

        # Group hits by doc_id, preserving element ordering within each doc.
        grouped: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        for cid, doc, meta in zip(ids, documents, metadatas, strict=True):
            grouped[meta["doc_id"]].append((meta["element_id"], cid, doc))  # type: ignore

        # Build groups in deterministic (doc_id, element_id) order, stopping once the
        # rendered size would exceed `_max_output_chars`. Dropped hits are reported via a
        # `truncation_note` so the agent knows to narrow its pattern / pass `limit` rather
        # than assuming it saw everything.
        groups: list[dict] = []
        used_chars = 0
        total_chunks = sum(len(v) for v in grouped.values())
        kept_chunks = 0
        truncated = False
        for doc_id in sorted(grouped):
            if truncated:
                break
            header = f"\n# doc_id={doc_id}"
            doc_chunks: list[dict] = []
            for _, cid, text in sorted(grouped[doc_id]):
                chunk_text = f"  [chunk_id={cid}] {text}"
                # Count the header only once we commit the first chunk of this doc.
                cost = len(chunk_text) + (len(header) if not doc_chunks else 0)
                if doc_chunks or groups:  # always allow the very first chunk through
                    if used_chars + cost > self._max_output_chars:
                        truncated = True
                        break
                used_chars += cost
                doc_chunks.append({"chunk_id": cid, "doc_id": doc_id, "text": chunk_text})
                kept_chunks += 1
            if doc_chunks:
                groups.append({"doc_id": doc_id, "header": header, "chunks": doc_chunks})

        result: dict = {GREP_RESULT_TAG: True, "groups": groups}
        if truncated:
            dropped = total_chunks - kept_chunks
            cap_k = self._max_output_chars // self._CHARS_PER_TOKEN // self._TOKENS_PER_K
            result["truncation_note"] = (
                f"[grep_corpus output truncated: showing {kept_chunks} of {total_chunks} matching "
                f"chunk(s) (~{cap_k}k-token cap reached); {dropped} chunk(s) omitted. Narrow the "
                f"pattern, add a metadata_filter, or pass limit=N to see specific hits.]"
            )
        return result

    doc = """\
### grep_corpus(pattern: str, metadata_filter: dict | None = None, limit: int | None = None)
This tool performs a regex search over the cleaned text of every chunk in the corpus and returns the matching chunks grouped by their `doc_id`. Each hit includes its `chunk_id` so you can later refer to it or prune it. By default (`limit=None`), every matching chunk is returned -- which is useful for "find every doc that mentions X" queries -- but you should pass `limit=N` for narrower exploratory searches. The same `metadata_filter` syntax as `search_corpus` is supported. Any chunks or docs you have previously pruned via `prune(...)` are automatically excluded from the results. The total output is capped: if a broad pattern matches more than the cap, the result is truncated with a note telling you how many hits were omitted -- narrow the pattern, add a `metadata_filter`, or pass `limit=N` to see the rest.

```python
# find every chunk that mentions "topic X" (case insensitive)
grep_corpus("(?i)topic X")

# find chunks matching the literal phrase "topic Y" within a filtered subset, capped at 50 hits
grep_corpus("topic Y", metadata_filter={"field_a": {"$in": ["value_1", "value_2"]}}, limit=50)
```"""


class ReadDocumentTool(Tool):
    name = "read_document"
    _DOC_TEMPLATE = """\
### read_document(doc_id: str | list[str])
This tool returns the full cleaned text of one or more documents, given their `doc_id`(s). Don't read more than ~{{ max_pages }} documents per tool call, as they may exceed your context window.

```python
# read two specific documents by id
read_document(["doc_id_1", "doc_id_2"])
```"""

    def __init__(self, document_map: dict[str, str], max_pages: int, max_output_chars: int):
        self._document_map = document_map
        # Hard cap on the rendered observation size (chars). A `read_document` over many
        # dense-table pages can otherwise dump 100s of K of tokens in one shot and 400 the
        # next request (see SkunkConfig.read_document_max_output_chars).
        self._max_output_chars = max_output_chars
        # Pre-substitute the jinja var: tool `doc`s may flow through a
        # StrictUndefined render, so no `{{ ... }}` may survive here.
        self.doc = self._DOC_TEMPLATE.replace("{{ max_pages }}", str(max_pages))

    def __call__(self, doc_id: str | list[str]) -> dict:
        doc_ids = [doc_id] if isinstance(doc_id, str) else list(doc_id)
        docs: list[dict] = []
        used = 0  # cumulative chars emitted across docs this call
        for n, did in enumerate(doc_ids):
            text = self._document_map.get(did)
            if text is None:
                body = "[no such document (or no content in document)]"
            else:
                body = text
            rendered = f"=== doc_id={did} ===\n{body}"
            remaining = self._max_output_chars - used
            if len(rendered) > remaining:
                # Truncate this doc to what's left, then stop — dropping any further docs with
                # a note so the agent reads fewer doc_ids (or narrows with search/grep) instead.
                dropped = len(doc_ids) - n - 1
                note = (
                    f"\n[truncated: read_document output exceeded "
                    f"{self._max_output_chars} chars"
                    + (f"; {dropped} more requested doc(s) not shown" if dropped else "")
                    + " — read fewer doc_ids per call, or narrow with search_corpus / "
                    "grep_corpus]"
                )
                docs.append({"doc_id": did, "text": rendered[: max(0, remaining)] + note})
                break
            docs.append({"doc_id": did, "text": rendered})
            used += len(rendered)
        return {READ_DOCUMENT_RESULT_TAG: True, "docs": docs}


class ViewFigureTool(Tool):
    """Render the full page containing a `<figure id=N>` placeholder the agent saw while
    reading a document, and hand it back as an image observation. We render the *whole*
    page (not a bbox crop) so the agent sees the figure in context plus any sibling
    figures on the page, and so OCR coordinate errors can't clip the chart."""

    name = "view_figure"

    def __init__(
        self,
        document_map: dict[str, str],
        pdf_dir: str | Path,
        *,
        dpi: int = 300,
        fmt: str = "png",
    ):
        self._document_map = document_map
        self._pdf_dir = pdf_dir
        self._dpi = dpi
        self._fmt = fmt

    @staticmethod
    def _figure_ids(page_text: str) -> list[str]:
        """The figure ids that appear as `<figure id=N>` placeholders in `page_text`."""
        return re.findall(r"<figure id=([^>]+)>", page_text)

    def __call__(self, doc_id: str, figure_id: int | str) -> dict:
        page_text = self._document_map.get(doc_id)
        if page_text is None:
            return {VIEW_FIGURE_RESULT_TAG: True, "error": f"no such document: {doc_id!r}"}
        if f"<figure id={figure_id}>" not in page_text:
            visible = self._figure_ids(page_text)
            hint = (
                f"figure ids on this page: {visible}" if visible else "this page has no figures"
            )
            return {
                VIEW_FIGURE_RESULT_TAG: True,
                "error": f"no figure with id={figure_id} on doc_id={doc_id}; {hint}",
            }
        # doc_id is `{year}_{month}_{page_id}`; render_page_b64 wants month as "YYYY-MM".
        try:
            year, month, page = doc_id.split("_")
        except ValueError:
            return {
                VIEW_FIGURE_RESULT_TAG: True,
                "error": f"doc_id {doc_id!r} is not in the expected '<year>_<month>_<page>' format",
            }
        try:
            img = render_page_b64(
                f"{year}-{month}", int(page),
                pdf_dir=self._pdf_dir, dpi=self._dpi, fmt=self._fmt,
            )
        except Exception as e:
            return {VIEW_FIGURE_RESULT_TAG: True, "error": f"view_figure render error: {e}"}
        if img is None:
            return {
                VIEW_FIGURE_RESULT_TAG: True,
                "error": f"could not render page for doc_id={doc_id} (PDF missing)",
            }
        return {
            VIEW_FIGURE_RESULT_TAG: True,
            "doc_id": doc_id,
            "figure_id": figure_id,
            "mime": img.mime,
            "data": img.data,
        }

    doc = """\
### view_figure(doc_id: str, figure_id: int | str)
When you read a document and see a `<figure id=N>` placeholder (a chart/figure that is NOT in the searchable text), call this tool to actually *see* it. It returns an image of the **entire page** that contains the figure — so you see the figure in context, along with any other figures on that page — appended to your messages, just like reading the page's text. Use it to judge whether a page whose answer may live in a chart is relevant. Pass the `doc_id` of the page you read and the `id` from the `<figure id=N>` placeholder.

```python
# you read doc_id "2002_12_8" and saw "<figure id=5>"; now view it
view_figure("2002_12_8", 5)
```"""


class PruneTool(Tool):
    name = "prune"

    def __init__(self, pruned_chunk_ids: set[str], pruned_doc_ids: set[str]):
        self._pruned_chunk_ids = pruned_chunk_ids
        self._pruned_doc_ids = pruned_doc_ids

    def __call__(
        self,
        chunk_ids: list[str] | None = None,
        doc_ids: list[str] | None = None,
    ) -> dict:
        # Single writer of the shared prune sets: mutate directly, then report
        # how many were newly added (the agent renders the summary, no re-apply).
        new_chunks = set(chunk_ids or ()) - self._pruned_chunk_ids
        new_docs = set(doc_ids or ()) - self._pruned_doc_ids
        self._pruned_chunk_ids.update(new_chunks)
        self._pruned_doc_ids.update(new_docs)
        return {
            PRUNE_RESULT_TAG: True,
            "new_chunk_count": len(new_chunks),
            "new_doc_count": len(new_docs),
            "total_chunks": len(self._pruned_chunk_ids),
            "total_docs": len(self._pruned_doc_ids),
        }

    doc = """\
### prune(chunk_ids: list[str] | None = None, doc_ids: list[str] | None = None)
This tool records `chunk_id`s and/or `doc_id`s that you've already inspected and deemed irrelevant. Pruned ids are excluded from the results of all subsequent `search_corpus` and `grep_corpus` calls, and are dropped from your visible context. Use this aggressively whenever a search surfaces clearly irrelevant chunks or docs, to keep later searches focused and your context window manageable.

```python
# mark some chunks and a whole doc as irrelevant
prune(chunk_ids=["chunk_id_1", "chunk_id_2"], doc_ids=["doc_id_3"])
```"""


class PageIndexSearchTool(Tool):
    """Query the PageIndex (concept-tree → year filter → semantic filter) from inside the
    search loop. Wraps `PageIndexRetriever.retrieve_all` over a single synthetic branch and
    returns the surviving CONTENT BLOCKS as compact catalog summaries (page key, title,
    summary, data interval, headers).

    Tool calls execute in a worker thread (see `MultiTurnAgent._execute_code`), but the
    retriever's downstream LLM client holds async objects bound to the agent's MAIN event
    loop (the async TPM limiter, the genai/aiohttp session). So we must NOT spin up a fresh
    loop here (`asyncio.run`) — that re-enters those objects from a foreign loop and raises
    "Future attached to a different loop". Instead we schedule the coroutine back onto the
    main loop via `run_coroutine_threadsafe` and block this thread on the result; the main
    loop keeps spinning (it is awaiting our `to_thread` call), so it services the work."""

    name = "page_index_search"

    # Token→char factor for the output cap (mirrors grep/read; ~4 chars/token).
    _CHARS_PER_TOKEN = 4

    def __init__(
        self,
        retriever: PageIndexRetriever,
        pdf_dir: str,
        ctx_getter: Callable[[], ExecutionContext],
        loop_getter: Callable[[], AbstractEventLoop],
        max_output_tokens: int,
        required_bulletins: list[str] | None = None,
    ):
        self._retriever = retriever
        self._pdf_dir = pdf_dir
        self._ctx_getter = ctx_getter
        self._loop_getter = loop_getter
        self._max_output_chars = max_output_tokens * self._CHARS_PER_TOKEN
        self._required_bulletins = list(required_bulletins) if required_bulletins else None

    def __call__(self, query: str, period: str | None = None) -> dict:
        # Imported here (not at module top) to keep the page_index subtree off the import
        # path of search_agent runs that never enable this tool.
        from skunk.plan import RetrieveBranch

        ctx = self._ctx_getter()
        branch = RetrieveBranch(kind="retrieve", key=query, period=period)
        # Hard-scope to human-required bulletins when set, matching the vector tools' filter.
        scopes = [self._required_bulletins] if self._required_bulletins else None
        try:
            # Run on the agent's main loop (see class docstring), not a fresh one.
            fut = asyncio.run_coroutine_threadsafe(
                self._retriever.retrieve_all(ctx, [branch], document_scopes=scopes),
                self._loop_getter(),
            )
            per_branch = fut.result()
        except Exception as e:
            return {PAGE_INDEX_RESULT_TAG: True, "results": [], "error": f"page_index_search error: {e}"}

        blocks = per_branch[0] if per_branch else []
        # Resolve each surviving block to its self-contained catalog metadata (title /
        # summary / interval / headers) — the same pool the page_index backend feeds selection.
        pool = self._retriever.pool_for_blocks(blocks, self._pdf_dir)

        results: list[dict] = []
        used = 0
        total = len(pool)
        truncated = False
        for entry in pool:
            try:
                page_key = pageref_to_page_key(entry.ref.page)
            except ValueError:
                continue
            interval = (
                f"{entry.interval[0]}..{entry.interval[1]}" if entry.interval else "n/a"
            )
            lines = [
                f"[page_key={page_key}] kind={entry.kind} | {entry.title or '(untitled)'}",
                f"  reports period: {interval}",
            ]
            if entry.summary:
                lines.append(f"  summary: {entry.summary}")
            if entry.cols:
                lines.append(f"  columns: {', '.join(entry.cols)}")
            if entry.rows_tail:
                lines.append(f"  row headers (tail): {', '.join(entry.rows_tail)}")
            text = "\n".join(lines)
            if results and used + len(text) > self._max_output_chars:
                truncated = True
                break
            results.append({"page_key": page_key, "text": text})
            used += len(text)

        out: dict = {PAGE_INDEX_RESULT_TAG: True, "results": results}
        if truncated:
            cap_k = self._max_output_chars // self._CHARS_PER_TOKEN // 1000
            out["truncation_note"] = (
                f"[page_index_search output truncated: showing {len(results)} of {total} "
                f"block(s) (~{cap_k}k-token cap reached). Add a `period`, or narrow the query.]"
            )
        return out

    doc = """\
### page_index_search(query: str, period: str | None = None)
This tool queries a separate **concept-aware page index** of the corpus and returns the pages whose tables/charts/prose most plausibly *report on* what you describe — regardless of when the bulletin was published. It is complementary to `search_corpus` (vector similarity) and `grep_corpus` (regex): use it when you want pages by *what data they contain* (e.g. "monthly federal debt subject to limit", "receipts by source"). Internally it picks candidate chapters from a taxonomy, filters by the data period, and runs a semantic filter, so each call costs a few LLM calls — use it deliberately rather than repeatedly.

`query` is a natural-language description of the data to find. `period` optionally restricts results to pages whose *reported data span* overlaps it (NOT publication date): a single month `"YYYY-MM"`, a range `"YYYY-MM..YYYY-MM"`, or a comma-list of either. Omit `period` to search all years.

Each result gives a `page_key` plus the block's title, reported period, summary, and table headers. Pass a `page_key` straight to `read_document(...)` to read the full page, to `view_figure(...)`, or include it in your final `page_keys` answer.

```python
# find pages reporting monthly public debt for fiscal year 2014
page_index_search("monthly outstanding public debt subject to statutory limit", period="2013-10..2014-09")
```"""
