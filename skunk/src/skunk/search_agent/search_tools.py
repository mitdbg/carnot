"""Tool implementations for the SearchAgent.

All retrieval tools are backed by a single ChromaDB collection in which each
row is one *element* (chunk) extracted from a *document* (e.g. a page of a
PDF report, or a scraped web page). The layout — produced by the corpus-prep
scripts that live with the benchmarks (e.g. qatfd's
`engaging-scripts/create_vector_db.py`) — is:

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

Shared state: one `RetrievalState` object (four sets) is shared BY REFERENCE
between the owning `SearchAgent` and its tools — its dataclass docstring spells
out the read/write contract per set. The agent creates it (with the tool
instances closing over it) in its `__init__`, and the orchestrator builds one
`SearchAgent` per question / branch, so state never crosses questions. A tool
constructed WITHOUT a state (one-shot use, e.g. a plain top-k vector search)
gets its own private fresh one.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from chromadb.api.models.Collection import Collection

from skunk.common import page_key_to_pageref
from skunk.common import render_page_b64
from skunk.multi_turn_agent import Tool

if TYPE_CHECKING:
    from skunk.common import ExecutionContext, PageRef
    from skunk.llm_client import LLMClient


@dataclass
class RetrievalState:
    """Per-question retrieval state, shared by reference between a `SearchAgent`
    and its tools (the one mutable contract in the retrieval loop):

    - `pruned_*`: material the agent ruled out. `PruneTool` is the SINGLE writer;
      the search/grep tools read them on every call (server-side `$nin`), so a
      prune takes effect immediately, and `SearchAgent._block_is_visible` reads
      them to redact already-emitted chunks at render time.
    - `seen_*`: material already fetched — chunks returned by search/grep, docs
      opened by `read_document` — auto-excluded from subsequent search/grep so
      each call surfaces NEW material. Unlike pruned chunks, fetched chunks stay
      VISIBLE (prune is reserved for ruling out irrelevant material). Search/grep
      both write (on return) and read (unioned with the pruned sets);
      `read_document` writes `seen_doc_ids`.
    """

    pruned_chunk_ids: set[str] = field(default_factory=set)
    pruned_doc_ids: set[str] = field(default_factory=set)
    seen_chunk_ids: set[str] = field(default_factory=set)
    seen_doc_ids: set[str] = field(default_factory=set)

# Tags identifying each tool's structured return payload to the SearchAgent.
PRUNE_RESULT_TAG = "__prune__"
SEARCH_RESULT_TAG = "__search_result__"
GREP_RESULT_TAG = "__grep_result__"
READ_DOCUMENT_RESULT_TAG = "__read_document_result__"
VIEW_FIGURE_RESULT_TAG = "__view_figure_result__"

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
    required_filter: dict | None = None,
    ignore_chunk_ids: set[str] | None,
    ignore_doc_ids: set[str] | None,
) -> dict | None:
    """Construct a ChromaDB ``where`` dict from a user filter + ignore inputs.

    ``metadata_filter`` is passed through as a ChromaDB-compatible where clause
    (e.g. ``{"year": "2010"}``, ``{"page_id": {"$in": [19, 26]}}``, or a
    compound ``{"$and": [...]}`` / ``{"$or": [...]}``). ``required_filter`` is a
    tool-construction-time clause the caller can never widen past (ANDed in when
    set). Both are ANDed with the server-side prune filters built from
    ``ignore_chunk_ids`` / ``ignore_doc_ids``.
    """
    clauses: list[dict] = []

    if required_filter:
        clauses.append(required_filter)
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
        llm_client: LLMClient,
        state: RetrievalState | None = None,
        required_metadata_filter: dict | None = None,
        ctx: ExecutionContext | None = None,
    ):
        # `state` is shared BY REFERENCE with the owning SearchAgent's other tools;
        # a one-shot caller (e.g. a plain top-k vector search with no agent) omits
        # it and gets a private fresh one.
        self._chroma_collection = chroma_collection
        self._emb_model_id = emb_model_id
        self._llm_client = llm_client
        self._ctx = ctx
        self._state = state if state is not None else RetrievalState()
        self._required_metadata_filter = required_metadata_filter

    def _embed_query(self, query: str) -> list[float]:
        """Embed `query` with the same model that produced the stored embeddings, via the
        LLMClient — which owns backend dispatch (OpenRouter / local SentenceTransformers),
        the process-wide "embed" rate bucket, retry, and usage accounting."""
        return self._llm_client.embed_query(query, model=self._emb_model_id, ctx=self._ctx)

    def __call__(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> dict:
        query_embedding = self._embed_query(query)

        where = _build_metadata_where(
            metadata_filter=metadata_filter,
            required_filter=self._required_metadata_filter,
            ignore_chunk_ids=self._state.pruned_chunk_ids | self._state.seen_chunk_ids,
            ignore_doc_ids=self._state.pruned_doc_ids | self._state.seen_doc_ids,
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
                "type": elt_type,
                "distance": dist,
                "text": (
                    f"[{rank}] chunk_id={cid} | doc_id={doc_id} | "
                    f"type={elt_type} | distance={dist:.4f}\n{doc or ''}"
                ),
            })

        self._state.seen_chunk_ids.update(c["chunk_id"] for c in chunks)
        return {SEARCH_RESULT_TAG: True, "chunks": chunks}

    doc = """\
### search_corpus(query: str, top_k: int, metadata_filter: dict | None = None)
This tool performs a vector search over all chunks in the corpus. The input `query` is embedded and the `top_k` most relevant chunks are returned, each labelled with its `chunk_id` and `doc_id`. You can optionally restrict the search to a subset of the corpus by passing a `metadata_filter`, which is a ChromaDB-style where clause over chunk metadata. Chunks already returned to you (by `search_corpus` / `grep_corpus`) and documents you have already `read_document`-ed are automatically excluded from the results — as is anything you have `prune(...)`-ed — so each call surfaces new material. (To revisit material you've already fetched, look back in your context or `read_document` it again.)

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
    # `SearchAgentConfig.grep_max_output_tokens`; this is just thousands-formatting.
    _TOKENS_PER_K = 1000

    def __init__(
        self,
        chroma_collection: Collection,
        max_output_tokens: int,
        state: RetrievalState | None = None,
        required_metadata_filter: dict | None = None,
    ):
        # `state` shared by reference with the owning agent's other tools; omitted
        # for one-shot use (private fresh one).
        self._chroma_collection = chroma_collection
        self._state = state if state is not None else RetrievalState()
        # Hard cap on the rendered observation size (chars). `limit=None` returns every
        # matching chunk, so a broad pattern can otherwise dump 100s of K of tokens into
        # the context in one shot and 400 the next request (see SearchAgentConfig.grep_max_output_tokens).
        self._max_output_chars = max_output_tokens * self._CHARS_PER_TOKEN
        self._required_metadata_filter = required_metadata_filter

    def __call__(
        self,
        pattern: str,
        metadata_filter: dict | None = None,
        limit: int | None = None,
    ) -> dict:
        where = _build_metadata_where(
            metadata_filter=metadata_filter,
            required_filter=self._required_metadata_filter,
            ignore_chunk_ids=self._state.pruned_chunk_ids | self._state.seen_chunk_ids,
            ignore_doc_ids=self._state.pruned_doc_ids | self._state.seen_doc_ids,
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

        for g in groups:
            self._state.seen_chunk_ids.update(ch["chunk_id"] for ch in g["chunks"])
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
This tool performs a regex search over the cleaned text of every chunk in the corpus and returns the matching chunks grouped by their `doc_id`. Each hit includes its `chunk_id` so you can later refer to it or prune it. By default (`limit=None`), every matching chunk is returned -- which is useful for "find every doc that mentions X" queries -- but you should pass `limit=N` for narrower exploratory searches. The same `metadata_filter` syntax as `search_corpus` is supported. Chunks already returned to you and documents you have already `read_document`-ed are automatically excluded from the results (as is anything you have `prune(...)`-ed), so each call surfaces new material. The total output is capped: if a broad pattern matches more than the cap, the result is truncated with a note telling you how many hits were omitted -- narrow the pattern, add a `metadata_filter`, or pass `limit=N` to see the rest.

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

    def __init__(self, document_map: dict[str, str], max_pages: int, max_output_chars: int,
                 state: RetrievalState | None = None):
        self._document_map = document_map
        self._state = state if state is not None else RetrievalState()
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
                self._state.seen_doc_ids.add(did)
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
        renders_dir: str | Path | None = None,
        dpi: int = 300,
        fmt: str = "png",
        page_ref_parser: Callable[[str], "PageRef"] | None = None,
    ):
        self._document_map = document_map
        self._pdf_dir = pdf_dir
        self._renders_dir = renders_dir
        self._dpi = dpi
        self._fmt = fmt
        # doc_id → PageRef, so the corpus's page-key scheme is injectable (raise
        # ValueError for an unparseable id). Default: the "<stem>_<page>" scheme.
        self._page_ref_parser = page_ref_parser or page_key_to_pageref

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
        # doc_id is a corpus page key; resolve to a PageRef for rendering.
        try:
            ref = self._page_ref_parser(doc_id)
        except ValueError:
            return {
                VIEW_FIGURE_RESULT_TAG: True,
                "error": f"doc_id {doc_id!r} is not a parseable page key for this corpus",
            }
        try:
            img = render_page_b64(
                ref.stem, ref.page,
                pdf_dir=self._pdf_dir, renders_dir=self._renders_dir, dpi=self._dpi, fmt=self._fmt,
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

    def __init__(self, state: RetrievalState):
        self._state = state

    def __call__(
        self,
        chunk_ids: list[str] | None = None,
        doc_ids: list[str] | None = None,
    ) -> dict:
        # Single writer of the shared prune sets: mutate directly, then report
        # how many were newly added (the agent renders the summary, no re-apply).
        new_chunks = set(chunk_ids or ()) - self._state.pruned_chunk_ids
        new_docs = set(doc_ids or ()) - self._state.pruned_doc_ids
        self._state.pruned_chunk_ids.update(new_chunks)
        self._state.pruned_doc_ids.update(new_docs)
        return {
            PRUNE_RESULT_TAG: True,
            "new_chunk_count": len(new_chunks),
            "new_doc_count": len(new_docs),
            "total_chunks": len(self._state.pruned_chunk_ids),
            "total_docs": len(self._state.pruned_doc_ids),
        }

    doc = """\
### prune(chunk_ids: list[str] | None = None, doc_ids: list[str] | None = None)
This tool records `chunk_id`s and/or `doc_id`s that you've already inspected and deemed irrelevant. Pruned ids are excluded from the results of all subsequent `search_corpus` and `grep_corpus` calls, and are dropped from your visible context. Use this aggressively whenever a search surfaces clearly irrelevant chunks or docs, to keep later searches focused and your context window manageable.

```python
# mark some chunks and a whole doc as irrelevant
prune(chunk_ids=["chunk_id_1", "chunk_id_2"], doc_ids=["doc_id_3"])
```"""
