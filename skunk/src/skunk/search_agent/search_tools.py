"""Tool implementations for the SearchAgent.

All retrieval tools are backed by a single ChromaDB collection in which each
row is one *chunk* (the embedded element) of a *document*. A "document" here
is the corpus's RETRIEVAL UNIT — the thing `read_document` fetches whole,
prune/seen state tracks, and the final answer returns — chosen per corpus
when the collection is built, NOT necessarily a whole source file: a single
page of a PDF report (OfficeQA: doc_id "1946_11_41" = the 1946-11 bulletin's
page 41; FinanceBench: "3M_2018_10K::p59"), a scraped web page
(BrowseComp-Plus), a PubMed abstract (TREC-BioGen), a Wikipedia article
(QAMPARI), or a source file (FreshStack). The layout — produced by the
corpus-prep scripts that live with the benchmarks (e.g. qatfd's
`engaging-scripts/create_vector_db.py`; taxonomy in qatfd's
CORPUS_MODEL.md) — is:

    id:        the `chunk_id` (e.g. "1946_11_41_6")
    document:  the cleaned chunk text (read from the `documents` field)
    metadata:  {
        doc_id:     str   -- the retrieval-unit key, e.g. "1946_11_41"
        chunk_id:   str   -- duplicate of the row id, for $nin filterability
        element_id: int   -- int ordering the chunk within its document
                             (a 0-based index, or any monotonic stand-in)
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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from chromadb.api.models.Collection import Collection

from skunk.common import page_key_to_pageref
from skunk.common import render_page_b64
from skunk.multi_turn_agent import Tool
from skunk.trace import truncate
from skunk.usage import match_model_entry

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

    Extra tools (constructed by the caller before the agent's state exists) opt in
    via a duck-typed `bind_retrieval_state(state)` that `SearchAgent.__init__` calls — e.g.
    `SemanticFilterTool`, which reads the pruned sets only (its corpus filter stays
    comprehensive over seen-but-not-pruned material) and writes `seen_chunk_ids`.
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
SEMFILTER_RESULT_TAG = "__semfilter_result__"

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
        LLMClient — which owns backend dispatch (OpenRouter / vLLM),
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
    """Fetch the full text of one or more retrieval units by `doc_id` from `document_map`.

    A "document" is the corpus's retrieval unit (see the module docstring), so what this
    returns is a PDF page for OfficeQA/FinanceBench, an abstract for TREC-BioGen, an
    assembled Wikipedia article for QAMPARI, or a source file for FreshStack."""

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


# ---------------------------------------------------------------------------
# Semantic filter: an LLM TRUE/FALSE judge applied per document. The judging
# rubric mirrors carnot's `sem_filter.yaml`. The core is synchronous
# (`filter_docs`) because tools run inside the `LocalPythonExecutor`, which
# calls them synchronously; the sync `LLMClient.call` is used per document,
# fanned out over a small thread pool.
# ---------------------------------------------------------------------------

# Per-doc text snippet cap in the structured trace event. Each filtered doc carries a
# preview so the trace viewer can expand it on click without a second corpus lookup; the
# cap keeps a 1,000-doc filter from bloating the event stream with full document bodies.
_SEMFILTER_DOC_PREVIEW_MAX = 2000

_SEMFILTER_SYSTEM = (
    "You determine whether a document satisfies a filter condition. You are given a "
    "filter condition and a document. Output TRUE if the document satisfies the "
    "condition, and FALSE otherwise. Don't overthink your reasoning. Your reply must be "
    "exactly TRUE or FALSE — a single word, with no explanation, reasoning, or any other text."
)

# Judge-request sizing. The judge call caps its output at `_JUDGE_OUTPUT_TOKENS` tokens by default
# (overridable per tool via `SearchAgentConfig.semantic_filter_max_output_tokens`); the reply is a
# single word, but a reasoning judge model needs enough budget to think before emitting the verdict —
# too small a cap makes it hit finish_reason=length with empty content, which retries + backs off and
# tanks throughput. When the judge model has a known context window (see `context_limits`) the
# document text is head-truncated so the whole request fits. Sizing uses the house ~4 chars/token
# estimate (there is no tokenizer) with `_JUDGE_CTX_SAFETY` headroom to absorb that estimate's error.
_JUDGE_OUTPUT_TOKENS = 2048
_JUDGE_CTX_SAFETY = 0.9
_JUDGE_CHARS_PER_TOKEN = 4
# Fixed scaffolding around the document in the judge `user` message (see `_judge_one`); its length is
# charged against the budget so the document gets what's left.
_JUDGE_USER_WRAPPER = "Filter Condition: \n\nDocument:\n"
_JUDGE_TRUNC_MARKER = "\n…(truncated to fit judge context)"


def _judge_doc_char_budget(context_limit: int, predicate: str, max_output_tokens: int = _JUDGE_OUTPUT_TOKENS) -> int:
    """Max chars of document text that fit in one judge request under `context_limit` (tokens),
    using the ~4 chars/token estimate with a safety factor and headroom reserved for the judge's
    output (`max_output_tokens`). Returns 0 when the fixed overhead alone already exceeds the limit."""
    overhead_tokens = max_output_tokens + (
        len(_SEMFILTER_SYSTEM) + len(_JUDGE_USER_WRAPPER) + len(predicate)
    ) / _JUDGE_CHARS_PER_TOKEN
    budget_tokens = context_limit * _JUDGE_CTX_SAFETY - overhead_tokens
    return max(0, int(budget_tokens * _JUDGE_CHARS_PER_TOKEN))


def _truncate_doc_for_judge(text: str, budget_chars: int) -> tuple[str, bool]:
    """Head-truncate `text` to `budget_chars` (marker included), returning (text, was_truncated)."""
    if len(text) <= budget_chars:
        return text, False
    keep = max(0, budget_chars - len(_JUDGE_TRUNC_MARKER))
    return text[:keep] + _JUDGE_TRUNC_MARKER, True


def _parse_bool(text: str) -> bool:
    """True/False from the judge reply. Recall-safe: default TRUE (keep) if unclear."""
    t = (text or "").strip().lower()
    if t.startswith("false"):
        return False
    if t.startswith("true"):
        return True
    if "false" in t and "true" not in t:
        return False
    return True


def _judge_one(
    llm_client, predicate: str, item_text: str, model: str, ctx, provider_order=None,
    max_output_tokens: int = _JUDGE_OUTPUT_TOKENS,
) -> bool:
    user = f"Filter Condition: {predicate}\n\nDocument:\n{item_text}"
    try:
        resp = llm_client.call(
            system=_SEMFILTER_SYSTEM, user=user, temperature=0.0, model=model, ctx=ctx, call_site="semfilter",
            provider_order=provider_order, max_output_tokens=max_output_tokens,
        )
    except Exception:
        return True  # recall-safe: keep on error
    return _parse_bool(resp.text)


def filter_docs(
    llm_client,
    predicate: str,
    doc_ids: list[str],
    document_map: dict[str, str],
    model: str,
    *,
    ctx=None,
    max_workers: int = 8,
    event_extra: dict | None = None,
    provider_order: list[str] | None = None,
    context_limit: int | None = None,
    judge_max_output_tokens: int = _JUDGE_OUTPUT_TOKENS,
) -> list[str]:
    """Return the subset of `doc_ids` whose document text satisfies `predicate`,
    preserving input order.

    When `context_limit` (the judge model's context window, in tokens) is set, each
    document's text is head-truncated so the judge request fits — otherwise a document
    larger than the judge's window would 400 and, via `_judge_one`'s recall-safe fallback,
    be kept unjudged. `context_limit=None` sends the full text (unchanged behavior).

    When `ctx` is provided, emits one structured `semantic_filter` observation event
    recording the predicate and every input doc's verdict + text preview, so the trace
    viewer can render the pass/fail breakdown (green/red) and expand each doc on click.
    `event_extra` keys are merged into the event's `data` (callers record e.g. the
    candidate-selection mode); the message format is load-bearing — downstream metrics
    key on `semantic_filter n_in=...`."""
    if not doc_ids:
        return []
    texts = [document_map.get(d, "") or "" for d in doc_ids]
    n_truncated = 0
    if context_limit:
        budget = _judge_doc_char_budget(context_limit, predicate, judge_max_output_tokens)
        truncated = [_truncate_doc_for_judge(t, budget) for t in texts]
        texts = [t for t, _ in truncated]
        n_truncated = sum(1 for _, was in truncated if was)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(doc_ids))) as pool:
        verdicts = list(pool.map(
            lambda it: _judge_one(llm_client, predicate, it, model, ctx, provider_order, judge_max_output_tokens),
            texts,
        ))
    kept = [d for d, keep in zip(doc_ids, verdicts, strict=True) if keep]

    if ctx is not None:
        data = {
            "predicate": predicate,
            "n_in": len(doc_ids),
            "n_out": len(kept),
            "n_truncated": n_truncated,
            "docs": [
                {"doc_id": d, "kept": bool(keep), "text": truncate(t, _SEMFILTER_DOC_PREVIEW_MAX, "\n…(truncated)")}
                for d, keep, t in zip(doc_ids, verdicts, texts, strict=True)
            ],
        }
        if event_extra:
            data.update(event_extra)
        ctx.emit(
            f"semantic_filter n_in={len(doc_ids)} n_out={len(kept)} predicate={truncate(predicate, 80)!r}",
            kind="observation",
            data=data,
        )
    return kept


class SemanticFilterTool(Tool):
    name = "semantic_filter"

    # Token→char factor for the corpus-mode output cap (mirrors GrepCorpusTool).
    _CHARS_PER_TOKEN = 4
    _TOKENS_PER_K = 1000
    # Page size for the metadata-only candidate scan (ids/metadatas only, no texts) and
    # for the post-judge text fetches — bounds per-request payloads against Chroma.
    _GET_PAGE_SIZE = 10_000

    def __init__(
        self,
        llm_client,
        document_map: dict[str, str],
        model: str,
        *,
        chroma_collection: Collection | None = None,
        emb_model_id: str | None = None,
        max_candidate_docs: int = 1000,
        max_output_tokens: int = 50_000,
        ctx=None,
        provider_order: list[str] | None = None,
        context_limits: dict[str, int] | None = None,
        judge_max_output_tokens: int = _JUDGE_OUTPUT_TOKENS,
    ) -> None:
        # `chroma_collection` (and, for top_k, `emb_model_id`) enable corpus mode; a
        # tool constructed without them supports only the doc_ids mode.
        self._llm_client = llm_client
        self._document_map = document_map
        self._model = model
        # Per-call OpenRouter provider order for the judge calls only (None => client default). Lets a
        # cheaper judge model route to specific providers while the agent model stays unpinned.
        self._provider_order = provider_order
        # Judge model's context window (tokens), resolved once from the `model id/substring -> limit`
        # map by the same matcher the price table uses; None (model not in the map) => no truncation.
        self._context_limit = match_model_entry(model, context_limits or {})
        # Output-token cap per judge call. Large enough that a reasoning judge finishes thinking and
        # emits its TRUE/FALSE verdict (a too-small cap => finish_reason=length + empty content => retry storm).
        self._judge_max_output_tokens = judge_max_output_tokens
        self._chroma_collection = chroma_collection
        self._emb_model_id = emb_model_id
        self._max_candidate_docs = max_candidate_docs
        self._max_output_chars = max_output_tokens * self._CHARS_PER_TOKEN
        self._ctx = ctx
        # Private fresh state for one-shot use; the owning SearchAgent replaces it via
        # `bind_retrieval_state` so corpus mode honors prunes and records seen chunks.
        self._state = RetrievalState()

    def bind_retrieval_state(self, state: RetrievalState) -> None:
        """Called by `SearchAgent.__init__` (duck-typed) to share the agent's per-question
        `RetrievalState`: corpus-mode candidate selection excludes PRUNED chunks/docs
        (never seen ones — the filter must stay comprehensive), and rendered chunks are
        recorded as seen."""
        self._state = state

    def _error(self, msg: str) -> dict:
        return {SEMFILTER_RESULT_TAG: True, "error": msg}

    def __call__(
        self,
        predicate: str,
        doc_ids: list[str] | None = None,
        metadata_filter: dict | None = None,
        top_k: int | None = None,
        search_str: str | None = None,
    ) -> dict:
        if not predicate or not str(predicate).strip():
            return self._error("semantic_filter error: `predicate` is required.")
        corpus_args = metadata_filter is not None or top_k is not None or search_str is not None
        if doc_ids is not None and corpus_args:
            return self._error(
                "semantic_filter error: pass EITHER doc_ids OR corpus-mode arguments "
                "(metadata_filter / top_k / search_str), not both."
            )
        if search_str is not None and top_k is None:
            return self._error(
                "semantic_filter error: search_str requires top_k (it is the query for the top_k vector search)."
            )
        if doc_ids is None and metadata_filter is None and top_k is None:
            return self._error(
                "semantic_filter error: provide doc_ids, or a metadata_filter and/or top_k "
                "to select candidates from the corpus."
            )

        if doc_ids is not None:
            if isinstance(doc_ids, str):
                doc_ids = [doc_ids]
            kept = filter_docs(
                self._llm_client, predicate, list(doc_ids), self._document_map, self._model,
                ctx=self._ctx, event_extra={"mode": "doc_ids"}, provider_order=self._provider_order,
                context_limit=self._context_limit, judge_max_output_tokens=self._judge_max_output_tokens,
            )
            return {SEMFILTER_RESULT_TAG: True, "kept_doc_ids": kept, "n_in": len(doc_ids), "n_out": len(kept)}

        return self._corpus_mode(predicate, metadata_filter, top_k, search_str)

    # ---- corpus mode -------------------------------------------------------------

    def _corpus_mode(
        self, predicate: str, metadata_filter: dict | None, top_k: int | None, search_str: str | None
    ) -> dict:
        if self._chroma_collection is None:
            return self._error(
                "semantic_filter error: corpus mode is not available "
                "(tool was constructed without a chroma collection)."
            )
        if top_k is not None and self._emb_model_id is None:
            return self._error(
                "semantic_filter error: top_k is not available (tool was constructed without an embedding model)."
            )

        # PRUNED exclusions only — never seen ones: the filter must stay comprehensive
        # over material the agent has already fetched but not ruled out.
        where = _build_metadata_where(
            metadata_filter=metadata_filter,
            ignore_chunk_ids=self._state.pruned_chunk_ids,
            ignore_doc_ids=self._state.pruned_doc_ids,
        )

        # Candidate chunks: (chunk_id, doc_id, text | None). Vector mode carries texts
        # (and relevance order); metadata mode defers texts to a post-judge fetch.
        if top_k is not None:
            try:
                emb = self._llm_client.embed_query(search_str or predicate, model=self._emb_model_id, ctx=self._ctx)
                query_kwargs: dict = {
                    "query_embeddings": [emb],
                    "n_results": top_k,
                    "include": ["metadatas", "documents"],
                }
                if where is not None:
                    query_kwargs["where"] = where
                res = self._chroma_collection.query(**query_kwargs)
            except Exception as e:
                return self._error(f"semantic_filter error: {e}")
            candidates = [
                (cid, meta["doc_id"], doc or "")
                for cid, doc, meta in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], strict=True)  # type: ignore
            ]
            candidate_doc_ids = list(dict.fromkeys(did for _, did, _ in candidates))  # relevance order
            if len(candidate_doc_ids) > self._max_candidate_docs:
                return self._error(
                    f"semantic_filter error: the top_k={top_k} search yielded {len(candidate_doc_ids)} candidate "
                    f"documents, over the {self._max_candidate_docs}-document cap. Lower top_k or narrow the "
                    f"metadata_filter."
                )
            mode = "vector"
        else:
            # Paged ids/metadatas-only scan so a broad filter cannot pull a huge chunk
            # set (let alone texts) client-side before the doc cap trips.
            candidates = []
            doc_id_set: set[str] = set()
            offset = 0
            try:
                while True:
                    page = self._chroma_collection.get(
                        where=where, include=["metadatas"], limit=self._GET_PAGE_SIZE, offset=offset
                    )
                    ids = page["ids"]
                    if not ids:
                        break
                    for cid, meta in zip(ids, page["metadatas"], strict=True):  # type: ignore
                        candidates.append((cid, meta["doc_id"], None))
                        doc_id_set.add(meta["doc_id"])
                    if len(doc_id_set) > self._max_candidate_docs:
                        return self._error(
                            f"semantic_filter error: the metadata_filter matched more than "
                            f"{self._max_candidate_docs} candidate documents (the per-call cap). Narrow the "
                            f"metadata_filter, or pass top_k (with an optional search_str) to prefilter "
                            f"candidates by vector relevance."
                        )
                    offset += len(ids)
                    if len(ids) < self._GET_PAGE_SIZE:
                        break
            except Exception as e:
                return self._error(f"semantic_filter error: {e}")
            candidate_doc_ids = sorted(doc_id_set)  # deterministic order, mirrors grep
            mode = "metadata"

        if not candidate_doc_ids:
            # Renders as EMPTY_RESULT_MESSAGE (a prune filter may have eliminated everything).
            return {SEMFILTER_RESULT_TAG: True, "summary": "", "chunks": [], "kept_doc_ids": [], "n_in": 0, "n_out": 0}

        kept = filter_docs(
            self._llm_client, predicate, candidate_doc_ids, self._document_map, self._model,
            ctx=self._ctx,
            event_extra={
                "mode": mode,
                "metadata_filter": metadata_filter,
                "top_k": top_k,
                "search_str": search_str,
                "n_candidate_chunks": len(candidates),
                "n_candidate_docs": len(candidate_doc_ids),
            },
            provider_order=self._provider_order,
            context_limit=self._context_limit, judge_max_output_tokens=self._judge_max_output_tokens,
        )
        kept_set = set(kept)
        kept_chunks = [(cid, did, text) for cid, did, text in candidates if did in kept_set]
        if mode == "metadata":
            # (doc_id, chunk_id) order within the sorted kept docs; texts fetched lazily below.
            kept_chunks.sort(key=lambda c: (c[1], c[0]))

        try:
            rendered, truncation_note = self._render_chunks(kept_chunks, mode)
        except Exception as e:
            # The judging is done — surface the verdict even if the snippet fetch failed.
            return {
                SEMFILTER_RESULT_TAG: True,
                "error": f"semantic_filter error: judged the candidates (kept_doc_ids={kept}) but "
                         f"failed to fetch their chunk texts: {e}",
                "kept_doc_ids": kept,
                "n_in": len(candidate_doc_ids),
                "n_out": len(kept),
            }
        self._state.seen_chunk_ids.update(c["chunk_id"] for c in rendered)
        summary = (
            f"[semantic_filter] kept {len(kept)}/{len(candidate_doc_ids)} candidate document(s) matching the "
            f"predicate. kept_doc_ids={kept}." + (" Chunks from kept documents follow." if rendered else "")
        )
        result = {
            SEMFILTER_RESULT_TAG: True,
            "summary": summary,
            "chunks": rendered,
            "kept_doc_ids": kept,
            "n_in": len(candidate_doc_ids),
            "n_out": len(kept),
        }
        if truncation_note:
            result["truncation_note"] = truncation_note
        return result

    def _render_chunks(self, kept_chunks: list[tuple], mode: str) -> tuple[list[dict], str | None]:
        """Rendered snippet dicts for the kept docs' candidate chunks, capped at
        `_max_output_chars` (the first chunk is always allowed through, mirroring grep).
        Metadata-mode texts are fetched here, one `_GET_PAGE_SIZE` batch at a time, so
        chunks past the cap are never pulled from Chroma."""
        rendered: list[dict] = []
        used_chars = 0
        truncated = False
        for start in range(0, len(kept_chunks), self._GET_PAGE_SIZE):
            if truncated:
                break
            batch = kept_chunks[start:start + self._GET_PAGE_SIZE]
            if mode == "metadata":
                fetched = self._chroma_collection.get(ids=[cid for cid, _, _ in batch], include=["documents"])
                texts = dict(zip(fetched["ids"], fetched["documents"] or [], strict=True))
                batch = [(cid, did, texts.get(cid) or "") for cid, did, _ in batch]
            for cid, did, text in batch:
                chunk_text = f"chunk_id={cid} | doc_id={did}\n{text}"
                if rendered and used_chars + len(chunk_text) > self._max_output_chars:
                    truncated = True
                    break
                used_chars += len(chunk_text)
                rendered.append({"chunk_id": cid, "doc_id": did, "text": chunk_text})
        note = None
        if truncated:
            dropped = len(kept_chunks) - len(rendered)
            cap_k = self._max_output_chars // self._CHARS_PER_TOKEN // self._TOKENS_PER_K
            note = (
                f"[semantic_filter output truncated: showing {len(rendered)} of {len(kept_chunks)} chunk(s) from "
                f"kept documents (~{cap_k}k-token cap reached); {dropped} chunk(s) omitted. All kept_doc_ids are "
                f"listed in the summary; use read_document or grep_corpus to see the omitted material.]"
            )
        return rendered, note

    doc = """\
### semantic_filter(predicate: str, doc_ids: list[str] | None = None, metadata_filter: dict | None = None, top_k: int | None = None, search_str: str | None = None)
Keep only the documents whose FULL text satisfies a natural-language `predicate`: each candidate document is judged independently by an LLM (TRUE/FALSE) and the survivors are returned. Candidates come from exactly one of two sources:
- **doc_ids mode**: pass `doc_ids` you already collected. Returns just the kept `doc_id`s.
- **corpus mode**: pass a `metadata_filter` (same ChromaDB where-clause syntax as `grep_corpus`) and/or `top_k`. With `top_k`, a vector search first selects the `top_k` most relevant chunks — embedding `search_str` if given, else the predicate; prefer a short focused `search_str` when the predicate is long or compound. With only a `metadata_filter`, every matching chunk is a candidate. Candidate chunks are deduped to their parent documents before judging. Returns the kept `doc_id`s PLUS the candidate chunks of the kept documents as text snippets. Anything you have `prune(...)`-ed is excluded from the candidates; a very large snippet output is truncated with a note.
Do not combine `doc_ids` with the corpus-mode arguments. `search_str` requires `top_k`. At most 1,000 candidate documents per call — narrow the filter or use `top_k` if you exceed it. The returned dict carries `kept_doc_ids` for programmatic use.

```python
# corpus mode, metadata only: judge every document from 1946
semantic_filter(predicate="mentions coal shortages affecting steel production", metadata_filter={"year": "1946"})

# corpus mode, vector prefilter: judge the documents behind the 200 chunks nearest a short query
semantic_filter(
    predicate="describes a government intervention in response to a labor strike, naming the statute invoked",
    top_k=200,
    search_str="government intervention strike",
)

# doc_ids mode: narrow ids you already collected (returns kept ids only)
semantic_filter(predicate="discusses topic X in relation to Y", doc_ids=["doc_id_1", "doc_id_2"])
```"""
