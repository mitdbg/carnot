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

Each tool is an instance of the `MultiTurnAgent.Tool` ABC: whose `doc` is spliced
into the system prompt, and whose `__call__` is registered with the sandbox under
`name` and invoked from model-emitted python.

Structured returns: every retrieval tool returns a *tagged dict* (never a bare
string — errors are carried in an "error" field of the tagged payload) so the
return shape is uniform. Each chunk carries its `chunk_id` / `doc_id`, which
`SearchAgent._blocks_from_output` turns into `ChunkBlock`s: this keeps the full
trajectory for reward computation while redacting pruned chunks from the
generation view (`_block_is_visible`). The tag constants below discriminate each
payload shape. The final answer is a JSON block (handled by the agent loop), not
a tool.

Shared state: one `RetrievalState` object is shared BY REFERENCE between the
`SearchAgent` and its tools. The `RetrievalState` maintains the set of retrieved,
seen, and pruned chunk/doc ids. The `SearchAgent` can fetch data into the
`RetrievalState` without immediately rendering it into its context window. The
`SearchAgent` may then issue additional queries over only the `RetrievalState`
to populate its context window. When invoking tools, the agent has the ability
to specify whether to:

  1. Execute them over the entire corpus or only the subset of data in `RetrievalState`
  2. Pull the data into the agent's context window or only into the `RetrievalState`
  3. Prune the data from the `RetrievalState` or only from the agent's context window
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from chromadb.api.models.Collection import Collection

from skunk.common import estimate_tokens, page_key_to_pageref, render_page_b64
from skunk.constants import CHARS_PER_TOKEN_EST
from skunk.multi_turn_agent import Tool
from skunk.search_agent.retrieval_state import RetrievalState
from skunk.storage.document_map import DocumentMap
from skunk.trace import truncate
from skunk.usage import match_model_entry

if TYPE_CHECKING:
    from skunk.common import ExecutionContext
    from skunk.llm_client import LLMClient

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
    doc = """\
### search_corpus(query: str, top_k: int, metadata_filter: dict | None = None)
This tool performs a vector search over all chunks in the corpus. The input `query` is embedded and the `top_k` most relevant chunks are returned, each labelled with its `chunk_id` and `doc_id`. You can optionally restrict the search to a subset of the corpus by passing a `metadata_filter`, which is a ChromaDB-style where clause over chunk metadata. Chunks already returned to you by earlier searches and documents you have already `read_document`-ed are automatically excluded from the results — as is anything you have `prune(...)`-ed — so each call surfaces new material. (To revisit material you've already fetched, look back in your context or `read_document` it again.)

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

    def __init__(
        self,
        chroma_collection: Collection,
        emb_model_id: str,
        llm_client: LLMClient,
        state: RetrievalState | None = None,
        required_metadata_filter: dict | None = None,
        ctx: ExecutionContext | None = None,
        usage_key: str = "default",
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
        self._usage_key = usage_key

    def _embed_query(self, query: str) -> list[float]:
        """Embed `query` with the same model that produced the stored embeddings, via the
        LLMClient — which owns backend dispatch (OpenRouter / vLLM),
        the process-wide "embed" rate bucket, retry, and usage accounting."""
        return self._llm_client.embed_query(query, model=self._emb_model_id, ctx=self._ctx, usage_key=self._usage_key)

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
            est_num_tokens = estimate_tokens(doc)
            chunks.append({
                "chunk_id": cid,
                "doc_id": doc_id,
                "type": elt_type,
                "distance": dist,
                "text": (
                    f"[{rank}] chunk_id={cid} | doc_id={doc_id} | "
                    f"type={elt_type} | est_num_tokens={est_num_tokens} | distance={dist:.4f}\n{doc or ''}"
                ),
            })

        self._state.seen_chunk_ids.update(c["chunk_id"] for c in chunks)
        return {SEARCH_RESULT_TAG: True, "chunks": chunks}


class GrepCorpusTool(Tool):
    name = "grep_corpus"
    doc = """\
### grep_corpus(pattern: str, metadata_filter: dict | None = None, limit: int | None = None, max_output_tokens: int | None = None)
This tool performs a regex search over the cleaned text of every chunk in the corpus and returns the matching chunks grouped by their `doc_id`. Each hit includes its `chunk_id` so you can later refer to it or prune it. By default (`limit=None`), every matching chunk is returned -- which is useful for "find every doc that mentions X" queries -- but you should pass `limit=N` for narrower exploratory searches. You can optionally restrict the search to a subset of the corpus by passing a `metadata_filter`, which is a ChromaDB-style where clause over chunk metadata. Chunks already returned to you and documents you have already `read_document`-ed are automatically excluded from the results (as is anything you have `prune(...)`-ed), so each call surfaces new material. The output is unlimited by default; pass `max_output_tokens` to cap it. If the matches exceed the cap, the result is truncated with a note telling you how many hits were omitted -- narrow the pattern, add a `metadata_filter`, pass `limit=N`, or adjust `max_output_tokens` to see the rest. A small `max_output_tokens` is a cheap way to peek at what a broad pattern surfaces before committing to the full output.

Supported `metadata_filter` syntax:
- Equality: `{"field": value}`
- Set membership: `{"field": {"$in": [v1, v2, ...]}}`
- Negation / not-in: `{"field": {"$nin": [...]}}`
- Compound: `{"$and": [clause1, clause2, ...]}` or `{"$or": [...]}`

```python
# find every chunk that mentions "topic X" (case insensitive)
grep_corpus("(?i)topic X")

# find chunks matching the literal phrase "topic Y" within a filtered subset, capped at 50 hits
grep_corpus("topic Y", metadata_filter={"field_a": {"$in": ["value_1", "value_2"]}}, limit=50)

# broad sweep, but keep the observation small: cap the output at ~5k tokens
grep_corpus("(?i)topic Z", max_output_tokens=5000)
```"""

    def __init__(
        self,
        chroma_collection: Collection,
        state: RetrievalState | None = None,
        required_metadata_filter: dict | None = None,
    ):
        # `state` shared by reference with the owning agent's other tools; omitted
        # for one-shot use (private fresh one).
        self._chroma_collection = chroma_collection
        self._state = state if state is not None else RetrievalState()
        self._required_metadata_filter = required_metadata_filter

    def __call__(
        self,
        pattern: str,
        metadata_filter: dict | None = None,
        limit: int | None = None,
        max_output_tokens: int | None = None,
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

        # Build groups in deterministic (doc_id, element_id) order. When the agent passed
        # `max_output_tokens`, stop once the rendered size (counted via `estimate_tokens`)
        # would exceed it; dropped hits are reported via a `truncation_note` so the agent
        # knows to narrow its pattern / adjust the cap rather than assuming it saw everything.
        groups: list[dict] = []
        used_tokens = 0
        total_chunks = sum(len(v) for v in grouped.values())
        kept_chunks = 0
        truncated = False
        for doc_id in sorted(grouped):
            if truncated:
                break
            header = f"\n# doc_id={doc_id}"
            doc_chunks: list[dict] = []
            for _, cid, text in sorted(grouped[doc_id]):
                est_num_tokens = estimate_tokens(text)
                chunk_text = f"  [chunk_id={cid} | est_num_tokens={est_num_tokens}] {text}"
                if max_output_tokens is not None:
                    # Count the header only once we commit the first chunk of this doc.
                    cost = estimate_tokens(chunk_text) + (estimate_tokens(header) if not doc_chunks else 0)
                    if doc_chunks or groups:  # always allow the very first chunk through
                        if used_tokens + cost > max_output_tokens:
                            truncated = True
                            break
                    used_tokens += cost
                doc_chunks.append({"chunk_id": cid, "doc_id": doc_id, "text": chunk_text})
                kept_chunks += 1
            if doc_chunks:
                groups.append({"doc_id": doc_id, "header": header, "chunks": doc_chunks})

        for g in groups:
            self._state.seen_chunk_ids.update(ch["chunk_id"] for ch in g["chunks"])
        result: dict = {GREP_RESULT_TAG: True, "groups": groups}
        if truncated:
            dropped = total_chunks - kept_chunks
            result["truncation_note"] = (
                f"[grep_corpus output truncated: showing {kept_chunks} of {total_chunks} matching "
                f"chunk(s) (~{max_output_tokens:,}-token cap reached); {dropped} chunk(s) omitted. Narrow the "
                f"pattern, add a metadata_filter, pass limit=N, or raise max_output_tokens to see the rest.]"
            )
        return result


class ReadDocumentTool(Tool):
    """Fetch the full (or partial) text of one or more retrieval units by `doc_id` from `document_map`.

    A "document" is the corpus's retrieval unit (see the module docstring), so what this
    returns is a PDF page for OfficeQA/FinanceBench, an abstract for TREC-BioGen, an
    assembled Wikipedia article for QAMPARI, or a source file for FreshStack."""

    name = "read_document"
    doc = """\
### read_document(doc_id: str | list[str], start_char_idx: int | None | list[int | None] = None, end_char_idx: int | None | list[int | None] = None)
This tool returns the cleaned text of one or more documents, given their `doc_id`(s). You can optionally specify `start_char_idx` and `end_char_idx` to fetch only a substring of each document. If you pass a list for `doc_id`, `start_char_idx` and `end_char_idx` must be lists of the same length or `None`, and each index pair will be applied to the corresponding document. If `end_char_idx` exceeds the document length, the document will be truncated to its actual length. If `start_char_idx` is negative, it counts from the end of the document. If `end_char_idx` is negative, it counts from the end of the document.

```python
# read two specific documents by id
read_document(["doc_id_1", "doc_id_2"])

# read the first 1000 characters of a document
read_document("doc_id_3", start_char_idx=0, end_char_idx=1000)

# read the last 1000 characters of a document and the full text of another document
read_document(["doc_id_4", "doc_id_5"], start_char_idx=[-1000, None])
```"""

    def __init__(self, document_map: DocumentMap, state: RetrievalState | None = None):
        self._document_map = document_map
        self._state = state if state is not None else RetrievalState()

    def __call__(self, doc_id: str | list[str], start_char_idx: int | None | list[int | None] = None, end_char_idx: int | None | list[int | None] = None) -> dict:
        if isinstance(doc_id, list):
            err_msg = "When doc_id is a list, start_char_idx and end_char_idx must be None or also be lists of the same length."
            assert start_char_idx is None or (isinstance(start_char_idx, list) and len(doc_id) == len(start_char_idx)), err_msg
            assert end_char_idx is None or (isinstance(end_char_idx, list) and len(doc_id) == len(end_char_idx)), err_msg
        doc_ids = list(doc_id) if isinstance(doc_id, list) else [doc_id]
        start_indices = list(start_char_idx) if isinstance(start_char_idx, list) else [start_char_idx]
        end_indices = list(end_char_idx) if isinstance(end_char_idx, list) else [end_char_idx]

        docs: list[dict] = []
        for did, start_idx, end_idx in zip(doc_ids, start_indices, end_indices, strict=True):
            text = self._document_map.get(did)
            total_chars = 0 if text is None else len(text)
            if text is None:
                body = "[no such document (or no content in document)]"
            else:
                body = text
                if start_idx is not None or end_idx is not None:
                    start_idx = start_idx or 0
                    end_idx = end_idx if end_idx is not None else len(text)
                    body = text[start_idx:end_idx]
                self._state.seen_doc_ids.add(did)
            rendered = f"=== doc_id={did} | total chars: {total_chars} ===\n{body}"
            docs.append({"doc_id": did, "text": rendered})
        return {READ_DOCUMENT_RESULT_TAG: True, "docs": docs}


class ViewFigureTool(Tool):
    """Render the full page for a doc_id the agent saw while reading — typically because
    the page text contains a `<figure id=N>` placeholder — and hand it back as an image
    observation. We render the *whole* page (not a bbox crop) so the agent sees any
    figures in context, and so OCR coordinate errors can't clip a chart."""

    name = "view_figure"
    doc = """\
### view_figure(doc_id: str)
When you read a document and see a `<figure id=N>` placeholder (a chart/figure that is NOT in the searchable text), call this tool to actually *see* it. It returns an image of the **entire page** — so you see every figure on the page in context — appended to your messages, just like reading the page's text. Use it to judge whether a page whose answer may live in a chart is relevant. Pass the `doc_id` of the page you read.

```python
# you read doc_id "2002_12_8" and saw "<figure id=5>"; now view the page
view_figure("2002_12_8")
```"""

    def __init__(
        self,
        document_map: DocumentMap,
        pdf_dir: str | Path,
        *,
        renders_dir: str | Path | None = None,
        dpi: int = 300,
        fmt: str = "png",
    ):
        self._document_map = document_map
        self._pdf_dir = pdf_dir
        self._renders_dir = renders_dir
        self._dpi = dpi
        self._fmt = fmt

    def __call__(self, doc_id: str) -> dict:
        if self._document_map.get(doc_id) is None:
            return {VIEW_FIGURE_RESULT_TAG: True, "error": f"no such document: {doc_id!r}"}
        # doc_id is a corpus page key; resolve to a PageRef for rendering.
        try:
            ref = page_key_to_pageref(doc_id)
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
            "mime": img.mime,
            "data": img.data,
        }


class PruneTool(Tool):
    name = "prune"
    doc = """\
### prune(chunk_ids: list[str] | None = None, doc_ids: list[str] | None = None)
This tool records `chunk_id`s and/or `doc_id`s that you've already inspected and deemed irrelevant. Pruned ids are excluded from the results of all subsequent corpus search/filter calls, and are dropped from your visible context. Use this aggressively whenever a search surfaces clearly irrelevant chunks or docs, to keep later searches focused and your context window manageable.

```python
# mark some chunks and a whole doc as irrelevant
prune(chunk_ids=["chunk_id_1", "chunk_id_2"], doc_ids=["doc_id_3"])
```"""

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
# document text is head-truncated so the whole request fits. Overhead is counted via
# `common.estimate_tokens`; the token budget converts back to a char cap with the house
# ~4 chars/token estimate, with `_JUDGE_CTX_SAFETY` headroom to absorb that estimate's error.
_JUDGE_OUTPUT_TOKENS = 2048
_JUDGE_CTX_SAFETY = 0.9
# Fixed scaffolding around the document in the judge `user` message (see `_judge_one`); its length is
# charged against the budget so the document gets what's left.
_JUDGE_USER_WRAPPER = "Filter Condition: \n\nDocument:\n"
_JUDGE_TRUNC_MARKER = "\n…(truncated to fit judge context)"


def _judge_doc_char_budget(context_limit: int, predicate: str, max_output_tokens: int = _JUDGE_OUTPUT_TOKENS) -> int:
    """Max chars of document text that fit in one judge request under `context_limit` (tokens):
    the fixed overhead (prompt scaffolding, counted via `estimate_tokens`) and the judge's output
    headroom (`max_output_tokens`) are subtracted, then the remaining token budget converts to
    chars at ~4 chars/token with a safety factor. Returns 0 when the fixed overhead alone already
    exceeds the limit."""
    overhead_tokens = max_output_tokens + estimate_tokens(
        _SEMFILTER_SYSTEM + _JUDGE_USER_WRAPPER + predicate
    )
    budget_tokens = context_limit * _JUDGE_CTX_SAFETY - overhead_tokens
    return max(0, int(budget_tokens * CHARS_PER_TOKEN_EST))


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
    llm_client: LLMClient, predicate: str, item_text: str, model: str, ctx: ExecutionContext | None,
    provider_order=None, max_output_tokens: int = _JUDGE_OUTPUT_TOKENS, usage_key: str = "default",
) -> bool:
    user = f"Filter Condition: {predicate}\n\nDocument:\n{item_text}"
    try:
        messages = [{"role": "user", "content": user}]
        resp = llm_client.call(
            system=_SEMFILTER_SYSTEM, messages=messages, temperature=0.0, model=model, ctx=ctx, call_site="semfilter",
            provider_order=provider_order, max_output_tokens=max_output_tokens, usage_key=usage_key,
        )
    except Exception:
        return True  # recall-safe: keep on error
    return _parse_bool(resp.text)


def filter_docs(
    llm_client,
    predicate: str,
    doc_ids: list[str],
    document_map: DocumentMap,
    model: str,
    *,
    ctx: ExecutionContext | None = None,
    max_workers: int = 8,
    event_extra: dict | None = None,
    provider_order: list[str] | None = None,
    context_limit: int | None = None,
    judge_max_output_tokens: int = _JUDGE_OUTPUT_TOKENS,
    usage_key: str = "default",
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
            lambda it: _judge_one(llm_client, predicate, it, model, ctx, provider_order, judge_max_output_tokens, usage_key=usage_key),
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
    doc = """\
### semantic_filter(predicate: str, metadata_filter: dict | None = None, top_k: int | None = None, search_str: str | None = None, exclude: bool = True)
Keep only the documents whose FULL text satisfies a natural-language `predicate`: candidates are selected from the corpus, each is judged independently by an LLM (TRUE/FALSE), and the survivors are returned. Select candidates with a `metadata_filter` (a ChromaDB-style where clause over chunk metadata) and/or `top_k`. With `top_k`, a vector search first selects the `top_k` most relevant chunks — embedding `search_str` if given, else the predicate; prefer a short focused `search_str` when the predicate is long or compound. With only a `metadata_filter`, every matching chunk is a candidate. Candidate chunks are deduped to their parent documents before judging. Returns the kept `doc_id`s PLUS the candidate chunks of the kept documents as text snippets. Anything you have `prune(...)`-ed is excluded from the candidates; a very large snippet output is truncated with a note. By default (`exclude=True`) documents already retrieved for judging by earlier tool calls are also excluded, so each call surfaces only new candidates; pass `exclude=False` to re-judge previously-seen documents under a new predicate.
`search_str` requires `top_k`. At most 1,000 candidate documents per call — narrow the filter or use `top_k` if you exceed it. The returned dict carries `kept_doc_ids` for programmatic use.

Supported `metadata_filter` syntax:
- Equality: `{"field": value}`
- Set membership: `{"field": {"$in": [v1, v2, ...]}}`
- Negation / not-in: `{"field": {"$nin": [...]}}`
- Compound: `{"$and": [clause1, clause2, ...]}` or `{"$or": [...]}`

```python
# metadata only: judge every document from 1946
semantic_filter(predicate="mentions coal shortages affecting steel production", metadata_filter={"year": "1946"})

# vector prefilter: judge the documents behind the 200 chunks nearest a short query
semantic_filter(
    predicate="describes a government intervention in response to a labor strike, naming the statute invoked",
    top_k=200,
    search_str="government intervention strike",
)
```"""

    _TOKENS_PER_K = 1000
    # Page size for the metadata-only candidate scan (ids/metadatas only, no texts) and
    # for the post-judge text fetches — bounds per-request payloads against Chroma.
    _GET_PAGE_SIZE = 10_000

    def __init__(
        self,
        llm_client: LLMClient,
        document_map: DocumentMap,
        model: str,
        *,
        chroma_collection: Collection | None = None,
        emb_model_id: str | None = None,
        max_candidate_docs: int = 1000,
        max_output_tokens: int = 50_000,
        ctx: ExecutionContext | None = None,
        provider_order: list[str] | None = None,
        context_limits: dict[str, int] | None = None,
        judge_max_output_tokens: int = _JUDGE_OUTPUT_TOKENS,
        usage_key: str = "default",
    ) -> None:
        # `chroma_collection` is required to select candidates from the corpus (and, for
        # top_k vector prefiltering, `emb_model_id`); without a collection every call errors.
        self._llm_client = llm_client
        self._document_map = document_map
        self._model = model
        self._usage_key = usage_key
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
        self._max_output_chars = max_output_tokens * CHARS_PER_TOKEN_EST
        self._ctx = ctx
        # Private fresh state for one-shot use; the owning SearchAgent replaces it via
        # `bind_retrieval_state` so corpus mode honors prunes and records seen chunks.
        self._state = RetrievalState()

    def bind_retrieval_state(self, state: RetrievalState) -> None:
        """Called by `SearchAgent.__init__` (duck-typed) to share the agent's per-question
        `RetrievalState`: corpus-mode candidate selection always excludes PRUNED chunks/docs
        and, by default (`exclude=True`), SEEN ones too so each call surfaces new material;
        `exclude=False` keeps the filter comprehensive over seen-but-not-pruned docs. Rendered
        chunks are recorded as seen."""
        self._state = state

    def _error(self, msg: str) -> dict:
        return {SEMFILTER_RESULT_TAG: True, "error": msg}

    def __call__(
        self,
        predicate: str,
        metadata_filter: dict | None = None,
        top_k: int | None = None,
        search_str: str | None = None,
        exclude: bool = True,
    ) -> dict:
        if not predicate or not str(predicate).strip():
            return self._error("semantic_filter error: `predicate` is required.")
        if search_str is not None and top_k is None:
            return self._error(
                "semantic_filter error: search_str requires top_k (it is the query for the top_k vector search)."
            )
        if metadata_filter is None and top_k is None:
            return self._error(
                "semantic_filter error: provide a metadata_filter and/or top_k to select "
                "candidates from the corpus."
            )

        return self._corpus_mode(predicate, metadata_filter, top_k, search_str, exclude)

    def _corpus_mode(
        self, predicate: str, metadata_filter: dict | None, top_k: int | None, search_str: str | None,
        exclude: bool,
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

        # Always exclude PRUNED material. When `exclude` (the default), also exclude SEEN
        # material — chunks already returned to the agent by any tool plus docs it has read —
        # so each call surfaces only NEW candidates, like grep/search_corpus. Pass exclude=False
        # to keep the filter comprehensive over seen-but-not-pruned docs, e.g. to re-judge
        # already-fetched material under a new predicate.
        ignore_chunk_ids = (
            self._state.pruned_chunk_ids | self._state.seen_chunk_ids if exclude
            else self._state.pruned_chunk_ids
        )
        ignore_doc_ids = (
            self._state.pruned_doc_ids | self._state.seen_doc_ids if exclude
            else self._state.pruned_doc_ids
        )
        where = _build_metadata_where(
            metadata_filter=metadata_filter,
            ignore_chunk_ids=ignore_chunk_ids,
            ignore_doc_ids=ignore_doc_ids,
        )

        # Candidate chunks: (chunk_id, doc_id, text | None). Vector mode carries texts
        # (and relevance order); metadata mode defers texts to a post-judge fetch.
        if top_k is not None:
            try:
                emb = self._llm_client.embed_query(search_str or predicate, model=self._emb_model_id, ctx=self._ctx, usage_key=self._usage_key)
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
            # str(): chroma types metadata values as a broad union, but doc_id is always a string.
            candidate_doc_ids = list(dict.fromkeys(str(did) for _, did, _ in candidates))  # relevance order
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
            context_limit=self._context_limit,
            judge_max_output_tokens=self._judge_max_output_tokens,
            usage_key=self._usage_key,
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
                # metadata mode is only reached from _corpus_mode, which guards the collection is present.
                assert self._chroma_collection is not None
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
            cap_k = self._max_output_chars // CHARS_PER_TOKEN_EST // self._TOKENS_PER_K
            note = (
                f"[semantic_filter output truncated: showing {len(rendered)} of {len(kept_chunks)} chunk(s) from "
                f"kept documents (~{cap_k}k-token cap reached); {dropped} chunk(s) omitted. All kept_doc_ids are "
                f"listed in the summary; use read_document to see the omitted material.]"
            )
        return rendered, note
