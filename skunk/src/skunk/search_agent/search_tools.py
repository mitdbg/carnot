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
from jinja2 import Environment, StrictUndefined

from skunk.common import estimate_tokens, page_key_to_pageref, render_page_b64
from skunk.multi_turn_agent import Tool
from skunk.prompts import load_prompts
from skunk.search_state.working_set import WorkingSet
from skunk.storage.document_map import DocumentMap
from skunk.trace import truncate
from skunk.usage import match_model_entry

if TYPE_CHECKING:
    from skunk.common import ExecutionContext
    from skunk.llm_client import LLMClient

_ENV = Environment(
    autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined,
    trim_blocks=True, lstrip_blocks=True,
)

# templates / strings for each tool's docstring
_PROMPTS = load_prompts("search_tools")

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
    in_chunk_ids: set[str] | None,
    in_doc_ids: set[str] | None,
    not_in_chunk_ids: set[str] | None,
    not_in_doc_ids: set[str] | None,
) -> dict | None:
    """Construct a ChromaDB ``where`` dict from a user filter + inclusion / exclusion sets.

    ``metadata_filter`` is passed through as a ChromaDB-compatible where clause
    (e.g. ``{"year": "2010"}``, ``{"page_id": {"$in": [19, 26]}}``, or a
    compound ``{"$and": [...]}`` / ``{"$or": [...]}``). ``required_filter`` is a
    tool-construction-time clause the caller can never widen past (ANDed in when
    set). Both are ANDed with the server-side prune filters built from
    ``not_in_chunk_ids`` / ``not_in_doc_ids``.
    """
    clauses: list[dict] = []

    if metadata_filter:
        clauses.append(metadata_filter)
    if in_doc_ids and in_chunk_ids:
        doc_id_clause = {"doc_id": {"$in": sorted(in_doc_ids)}}
        chunk_id_clause = {"chunk_id": {"$in": sorted(in_chunk_ids)}}
        clauses.append({"$or": [doc_id_clause, chunk_id_clause]})
    elif in_doc_ids:
        clauses.append({"doc_id": {"$in": sorted(in_doc_ids)}})
    elif in_chunk_ids:
        clauses.append({"chunk_id": {"$in": sorted(in_chunk_ids)}})

    if not_in_doc_ids:
        clauses.append({"doc_id": {"$nin": sorted(not_in_doc_ids)}})
    if not_in_chunk_ids:
        clauses.append({"chunk_id": {"$nin": sorted(not_in_chunk_ids)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

# TODO: have claude template the prompts on a preliminary sample so you can see what things look like in the 4 scenarios
class SearchCorpusTool(Tool):
    name = "search_corpus"
    doc_template = _PROMPTS["search_corpus"]
    doc: str

    def __init__(
        self,
        chroma_collection: Collection,
        llm_client: LLMClient,
        working_set: WorkingSet,
        ctx: ExecutionContext,
        usage_key: str = "default",
        working_set_collection_off: bool = False,
        id_tracking_off: bool = False,
    ):
        self._chroma_collection = chroma_collection
        self._llm_client = llm_client
        self._ctx = ctx
        self._working_set = working_set
        self._usage_key = usage_key
        self._working_set_collection_off = working_set_collection_off
        self._id_tracking_off = id_tracking_off
        self.doc = _ENV.from_string(self.doc_template).render(
            working_set_collection_on=not self._working_set_collection_off,
            id_tracking_on=not self._id_tracking_off,
        )

    def _embed_query(self, query: str) -> list[float]:
        """Embed `query` with the same model that produced the stored embeddings, via the
        LLMClient — which owns backend dispatch (OpenRouter / vLLM), the process-wide "embed"
        rate bucket, retry, and usage accounting."""
        return self._llm_client.embed_query(query, ctx=self._ctx, usage_key=self._usage_key)

    def __call__(
        self,
        query: str,
        top_k: int,
        read: bool = False,
        fetch: bool = False,
        metadata_filter: dict | None = None,
    ) -> dict:
        # if the working set collection and id tracking are turned off, then every query
        # is a fetch + read query that goes directly to self._chroma_collection
        if self._working_set_collection_off and self._id_tracking_off:
            fetch = read = True

        assert fetch or read, "At least one of fetch or read must be present."
        assert fetch or not self._working_set.empty(use_id_state=self._working_set_collection_off), "Cannot read from empty working set"
        query_embedding = self._embed_query(query)

        # rephrase booleans to improve readability below
        working_set_collection_on = not self._working_set_collection_off
        id_tracking_on = not self._id_tracking_off

        # set the inclusion and exclusion filters based on the scenario if id tracking is on
        in_chunk_ids = in_doc_ids = not_in_chunk_ids = not_in_doc_ids = None
        if id_tracking_on:
            if fetch and read:
                not_in_chunk_ids = self._working_set.pruned_chunk_ids | self._working_set.read_chunk_ids
                not_in_doc_ids = self._working_set.pruned_doc_ids | self._working_set.read_doc_ids
            elif fetch:
                not_in_chunk_ids = self._working_set.fetched_chunk_ids | self._working_set.pruned_chunk_ids
                not_in_doc_ids = self._working_set.fetched_doc_ids | self._working_set.pruned_doc_ids
            elif read:
                in_chunk_ids = self._working_set.fetched_chunk_ids
                in_doc_ids = self._working_set.fetched_doc_ids
                not_in_chunk_ids = self._working_set.pruned_chunk_ids | self._working_set.read_chunk_ids
                not_in_doc_ids = self._working_set.pruned_doc_ids | self._working_set.read_doc_ids

        where = _build_metadata_where(
            metadata_filter=metadata_filter,
            in_chunk_ids=in_chunk_ids,
            in_doc_ids=in_doc_ids,
            not_in_chunk_ids=not_in_chunk_ids,
            not_in_doc_ids=not_in_doc_ids,
        )
        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": (
                ["metadatas", "documents", "distances", "embeddings"]
                if working_set_collection_on
                else ["metadatas", "documents", "distances"]
            ),
        }
        if where is not None:
            query_kwargs["where"] = where

        # query results from the appropriate collection
        try:
            # route the query: fetch always reads the full corpus; read-only is served by the
            # working-set collection when it exists, and by the corpus — restricted to fetched
            # material through the inclusion filters — when it does not
            if fetch or self._working_set_collection_off:
                results = self._chroma_collection.query(**query_kwargs)
            else:
                results = self._working_set.collection.query(**query_kwargs)

        except Exception as e:
            return {SEARCH_RESULT_TAG: True, "read_chunks": [], "fetched_chunks": [], "error": f"search_corpus error: {e}"}

        ids = results["ids"][0]
        documents = results["documents"][0]  # type: ignore
        metadatas = results["metadatas"][0]  # type: ignore
        distances = results["distances"][0]  # type: ignore

        # insert results into working set collection (if turned on)
        # TODO: figure out a way to run this in the background (off critical path)
        if fetch and working_set_collection_on and len(ids) > 0:
            embeddings = results["embeddings"][0]  # type: ignore
            self._working_set.insert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,  # type: ignore
                embeddings=embeddings, # type: ignore
            )

        # update id state (if id tracking on)
        if id_tracking_on:
            if fetch:
                self._working_set.fetched_chunk_ids.update(ids)
            if read:
                self._working_set.read_chunk_ids.update(ids)

        # add action to the working set
        if working_set_collection_on or id_tracking_on:
            self._working_set.add_action(
                tool=self.name,
                tool_kwargs={"query": query, "top_k": top_k, "read": read, "fetch": fetch, "metadata_filter": metadata_filter},
            )

        # return results to agent
        chunks: list[dict] = []
        for rank, (cid, doc, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True), 1
        ):
            doc_id = meta.get("doc_id", "?")
            elt_type = meta.get("type", "?")
            est_num_tokens = estimate_tokens(doc)
            header = f"[{rank}] chunk_id={cid} | doc_id={doc_id} | type={elt_type} | est_num_tokens={est_num_tokens} | distance={dist:.4f}"
            chunks.append({
                "chunk_id": cid,
                "doc_id": doc_id,
                "type": elt_type,
                "est_num_tokens": est_num_tokens,
                "distance": dist,
                "header": header,
                "text": f"{header}\n{doc or ''}",
            })

        return {SEARCH_RESULT_TAG: True, "read_chunks": chunks if read else [], "fetched_chunks": chunks if fetch else []}


class GrepCorpusTool(Tool):
    name = "grep_corpus"
    doc_template = _PROMPTS["grep_corpus"]
    doc: str

    def __init__(
        self,
        chroma_collection: Collection,
        working_set: WorkingSet,
        working_set_collection_off: bool = False,
        id_tracking_off: bool = False,
    ):
        self._chroma_collection = chroma_collection
        self._working_set = working_set
        self._working_set_collection_off = working_set_collection_off
        self._id_tracking_off = id_tracking_off
        self.doc = _ENV.from_string(self.doc_template).render(
            working_set_collection_on=not self._working_set_collection_off,
            id_tracking_on=not self._id_tracking_off,
        )

    def __call__(
        self,
        pattern: str,
        read: bool = False,
        fetch: bool = False,
        metadata_filter: dict | None = None,
        limit: int | None = None,
        max_output_tokens: int | None = None,
    ) -> dict:
        # if the working set collection and id tracking are turned off, then every query
        # is a fetch + read query that goes directly to self._chroma_collection
        if self._working_set_collection_off and self._id_tracking_off:
            fetch = read = True

        assert fetch or read, "At least one of fetch or read must be present."
        assert fetch or not self._working_set.empty(use_id_state=self._working_set_collection_off), "Cannot read from empty working set"

        # rephrase booleans to improve readability below
        working_set_collection_on = not self._working_set_collection_off
        id_tracking_on = not self._id_tracking_off

        # set the inclusion and exclusion filters based on the scenario
        in_chunk_ids = in_doc_ids = not_in_chunk_ids = not_in_doc_ids = None
        if id_tracking_on:
            if fetch and read:
                not_in_chunk_ids = self._working_set.pruned_chunk_ids | self._working_set.read_chunk_ids
                not_in_doc_ids = self._working_set.pruned_doc_ids | self._working_set.read_doc_ids
            elif fetch:
                not_in_chunk_ids = self._working_set.fetched_chunk_ids | self._working_set.pruned_chunk_ids
                not_in_doc_ids = self._working_set.fetched_doc_ids | self._working_set.pruned_doc_ids
            elif read:
                in_chunk_ids = self._working_set.fetched_chunk_ids
                in_doc_ids = self._working_set.fetched_doc_ids
                not_in_chunk_ids = self._working_set.pruned_chunk_ids | self._working_set.read_chunk_ids
                not_in_doc_ids = self._working_set.pruned_doc_ids | self._working_set.read_doc_ids

        where = _build_metadata_where(
            metadata_filter=metadata_filter,
            in_chunk_ids=in_chunk_ids,
            in_doc_ids=in_doc_ids,
            not_in_chunk_ids=not_in_chunk_ids,
            not_in_doc_ids=not_in_doc_ids,
        )
        get_kwargs: dict = {
            "where_document": {"$regex": pattern},
            "include": ["metadatas", "documents", "embeddings"] if working_set_collection_on else ["metadatas", "documents"],
        }
        if where is not None:
            get_kwargs["where"] = where
        if limit is not None:
            get_kwargs["limit"] = limit

        # query results from the appropriate collection
        try:
            # route the query: fetch always reads the full corpus; read-only is served by the
            # working-set collection when it exists, and by the corpus — restricted to fetched
            # material through the inclusion filters — when it does not
            if fetch or self._working_set_collection_off:
                res = self._chroma_collection.get(**get_kwargs)
            else:
                res = self._working_set.collection.get(**get_kwargs)

        except Exception as e:
            return {GREP_RESULT_TAG: True, "read_groups": [], "fetched_groups": [], "error": f"grep_corpus error: {e}"}

        ids = res["ids"]
        documents = res["documents"] or [None] * len(ids)
        metadatas = res["metadatas"] or [None] * len(ids)

        # if we got an empty result, return early
        if not ids:
            return {GREP_RESULT_TAG: True, "read_groups": [], "fetched_groups": []}

        # insert results into working set collection (if turned on)
        # TODO: figure out a way to run this in the background (off critical path)
        if fetch and working_set_collection_on and len(ids) > 0:
            embeddings = res["embeddings"]  # type: ignore
            self._working_set.insert(
                ids=ids,
                documents=documents,  # type: ignore
                metadatas=metadatas,  # type: ignore
                embeddings=embeddings,  # type: ignore
            )

        # add action to the working set
        if working_set_collection_on or id_tracking_on:
            self._working_set.add_action(
                tool=self.name,
                tool_kwargs={
                    "pattern": pattern,
                    "read": read,
                    "fetch": fetch,
                    "metadata_filter": metadata_filter,
                    "limit": limit,
                    "max_output_tokens": max_output_tokens,
                },
            )

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
            header = f"\n# doc_id={doc_id}"
            read_doc_chunks: list[dict] = []
            fetched_doc_chunks: list[dict] = []
            for _, cid, text in sorted(grouped[doc_id]):
                est_num_tokens = estimate_tokens(text)
                chunk_header = f"  [chunk_id={cid} | est_num_tokens={est_num_tokens}]"
                chunk_text = f"{chunk_header} {text}"
                fetched_doc_chunks.append({"chunk_id": cid, "doc_id": doc_id, "est_num_tokens": est_num_tokens, "header": chunk_header, "text": chunk_text})
                if max_output_tokens is not None:
                    # Count the header only once we commit the first chunk of this doc.
                    cost = estimate_tokens(chunk_text) + (estimate_tokens(header) if not read_doc_chunks else 0)
                    if read_doc_chunks or groups:  # always allow the very first chunk through
                        if used_tokens + cost > max_output_tokens:
                            truncated = True
                    used_tokens += cost

                if not truncated:
                    read_doc_chunks.append({"chunk_id": cid, "doc_id": doc_id, "header": chunk_header, "text": chunk_text})
                    kept_chunks += 1

            groups.append({"doc_id": doc_id, "header": header, "read_chunks": read_doc_chunks, "fetched_chunks": fetched_doc_chunks})

        # update id state (if id tracking on)
        if id_tracking_on:
            for g in groups:
                if fetch:
                    self._working_set.fetched_chunk_ids.update(c["chunk_id"] for c in g["fetched_chunks"])
                if read:
                    self._working_set.read_chunk_ids.update(c["chunk_id"] for c in g["read_chunks"])

        # return results to agent
        result: dict = {GREP_RESULT_TAG: True, "read_groups": groups if read else [], "fetched_groups": groups if fetch else []}
        if read and truncated:
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
    doc = _PROMPTS["read_document"]

    def __init__(self, document_map: DocumentMap, working_set: WorkingSet, id_tracking_off: bool):
        self._document_map = document_map
        self._working_set = working_set
        self._id_tracking_off = id_tracking_off

    def __call__(self, doc_id: str | list[str], start_char_idx: int | None | list[int | None] = None, end_char_idx: int | None | list[int | None] = None) -> dict:
        if isinstance(doc_id, list):
            err_msg = "When doc_id is a list, start_char_idx and end_char_idx must be None or also be lists of the same length."
            assert start_char_idx is None or (isinstance(start_char_idx, list) and len(doc_id) == len(start_char_idx)), err_msg
            assert end_char_idx is None or (isinstance(end_char_idx, list) and len(doc_id) == len(end_char_idx)), err_msg
        doc_ids = list(doc_id) if isinstance(doc_id, list) else [doc_id]
        start_indices = list(start_char_idx) if isinstance(start_char_idx, list) else [start_char_idx] * len(doc_ids)
        end_indices = list(end_char_idx) if isinstance(end_char_idx, list) else [end_char_idx] * len(doc_ids)

        docs: list[dict] = []
        for did, start_idx, end_idx in zip(doc_ids, start_indices, end_indices, strict=True):
            text = self._document_map.get(did)
            est_tokens = 0 if text is None else estimate_tokens(text)
            total_chars = 0 if text is None else len(text)
            if text is None:
                body = "[no such document (or no content in document)]"
            else:
                body = text
                if start_idx is not None or end_idx is not None:
                    start_idx = start_idx or 0
                    end_idx = end_idx if end_idx is not None else len(text)
                    body = text[start_idx:end_idx]

                if not self._id_tracking_off:
                    self._working_set.read_doc_ids.add(did)
                    self._working_set.fetched_doc_ids.add(did)
                    self._working_set.pruned_doc_ids.discard(did)
                    self._working_set.redacted_doc_ids.discard(did)

                self._working_set.add_action(self.name, {"doc_id": doc_id, "start_char_idx": start_char_idx, "end_char_idx": end_char_idx})

            rendered = f"=== doc_id={did} | total chars: {total_chars} | est. tokens: {est_tokens} ===\n{body}"
            docs.append({"doc_id": did, "text": rendered})
        return {READ_DOCUMENT_RESULT_TAG: True, "docs": docs}


class ViewFigureTool(Tool):
    """Render the full page for a doc_id the agent saw while reading — typically because
    the page text contains a `<figure id=N>` placeholder — and hand it back as an image
    observation. We render the *whole* page (not a bbox crop) so the agent sees any
    figures in context, and so OCR coordinate errors can't clip a chart."""

    name = "view_figure"
    doc = _PROMPTS["view_figure"]

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
    doc = _PROMPTS["prune"]

    def __init__(self, working_set: WorkingSet):
        self._working_set = working_set

    def __call__(
        self,
        chunk_ids: list[str] | None = None,
        doc_ids: list[str] | None = None,
    ) -> dict:
        # prune the chunks and docs from the working set and add an action to the working set
        new_pruned_chunks, new_pruned_docs = self._working_set.prune(chunk_ids, doc_ids)
        self._working_set.add_action(tool=self.name, tool_kwargs={"chunk_ids": chunk_ids, "doc_ids": doc_ids})

        return {
            PRUNE_RESULT_TAG: True,
            "new_chunk_count": len(new_pruned_chunks),
            "new_doc_count": len(new_pruned_docs),
            "total_chunks": len(self._working_set.pruned_chunk_ids),
            "total_docs": len(self._working_set.pruned_doc_ids),
        }


# Per-doc text snippet cap in the structured trace event. Each filtered doc carries a
# preview so the trace viewer can expand it on click without a second corpus lookup; the
# cap keeps a 1,000-doc filter from bloating the event stream with full document bodies.

class SemanticFilterTool(Tool):
    name = "semantic_filter"
    doc_template = _PROMPTS["sem_filter_tool"]
    doc: str

    _SEMFILTER_JUDGE_PROMPT = _PROMPTS["sem_filter_judge"]
    _SEMFILTER_MAX_WORKERS = 16
    _JUDGE_MAX_OUTPUT_TOKENS = 4
    _MAX_CANDIDATE_DOCS = 1000
    _JUDGE_CTX_SAFETY = 0.9
    _JUDGE_USER_WRAPPER = "Filter Condition: \n\nDocument:\n"
    _JUDGE_TRUNC_MARKER = "\n…(truncated to fit judge context)"
    _SEMFILTER_DOC_PREVIEW_MAX = 2000

    def __init__(
        self,
        chroma_collection: Collection,
        llm_client: LLMClient,
        document_map: DocumentMap,
        working_set: WorkingSet,
        model: str | None = None,
        *,
        ctx: ExecutionContext | None = None,
        max_workers: int | None = None,
        provider_order: list[str] | None = None,
        judge_max_output_tokens: int | None = None,
        disable_judge_reasoning: bool = True,
        usage_key: str = "default",
        working_set_collection_off: bool = False,
        id_tracking_off: bool = False,
    ) -> None:
        self._chroma_collection = chroma_collection
        self._llm_client = llm_client
        self._document_map = document_map
        self._working_set = working_set
        self._model = model or llm_client.config.llm_model
        self._ctx = ctx
        self._max_workers = max_workers or self._SEMFILTER_MAX_WORKERS
        self._judge_max_output_tokens = judge_max_output_tokens or self._JUDGE_MAX_OUTPUT_TOKENS
        self._disable_judge_reasoning = disable_judge_reasoning
        self._usage_key = usage_key
        self._working_set_collection_off = working_set_collection_off
        self._id_tracking_off = id_tracking_off
        self.doc = _ENV.from_string(self.doc_template).render(
            working_set_collection_on=not self._working_set_collection_off,
            id_tracking_on=not self._id_tracking_off,
        )

        # per-call OpenRouter provider order for the judge calls only (None => client default). Lets a
        # cheaper judge model route to specific providers while the agent model stays unpinned.
        self._provider_order = provider_order

        # resolve the model's context limit
        self._context_limit = match_model_entry(self._model, llm_client.config.llm_context_limits)

    @staticmethod
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

    def _error(self, msg: str) -> dict:
        return {SEMFILTER_RESULT_TAG: True, "error": msg}

    def _metadata_only_selection(self, where: dict, fetch: bool) -> tuple[list[tuple], list[str]]:
        """Use metadata filtering to return a set of candidates and candidate ids."""
        working_set_collection_on = not self._working_set_collection_off
        get_kwargs: dict = {
            "where": where,
            "include": ["metadatas", "documents", "embeddings"] if working_set_collection_on else ["metadatas", "documents"],
        }

        # route the query: fetch always reads the full corpus; read-only is served by the
        # working-set collection when it exists, and by the corpus — restricted to fetched
        # material through the inclusion filters — when it does not
        if fetch or self._working_set_collection_off:
            res = self._chroma_collection.get(**get_kwargs)
        else:
            res = self._working_set.collection.get(**get_kwargs)

        ids = res["ids"]
        documents = res["documents"] or [None] * len(ids)
        metadatas = res["metadatas"] or [None] * len(ids)
        embeddings = res["embeddings"] if working_set_collection_on else [None] * len(ids)

        candidates = [
            (cid, meta["doc_id"], doc or "", meta, emb if emb is not None else [])
            for cid, doc, meta, emb in zip(ids, documents, metadatas, embeddings, strict=True)  # type: ignore
        ]
        candidate_doc_ids = list(dict.fromkeys(str(did) for _, did, _, _, _ in candidates))
        return candidates, candidate_doc_ids

    def _vector_search_selection(self, search_str: str, top_k: int, where: dict | None, fetch: bool) -> tuple[list[tuple], list[str]]:
        """Use vector search to return a set of candidates and candidate ids."""
        working_set_collection_on = not self._working_set_collection_off
        emb = self._llm_client.embed_query(search_str, ctx=self._ctx, usage_key=self._usage_key)
        query_kwargs: dict = {
            "query_embeddings": [emb],
            "n_results": top_k,
            "include": ["metadatas", "documents", "embeddings"] if working_set_collection_on else ["metadatas", "documents"],
        }
        if where is not None:
            query_kwargs["where"] = where

        # route the query: fetch always reads the full corpus; read-only is served by the
        # working-set collection when it exists, and by the corpus — restricted to fetched
        # material through the inclusion filters — when it does not
        if fetch or self._working_set_collection_off:
            res = self._chroma_collection.query(**query_kwargs)
        else:
            res = self._working_set.collection.query(**query_kwargs)

        ids = res["ids"][0]
        documents = res["documents"][0]  # type: ignore
        metadatas = res["metadatas"][0]  # type: ignore
        embeddings = res["embeddings"][0] if working_set_collection_on else [None] * len(ids)  # type: ignore

        candidates = [
            (cid, meta["doc_id"], doc or "", meta, emb if emb is not None else [])
            for cid, doc, meta, emb in zip(ids, documents, metadatas, embeddings, strict=True)  # type: ignore
        ]
        candidate_doc_ids = list(dict.fromkeys(str(did) for _, did, _, _, _ in candidates))
        return candidates, candidate_doc_ids

    def _grep_selection(self, pattern: str, limit: int | None, where: dict | None, fetch: bool) -> tuple[list[tuple], list[str]]:
        """Use grep to return a set of candidates and candidate ids."""
        working_set_collection_on = not self._working_set_collection_off
        get_kwargs: dict = {
            "where_document": {"$regex": pattern},
            "include": ["metadatas", "documents", "embeddings"] if working_set_collection_on else ["metadatas", "documents"],
        }
        if where is not None:
            get_kwargs["where"] = where
        if limit is not None:
            get_kwargs["limit"] = limit

        # route the query: fetch always reads the full corpus; read-only is served by the
        # working-set collection when it exists, and by the corpus — restricted to fetched
        # material through the inclusion filters — when it does not
        if fetch or self._working_set_collection_off:
            res = self._chroma_collection.get(**get_kwargs)
        else:
            res = self._working_set.collection.get(**get_kwargs)

        ids = res["ids"]
        documents = res["documents"] or [None] * len(ids)
        metadatas = res["metadatas"] or [None] * len(ids)
        embeddings = res["embeddings"] if working_set_collection_on else [None] * len(ids)

        candidates = [
            (cid, meta["doc_id"], doc or "", meta, emb if emb is not None else [])
            for cid, doc, meta, emb in zip(ids, documents, metadatas, embeddings, strict=True)  # type: ignore
        ]
        candidate_doc_ids = list(dict.fromkeys(str(did) for _, did, _, _, _ in candidates))
        return candidates, candidate_doc_ids

    def _truncate_doc_for_judge(self, text: str, budget_tokens: int) -> tuple[str, bool]:
        """Head-truncate `text` to `budget_tokens` (marker included), returning (text, was_truncated)."""
        token_est = estimate_tokens(text)
        if token_est > budget_tokens:
            ratio = budget_tokens / token_est
            return text[:int(ratio * len(text))] + self._JUDGE_TRUNC_MARKER, True
        return text, False

    def _judge_doc_token_budget(self, predicate: str) -> int:
        """Max tokens of document text that fit in one judge request under `context_limit` (tokens):
        the fixed overhead (prompt scaffolding, counted via `estimate_tokens`) and the judge's output
        headroom (`max_output_tokens`) are subtracted, then the remaining token budget is returned
        with a safety factor. Returns 0 when the fixed overhead alone already exceeds the limit."""
        assert isinstance(self._context_limit, int)
        overhead_tokens = self._judge_max_output_tokens + estimate_tokens(
            self._SEMFILTER_JUDGE_PROMPT + self._JUDGE_USER_WRAPPER + predicate
        )
        budget_tokens = self._context_limit * self._JUDGE_CTX_SAFETY - overhead_tokens
        return max(0, int(budget_tokens))

    def _judge_one(self, predicate: str, item_text: str) -> bool:
        user = f"Filter Condition: {predicate}\n\nDocument:\n{item_text}"
        try:
            messages = [{"role": "user", "content": user}]
            resp = self._llm_client.call(
                system=self._SEMFILTER_JUDGE_PROMPT,
                messages=messages,
                temperature=0.0,
                model=self._model,
                ctx=self._ctx,
                call_site="semfilter",
                provider_order=self._provider_order,
                max_output_tokens=self._judge_max_output_tokens,
                disable_reasoning=self._disable_judge_reasoning,
                usage_key=self._usage_key,
            )
        except Exception:
            return True  # recall-safe: keep on error
        return self._parse_bool(resp.text)

    def _filter_docs(self, predicate: str, doc_ids: list[str], event_extra: dict | None = None) -> list[str]:
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
        texts = [self._document_map.get(d, "") or "" for d in doc_ids]
        n_truncated = 0
        if self._context_limit:
            budget = self._judge_doc_token_budget(predicate)
            truncated = [self._truncate_doc_for_judge(t, budget) for t in texts]
            texts = [t for t, _ in truncated]
            n_truncated = sum(1 for _, was in truncated if was)

        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(doc_ids))) as pool:
            verdicts = list(pool.map(
                lambda text: self._judge_one(predicate, text),
                texts,
            ))

        kept = [d for d, keep in zip(doc_ids, verdicts, strict=True) if keep]
    
        if self._ctx is not None:
            data = {
                "predicate": predicate,
                "n_in": len(doc_ids),
                "n_out": len(kept),
                "n_truncated": n_truncated,
                "docs": [
                    {"doc_id": d, "kept": bool(keep), "text": truncate(t, self._SEMFILTER_DOC_PREVIEW_MAX, "\n…(truncated)")}
                    for d, keep, t in zip(doc_ids, verdicts, texts, strict=True)
                ],
            }
            if event_extra:
                data.update(event_extra)
            self._ctx.emit(
                f"semantic_filter n_in={len(doc_ids)} n_out={len(kept)} predicate={truncate(predicate, 80)!r}",
                kind="observation",
                data=data,
            )

        return kept

    def __call__(
        self,
        predicate: str,
        read: bool = False,
        fetch: bool = False,
        metadata_filter: dict | None = None,
        top_k: int | None = None,
        search_str: str | None = None,
        pattern: str | None = None,
        limit: int | None = None,
    ) -> dict:
        # if the working set collection and id tracking are turned off, then every query
        # is a fetch + read query that goes directly to self._chroma_collection
        if self._working_set_collection_off and self._id_tracking_off:
            fetch = read = True

        assert fetch or read, "At least one of fetch or read must be present."
        assert fetch or not self._working_set.empty(use_id_state=self._working_set_collection_off), "Cannot read from empty working set"

        if not predicate or not str(predicate).strip():
            return self._error("semantic_filter error: `predicate` is required.")
        if fetch and metadata_filter is None and top_k is None and pattern is None:
            return self._error(
                "semantic_filter error: if fetching from the full corpus, must provide one of " \
                "`metadata_filter`, `top_k`, or `pattern` for candidate selection."
            )
        if search_str is not None and top_k is None:
            return self._error(
                "semantic_filter error: search_str requires top_k (it is the query for the top_k vector search)."
            )
        if limit is not None and pattern is None:
            return self._error(
                "semantic_filter error: limit requires pattern (it is the limit for the grep expression)."
            )
        if top_k is not None and pattern is not None:
            # TODO: add a warning / notice that we will only execute the vector search
            pass

        # rephrase booleans to improve readability below
        working_set_collection_on = not self._working_set_collection_off
        id_tracking_on = not self._id_tracking_off

        # set the inclusion and exclusion filters based on the scenario; this tool differs a bit
        # from vector search and grep because we want to allow the tool to re-read chunks / documents
        # while applying new predicates; we exclude pruned chunks when read is true; we additionally
        # exclude previously fetched chunks when the filter is fetch-only (the are already in the
        # working set)
        in_chunk_ids = in_doc_ids = not_in_chunk_ids = not_in_doc_ids = None
        if id_tracking_on:
            if fetch and read:
                not_in_chunk_ids = self._working_set.pruned_chunk_ids
                not_in_doc_ids = self._working_set.pruned_doc_ids
            elif fetch:
                not_in_chunk_ids = self._working_set.fetched_chunk_ids | self._working_set.pruned_chunk_ids
                not_in_doc_ids = self._working_set.fetched_doc_ids | self._working_set.pruned_doc_ids
            else:
                in_chunk_ids = self._working_set.fetched_chunk_ids
                in_doc_ids = self._working_set.fetched_doc_ids
                not_in_chunk_ids = self._working_set.pruned_chunk_ids
                not_in_doc_ids = self._working_set.pruned_doc_ids

        where = _build_metadata_where(
            metadata_filter=metadata_filter,
            in_chunk_ids=in_chunk_ids,
            in_doc_ids=in_doc_ids,
            not_in_chunk_ids=not_in_chunk_ids,
            not_in_doc_ids=not_in_doc_ids,
        )

        # select candidate chunks: (chunk_id, doc_id, text | None, meta, emb) using vector search, grep, or metadata-only filtering
        if top_k is not None:
            mode = "vector"
            try:
                candidates, candidate_doc_ids = self._vector_search_selection(search_str or predicate, top_k, where, fetch)
            except Exception as e:
                return self._error(f"semantic_filter error while executing vector search: {e}")

            if len(candidate_doc_ids) > self._MAX_CANDIDATE_DOCS:
                return self._error(
                    f"semantic_filter error: the top_k={top_k} vector search yielded {len(candidate_doc_ids)} candidate "
                    f"documents, over the {self._MAX_CANDIDATE_DOCS}-document cap. Lower top_k or narrow the metadata_filter."
                )
        elif pattern is not None:
            mode = "grep"
            try:
                candidates, candidate_doc_ids = self._grep_selection(pattern, limit, where, fetch)
            except Exception as e:
                return self._error(f"semantic_filter error while executing grep: {e}")

            if len(candidate_doc_ids) > self._MAX_CANDIDATE_DOCS:
                return self._error(
                    f"semantic_filter error: the pattern={pattern}, limit={limit} grep yielded {len(candidate_doc_ids)} candidate "
                    f"documents, over the {self._MAX_CANDIDATE_DOCS}-document cap. Lower limit or narrow the metadata_filter."
                )
        else:
            mode = "metadata"
            try:
                assert isinstance(where, dict)
                candidates, candidate_doc_ids = self._metadata_only_selection(where, fetch)
            except Exception as e:
                return self._error(f"semantic_filter error while executing metadata: {e}")
                
            if len(candidate_doc_ids) > self._MAX_CANDIDATE_DOCS:
                return self._error(
                    f"semantic_filter error: the where={where} metadata_filter yielded {len(candidate_doc_ids)} candidate "
                    f"documents, over the {self._MAX_CANDIDATE_DOCS}-document cap. Narrow the metadata_filter or combine with vector search or grep."
                )
            
        if not candidate_doc_ids:
            # renders as EMPTY_RESULT_MESSAGE (a prune filter or bad grep may lead to no results)
            return {SEMFILTER_RESULT_TAG: True, "summary": "", "read_chunks": [], "fetched_chunks": [], "kept_doc_ids": [], "n_in": 0, "n_out": 0}

        # filter documents using judge model
        kept = self._filter_docs(
            predicate,
            candidate_doc_ids,
            event_extra={
                "mode": mode,
                "metadata_filter": metadata_filter,
                "top_k": top_k,
                "search_str": search_str,
                "n_candidate_chunks": len(candidates),
                "n_candidate_docs": len(candidate_doc_ids),
            },
        )
        kept_set = set(kept)
        kept_chunks = [
            {
                "chunk_id": cid,
                "doc_id": did,
                "est_num_tokens": estimate_tokens(text),
                "header": f"  [chunk_id={cid} | doc_id={did} | est_num_tokens={estimate_tokens(text)}]",
                "text": f"chunk_id={cid} | doc_id={did}\n{text}"
            }
            for cid, did, text, _, _ in candidates
            if did in kept_set
        ]

        # insert results into working set collection (if turned on)
        # TODO: figure out a way to run this in the background (off critical path)
        if fetch and working_set_collection_on and len(kept_set) > 0:
            ids, documents, metadatas, embeddings = [], [], [], []
            for id, doc_id, doc, meta, emb in candidates:
                if doc_id in kept_set:
                    ids.append(id)
                    documents.append(doc)
                    metadatas.append(meta)
                    embeddings.append(emb)

            self._working_set.insert(
                ids=ids,  # type: ignore
                documents=documents,  # type: ignore
                metadatas=metadatas,  # type: ignore
                embeddings=embeddings, # type: ignore
            )

        # update id state (if id tracking on)
        # (doc_id, chunk_id) order within the sorted kept docs
        kept_chunks.sort(key=lambda c: (c["doc_id"], c["chunk_id"]))
        if id_tracking_on:
            if fetch:
                self._working_set.fetched_chunk_ids.update(c["chunk_id"] for c in kept_chunks)
                self._working_set.fetched_doc_ids.update(kept)
            if read:
                self._working_set.read_chunk_ids.update(c["chunk_id"] for c in kept_chunks)
                self._working_set.read_doc_ids.update(kept)
                # A re-surfaced chunk must actually render: `_block_is_visible` hides ChunkBlocks by
                # chunk_id/doc_id, so a kept chunk the context trimmer once redacted would come back
                # invisible while the summary says its doc passed. Drop kept ids from the redaction
                # sets (mirrors `read_document`); the trimmer can always re-redact on a later step.
                for c in kept_chunks:
                    self._working_set.redacted_chunk_ids.discard(c["chunk_id"])
                for did in kept:
                    self._working_set.redacted_doc_ids.discard(did)

        # add action to the working set
        if working_set_collection_on or id_tracking_on:
            self._working_set.add_action(
                tool=self.name,
                tool_kwargs={
                    "predicate": predicate,
                    "read": read,
                    "fetch": fetch, 
                    "metadata_filter": metadata_filter,
                    "top_k": top_k,
                    "search_str": search_str,
                    "pattern": pattern,
                    "limit": limit,
                },
            )

        # return results to the agent
        summary = (
            f"[semantic_filter] kept {len(kept)}/{len(candidate_doc_ids)} candidate document(s) matching the "
            f"predicate. kept_doc_ids={kept}." + (" Chunks from kept documents follow." if kept_chunks else "")
        )
        result = {
            SEMFILTER_RESULT_TAG: True,
            "summary": summary,
            "read_chunks": kept_chunks if read else [],
            "fetched_chunks": kept_chunks if fetch else [],
            "kept_doc_ids": kept,
            "n_in": len(candidate_doc_ids),
            "n_out": len(kept),
        }

        return result
