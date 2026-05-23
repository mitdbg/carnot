"""Tool implementations for the SearchAgent.

All retrieval tools are backed by a single ChromaDB collection in which each
row is one *element* (chunk) extracted from a page (= "document") of a
treasury bulletin.  Expected per-row layout:

    id:        "{page_key}_{element_id}"  (also stored as metadata.chunk_id
                                           so it can be used in $nin filters)
    document:  the cleaned element text
    metadata:  {
        doc_id:     str   -- the page_key, e.g. "1946_11_41"
        chunk_id:   str   -- duplicate of the row id for filterability
        year:       str   -- zero-padded e.g. "1941"
        month:      str   -- zero-padded e.g. "01"
        page_id:    int
        file_id:    str
        element_id: int
        type:       str   -- "text" | "title" | "table" | ...
    }

The collection is produced by `create_vector_db.py`.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable

from chromadb.api.models.Collection import Collection
from google import genai
from openrouter import OpenRouter

# tags to help the SearchAgent identify tool return values
PRUNE_RESULT_TAG = "__prune__"
SEARCH_RESULT_TAG = "__search_result__"
GREP_RESULT_TAG = "__grep_result__"
READ_DOCUMENT_RESULT_TAG = "__read_document_result__"

# ---------------------------------------------------------------------------
# Internal where-clause helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# search_corpus
# ---------------------------------------------------------------------------


def _make_search_corpus(
    chroma_collection: Collection,
    emb_model_id: str,
    openrouter_client: OpenRouter | genai.Client,
    pruned_chunk_ids: set[str],
    pruned_doc_ids: set[str],
) -> Callable:
    """Build a `search_corpus` tool bound to a collection + embedding client.

    The returned tool excludes any chunks / docs in the shared
    `pruned_chunk_ids` / `pruned_doc_ids` sets at query time.
    """

    def search_corpus(
        query: str,
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> dict | str:
        """Perform a vector search over the corpus and return a formatted list of chunks.

        Each returned chunk is labelled with its chunk_id and doc_id so the
        agent can later pass them to `prune(...)`.
        """
        # embed the query using the same model that produced the stored embeddings.
        resp = openrouter_client.embeddings.generate(input=query, model=emb_model_id)  # type: ignore
        query_embedding = resp.data[0].embedding  # type: ignore

        where = _build_metadata_where(
            metadata_filter=metadata_filter,
            ignore_chunk_ids=pruned_chunk_ids,
            ignore_doc_ids=pruned_doc_ids,
        )

        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["metadatas", "documents", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where

        try:
            results = chroma_collection.query(**query_kwargs)
        except Exception as e:
            return f"[search_corpus error: {e}]"

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
            text = doc or ""
            chunks.append(
                {
                    "chunk_id": cid,
                    "doc_id": doc_id,
                    "text": (
                        f"[{rank}] chunk_id={cid} | doc_id={doc_id} | "
                        f"type={elt_type} | distance={dist:.4f}\n{text}"
                    ),
                }
            )

        return {
            SEARCH_RESULT_TAG: True,
            "chunks": chunks,
        }

    return search_corpus

# ---------------------------------------------------------------------------
# grep_corpus
# ---------------------------------------------------------------------------


def _make_grep_corpus(
    chroma_collection: Collection,
    pruned_chunk_ids: set[str],
    pruned_doc_ids: set[str],
) -> Callable:
    """Build a `grep_corpus` tool that regex-matches chunk text via ChromaDB."""

    def grep_corpus(
        pattern: str,
        metadata_filter: dict | None = None,
        limit: int | None = None,
    ) -> dict | str:
        """Regex-search every chunk's text in the corpus.

        Returns hits grouped by `doc_id`.  When ``limit`` is ``None`` all
        matches are returned (use a ``limit`` for narrower exploration).
        """
        where = _build_metadata_where(
            metadata_filter=metadata_filter,
            ignore_chunk_ids=pruned_chunk_ids,
            ignore_doc_ids=pruned_doc_ids,
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
            res = chroma_collection.get(**get_kwargs)
        except Exception as e:
            return f"[grep_corpus error: {e}]"

        ids = res["ids"]
        documents = res["documents"] or []
        metadatas = res["metadatas"] or []

        if not ids:
            return {
                GREP_RESULT_TAG: True,
                "groups": [],
            }

        # group by doc_id; preserve element ordering within each doc.
        grouped: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        for cid, doc, meta in zip(ids, documents, metadatas, strict=True):
            doc_id: str = meta["doc_id"]  # type: ignore
            elt_id: int = meta["element_id"]  # type: ignore
            grouped[doc_id].append((elt_id, cid, doc))

        groups: list[dict] = []
        for doc_id in sorted(grouped.keys()):
            doc_chunks: list[dict] = []
            for _, cid, text in sorted(grouped[doc_id]):
                doc_chunks.append(
                    {
                        "chunk_id": cid,
                        "doc_id": doc_id,
                        "text": f"  [chunk_id={cid}] {text}",
                    }
                )
            groups.append(
                {
                    "doc_id": doc_id,
                    "header": f"\n# doc_id={doc_id}",
                    "chunks": doc_chunks,
                }
            )

        return {
            GREP_RESULT_TAG: True,
            "groups": groups,
        }

    return grep_corpus

# ---------------------------------------------------------------------------
# read_document
# ---------------------------------------------------------------------------


def _make_read_document(
    clean_page_map: dict[str, list],
    bulletins_dir: str | None = None,
) -> Callable:
    """Build a `read_document` tool that returns full cleaned page text.

    Uses the LLM-reordered text stored on disk and pointed to by
    `clean_page_map` (we cannot reliably reconstruct a page by simply
    concatenating its chunks ordered by `element_id`).  `clean_page_map` is
    keyed by `doc_id` (= page_key, e.g. "2002_12_25").
    """

    def _resolve_path(rel_path: str) -> str:
        if bulletins_dir is None or os.path.isabs(rel_path):
            return rel_path

        basename = os.path.basename(rel_path)
        return os.path.join(bulletins_dir, basename)

    def read_document(doc_id: str | list[str]) -> dict:
        """Read the cleaned text for one or more pages, given their doc_ids.

        Returns a tagged dict with one entry per requested doc_id, so the
        SearchAgent can wrap each in a per-doc block that gets redacted if
        the doc_id is later pruned.
        """
        doc_ids = [doc_id] if isinstance(doc_id, str) else list(doc_id)

        docs: list[dict] = []
        for did in doc_ids:
            entry = clean_page_map.get(did)
            if entry is None:
                docs.append(
                    {
                        "doc_id": did,
                        "text": (
                            f"=== doc_id={did} ===\n"
                            f"[no such page (or no content on page)]"
                        ),
                    }
                )
                continue

            filepath = _resolve_path(entry[0])
            try:
                with open(filepath) as f:
                    text = f.read()
            except OSError as e:
                docs.append(
                    {
                        "doc_id": did,
                        "text": f"=== doc_id={did} ===\n[error reading file: {e}]",
                    }
                )
                continue

            docs.append(
                {
                    "doc_id": did,
                    "text": f"=== doc_id={did} ({filepath}) ===\n{text}",
                }
            )

        return {READ_DOCUMENT_RESULT_TAG: True, "docs": docs}

    return read_document

# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------


def prune(
    chunk_ids: list[str] | None = None,
    doc_ids: list[str] | None = None,
) -> dict:
    """Pass-through tool: returns the requested ids tagged for the SearchAgent.

    The actual pruning state is owned by the SearchAgent, which detects the
    tagged return value after the step completes and updates its prune sets
    accordingly.  Subsequent `search_corpus` / `grep_corpus` calls then
    exclude the pruned ids server-side.
    """
    return {
        PRUNE_RESULT_TAG: True,
        "chunk_ids": list(chunk_ids) if chunk_ids else [],
        "doc_ids": list(doc_ids) if doc_ids else [],
    }


# ---------------------------------------------------------------------------
# Bundle factory and final_answer
# ---------------------------------------------------------------------------


def make_search_tools(
    chroma_collection: Collection,
    emb_model_id: str,
    openrouter_client: OpenRouter | genai.Client,
    clean_page_map: dict[str, list],
    pruned_chunk_ids: set[str],
    pruned_doc_ids: set[str],
    bulletins_dir: str | None = None,
) -> dict[str, Callable]:
    """Build the full toolset for a SearchAgent invocation.

    The ``pruned_chunk_ids`` / ``pruned_doc_ids`` sets are owned and updated
    by the caller (the SearchAgent).  ``search_corpus`` / ``grep_corpus``
    close over them and read their current contents on every call, so any
    mutations the SearchAgent makes between steps take effect immediately.
    """
    return {
        "search_corpus": _make_search_corpus(
            chroma_collection, emb_model_id, openrouter_client,
            pruned_chunk_ids, pruned_doc_ids,
        ),
        "grep_corpus": _make_grep_corpus(
            chroma_collection, pruned_chunk_ids, pruned_doc_ids,
        ),
        "read_document": _make_read_document(clean_page_map, bulletins_dir),
        "prune": prune,
        "final_answer": final_answer,
    }


def final_answer(page_keys):
    return page_keys
