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
DATAGEN_FINAL_ANSWER_TAG = "__datagen_final_answer__"
QUALITY_FILTER_FINAL_ANSWER_TAG = "__quality_filter_final_answer__"
TASK_SOLVER_FINAL_ANSWER_TAG = "__task_solver_final_answer__"

# Generic message surfaced when search_corpus/grep_corpus return zero hits.
# ChromaDB does not tell us whether a server-side `$nin` prune filter was
# what eliminated all candidates, so we use a generic message that mentions
# both possibilities.
EMPTY_RESULT_MESSAGE = (
    "No results found; it is possible that exclusion filters on all "
    "previously returned chunks prevented any results from being returned."
)

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


def _make_read_document(document_map: dict[str, str]) -> Callable:
    """Build a `read_document` tool that returns full document text.

    `document_map` is keyed by `doc_id` and maps to the full document text.
    """

    def read_document(doc_id: str | list[str]) -> dict:
        """Read the text for one or more documents, given their doc_ids.

        Returns a tagged dict with one entry per requested doc_id, so the
        SearchAgent can wrap each in a per-doc block that gets redacted if
        the doc_id is later pruned.
        """
        doc_ids = [doc_id] if isinstance(doc_id, str) else list(doc_id)

        docs: list[dict] = []
        for did in doc_ids:
            text = document_map.get(did)
            if text is None:
                docs.append(
                    {
                        "doc_id": did,
                        "text": (
                            f"=== doc_id={did} ===\n"
                            f"[no such document (or no content in document)]"
                        ),
                    }
                )
                continue

            docs.append(
                {
                    "doc_id": did,
                    "text": f"=== doc_id={did} ===\n{text}",
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
    document_map: dict[str, str],
    pruned_chunk_ids: set[str],
    pruned_doc_ids: set[str],
    final_answer_fn: Callable | None = None,
) -> dict[str, Callable]:
    """Build the full toolset for a SearchAgent invocation.

    The ``pruned_chunk_ids`` / ``pruned_doc_ids`` sets are owned and updated
    by the caller (the SearchAgent).  ``search_corpus`` / ``grep_corpus``
    close over them and read their current contents on every call, so any
    mutations the SearchAgent makes between steps take effect immediately.

    ``final_answer_fn``, if provided, replaces the default ``final_answer``
    tool. Use this to inject a datagen-specific ``final_answer`` that accepts
    a different signature (e.g. ``question``, ``answer``, ``chunk_ids``,
    ``doc_ids`` instead of ``page_keys``).
    """
    return {
        "search_corpus": _make_search_corpus(
            chroma_collection, emb_model_id, openrouter_client,
            pruned_chunk_ids, pruned_doc_ids,
        ),
        "grep_corpus": _make_grep_corpus(
            chroma_collection, pruned_chunk_ids, pruned_doc_ids,
        ),
        "read_document": _make_read_document(document_map),
        "prune": prune,
        "final_answer": final_answer_fn if final_answer_fn is not None else final_answer,
    }


def final_answer(page_keys):
    return page_keys


def datagen_final_answer(qa_pairs: list[dict]) -> dict:
    """final_answer variant for the SearchAgent when performing QA synthesis.

    Each entry of ``qa_pairs`` is a dict with keys:
      - ``question`` (str): the synthesised question.
      - ``answer`` (list[str]): the answer expressed as a list of *nuggets*,
        each nugget being a key piece of information essential to the answer.
      - ``chunk_ids`` (list[str]): the chunk_ids needed to answer the question
        (doc_ids can be derived downstream from each chunk_id).

    `SearchAgent.qa_synthesis()` reads this dict from the final `_StepOutcome`
    and returns the list of pairs directly.
    """
    normalized: list[dict] = []
    for pair in qa_pairs:
        normalized.append({
            "question": str(pair.get("question", "")),
            "answer": [str(n) for n in pair.get("answer", [])],
            "chunk_ids": [str(c) for c in pair.get("chunk_ids", [])],
        })
    return {
        DATAGEN_FINAL_ANSWER_TAG: True,
        "qa_pairs": normalized,
    }


def quality_filter_final_answer(valid: bool, reasoning: str) -> dict:
    """final_answer variant for the QualityFilter SearchAgent.

    The QF agent uses the corpus tools to investigate a candidate QA pair
    against the rollout attempts and the ground truth, then calls this
    tool exactly once with:
      - ``valid`` (bool): True iff the pair is unambiguous AND its
        provided answer is correct (and non-trivially answerable).
      - ``reasoning`` (str): 1-3 sentence justification.

    `quality_filter.run_quality_filter_for_pair` reads this dict from the
    final `_StepOutcome.raw_output`.
    """
    return {
        QUALITY_FILTER_FINAL_ANSWER_TAG: True,
        "valid": bool(valid),
        "reasoning": str(reasoning),
    }


def task_solver_final_answer(answer: str) -> dict:
    """final_answer variant for the TaskSolverAgent.

    The TaskSolver calls this exactly once with the plain-string answer it
    derived from the provided documents (and any Python computations).  Pass
    the exact fallback string when the documents are insufficient::

        final_answer("I cannot answer this question with the provided documents.")

    ``task_solver.run_task_solver`` reads this dict from the final
    ``_StepOutcome.raw_output``.
    """
    return {
        TASK_SOLVER_FINAL_ANSWER_TAG: True,
        "answer": str(answer),
    }
