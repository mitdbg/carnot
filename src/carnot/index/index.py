from __future__ import annotations

import json
import logging
import random
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import chromadb
import faiss
import litellm
import numpy as np
from tqdm import tqdm

from carnot.index.sem_indices import FlatFileIndex, HierarchicalFileIndex
from carnot.storage.config import StorageConfig

if TYPE_CHECKING:
    from carnot.agents.models import LiteLLMModel

logger = logging.getLogger(__name__)

# ── Embedding model per-input token limits ──────────────────────────────
# Duplicated from optimizer.model_ids to avoid a circular import
# (index → optimizer → rules → … → index).
_EMBEDDING_MAX_INPUT_TOKENS: dict[str, int] = {
    "text-embedding-3-small": 8191,
    "openai/text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "openai/text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191,
    "openai/text-embedding-ada-002": 8191,
}
_DEFAULT_EMBEDDING_MAX_INPUT_TOKENS = 8191
_MAX_TOKENS_PER_REQUEST = 300_000
_CHROMA_INSERT_BATCH_SIZE = 2048

def _resolve_model(model, api_key) -> LiteLLMModel:
    """Resolve *model* to a :class:`LiteLLMModel` instance.

    Accepts a ``LiteLLMModel``, a model-id string (for backward-compat
    with callers that pass e.g. ``model="openai/text-embedding-3-small"``),
    or ``None`` (creates a default).

    Requires:
        - *model* is a :class:`LiteLLMModel`, a ``str``, or ``None``.

    Returns:
        A :class:`LiteLLMModel` instance.

    Raises:
        None.
    """
    from carnot.agents.models import LiteLLMModel as _LiteLLMModel

    if isinstance(model, _LiteLLMModel):
        return model
    model_id = model if isinstance(model, str) else "openai/text-embedding-3-small"
    return _LiteLLMModel(model_id=model_id, api_key=api_key)


class CarnotIndex(ABC):
    """Abstract base for all Carnot indices.

    A ``CarnotIndex`` wraps a concrete index implementation (vector DB,
    file-summary list, etc.) and exposes a uniform ``search`` interface.

    Subclasses must implement :meth:`_get_or_create_index` and
    :meth:`search`.

    Construction parameters:

    - *name* (``str``): unique name for this index (used for on-disk
      persistence).  Callers should ensure uniqueness across datasets,
      e.g. ``"ds{dataset_id}_{kind}"``.
    - *items* (``list[dict] | None``): the data items to be indexed.
      Will be raw materialized ``dict`` objects. When *index* is provided
      (pre-built index reuse), ``items`` enables result mapping (e.g., URI → item).
    - *index*: an optional pre-built inner index object.  When
      provided, subclasses should assign it to ``self._index`` and
      skip the expensive build step.  **Must not be provided without
      ``items``**: the item list is required to map search results
      back to the original objects.

    Public attributes after construction:

    - ``name`` (``str``)
    - ``items`` (``list``)
    - ``description`` (``str``, class-level): human-readable sentence
      describing the index type.

    Representation invariant:
        - ``_index`` is ``None`` before :meth:`_get_or_create_index`
          completes; non-``None`` afterwards.
        - If ``_index`` is non-``None`` at construction time (pre-built
          reuse), ``items`` is non-empty.

    Abstraction function:
        Represents a searchable index over ``items`` whose concrete
        implementation is held in ``_index``.
    """

    description: str = "Base CarnotIndex class. Not meant to be used directly."

    def __init__(
        self,
        name: str,
        items: list[dict] | None = None,
        index = None,
    ):
        if index is not None and not items:
            raise ValueError(
                "CarnotIndex: 'index' was provided without 'items'. "
                "Providing a pre-built index without items makes search results "
                "unmappable (URI → item lookups will always miss). "
                "Pass the original item list alongside the pre-built index."
            )
        self.name = name
        self.items = items or []
        self._index = index

    @abstractmethod
    def _get_or_create_index(self):
        """Build or retrieve the underlying index object.

        Subclasses must populate ``self._index`` and return it.
        """
        ...

    @abstractmethod
    def search(self, query: str, k: int) -> list:
        """Return up to *k* items most relevant to *query*.

        Requires:
            - *query* is a non-empty string.
            - *k* >= 1.

        Returns:
            A list of at most *k* items from ``self.items``, ordered by
            descending relevance.

        Raises:
            Subclass-dependent.
        """
        ...


class HierarchicalCarnotIndex(CarnotIndex):
    """CarnotIndex backed by a :class:`HierarchicalFileIndex`.

    Builds a B-tree-like hierarchy of file summaries and uses top-down
    traversal to retrieve the most relevant files for a query.

    Construction eagerly triggers index creation (or catalog look-up)
    so that the index is ready for ``search`` immediately after
    ``__init__``.

    Items are materialized ``dict`` objects with a ``"uri"`` key.
    The underlying ``HierarchicalFileIndex`` operates on file URIs;
    ``search`` maps the returned URIs back to the original items.

    Representation invariant:
        - ``_index`` is ``None`` iff no items with a valid URI were
          provided **and** no pre-built index was supplied.
        - ``_uri_to_idx`` maps every URI present in ``items`` to the
          item's position in ``self.items``.

    Abstraction function:
        Represents a hierarchical file-summary index over
        ``self.items``, enabling LLM-routed top-k retrieval by file
        relevance.
    """

    description: str = (
        "Tree-structured index with internal nodes summarizing clusters of files or child nodes. "
        "Search uses top-down traversal by query-embedding similarity. "
        "Best for larger datasets; reduces context use by routing through the hierarchy."
    )

    def __init__(
        self,
        name: str,
        items: list[dict] | None = None,
        model: LiteLLMModel | str | None = None,
        config=None,
        api_key: str | None = None,
        index = None,
        **kwargs,
    ):
        super().__init__(name=name, items=items, index=index)
        self._config = config
        self._api_key = api_key
        self._llm_model = _resolve_model(model, api_key)
        self._uri_to_idx: dict[str, int] = _build_uri_to_idx(self.items)
        if self._index is None and self._uri_to_idx:
            self._get_or_create_index()

    def _get_or_create_index(self) -> HierarchicalFileIndex | None:
        """Build or retrieve the hierarchical file index.

        On first call, builds a :class:`HierarchicalFileIndex` from
        ``self.items``.  Subsequent calls return the cached index.

        Requires:
            - ``self.items`` contains at least one item with a non-empty URI.

        Returns:
            The :class:`HierarchicalFileIndex`.

        Raises:
            ValueError: if no items have a valid URI or the index
            cannot be built.
        """
        if self._index is not None:
            return self._index

        if not self.items:
            raise ValueError("HierarchicalCarnotIndex: no items with uri to build index")

        index = HierarchicalFileIndex.from_items(
            name=self.name,
            items=self.items,
            model=self._llm_model,
            config=self._config,
            api_key=self._api_key,
        )
        if index is None:
            raise ValueError("Could not build hierarchical index from items")

        self._index = index
        return self._index

    def search(self, query: str, k: int) -> list:
        """Return up to *k* items most relevant to *query*.

        Delegates to the underlying ``HierarchicalFileIndex.search``
        and maps returned URIs back to the original items.

        Requires:
            - *query* is a non-empty string.
            - *k* >= 1.

        Returns:
            A list of at most *k* items (same type as ``self.items``
            elements), ordered by descending relevance.  Items whose
            URI was not found in the original item list are silently
            skipped.

        Raises:
            ValueError: if the index has not been built and cannot be
            created.
        """
        if self._index is None:
            self._get_or_create_index()
        uris = self._index.search(query, k)
        return [self.items[self._uri_to_idx[p]] for p in uris if p in self._uri_to_idx][:k]


class FlatCarnotIndex(CarnotIndex):
    """CarnotIndex backed by a :class:`FlatFileIndex`.

    All file summaries live in a flat list.  At query time the LLM
    (or embedding pre-filter) selects the top-k most relevant files.

    Items will have previously been materialized to dictionaries.

    Representation invariant:
        - ``_index`` is ``None`` iff no items with a valid URI were
          provided **and** no pre-built index was supplied.
        - ``_uri_to_idx`` maps every URI present in ``items`` to the
          item's position in ``self.items``.

    Abstraction function:
        Represents a flat file-summary index over ``self.items``,
        enabling LLM-selected top-k retrieval.
    """

    description: str = (
        "Single-level index where all file summaries are in one list. "
        "At query time, the LLM receives summaries (or an embedding-pre-filtered subset) "
        "and selects the top-K most relevant. "
        "Best for smaller datasets where summaries fit in context."
    )

    def __init__(
        self,
        name: str,
        items: list[dict] | None = None,
        model: LiteLLMModel | str | None = None,
        config=None,
        api_key: str | None = None,
        index = None,
        **kwargs,
    ):
        super().__init__(name=name, items=items, index=index)
        self._config = config
        self._api_key = api_key
        self._llm_model = _resolve_model(model, api_key)
        self._uri_to_idx: dict[str, int] = _build_uri_to_idx(self.items)
        if self._index is None and self._uri_to_idx:
            self._get_or_create_index()

    def _get_or_create_index(self) -> FlatFileIndex | None:
        """Build or retrieve the flat file index.

        On first call, builds a :class:`FlatFileIndex` from
        ``self.items``.  Subsequent calls return the cached index.

        Requires:
            - ``self.items`` contains at least one item with a non-empty URI.

        Returns:
            The :class:`FlatFileIndex`.

        Raises:
            ValueError: if no items have a valid URI or the index
            cannot be built.
        """
        if self._index is not None:
            return self._index

        if not self.items:
            raise ValueError("FlatCarnotIndex: no items with uri to build index")

        index = FlatFileIndex.from_items(
            name=self.name,
            items=self.items,
            model=self._llm_model,
            config=self._config,
            api_key=self._api_key,
        )
        if index is None:
            raise ValueError("Could not build flat index from items")

        self._index = index
        return self._index

    def search(self, query: str, k: int) -> list:
        """Return up to *k* items most relevant to *query*.

        Delegates to the underlying ``FlatFileIndex.search`` and maps
        returned URIs back to the original items.

        Requires:
            - *query* is a non-empty string.
            - *k* >= 1.

        Returns:
            A list of at most *k* items (same type as ``self.items``
            elements), ordered by descending relevance.

        Raises:
            ValueError: if the index has not been built and cannot be
            created.
        """
        if self._index is None:
            self._get_or_create_index()
        uris = self._index.search(query, k)
        return [self.items[self._uri_to_idx[p]] for p in uris if p in self._uri_to_idx][:k]


class ChromaIndex(CarnotIndex):
    """CarnotIndex backed by a ChromaDB persistent collection.

    Embeds items using a configurable embedding model and stores them
    in a ChromaDB persistent collection.  Search performs vector
    similarity retrieval.

    Items must be ``str`` or ``dict`` (serialised to JSON).

    Representation invariant:
        - ``_index`` is a ``chromadb.Collection`` after construction.
        - ``self.ids`` has the same length as ``self.items``.

    Abstraction function:
        Represents a ChromaDB vector-similarity index over
        ``self.items``.
    """

    description: str = (
        "Index that uses ChromaDB to find the top-K most relevant items. "
        "Uses a ChromaDB vector similarity search index. "
        "Best if you need to find the top-K most relevant items very quickly but has lower accuracy."
    )

    def __init__(
        self,
        name: str,
        items: list[dict] | None = None,
        model: LiteLLMModel | str | None = "openai/text-embedding-3-small",
        api_key: str = None,
        index = None,
    ):
        super().__init__(name=name, items=items, index=index)
        self.ids = [f"{idx}" for idx in range(len(self.items))]
        self.model = model if isinstance(model, str) else (model.model_id if model else "openai/text-embedding-3-small")
        self.api_key = api_key
        self._llm_model = _resolve_model(model, api_key)
        self._llm_call_stats: list = []
        if self._index is None:
            self._index = self._get_or_create_index()

    def _get_or_create_index(self):
        """Build or retrieve the ChromaDB collection.

        Uses ``StorageConfig().chroma_dir`` for persistent storage.
        If a collection with ``self.name`` already exists and is
        non-empty, it is returned without re-indexing.

        Requires:
            - ``self.items`` elements are ``dict``.

        Returns:
            A ``chromadb.Collection`` containing all items.

        Raises:
            ValueError: if an item is neither ``str`` nor ``dict``.
        """
        config = StorageConfig()
        chroma_directory = config.chroma_dir
        chroma_directory.mkdir(parents=True, exist_ok=True)

        chroma_client = chromadb.PersistentClient(str(chroma_directory))
        collection = chroma_client.get_or_create_collection(name=self.name)

        if collection.count() > 0:
            return collection

        item_strs = []
        for item in self.items:
            if isinstance(item, dict):
                item_strs.append(json.dumps(item))
            else:
                raise ValueError("ChromaIndex currently only supports items of type: [dict].")

        all_embeddings, stats = _embed_with_chunking(item_strs, self._llm_model)
        self._llm_call_stats.extend(stats)

        # Add in batches to avoid oversized ChromaDB inserts.
        for start in range(0, len(item_strs), _CHROMA_INSERT_BATCH_SIZE):
            end = min(start + _CHROMA_INSERT_BATCH_SIZE, len(item_strs))
            collection.add(
                documents=item_strs[start:end],
                embeddings=all_embeddings[start:end],
                ids=self.ids[start:end],
            )

        return collection

    def search(self, query: str, k: int) -> list:
        """Return up to *k* items most similar to *query*.

        Embeds the query and performs a ChromaDB vector search.

        Requires:
            - *query* is a non-empty string.
            - *k* >= 1.

        Returns:
            A list of at most *k* items from ``self.items``, ordered
            by vector similarity.

        Raises:
            None.
        """
        query_embedding, embed_stats = self._llm_model.embed(texts=[query])
        self._llm_call_stats.append(embed_stats)
        query_embedding = query_embedding[0]

        results = self._index.query(query_embeddings=[query_embedding], n_results=k)
        return [self.items[int(idx)] for idx in results["ids"][0]]


class FaissIndex(CarnotIndex):
    """CarnotIndex backed by a FAISS ``IndexFlatL2`` vector index.

    Embeds items and builds a FAISS flat L2 index, persisted to disk.
    Search performs nearest-neighbour retrieval.

    Items must be ``str`` (or string-coercible).

    Representation invariant:
        - ``_index`` is a ``faiss.Index`` after construction.
        - ``self.ids`` has the same length as ``self.items``.

    Abstraction function:
        Represents a FAISS flat-L2 vector index over ``self.items``.
    """

    description: str = (
        "Index that uses Faiss to find the top-K most relevant items. "
        "Uses a Faiss vector similarity search index. "
        "Best if you need to find the top-K most relevant items very quickly but has lower accuracy."
    )

    def __init__(
        self,
        name: str,
        items: list[dict] | None = None,
        model: LiteLLMModel | str | None = "openai/text-embedding-3-small",
        api_key: str = None,
        index = None,
    ):
        super().__init__(name=name, items=items, index=index)
        self.ids = [f"{idx}" for idx in range(len(self.items))]
        self.model = model if isinstance(model, str) else (model.model_id if model else "openai/text-embedding-3-small")
        self.api_key = api_key
        self._llm_model = _resolve_model(model, api_key)
        self._llm_call_stats: list = []
        if self._index is None:
            self._index = self._get_or_create_index()

    def _get_or_create_index(self):
        """Build or retrieve the FAISS index.

        Uses ``StorageConfig().faiss_dir`` for persistent storage.  If
        an index file for ``self.name`` already exists on disk, it is
        loaded instead of re-built.

        Requires:
            - ``self.items`` elements are embeddable strings.

        Returns:
            A ``faiss.Index`` containing embeddings for all items.

        Raises:
            None.
        """
        config = StorageConfig()
        faiss_directory = config.faiss_dir
        faiss_directory.mkdir(parents=True, exist_ok=True)

        index_path = faiss_directory / f"{self.name}.index"
        if index_path.exists():
            return faiss.read_index(str(index_path))

        item_strs = [json.dumps(item) if isinstance(item, dict) else str(item) for item in self.items]
        vectors, stats = _embed_with_chunking(item_strs, self._llm_model)
        self._llm_call_stats.extend(stats)

        matrix = np.asarray(vectors, dtype="float32")
        index = faiss.IndexFlatL2(matrix.shape[1])
        index.add(matrix)
        faiss.write_index(index, str(index_path))

        return index

    def search(self, query: str, k: int) -> list:
        """Return up to *k* items nearest to *query* in embedding space.

        Embeds the query and performs a FAISS L2 nearest-neighbour
        search.

        Requires:
            - *query* is a non-empty string.
            - *k* >= 1.

        Returns:
            A list of at most *k* items from ``self.items``, ordered
            by L2 distance (ascending).

        Raises:
            None.
        """
        query_embedding, embed_stats = self._llm_model.embed(texts=[query])
        self._llm_call_stats.append(embed_stats)
        query_embedding = query_embedding[0]

        query_vector = np.asarray([query_embedding], dtype="float32")
        _, indices = self._index.search(query_vector, k)

        return [self.items[int(idx)] for idx in indices[0]]


class SemanticIndex(CarnotIndex):
    """Placeholder index that returns items in insertion order.

    Not yet implemented — exists as a stub so that callers can
    reference the class without error.

    Representation invariant:
        - ``_index`` is always ``self`` (identity).

    Abstraction function:
        Represents a no-op index that ignores the query and returns the
        first *k* items.
    """

    description: str = "Placeholder semantic index (not yet implemented)."

    def __init__(self, name: str = "", items: list | None = None, **kwargs):
        super().__init__(name=name, items=items or [])
        self._index = self

    def _get_or_create_index(self):
        """No-op — the index is ``self``.

        Returns:
            ``self``.

        Raises:
            None.
        """
        return self

    def search(self, query: str, k: int) -> list:
        """Return the first *k* items (ignores *query*).

        Requires:
            - *k* >= 0.

        Returns:
            ``self.items[:k]``.

        Raises:
            None.
        """
        return self.items[:k]


# ── Module-private helpers ──────────────────────────────────────────────


def _build_token_aware_batches(
    token_counts: list[int],
    max_tokens_per_input: int,
    max_tokens_per_request: int = _MAX_TOKENS_PER_REQUEST,
) -> list[list[int]]:
    """Create batches that respect both per-input and per-request token limits.

    Batches are filled greedily: texts are added to the current batch until
    adding the next text would exceed either limit.

    Requires:
        - *token_counts* is a non-empty list of non-negative integers.
        - *max_tokens_per_input* > 0.
        - *max_tokens_per_request* > 0.

    Returns:
        A list of batches, where each batch is a list of integer indices.
        Every index in ``range(len(token_counts))`` appears in exactly one batch.

    Raises:
        None.
    """
    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_tokens = 0

    for idx, count in enumerate(token_counts):
        if current_batch and (current_tokens + count > max_tokens_per_request or count > max_tokens_per_input):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(idx)
        current_tokens += count

    if current_batch:
        batches.append(current_batch)

    return batches


def _embed_with_chunking(
    texts: list[str],
    llm_model: LiteLLMModel,
    max_workers: int = 16,
) -> tuple[list[list[float]], list]:
    """Embed *texts*, chunking any that exceed the model's per-input token limit.

    Texts that fit within the limit are batched together in a single
    embedding call.  Texts that exceed the limit are split into
    token-safe chunks, each chunk is embedded separately, and the
    resulting vectors are averaged to produce one embedding per
    original text.

    Requires:
        - *texts* is a list of strings (may be empty).
        - *llm_model* exposes an ``embed(texts, model)`` method and ``model_id``.

    Returns:
        A tuple ``(embeddings, stats)`` where *embeddings* is a list
        of float vectors (one per input text, in the same order) and
        *stats* is a list of ``LLMCallStats`` objects from all
        embedding calls made.

    Raises:
        Whatever ``llm_model.embed`` raises on unrecoverable failure.
    """
    if not texts:
        return [], []

    # Suppress litellm / openai / httpx log output during bulk embedding.
    litellm.suppress_debug_info = True

    # Maximum tokens the embedding model accepts per call.
    max_tokens = _EMBEDDING_MAX_INPUT_TOKENS.get(llm_model.model_id, _DEFAULT_EMBEDDING_MAX_INPUT_TOKENS)

    # Optimistic token estimates — used only for batching, not for correctness.
    from carnot.utils.model_helpers import _CHARS_PER_TOKEN
    est_token_counts = [len(t) // _CHARS_PER_TOKEN + 1 for t in texts]

    # Build batches from the estimates.  If any batch ends up exceeding the
    # real token limit the _process_batch fallback will handle it.
    batches = _build_token_aware_batches(est_token_counts, max_tokens)
    print(f"Embedding {len(texts)} texts in {len(batches)} batches (max_workers={max_workers})")

    # Pre-allocate result list.
    embeddings: list[list[float] | None] = [None] * len(texts)
    all_stats: list = []

    def _embed_with_retry(embed_texts: list[str], max_attempts: int = 10) -> tuple[list[list[float]], object]:
        """Call llm_model.embed with exponential backoff on rate-limit errors."""
        for attempt in range(max_attempts):
            try:
                return llm_model.embed(texts=embed_texts)
            except Exception as exc:
                if "rate_limit" in str(exc).lower() or "429" in str(exc):
                    delay = min(2 ** attempt + random.random(), 60.0)
                    time.sleep(delay)
                    continue
                raise
        # Final attempt — let it raise if it fails.
        return llm_model.embed(texts=embed_texts)

    def _process_batch(batch_indices: list[int]) -> tuple[list[tuple[int, list[float]]], list]:
        batch = [texts[i] for i in batch_indices]
        try:
            batch_embeddings, stats = _embed_with_retry(batch)
            return [(bi, emb) for bi, emb in zip(batch_indices, batch_embeddings, strict=True)], [stats]
        except Exception as exc:
            exc_str = str(exc).lower()
            if "maximum input length" not in exc_str and "max_tokens_per_request" not in exc_str:
                raise
            # Fallback: embed one-by-one, chunking any that still exceed the limit.
            results: list[tuple[int, list[float]]] = []
            stats_list: list = []
            for bi in batch_indices:
                try:
                    emb_list, stats = _embed_with_retry([texts[bi]])
                    results.append((bi, emb_list[0]))
                    stats_list.append(stats)
                except Exception:
                    chunks = _split_text_to_token_limit(texts[bi], max_tokens, llm_model.model_id)
                    chunk_embs: list[list[float]] = []
                    for chunk in chunks:
                        emb_list, stats = _embed_with_retry([chunk])
                        chunk_embs.append(emb_list[0])
                        stats_list.append(stats)
                    results.append((bi, np.mean(chunk_embs, axis=0).tolist()))
            return results, stats_list

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_batch, bi): bi for bi in batches}
        try:
            with tqdm(total=len(batches), desc="Embedding texts", unit="batch") as pbar:
                for fut in as_completed(futures):
                    pairs, batch_stats = fut.result()
                    all_stats.extend(batch_stats)
                    for bi, emb in pairs:
                        embeddings[bi] = emb
                    pbar.update(1)
        except Exception:
            # Cancel remaining futures so shutdown doesn't block.
            for f in futures:
                f.cancel()
            raise

    return embeddings, all_stats


def _split_text_to_token_limit(text: str, max_tokens: int, model: str) -> list[str]:
    """Split *text* into chunks each within *max_tokens*.

    Iteratively halves the character budget until each chunk fits.

    Requires:
        - *text* is a non-empty string.
        - *max_tokens* >= 1.

    Returns:
        A list of non-empty string chunks whose token counts are
        each <= *max_tokens*.

    Raises:
        None.
    """
    # Start with a character budget estimated from the token limit.
    # ~4 chars per token is a rough estimate; we validate and shrink.
    from carnot.utils.model_helpers import _CHARS_PER_TOKEN
    chunk_chars = max_tokens * _CHARS_PER_TOKEN
    while chunk_chars > 0:
        chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]
        try:
            counts = [litellm.token_counter(model=model, text=c) for c in chunks]
        except Exception:
            counts = [len(c) // _CHARS_PER_TOKEN + 1 for c in chunks]

        if all(c <= max_tokens for c in counts):
            return chunks
        chunk_chars //= 2

    # Last resort: return one-character chunks (should never happen).
    return list(text)


def _build_uri_to_idx(items: list) -> dict[str, int]:
    """Build a mapping from URI to index position in *items*.

    Requires:
        - *items* is a list.

    Returns:
        A ``dict`` mapping each non-empty URI to its first position
        in *items*.

    Raises:
        None.
    """
    mapping: dict[str, int] = {}
    for i, item in enumerate(items):
        uri = item.uri if hasattr(item, "uri") else (item.get("uri") if isinstance(item, dict) else None)
        if uri:
            mapping[uri] = i
    return mapping
