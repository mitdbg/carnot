from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

import chromadb
import faiss
import litellm
import numpy as np

from carnot.index.sem_indices import FlatFileIndex, HierarchicalFileIndex
from carnot.storage.config import StorageConfig

logger = logging.getLogger(__name__)

INDEX_BATCH_SIZE = 1000


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
        config=None,
        api_key: str | None = None,
        index = None,
        **kwargs,
    ):
        super().__init__(name=name, items=items, index=index)
        self._config = config
        self._api_key = api_key
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
        config=None,
        api_key: str | None = None,
        index = None,
        **kwargs,
    ):
        super().__init__(name=name, items=items, index=index)
        self._config = config
        self._api_key = api_key
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
        model: str = "openai/text-embedding-3-small",
        api_key: str = None,
        index = None,
    ):
        super().__init__(name=name, items=items, index=index)
        self.ids = [f"{idx}" for idx in range(len(self.items))]
        self.model = model
        self.api_key = api_key
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

        for start in range(0, len(item_strs), INDEX_BATCH_SIZE):
            batch = item_strs[start : start + INDEX_BATCH_SIZE]
            response = litellm.embedding(model=self.model, input=batch, api_key=self.api_key)
            embeddings = [item["embedding"] for item in response.data]

            collection.add(
                documents=item_strs[start : start + INDEX_BATCH_SIZE],
                embeddings=embeddings,
                ids=self.ids[start : start + INDEX_BATCH_SIZE],
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
        response = litellm.embedding(model=self.model, input=[query], api_key=self.api_key)
        query_embedding = response.data[0]["embedding"]

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
        model: str = "openai/text-embedding-3-small",
        api_key: str = None,
        index = None,
    ):
        super().__init__(name=name, items=items, index=index)
        self.ids = [f"{idx}" for idx in range(len(self.items))]
        self.model = model
        self.api_key = api_key
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

        vectors = []
        for start in range(0, len(self.items), INDEX_BATCH_SIZE):
            batch = self.items[start : start + INDEX_BATCH_SIZE]
            response = litellm.embedding(model=self.model, input=batch, api_key=self.api_key)
            vectors.extend(item["embedding"] for item in response.data)

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
        response = litellm.embedding(model=self.model, input=[query], api_key=self.api_key)
        query_embedding = response.data[0]["embedding"]

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
