from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import logging
import chromadb
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Optional

from sentence_transformers import SentenceTransformer

from ..config import Config
from ..types import DocumentChunk, VectorQueryResult

from .concept_generator import LLMConceptGenerator

logger = logging.getLogger(__name__)

# ---------- Abstract interfaces (ABCs) ----------

class BaseVectorIndex(ABC):
    """Abstract interface for dense vector similarity search."""

    @abstractmethod
    def add_documents(self, docs: Iterable[DocumentChunk]) -> None:
        """Ingest documents into the vector index."""
        pass

    @abstractmethod
    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        include: Optional[Sequence[str]] = None,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> VectorQueryResult:
        """
        Return a projection of the top-k nearest neighbors. Implementations
        should honor `include` to avoid constructing expensive fields when the
        caller does not need them.
        """
        pass


class BaseKeywordIndex(ABC):
    """Abstract interface for keyword search."""

    @abstractmethod
    def filter(self, terms: Sequence[str], top_k: int | None = None) -> Sequence[str]:
        """Return doc_ids that satisfy the given keywords."""
        pass


class BaseMetadataStore(ABC):
    """Abstract interface for structural and semantic metadata tables."""

    @abstractmethod
    def upsert_metadata(self, doc_id: str, metadata: Mapping[str, Any]) -> None:
        """Insert or update metadata for a document."""
        pass

    @abstractmethod
    def replace_metadata(self, doc_id: str, metadata: Mapping[str, Any]) -> None:
        """Replace metadata for a document (overwrite)."""
        pass

    @abstractmethod
    def filter(self, predicates: Mapping[str, Any]) -> Sequence[str]:
        """Return doc_ids that satisfy structured metadata predicates."""
        pass


class BaseDocumentCatalog(ABC):
    """Abstract interface for the source-of-truth document store."""

    @abstractmethod
    def add_documents(self, docs: Iterable[DocumentChunk]) -> None:
        """Upsert documents (text + base metadata) into the store."""
        pass

    @abstractmethod
    def get_documents(self, doc_ids: Sequence[str]) -> Sequence[DocumentChunk]:
        """Retrieve documents by ID (must return text + metadata)."""
        pass

    @abstractmethod
    def list_document_ids(self) -> Sequence[str]:
        """List all known document IDs (eager)."""
        pass


# ---------- Concrete Chroma-backed implementations ----------

# ---- Embedding function for Chroma ----

class STEmbeddingFn:
    """SentenceTransformer-based embedding function for Chroma."""

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 64):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embed a batch of texts for Chroma."""
        if not input:
            return []
        embs = self.model.encode(
            input,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embs.tolist()

    def name(self) -> str:
        return f"sentence-transformers:{self.model_name}"


# ---- Chroma-backed catalog + vector index (one class, one collection) ----

class ChromaVectorIndex(BaseVectorIndex, BaseDocumentCatalog):
    """
    Dense vector index + document catalog backed by a single Chroma collection.

    Each row in the collection corresponds to a *chunk* (or whole document),
    with:
      - id
      - embedding   (managed by Chroma via embedding_function)
      - document    (chunk text)
      - metadata    (JSON dict with dataset + pipeline fields)
    """

    def __init__(self, config: Config) -> None:
        self._persist_dir = config.chroma_persist_dir
        self._collection_name = config.chroma_collection_name
        self._embedding_model_name = getattr(
            config, "embedding_model_name", "BAAI/bge-small-en-v1.5"
        )
        # Match legacy script behavior: larger batches for first-512 mode
        self._batch_size = 2048 if getattr(config, "index_first_512", False) else 256

        os.makedirs(self._persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._embed_fn = STEmbeddingFn(self._embedding_model_name)

        # Chroma will call _embed_fn when we upsert documents.
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embed_fn,
        )
        logger.info(f"Initialized ChromaVectorIndex (collection={self._collection_name}, persist_dir={self._persist_dir})")
        logger.info(f"ChromaVectorIndex: using upsert batch_size={self._batch_size} (index_first_512={getattr(config, 'index_first_512', False)})")

    # ---------- BaseDocumentCatalog API ----------

    def reset_collection(self) -> None:
        """
        Delete and recreate the underlying Chroma collection.
        """
        logger.info(f"ChromaVectorIndex: resetting collection '{self._collection_name}'.")
        try:
            # Delete existing collection if present
            self._client.delete_collection(self._collection_name)
        except Exception:
            # Ignore if collection does not exist
            pass
        # Recreate collection with the same embedding function
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embed_fn,
        )

    def add_documents(self, docs: Iterable[DocumentChunk]) -> None:
        """
        Add chunks/documents to Chroma.

        Each element in `docs` is expected to be:
          {
            "id": <chunk_id or doc_id>,
            "text": <chunk text>,
            "metadata": { ... arbitrary JSON-serializable fields ... }
          }
        """
        ids_batch: List[str] = []
        texts_batch: List[str] = []
        metas_batch: List[Mapping[str, Any]] = []
        pos_by_id: Dict[str, int] = {}
        total = 0

        def flush() -> None:
            nonlocal ids_batch, texts_batch, metas_batch, pos_by_id, total
            if not ids_batch:
                return
            self._collection.upsert(ids=ids_batch, documents=texts_batch, metadatas=metas_batch)
            total += len(ids_batch)
            logger.info(f"ChromaVectorIndex: upserted {len(ids_batch)} documents (total={total}).")
            ids_batch = []
            texts_batch = []
            metas_batch = []
            pos_by_id = {}

        for d in docs:
            doc_id = str(d.get("id") or d.get("doc_id"))
            text = d.get("text", "")
            metadata = dict(d.get("metadata", {}))

            if not doc_id or not text:
                continue

            if doc_id in pos_by_id:
                pos = pos_by_id[doc_id]
                texts_batch[pos] = text
                metas_batch[pos] = metadata
            else:
                pos_by_id[doc_id] = len(ids_batch)
                ids_batch.append(doc_id)
                texts_batch.append(text)
                metas_batch.append(metadata)

            if len(ids_batch) >= self._batch_size:
                flush()

        flush()

    def get_documents(self, doc_ids: Sequence[str]) -> Sequence[DocumentChunk]:
        """Retrieve documents (text + metadata) by ID from Chroma."""
        if not doc_ids:
            return []

        results = self._collection.get(
            ids=list(doc_ids),
            include=["documents", "metadatas"],
        )

        found_ids = results.get("ids", [])
        found_texts = results.get("documents", [])
        found_metas = results.get("metadatas", [])

        out: List[DocumentChunk] = []
        for i, doc_id in enumerate(found_ids):
            out.append(
                {
                    "id": doc_id,
                    "text": found_texts[i],
                    "metadata": found_metas[i] if found_metas else {},
                }
            )
        return out

    def list_document_ids(self) -> Sequence[str]:
        """List all stored chunk IDs."""
        results = self._collection.get(include=[])
        return results.get("ids", [])

    def upsert_metadata(self, doc_id: str, metadata: Mapping[str, Any]) -> None:
        """
        Update metadata for an existing document in Chroma.
        Performs a read-modify-write to preserve existing fields.
        """
        # Fetch existing metadata
        results = self._collection.get(ids=[doc_id], include=["metadatas"])
        existing_metas = results.get("metadatas")
        
        current_meta = {}
        if existing_metas and len(existing_metas) > 0 and existing_metas[0]:
            current_meta = existing_metas[0]

        # Merge new metadata
        current_meta.update(metadata)

        # Write back
        self._collection.update(
            ids=[doc_id],
            metadatas=[current_meta]
        )
    
    def replace_metadata(self, doc_id: str, metadata: Mapping[str, Any]) -> None:
        """
        Replace metadata for an existing document in Chroma (overwrite).
        """
        self._collection.update(
            ids=[doc_id],
            metadatas=[dict(metadata)]
        )

    def filter_by_metadata(self, predicates: Mapping[str, Any]) -> Sequence[str]:
        """
        Return doc_ids that satisfy the given metadata predicates (AND).
        """
        if not predicates:
            return self.list_document_ids()

        # Chroma expects 'where' clause
        # If single predicate: {"key": value}
        # If multiple: {"$and": [{"key": val}, ...]}
        
        where_clause: Dict[str, Any] = {}
        items = list(predicates.items())
        
        if len(items) == 1:
            where_clause = {items[0][0]: items[0][1]}
        else:
            where_clause = {"$and": [{k: v} for k, v in items]}

        results = self._collection.get(where=where_clause, include=[])
        return results.get("ids", [])
    
    def filter_by_keyword(
        self,
        terms: Sequence[str],
        top_k: int | None = None,
    ) -> Sequence[str]:
        """
        Return doc_ids whose *document text* matches the given keywords.
        This uses Chroma's `where_document` + `$contains` for full-text search.

        - Currently uses AND semantics: all terms must appear in the text.
        - If `top_k` is None, returns all matching ids (subject to Chroma limits).
        """
        # Normalize terms
        clean_terms = [t.strip() for t in terms if t and t.strip()]
        if not clean_terms:
            return []

        # Build where_document:
        # 1 term: {"$contains": "foo"}
        # >1 term: {"$and": [{"$contains": "foo"}, {"$contains": "bar"}]}
        if len(clean_terms) == 1:
            where_document: dict[str, Any] = {"$contains": clean_terms[0]}
        else:
            where_document = {"$and": [{"$contains": t} for t in clean_terms]}

        # Chroma's get() supports where_document + limit
        results = self._collection.get(
            where_document=where_document,
            limit=top_k,           # None means "no explicit limit"
            include=[],            # we only need ids
        )
        return results.get("ids", []) or []

    # ---------- BaseVectorIndex API ----------

    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        include: Optional[Sequence[str]] = None,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> VectorQueryResult:
        """
        Query Chroma for nearest neighbors given a precomputed embedding.

        Returns a list of (id, distance) pairs.
        """
        logger.info(f"ChromaVectorIndex: querying with top_k={top_k}, filters={filters}")
        include_fields = list(include) if include is not None else ["distances"]
        results = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            include=include_fields,
            where=filters,
        )

        ids = results.get("ids", [[]])[0]
        has_documents = "documents" in include_fields
        has_metadatas = "metadatas" in include_fields
        has_distances = "distances" in include_fields

        documents = results.get("documents", [[]])[0] if has_documents else None
        metadatas = results.get("metadatas", [[]])[0] if has_metadatas else None
        distances = results.get("distances", [[]])[0] if has_distances else None

        if not ids:
            return VectorQueryResult(ids=[], documents=documents, metadatas=metadatas, distances=distances)

        return VectorQueryResult(ids=ids, documents=documents, metadatas=metadatas, distances=distances)


# ---- Simple keyword index stub (kept minimal for now) ----
class ChromaKeywordIndex(BaseKeywordIndex):
    """
    Thin façade over ChromaVectorIndex's keyword filter.
    """

    def __init__(self, vector_index: ChromaVectorIndex) -> None:
        self._vector_index = vector_index

    def add_documents(self, docs: Iterable[DocumentChunk]) -> None:
        # Vector index already handles ingest; nothing to do here.
        return

    def filter(self, terms: Sequence[str], top_k: int | None = None) -> Sequence[str]:
        """Keyword filter façade over the vector index."""
        return self._vector_index.filter_by_keyword(terms, top_k=top_k)


# ---- Structured metadata table with uniform schema ----

class TableMetadataStore(BaseMetadataStore):
    """
    Metadata store backed directly by Chroma.
    
    Instead of maintaining an in-memory dict, this delegates storage and 
    filtering to the underlying Chroma collection.
    """

    def __init__(self, vector_index: ChromaVectorIndex) -> None:
        self._vector_index = vector_index

    def upsert_metadata(self, doc_id: str, metadata: Mapping[str, Any]) -> None:
        """
        Insert or update metadata for a document via Chroma.
        """
        self._vector_index.upsert_metadata(doc_id, metadata)
    
    def replace_metadata(self, doc_id: str, metadata: Mapping[str, Any]) -> None:
        """
        Replace metadata for a document via Chroma (overwrite).
        """
        self._vector_index.replace_metadata(doc_id, metadata)

    def filter(self, predicates: Mapping[str, Any]) -> Sequence[str]:
        """Filter documents using Chroma's 'where' clause."""
        return self._vector_index.filter_by_metadata(predicates)

    def list_metadata_keys(self, sample_size: int = 1) -> Sequence[str]:
        """
        Best-effort list of metadata keys present in the collection.

        Since all enriched documents share the same schema, inspecting a single
        document is sufficient to discover the available concept keys.
        """
        # Get up to `sample_size` IDs
        all_ids = self._vector_index.list_document_ids()
        if not all_ids:
            return []

        sample_ids = all_ids[:sample_size]
        docs = self._vector_index.get_documents(sample_ids)

        keys: set[str] = set()
        for d in docs:
            meta = d.get("metadata", {}) or {}
            keys.update(meta.keys())

        return sorted(keys)


# ---- High-level index management pipeline ----

@dataclass
class IndexManagementPipeline:
    """
    Orchestrates ingestion, metadata, concept generation, and indexes on Chroma.

    Flow:
      - add_documents: store chunks in Chroma + register base metadata in TableMetadataStore
      - enrich_documents: assign concepts, update metadata

    This class is intentionally a thin orchestrator over composable building
    blocks (catalog, vector index, keyword index, metadata store, concept
    generator) so that backends can be swapped without changing the pipeline
    facade.
    """

    document_catalog: BaseDocumentCatalog
    vector_index: BaseVectorIndex
    keyword_index: BaseKeywordIndex
    metadata_store: TableMetadataStore
    concept_generator: LLMConceptGenerator

    @classmethod
    def from_config(cls, config: Config) -> "IndexManagementPipeline":
        # Single Chroma-backed object used as both catalog and vector index
        chroma_index = ChromaVectorIndex(config)
        keyword = ChromaKeywordIndex(chroma_index)
        
        # Metadata store now wraps the vector index (Chroma)
        metadata = TableMetadataStore(chroma_index)
        
        concept_gen = LLMConceptGenerator(config)

        return cls(
            document_catalog=chroma_index,
            vector_index=chroma_index,
            keyword_index=keyword,
            metadata_store=metadata,
            concept_generator=concept_gen,
        )

    def add_documents(self, docs: Iterable[DocumentChunk], reset: bool = False) -> None:
        """
        Ingest documents/chunks into the system (Chroma + metadata + keyword index).

        Each input doc is treated as a *chunk*:
          {
            "id": <chunk_id>,
            "text": <chunk text>,
            "metadata": { ... base fields from dataset / pipeline ... }
          }

        After this call:
          - Chroma has (id, embedding, document text, base metadata).
        """
        docs_list = list(docs)
        if not docs_list:
            return

        logger.info(f"IndexManagementPipeline: adding {len(docs_list)} documents.")

        # Optionally clear the Chroma collection before ingesting
        if reset and hasattr(self.document_catalog, "reset_collection"):
            logger.info("IndexManagementPipeline: clear_chroma_collection=True; resetting underlying Chroma collection...")
            self.document_catalog.reset_collection()

        # Source of truth for text + embeddings + raw metadata: Chroma
        self.document_catalog.add_documents(docs_list)

        # Keyword index (if implemented)
        self.keyword_index.add_documents(docs_list)

    def enrich_documents(self, concept_vocabulary: Optional[List[str]] = None, batch_size: int = 128) -> None:
        """
        Run concept assignment and update metadata for EXISTING documents.

        After this call:
          - concept:* fields are added to TableMetadataStore (same schema for all docs).
        """
        doc_ids = self.document_catalog.list_document_ids()
        total_docs = len(doc_ids)
        if total_docs == 0:
            logger.info("IndexManagementPipeline: no documents to enrich.")
            return

        logger.info(f"IndexManagementPipeline: enriching {total_docs} documents in batches of {batch_size}.")

        if concept_vocabulary:
             concept_schemas = self.concept_generator._infer_concept_schemas(concept_vocabulary)
        else:
             concept_schemas = {}
        
        logger.info(f"IndexManagementPipeline: concept_schemas={concept_schemas}")

        for i in range(0, total_docs, batch_size):
            batch_ids = doc_ids[i : i + batch_size]
            docs = self.document_catalog.get_documents(batch_ids)
            
            if not docs:
                continue

            # Run concept mapping (produces a dict: doc_id -> { "concept:*": value, ... })
            assignments = self.concept_generator.concept_map(docs, concept_vocabulary=concept_vocabulary, concept_schemas=concept_schemas)

            # Update structured metadata store with concept fields
            # We first remove old concept fields (flat strings) and then apply new ones.
            # Since Chroma update is a replace operation on metadata, we fetch current, clean it, and write it back.
            
            for doc in docs:
                doc_id = str(doc.get("id"))
                if not doc_id:
                     continue
                
                # Fetch current metadata from the doc object we just retrieved
                current_meta = dict(doc.get("metadata", {}))
                
                # Identify old flat concept keys to remove
                # (any key starting with "concept:" that is not in the new assignments?)
                # Actually, safest to remove ALL "concept:" keys and apply new ones from scratch
                # to avoid mixing old flat keys with new hierarchical ones.
                keys_to_remove = [k for k in current_meta.keys() if k.startswith("concept:")]
                for k in keys_to_remove:
                    del current_meta[k]

                # Merge new assignments
                if doc_id in assignments:
                    current_meta.update(assignments[doc_id])

                # Replace the metadata in Chroma
                self.metadata_store.replace_metadata(doc_id, current_meta)
            
            logger.info(f"IndexManagementPipeline: processed batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size}")

    def get_vector_index(self) -> BaseVectorIndex:
        return self.vector_index

    def get_keyword_index(self) -> BaseKeywordIndex:
        return self.keyword_index

    def get_metadata_store(self) -> BaseMetadataStore:
        return self.metadata_store
