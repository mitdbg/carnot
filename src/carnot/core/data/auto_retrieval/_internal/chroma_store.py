import logging
import uuid
from typing import Any, Dict, List, Optional

import chromadb
# from sentence_transformers import SentenceTransformer
from openai import OpenAI

logger = logging.getLogger(__name__)

OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"


class ChromaStore:
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: str = "./chroma_db",
        # embedding_model_name: str = "Qwen/Qwen3-Embedding-4B",
        embedding_model_name: str = OPENAI_EMBEDDING_MODEL,
        distance_metric: str = "cosine",
        collection: Optional[Any] = None,
        normalize_embeddings: bool = True,
        embed: bool = True,
    ):
        self.embed = embed

        if collection is not None:
            self.collection = collection
            self.embedding_model_name = embedding_model_name
            self.normalize_embeddings = normalize_embeddings
            # self.model = SentenceTransformer(embedding_model_name) if self.embed else None
            self._openai_client = OpenAI() if self.embed else None
            return

        if not collection_name:
            raise ValueError("collection_name is required if no existing collection is provided.")

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_model_name = embedding_model_name
        self.normalize_embeddings = normalize_embeddings
        # self.model = SentenceTransformer(embedding_model_name) if self.embed else None
        self._openai_client = OpenAI() if self.embed else None

        # Note: when supplying embeddings manually, do NOT pass embedding_function.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )
        logger.info(
            f"Connected to collection '{collection_name}' at '{persist_directory}' "
            f"using embedding model '{embedding_model_name}'"
        )

    def _embed_documents(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not self.embed:
            return None
        if not texts:
            return []
        # embeddings = self.model.encode(
        #     texts,
        #     normalize_embeddings=self.normalize_embeddings,
        #     convert_to_numpy=True,
        #     show_progress_bar=False,
        # )
        # return embeddings.tolist()
        response = self._openai_client.embeddings.create(
            input=texts,
            model=self.embedding_model_name,
        )
        return [item.embedding for item in response.data]

    def _embed_query(self, query_text: str) -> Optional[List[float]]:
        if not self.embed:
            return None
        # # Qwen3 embedding models support query-specific prompting.
        # encode_kwargs = {
        #     "normalize_embeddings": self.normalize_embeddings,
        #     "convert_to_numpy": True,
        #     "show_progress_bar": False,
        # }
        #
        # if self.embedding_model_name.startswith("Qwen/"):
        #     query_embedding = self.model.encode(
        #         [query_text],
        #         prompt_name="query",
        #         **encode_kwargs,
        #     )[0]
        # else:
        #     # For BGE / MiniLM / most standard ST models, regular encode is fine.
        #     query_embedding = self.model.encode(
        #         [query_text],
        #         **encode_kwargs,
        #     )[0]
        #
        # return query_embedding.tolist()
        response = self._openai_client.embeddings.create(
            input=query_text,
            model=self.embedding_model_name,
        )
        return response.data[0].embedding

    def upsert_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 512,
    ) -> List[str]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        if len(documents) != len(ids):
            raise ValueError(f"Count mismatch: {len(documents)} docs vs {len(ids)} IDs")

        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError(f"Count mismatch: {len(metadatas)} metadata entries vs {len(ids)} IDs")

        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            batch_docs = documents[start:end]
            batch_ids = ids[start:end]
            batch_metas = metadatas[start:end] if metadatas is not None else None
            batch_embeddings = self._embed_documents(batch_docs)

            upsert_kwargs = dict(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
            )
            if batch_embeddings is not None:
                upsert_kwargs["embeddings"] = batch_embeddings
            self.collection.upsert(**upsert_kwargs)

        return ids

    def query(
        self,
        query_text: str,
        n_results: int = 100,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query_embedding = self._embed_query(query_text)

        query_kwargs = dict(
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        if query_embedding is not None:
            query_kwargs["query_embeddings"] = [query_embedding]
        else:
            query_kwargs["query_texts"] = [query_text]
        results = self.collection.query(**query_kwargs)

        clean_results = []
        if results["ids"] and len(results["ids"]) > 0:
            ids_list = results["ids"][0]
            docs_list = results["documents"][0] if results.get("documents") else []
            metas_list = results["metadatas"][0] if results.get("metadatas") else []
            dists_list = results["distances"][0] if results.get("distances") else []

            for i in range(len(ids_list)):
                clean_results.append(
                    {
                        "id": ids_list[i],
                        "document": docs_list[i] if i < len(docs_list) else None,
                        "metadata": metas_list[i] if i < len(metas_list) else None,
                        "distance": dists_list[i] if i < len(dists_list) else None,
                    }
                )

        return clean_results

    def update_document_metadata(
        self,
        doc_id: str,
        new_metadata: Dict[str, Any],
        id_field: str = "entity_id",
    ):
        results = self.collection.get(
            where={id_field: doc_id},
            include=["metadatas"],
        )

        if not results["ids"]:
            raise ValueError(f"Document with {id_field}='{doc_id}' not found.")
        if len(results["ids"]) > 1:
            raise ValueError(f"Multiple documents found for {id_field}='{doc_id}'.")

        internal_id = results["ids"][0]
        current_meta = (
            results["metadatas"][0]
            if results.get("metadatas") and results["metadatas"][0]
            else {}
        )

        self.collection.update(
            ids=[internal_id],
            metadatas=[{**current_meta, **new_metadata}],
        )

    def get(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        results = self.collection.get(
            where=where,
            limit=limit,
            include=["metadatas"],
            # include=["documents", "metadatas"],
        )

        clean = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                clean.append(
                    {
                        "id": results["ids"][i],
                        "document": (
                            results["documents"][i]
                            if results.get("documents") and i < len(results["documents"])
                            else None
                        ),
                        "metadata": (
                            results["metadatas"][i]
                            if results.get("metadatas") and i < len(results["metadatas"])
                            else None
                        ),
                    }
                )
        return clean

    def get_filtered_ids(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """Return only IDs matching *where* (no metadata deserialization)."""
        results = self.collection.get(
            where=where,
            limit=limit,
            include=[],
        )
        return results["ids"] if results["ids"] else []

    def build_title_index(self) -> Dict[str, Dict[str, Any]]:
        """Build an in-memory id -> {title, source} mapping for the whole collection."""
        results = self.collection.get(include=["metadatas"])
        index: Dict[str, Dict[str, Any]] = {}
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                meta = results["metadatas"][i] if results.get("metadatas") else {}
                index[doc_id] = {
                    "title": meta.get("title") if meta else None,
                    "source": meta.get("source") if meta else None,
                }
        return index

    def delete(self, ids: List[str]):
        self.collection.delete(ids=ids)

    def count(self) -> int:
        return self.collection.count()

    @classmethod
    def reset_collection(cls, collection_name: str, persist_directory: str):
        client = chromadb.PersistentClient(path=persist_directory)
        try:
            client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
        except Exception:
            logger.info(f"Collection '{collection_name}' not found, nothing to delete.")