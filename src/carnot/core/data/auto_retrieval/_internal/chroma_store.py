import logging
import uuid
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class ChromaStore:
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: str = "./chroma_db",
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        distance_metric: str = "cosine",
        collection: Optional[Any] = None,
    ):
        if collection is not None:
            self.collection = collection
            return

        if not collection_name:
            raise ValueError("collection_name is required if no existing collection is provided.")

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={
                "embedding_function": self.embedding_fn,
                "hnsw": {"space": distance_metric},
            }
        )
        logger.info(f"Connected to collection '{collection_name}' at '{persist_directory}'")

    def upsert_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 5000,
    ) -> List[str]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        if len(documents) != len(ids):
            raise ValueError(f"Count mismatch: {len(documents)} docs vs {len(ids)} IDs")
        if metadatas and len(metadatas) != len(ids):
            raise ValueError(f"Count mismatch: {len(metadatas)} metadata entries vs {len(ids)} IDs")

        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            self.collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end] if metadatas is not None else None,
            )
        return ids

    def query(
        self,
        query_text: str,
        n_results: int = 100,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        clean_results = []
        if results["ids"] and len(results["ids"]) > 0:
            ids_list = results["ids"][0]
            docs_list = results["documents"][0] if results["documents"] else []
            metas_list = results["metadatas"][0] if results["metadatas"] else []
            dists_list = results["distances"][0] if results["distances"] else []
            for i in range(len(ids_list)):
                clean_results.append({
                    "id": ids_list[i],
                    "document": docs_list[i] if i < len(docs_list) else None,
                    "metadata": metas_list[i] if i < len(metas_list) else None,
                    "distance": dists_list[i] if i < len(dists_list) else None,
                })
        return clean_results

    def update_document_metadata(self, doc_id: str, new_metadata: Dict[str, Any], id_field: str = "entity_id"):
        results = self.collection.get(
            where={id_field: doc_id},
            include=["metadatas"],
        )
        if not results["ids"]:
            raise ValueError(f"Document with {id_field}='{doc_id}' not found.")
        if len(results["ids"]) > 1:
            raise ValueError(f"Multiple documents found for {id_field}='{doc_id}'.")

        internal_id = results["ids"][0]
        current_meta = results["metadatas"][0] if results["metadatas"] and results["metadatas"][0] else {}
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
            include=["documents", "metadatas"],
        )
        clean = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                clean.append({
                    "id": results["ids"][i],
                    "document": results["documents"][i] if results.get("documents") and i < len(results["documents"]) else None,
                    "metadata": results["metadatas"][i] if results.get("metadatas") and i < len(results["metadatas"]) else None,
                })
        return clean

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
            
