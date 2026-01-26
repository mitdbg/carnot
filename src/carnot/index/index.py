from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import chromadb
import faiss
import litellm
import numpy as np

INDEX_BATCH_SIZE = 1000

# class IndexCatalog:
#     _instance = None

#     def __new__(cls):
#         if cls._instance is None:
#             instance = super().__new__(cls)
#             instance.indices = None
#             instance.indices = instance._load_indices()
#             cls._instance = instance
#         return cls._instance

#     def _load_indices(self) -> dict[str, CarnotIndex]:
#         if self.indices is None:
#             home_directory = Path.home() / ".carnot" if os.getenv("CARNOT_HOME") is None else Path(os.getenv("CARNOT_HOME"))
#             indices_file = home_directory / "index_catalog.json"
#             with open()



class CarnotIndex(ABC):
    def __init__(self, name: str, items: list):
        self.name = name
        self.items = items
        # self.index_catalog = IndexCatalog()

    @abstractmethod
    def _add_index_to_catalog(self):
        pass

    @abstractmethod
    def _get_or_create_index(self):
        pass

    @abstractmethod
    def search(self, query: str, k: int) -> list:
        pass


class ChromaIndex(CarnotIndex):
    def __init__(self, name: str, items: list, model: str = "openai/text-embedding-3-small"):
        # construct the index
        super().__init__(name, items)
        self.ids = [f"{idx}" for idx in range(len(items))]
        self.model = model
        self.index = self._get_or_create_index()
        self._add_index_to_catalog()

    def _add_index_to_catalog(self):
        pass

    def _get_or_create_index(self):
        # retrieve the location of the chroma database
        home_directory = Path.home() / ".carnot" if os.getenv("CARNOT_HOME") is None else Path(os.getenv("CARNOT_HOME"))
        chroma_directory = home_directory / "chroma"
        chroma_directory.mkdir(parents=True, exist_ok=True)

        # create the chroma client
        chroma_client = chromadb.PersistentClient(str(chroma_directory))

        # call get_or_create_collection to either retrieve or create the collection
        collection = chroma_client.get_or_create_collection(name=self.name)

        # if the collection already exists, return it
        if collection.count() > 0:
            return collection

        # convert items to strings and generate embeddings in batches,
        item_strs = []
        for item in self.items:
            if isinstance(item, str):
                item_strs.append(item)
            elif isinstance(item, dict):
                item_strs.append(json.dumps(item))
            else:
                raise ValueError("ChromaIndex currently only supports items of type: [str, dict].")

        # add items to the collection
        for start in range(0, len(item_strs), INDEX_BATCH_SIZE):
            # generate embeddings for the batch
            # TODO: check that input does not exceed maximum input token limit for model and is not an empty string
            batch = item_strs[start : start + INDEX_BATCH_SIZE]
            response = litellm.embedding(model=self.model, input=batch)
            embeddings = [item["embedding"] for item in response.data]

            # insert embeddings into the collection
            collection.add(
                documents=item_strs[start : start + INDEX_BATCH_SIZE],
                embeddings=embeddings,
                ids=self.ids[start : start + INDEX_BATCH_SIZE],
            )

        return collection

    def search(self, query: str, k: int) -> list:
        # embed the query
        response = litellm.embedding(model=self.model, input=[query])
        query_embedding = response.data[0]["embedding"]

        # perform the search
        results = self.index.query(query_embeddings=[query_embedding], n_results=k)
        return [self.items[int(idx)] for idx in results["ids"][0]]


class FaissIndex(CarnotIndex):
    def __init__(self, name: str, items: list, model: str = "openai/text-embedding-3-small"):
        # construct the index
        super().__init__(name, items)
        self.ids = [f"{idx}" for idx in range(len(items))]
        self.model = model
        self.index = self._get_or_create_index()
        self._add_index_to_catalog()
    
    def _add_index_to_catalog(self):
        pass

    def _get_or_create_index(self):
        # retrieve the location of the vector database
        home_directory = Path.home() / ".carnot" if os.getenv("CARNOT_HOME") is None else Path(os.getenv("CARNOT_HOME"))
        faiss_directory = home_directory / "faiss"
        faiss_directory.mkdir(parents=True, exist_ok=True)

        # check if the index already exists and return it if so
        index_path = faiss_directory / f"{self.name}.index"
        if index_path.exists():
            return faiss.read_index(str(index_path))

        # otherwise, generate the embedding vectors in batches
        vectors = []
        for start in range(0, len(self.items), INDEX_BATCH_SIZE):
            batch = self.items[start : start + INDEX_BATCH_SIZE]
            response = litellm.embedding(model=self.model, input=batch)
            vectors.extend(item["embedding"] for item in response.data)

        # create the faiss index and save it to disk
        matrix = np.asarray(vectors, dtype="float32")
        index = faiss.IndexFlatL2(matrix.shape[1])
        index.add(matrix)
        faiss.write_index(index, str(index_path))

        return index

    def search(self, query: str, k: int) -> list:
        # embed the query
        response = litellm.embedding(model=self.model, input=[query])
        query_embedding = response.data[0]["embedding"]

        # perform the search
        query_vector = np.asarray([query_embedding], dtype="float32")
        _, indices = self.index.search(query_vector, k)

        return [self.items[int(idx)] for idx in indices[0]]


class SemanticIndex(CarnotIndex):
    def search(self, query: str, k: int) -> list:
        return self.items[:k]
