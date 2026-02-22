from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import chromadb

from carnot.index.summary_indices import FlatFileIndex, HierarchicalFileIndex
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
    def get_index(self) -> HierarchicalFileIndex | FlatFileIndex | ChromaIndex | FaissIndex | SemanticIndex:
        pass

    @abstractmethod
    def get_index_type_string(self) -> str:
        pass

    @abstractmethod
    def _get_or_create_index(self):
        pass

    @abstractmethod
    def search(self, query: str, k: int) -> list:
        pass


class HierarchicalCarnotIndex(CarnotIndex):
    """CarnotIndex adapter for HierarchicalFileIndex. Maps paths to DataItems."""

    def __init__(
        self,
        name: str,
        items: list,
        hierarchical_index: "HierarchicalFileIndex | None" = None,
        config=None,
        api_key: str | None = None,
        use_persistence: bool = True,
        **kwargs
    ):
        super().__init__(name=name, items=items)
        self._hierarchical = hierarchical_index
        self._config = config
        self._api_key = api_key
        self._use_persistence = use_persistence
        self._path_to_item = {}
        for item in items:
            p = item.path if hasattr(item, "path") else (item.get("path") if isinstance(item, dict) else None)
            if p:
                self._path_to_item[p] = item
        if self._hierarchical is None and self._path_to_item:
            self._get_or_create_index()

    def _add_index_to_catalog(self):
        pass

    def get_index(self) -> HierarchicalFileIndex:
        if self._hierarchical is None:
            self._get_or_create_index()
        return self._hierarchical

    def get_index_type_string(self) -> str:
        return "HierarchicalCarnotIndex"

    def _get_or_create_index(self) -> HierarchicalFileIndex | None:
        if self._hierarchical is not None:
            return self._hierarchical
        from carnot.data.item import DataItem

        def to_data_item(i) -> DataItem | None:
            if isinstance(i, DataItem) and i.path:
                return i
            if isinstance(i, dict) and i.get("path"):
                di = DataItem(path=i["path"])
                di._dict = i
                return di
            return None

        data_items = [x for x in (to_data_item(i) for i in self.items) if x is not None]
        if not data_items:
            raise ValueError("HierarchicalCarnotIndex: no items with path to build index")
        index = HierarchicalFileIndex.from_items(
            name=self.name,
            items=data_items,
            config=self._config,
            api_key=self._api_key,
            use_persistence=self._use_persistence,
        )
        if index is None:
            raise ValueError("Could not build hierarchical index from items")
        self._hierarchical = index
        path_to_item = {}
        for item in self.items:
            p = item.path if hasattr(item, "path") else (item.get("path") if isinstance(item, dict) else None)
            if p:
                path_to_item[p] = item
        self._path_to_item = path_to_item
        return self._hierarchical

    def search(self, query: str, k: int) -> list:
        if self._hierarchical is None:
            self._get_or_create_index()
        paths = self._hierarchical.search(query, k)
        return [self._path_to_item[p] for p in paths if p in self._path_to_item][:k]


class FlatCarnotIndex(CarnotIndex):
    """CarnotIndex adapter for FlatFileIndex. Single-level, LLM selects top-K."""

    def __init__(
        self,
        name: str,
        items: list,
        flat_index: "FlatFileIndex | None" = None,
        config=None,
        api_key: str | None = None,
        use_persistence: bool = True,
        **kwargs
    ):
        super().__init__(name=name, items=items)
        self._flat = flat_index
        self._config = config
        self._api_key = api_key
        self._use_persistence = use_persistence
        self._path_to_item = {}
        for item in items:
            p = item.path if hasattr(item, "path") else (item.get("path") if isinstance(item, dict) else None)
            if p:
                self._path_to_item[p] = item
        if self._flat is None and self._path_to_item:
            self._get_or_create_index()

    def _add_index_to_catalog(self):
        pass

    def get_index(self) -> FlatFileIndex:
        if self._flat is None:
            self._get_or_create_index()
        return self._flat

    def get_index_type_string(self) -> str:
        return "FlatCarnotIndex"

    def _get_or_create_index(self) -> FlatFileIndex | None:
        if self._flat is not None:
            return self._flat
        from carnot.data.item import DataItem

        def to_data_item(i):
            if isinstance(i, DataItem) and i.path:
                return i
            if isinstance(i, dict) and i.get("path"):
                di = DataItem(path=i["path"])
                di._dict = i
                return di
            return None

        data_items = [x for x in (to_data_item(i) for i in self.items) if x is not None]
        if not data_items:
            raise ValueError("FlatCarnotIndex: no items with path to build index")
        index = FlatFileIndex.from_items(
            name=self.name,
            items=data_items,
            config=self._config,
            api_key=self._api_key,
            use_persistence=self._use_persistence,
        )
        if index is None:
            raise ValueError("Could not build flat index from items")
        self._flat = index
        return self._flat

    def search(self, query: str, k: int) -> list:
        if self._flat is None:
            self._get_or_create_index()
        paths = self._flat.search(query, k)
        return [self._path_to_item[p] for p in paths if p in self._path_to_item][:k]


class ChromaIndex(CarnotIndex):
    def __init__(self, name: str, items: list, model: str = "openai/text-embedding-3-small", api_key: str = None):
        # construct the index
        super().__init__(name, items)
        self.ids = [f"{idx}" for idx in range(len(items))]
        self.model = model
        self.api_key = api_key
        self.index = self._get_or_create_index()
        self._add_index_to_catalog()

    def _add_index_to_catalog(self):
        pass

    def get_index(self) -> ChromaIndex:
        if self._chroma is None:
            self._get_or_create_index()
        return self._chroma

    def get_index_type_string(self) -> str:
        return "ChromaIndex"

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
            response = litellm.embedding(model=self.model, input=batch, api_key=self.api_key)
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
        response = litellm.embedding(model=self.model, input=[query], api_key=self.api_key)
        query_embedding = response.data[0]["embedding"]

        # perform the search
        results = self.index.query(query_embeddings=[query_embedding], n_results=k)
        return [self.items[int(idx)] for idx in results["ids"][0]]


class FaissIndex(CarnotIndex):
    def __init__(self, name: str, items: list, model: str = "openai/text-embedding-3-small", api_key: str = None):
        # construct the index
        super().__init__(name, items)
        self.ids = [f"{idx}" for idx in range(len(items))]
        self.model = model
        self.api_key = api_key
        self.index = self._get_or_create_index()
        self._add_index_to_catalog()
    
    def _add_index_to_catalog(self):
        pass

    def get_index(self) -> FaissIndex:
        if self._faiss is None:
            self._get_or_create_index()
        return self._faiss

    def get_index_type_string(self) -> str:
        return "FaissIndex"

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
            response = litellm.embedding(model=self.model, input=batch, api_key=self.api_key)
            vectors.extend(item["embedding"] for item in response.data)

        # create the faiss index and save it to disk
        matrix = np.asarray(vectors, dtype="float32")
        index = faiss.IndexFlatL2(matrix.shape[1])
        index.add(matrix)
        faiss.write_index(index, str(index_path))

        return index

    def search(self, query: str, k: int) -> list:
        # embed the query
        response = litellm.embedding(model=self.model, input=[query], api_key=self.api_key)
        query_embedding = response.data[0]["embedding"]

        # perform the search
        query_vector = np.asarray([query_embedding], dtype="float32")
        _, indices = self.index.search(query_vector, k)

        return [self.items[int(idx)] for idx in indices[0]]


class SemanticIndex(CarnotIndex):
    def get_index(self) -> SemanticIndex:
        return self

    def get_index_type_string(self) -> str:
        return "SemanticIndex"

    def search(self, query: str, k: int) -> list:
        return self.items[:k]
