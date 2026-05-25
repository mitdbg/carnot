from __future__ import annotations

import hashlib
import importlib.util
import os
import pickle
import re

import faiss
import numpy as np

from skunk.retrieval.nodes import PageNode, PlotNode, TableNode, TextNode
from skunk.retrieval.nodes.document_node import DocumentNode
from skunk.retrieval.nodes.llm_wrapper import DEFAULT_EMBEDDING_MODEL, get_llm_wrapper

SKUNK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE_DIR = os.path.join(SKUNK_DIR, "cache")
DEFAULT_CACHE_PATH = os.path.join(DEFAULT_CACHE_DIR, "semantic_document_index.pckl")
EMBEDDING_CACHE_VERSION = 1
EMBEDDING_WORKERS = 16

class SemanticDocumentIndex:

    def __init__(
        self,
        pdf_dir: str,
        ocr_text_dir: str,
        render_binary: bool = True,
        render_zoom: float = 1.0,
        cache_path: str = DEFAULT_CACHE_PATH,
        cache_flush_idx: int = 12,
        embedding_workers: int = EMBEDDING_WORKERS,
        use_cache: bool = True,
    ):
        self.cache_path = cache_path
        self.documents: dict[str, DocumentNode] = {}
        self._sha_to_document_id: dict[str, str] = {}
        self.initialized = False
        self.embedding_idx_map: dict[str, int] = {}
        self.embedding_node_ids: list[str] = []
        self.embedding_matrix: np.ndarray | None = None
        self.embedding_model: str | None = None
        self.embedding_faiss_index: faiss.IndexFlatIP | None = None
        self.cache_flush_idx = cache_flush_idx
        self.embedding_workers = embedding_workers
        if os.path.exists(self.cache_path) and use_cache:
            try:
                with open(self.cache_path, "rb") as cache_file:
                    cached_data = pickle.load(cache_file)
                    self.documents = cached_data["documents"]
                    self._sha_to_document_id = cached_data.get("sha_to_document_id", {})
                    self.embedding_idx_map = cached_data.get("embedding_idx_map", {})
                    self.embedding_node_ids = cached_data.get("embedding_node_ids", [])
                    self.embedding_matrix = cached_data.get("embedding_matrix")
                    self.embedding_model = cached_data.get("embedding_model")
                    # for document in self.documents.values():
                    # for page_node in document.page_nodes:
                    # if isinstance(page_node, Exception):
                    # raise ValueError("Cached semantic index contains a failed page node result.")
                    # if self.embedding_matrix is not None:
                        # self._rebuild_faiss_index()
                    # self.initialized = True
                    # self.initialize()
            except (EOFError, OSError, pickle.PickleError, TypeError, AttributeError, KeyError, ValueError):
                self.documents: dict[str, DocumentNode] = {}
                self._sha_to_document_id: dict[str, str] = {}
                self.embedding_idx_map = {}
                self.embedding_node_ids = []
                self.embedding_matrix = None
                self.embedding_model = None
                self.embedding_faiss_index = None

        self.pdf_dir = pdf_dir
        self.ocr_text_dir = ocr_text_dir
        self.render_binary = render_binary
        self.render_zoom = render_zoom
        self.cache_path = cache_path
        self.use_cache = use_cache

    def _save_cache(self) -> None:
        if not self.use_cache:
            return

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        temp_cache_path = f"{self.cache_path}.tmp"
        with open(temp_cache_path, "wb") as cache_file:
            pickle.dump(
                {
                    "documents": self.documents,
                    "sha_to_document_id": self._sha_to_document_id,
                    "embedding_cache_version": EMBEDDING_CACHE_VERSION,
                    "embedding_idx_map": self.embedding_idx_map,
                    "embedding_node_ids": self.embedding_node_ids,
                    "embedding_matrix": self.embedding_matrix,
                    "embedding_model": self.embedding_model,
                },
                cache_file,
            )
        os.replace(temp_cache_path, self.cache_path)

    def add(self, documents: list[str] | list[bytes]) -> str:
        if importlib.util.find_spec("fitz") is None:
            raise ImportError("SemanticDocumentIndex requires PyMuPDF; install the project dependencies.")

        ids = []
        for idx, document in enumerate(documents):
            # print("Adding document to index:", document)
            source_uri = "in_memory.pdf"
            if isinstance(document, str):
                source_uri = document
                with open(document, "rb") as pdf_file:
                    raw_pdf = pdf_file.read()
            elif isinstance(document, bytes):
                raw_pdf = document
            else:
                raise TypeError("document must be a PDF path or raw PDF bytes")

            sha256 = hashlib.sha256(raw_pdf).hexdigest()
            if sha256 in self._sha_to_document_id:
                ids.append(self._sha_to_document_id[sha256])
                continue

            document_node = DocumentNode(
                filename=source_uri,
                pdf_bytes=raw_pdf,
                ocr_text_dir=self.ocr_text_dir,
                use_cache=self.use_cache,
            )
            self.documents[document_node.document_id] = document_node
            self._sha_to_document_id[sha256] = document_node.document_id
            self.initialized = False

            if self.use_cache:
                if idx == len(documents)-1 or (idx > 0 and (idx % self.cache_flush_idx) == 0):
                    self._save_cache()
                    get_llm_wrapper().flush_cache()

            ids.append(document_node.document_id)
        return ids

    def initialize(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_workers: int | None = None,
    ) -> None:
        node_ids = []
        descriptions = []
        for document in self.documents.values():
            document_description = document.description or f"{document.document_title} {document.document_date} {document.filename}".strip()
            node_ids.append(document.document_id)
            descriptions.append(document_description)
            for page in document.page_nodes:
                page_description = page.description or f"Page {page.page_pdf_number}".strip()
                node_ids.append(page.page_id)
                descriptions.append(page_description)

                for text_node in page.text_nodes:
                    node_ids.append(text_node.text_id)
                    descriptions.append(text_node.description or text_node.text)
                for table_node in page.table_nodes:
                    node_ids.append(table_node.table_id)
                    descriptions.append(table_node.description or f"{table_node.table_title}\n{table_node.table_data}".strip())
                for plot_node in page.plot_nodes:
                    node_ids.append(plot_node.plot_id)
                    descriptions.append(plot_node.description or plot_node.plot_title)

        if not node_ids:
            self.initialized = True
            self.embedding_idx_map = {}
            self.embedding_node_ids = []
            self.embedding_matrix = None
            self.embedding_model = embedding_model
            self.embedding_faiss_index = None
            self._save_cache()
            return

        n_embedding_workers = self.embedding_workers if embedding_workers is None else embedding_workers
        n_embedding_workers = max(1, min(n_embedding_workers, len(descriptions)))
        embeddings = get_llm_wrapper().embed_texts(
            descriptions,
            model=embedding_model,
            use_cache=self.use_cache,
            n_workers=n_embedding_workers,
        )

        faiss.normalize_L2(embeddings)
        self.embedding_idx_map = {node_id: node_idx for node_idx, node_id in enumerate(node_ids)}
        self.embedding_node_ids = node_ids
        self.embedding_matrix = embeddings
        self.embedding_model = embedding_model
        self._rebuild_faiss_index()
        if self.use_cache:
            get_llm_wrapper().flush_cache(evict_generated_embeddings=True)
        self._save_cache()
        self.initialized = True

    def _rebuild_faiss_index(self) -> None:
        if self.embedding_matrix is None or len(self.embedding_matrix) == 0:
            self.embedding_faiss_index = None
            self.initialized = False
            return

        matrix = np.asarray(self.embedding_matrix, dtype="float32")
        faiss.normalize_L2(matrix)
        self.embedding_matrix = matrix
        self.embedding_faiss_index = faiss.IndexFlatIP(matrix.shape[1])
        self.embedding_faiss_index.add(matrix)
        self.initialized = True

    def get_node_id(self, node: DocumentNode | PageNode | TextNode | TableNode | PlotNode) -> str:
        if isinstance(node, DocumentNode):
            return node.document_id
        if isinstance(node, PageNode):
            return node.page_id
        if isinstance(node, TextNode):
            return node.text_id
        if isinstance(node, TableNode):
            return node.table_id
        return node.plot_id

    def search_description_embeddings(self, query_embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.embedding_faiss_index is None:
            raise RuntimeError("SemanticDocumentIndex must be initialized before retrieval.")

        query_matrix = np.asarray([query_embedding], dtype="float32")
        faiss.normalize_L2(query_matrix)
        scores, indexes = self.embedding_faiss_index.search(query_matrix, min(k, len(self.embedding_node_ids)))
        results = []
        for score, node_idx in zip(scores[0], indexes[0], strict=True):
            if node_idx == -1:
                continue
            results.append((self.embedding_node_ids[node_idx], float(score)))
        return results

    def get_node(self, node_id: str) -> DocumentNode | PageNode | TextNode | TableNode | PlotNode:
        doc_id = node_id.split("_")[0]
        doc = self.documents.get(doc_id)
        assert doc is not None, f"Document {doc_id} not found in index"
        if doc_id == node_id:
            return doc

        # if node_id is doc:9999_page:5_text:234... the page id is doc:9999_page:5
        page_id = "_".join(node_id.split("_")[:2])
        page = doc.page_nodes.get(page_id)
        assert page is not None, f"Page {page_id} not found in document {doc_id}"
        if page_id == node_id:
            return page

        if 'text' in node_id:
            text_node = page.text_nodes.get(node_id)
            if text_node is not None:
                return text_node
        elif 'table' in node_id:
            table_node = page.table_nodes.get(node_id)
            if table_node is not None:
                return table_node
        elif 'plot' in node_id:
            plot_node = page.plot_nodes.get(node_id)
            if plot_node is not None:
                return plot_node

        raise KeyError(node_id)

    def get_document_tree(self, document_id: str) -> DocumentNode:
        return self.documents[document_id]

    def counts(self) -> dict[str, int]:
        output = {
            "documents": len(self.documents),
            "document_nodes": len(self.documents),
            "page_nodes": 0,
            "content_nodes": 0,
            "text_nodes": 0,
            "table_nodes": 0,
            "plot_nodes": 0,
        }
        for document in self.documents.values():
            output["page_nodes"] += len(document.page_nodes)
            for page in document.page_nodes:
                output["text_nodes"] += len(page.text_nodes)
                output["table_nodes"] += len(page.table_nodes)
                output["plot_nodes"] += len(page.plot_nodes)
        output["content_nodes"] = output["text_nodes"] + output["table_nodes"] + output["plot_nodes"]
        return output

    def __str__(
        self,
    ) -> str:

        counts = self.counts()
        output = [
            (
                f"SemanticDocumentIndex documents={counts['documents']} "
                f"pages={counts['page_nodes']} content={counts['content_nodes']} "
                f"text={counts['text_nodes']} tables={counts['table_nodes']} plots={counts['plot_nodes']}"
            )
        ]

        documents = list(self.documents.values())

        for document_idx, document in enumerate(documents):
            document_is_last = document_idx == len(documents) - 1
            document_branch = "`-- " if document_is_last else "|-- "
            document_prefix = "    " if document_is_last else "|   "
            document_title = document.document_title or os.path.basename(document.filename) or document.document_id
            document_desc = document.description.strip() if document.description else ""
            document_desc = re.sub(r"\s+", " ", document_desc)

            output.append(
                f"{document_branch}doc {document.document_id} "
                f"[{document_title}; date={document.document_date}; pages={document.num_pdf_pages}]"
            )
            if document_desc:
                output.append(f"{document_prefix}|-- description: {document_desc}")

            pages = document.page_nodes
            if not pages:
                output.append(f"{document_prefix}`-- pages: empty")
                continue

            for page_idx, page in enumerate(pages):
                page_is_last = page_idx == len(pages) - 1
                page_branch = "`-- " if page_is_last else "|-- "
                page_prefix = "    " if page_is_last else "|   "
                page_desc = page.description.strip() if page.description else ""
                page_desc = re.sub(r"\s+", " ", page_desc)

                output.append(
                    f"{document_prefix}{page_branch}page {page.page_id} "
                    f"[pdf={page.page_pdf_number}; doc={page.page_pdf_number}]"
                )
                if page_desc:
                    output.append(f"{document_prefix}{page_prefix}|-- description: {page_desc}")

                content_groups = [
                    ("text", page.text_nodes),
                    ("tables", page.table_nodes),
                    ("plots", page.plot_nodes),
                ]
                non_empty_groups = [(label, nodes) for label, nodes in content_groups if nodes]
                if not non_empty_groups:
                    output.append(f"{document_prefix}{page_prefix}`-- content: empty")
                    continue

                for group_idx, (group_label, nodes) in enumerate(non_empty_groups):
                    group_is_last = group_idx == len(non_empty_groups) - 1
                    group_branch = "`-- " if group_is_last else "|-- "
                    group_prefix = "    " if group_is_last else "|   "
                    output.append(f"{document_prefix}{page_prefix}{group_branch}{group_label} ({len(nodes)})")

                    for node_idx, node in enumerate(nodes):
                        node_is_last = node_idx == len(nodes) - 1
                        node_branch = "`-- " if node_is_last else "|-- "

                        if isinstance(node, TextNode):
                            node_id = node.text_id
                            node_kind = "text"
                            node_title = node.text.strip()
                        elif isinstance(node, TableNode):
                            node_id = node.table_id
                            node_kind = "table"
                            node_title = node.table_title or node.table_data.strip()
                        else:
                            node_id = node.plot_id
                            node_kind = "plot"
                            node_title = node.plot_title

                        node_title = re.sub(r"\s+", " ", node_title or "")
                        if len(node_title) > 60:
                            node_title = f"{node_title[:57].rstrip()}..."

                        node_desc = node.description.strip() if node.description else ""
                        node_desc = re.sub(r"\s+", " ", node_desc)

                        node_line = f"{document_prefix}{page_prefix}{group_prefix}{node_branch}{node_kind} {node_id}"
                        if node_title:
                            node_line = f"{node_line} [{node_title}]"
                        if node_desc:
                            node_line = f"{node_line}\n{document_prefix}{page_prefix}{group_prefix}|-- description: {node_desc}"
                        output.append(node_line)

        rendered = "\n".join(output)
        return rendered
