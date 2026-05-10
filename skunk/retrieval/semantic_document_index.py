from __future__ import annotations

import hashlib
import os
import pickle
import re
import sys
from collections.abc import Iterator
from typing import Literal, TextIO
from skunk.retrieval.nodes.document_node import DocumentNode
from skunk.retrieval.nodes import *

ContentType = Literal["text", "plot", "table"]

DEFAULT_PDF_DIR = "data/officeqa/treasury_bulletin_pdfs"
DEFAULT_OCR_TEXT_DIR = "data/officeqa/treasury_bulletins_parsed/transformed"
SKUNK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE_DIR = os.path.join(SKUNK_DIR, "cache")
DEFAULT_CACHE_PATH = os.path.join(DEFAULT_CACHE_DIR, "semantic_document_index.pckl")

class SemanticDocumentIndex:
    def __init__(
        self,
        pdf_dir: str = DEFAULT_PDF_DIR,
        ocr_text_dir: str = DEFAULT_OCR_TEXT_DIR,
        render_binary: bool = True,
        render_zoom: float = 1.0,
        cache_path: str = DEFAULT_CACHE_PATH,
    ):
        self.cache_path = cache_path
        self.documents: dict[str, DocumentNode] = {}
        self._sha_to_document_id: dict[str, str] = {}
        if os.path.exists(self.cache_path) and False:
            try:
                with open(self.cache_path, "rb") as cache_file:
                    cached_data = pickle.load(cache_file)
                    self.documents = cached_data["documents"]
                    self._sha_to_document_id = cached_data.get("sha_to_document_id", {})
            except (EOFError, OSError, pickle.PickleError, TypeError, AttributeError, KeyError):
                self.documents: dict[str, DocumentNode] = {}
                self._sha_to_document_id: dict[str, str] = {}

        self.pdf_dir = pdf_dir
        self.ocr_text_dir = ocr_text_dir
        self.render_binary = render_binary
        self.render_zoom = render_zoom
        self.cache_path = cache_path

    def _save_cache(self) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        temp_cache_path = f"{self.cache_path}.tmp"
        with open(temp_cache_path, "wb") as cache_file:
            pickle.dump(
                {
                    "documents": self.documents,
                    "sha_to_document_id": self._sha_to_document_id,
                },
                cache_file,
            )
        os.replace(temp_cache_path, self.cache_path)

    def add(self, document: str | bytes) -> str:
        try:
            import fitz
        except ImportError as exc:
            raise ImportError("SemanticDocumentIndex requires PyMuPDF; install the project dependencies.") from exc

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
            return self._sha_to_document_id[sha256]

        document_node = DocumentNode(filename=source_uri, content=raw_pdf)
        self.documents[document_node.document_id] = document_node
        self._sha_to_document_id[sha256] = document_node.document_id
        self._save_cache()
        return document_node.document_id

    def retrieve(self, keyword: str) -> list[TextNode | TableNode | PlotNode]:
        if not isinstance(keyword, str) or not keyword.strip():
            raise ValueError("keyword must be a non-empty string")
        return []

    def get_node(self, node_id: str) -> DocumentNode | PageNode | TextNode | TableNode | PlotNode:
        for document in self.documents.values():
            if document.document_id == node_id:
                return document
            for page in document.page_nodes:
                if page.page_id == node_id:
                    return page
                for text_node in page.text_nodes:
                    if text_node.text_id == node_id:
                        return text_node
                for table_node in page.table_nodes:
                    if table_node.table_id == node_id:
                        return table_node
                for plot_node in page.plot_nodes:
                    if plot_node.plot_id == node_id:
                        return plot_node
        raise KeyError(node_id)

    def get_document_tree(self, document_id: str) -> DocumentNode:
        return self.documents[document_id]

    def iter_content(self, document_id: str, content_type: ContentType | None = None) -> Iterator[TextNode | TableNode | PlotNode]:
        document = self.get_document_tree(document_id)
        for page in document.page_nodes:
            if content_type in (None, "text"):
                yield from page.text_nodes
            if content_type in (None, "table"):
                yield from page.table_nodes
            if content_type in (None, "plot"):
                yield from page.plot_nodes

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
                (
                    f"{document_branch}doc {document.document_id} "
                    f"[{document_title}; date={document.document_date}; pages={document.page_count}]"
                )
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
                    (
                        f"{document_prefix}{page_branch}page {page.page_id} "
                        f"[pdf={page.page_pdf_number}; doc={page.page_doc_number}]"
                    )
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
                            node_line = f"{node_line}: {node_desc}"
                        output.append(node_line)

        rendered = "\n".join(output)
        return rendered

