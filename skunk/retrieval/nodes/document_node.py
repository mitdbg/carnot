import hashlib
import json
import os
from dataclasses import dataclass, field
import concurrent.futures
import fitz
from tqdm import tqdm

from skunk.retrieval.nodes.pdf_file import parse_pdf_pages

from .llm_wrapper import get_llm_wrapper
from .page_node import PageNode

MAX_DOCUMENT_PROMPT_CHARS = 2000
PAGE_PROCESS_WORKERS = 128
PAGE_RENDER_DPI = 150


@dataclass
class DocumentNode:
    document_id: str
    filename: str
    document_title: str
    document_date: str
    num_pdf_pages: int
    ocr_text_dir: str
    json_dir: str
    page_nodes: dict[str, PageNode] = field(default_factory=dict)
    description: str = ""
    page_process_workers: int = PAGE_PROCESS_WORKERS

    def __init__(
        self,
        filename: str,
        pdf_bytes: bytes,
        ocr_text_dir: str,
        json_dir: str,
        page_process_workers: int = PAGE_PROCESS_WORKERS,
        use_cache: bool = True,
    ):

        sha256 = hashlib.sha256(pdf_bytes).hexdigest()
        self.document_id = f"doc:{sha256[:16]}"
        pdf_name = os.path.basename(filename)
        self.filename = pdf_name
        self.ocr_text_dir = ocr_text_dir
        self.json_dir = json_dir
        self.page_process_workers = page_process_workers
        self.use_cache = use_cache

        stem = os.path.splitext(pdf_name)[0]
        ocr_text_uri = os.path.join(self.ocr_text_dir, f"{stem}.txt")
        ocr_text = ""
        if os.path.exists(ocr_text_uri):
            with open(ocr_text_uri, errors="replace") as ocr_file:
                ocr_text = ocr_file.read()
        else:
            ocr_text_uri = None

        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        self.num_pdf_pages = len(pdf)
        self.document_title = self.extract_document_title(ocr_text)
        self.document_date = self.extract_document_date(ocr_text)
        self.description = self.describe_text(ocr_text)
        pdf.close()

        stem = os.path.splitext(self.filename)[0]
        source_json_path = os.path.join(self.json_dir, f"{stem}.json")

        with open(source_json_path, encoding="utf-8") as source_json_file:
            source_json = json.load(source_json_file)

        page_elements = {idx: [] for idx in range(self.num_pdf_pages)}
        for e in source_json["document"]["elements"]:
            page_ids = [bb["page_id"] for bb in e["bbox"]]
            page_ids = list(set(page_ids))
            if len(page_ids) > 1:
                print(
                    f"Warning: element appears on multiple pages {page_ids}, assigning to first page"
                )
            # json is 1-indexed
            page_id = page_ids[0] - 1 if page_ids else None
            if page_id is not None:
                if page_id >= self.num_pdf_pages:
                    continue
                else:
                    page_elements[page_id].append(e)

        figure_page_indices = [
            idx
            for idx, elements in page_elements.items()
            if [e for e in elements if e["type"] == "figure"]
        ]
        page_images = parse_pdf_pages(
            filename,
            n_workers=self.page_process_workers,
            dpi=PAGE_RENDER_DPI,
            selected_page_indices=figure_page_indices,
        )

        page_args = []
        for idx in range(self.num_pdf_pages):
            if len(page_elements[idx]) > 0:
                if [e for e in page_elements[idx] if e["type"] == "figure"]:
                    img = page_images[idx] 
                else:
                    img = None
                page_args.append(
                    (
                        self.document_id,
                        idx,
                        page_elements.get(idx, []),
                        img,
                        self.use_cache,
                    )
                )

        self.page_nodes = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.page_process_workers, thread_name_prefix="llm-text"
        ) as executor:
            futures = {
                executor.submit(PageNode, *arg): arg_idx
                for arg_idx, arg in enumerate(page_args)
            }
            completed_futures = concurrent.futures.as_completed(futures)
            completed_futures = tqdm(
                completed_futures,
                total=len(futures),
                desc="Processing pages",
                unit="request",
            )
            for future in completed_futures:
                arg_idx = futures[future]
                page_node = future.result()
                if isinstance(page_node, Exception):
                    raise RuntimeError(
                        f"Failed to process page {page_args[arg_idx][1]} in {self.filename}"
                    ) from page_node
                self.page_nodes[page_node.page_id] = page_node

    def extract_document_title(self, text: str) -> str:
        if not text.strip():
            return "Untitled document"

        prompt = f"""
Extract the best human-readable title for this document.

Rules:
- Return only the title text.
- Do not include labels, quotes, markdown, or explanation.
- Prefer the title printed in the document over inventing one.
- If no title is visible, return: Untitled document

Document OCR text:
{text[:MAX_DOCUMENT_PROMPT_CHARS]}
""".strip()

        return get_llm_wrapper().call_llm(prompt, use_cache=self.use_cache)

    def extract_document_date(self, text: str) -> str:
        if not text.strip():
            return "Unknown date"

        prompt = f"""
Extract the document date from this OCR text.

Rules:
- Return only the date.
- Do not include labels, quotes, markdown, or explanation.
- Prefer the publication date, issue date, report date, or cover date.
- Preserve useful precision from the document, such as "June 2018" if no day appears.
- If no date is visible, return: Unknown date

Document OCR text:
{text[:MAX_DOCUMENT_PROMPT_CHARS]}
""".strip()

        return get_llm_wrapper().call_llm(prompt, use_cache=self.use_cache)

    def describe_text(self, text: str) -> str:
        fallback = f"{self.document_title} from {self.document_date} contains {self.num_pdf_pages} pages."
        if not text.strip():
            return fallback

        prompt = f"""
Write a concise semantic index description for this document.

Rules:
- Return exactly one sentence.
- Maximum 240 characters.
- Describe the document's subject matter and the kinds of information it contains.
- Do not include labels, quotes, markdown, or explanation.
- Do not mention OCR or that you are reading extracted text.

Document title: {self.document_title}
Document date: {self.document_date}
Page count: {self.num_pdf_pages}

Document OCR text:
{text[:MAX_DOCUMENT_PROMPT_CHARS]}
""".strip()

        desc = get_llm_wrapper().call_llm(prompt, use_cache=self.use_cache)
        return desc

    def get_page(self, page_id: str) -> PageNode | None:
        if page_id in self.page_nodes:
            return self.page_nodes[page_id]
        raise KeyError(f"Page {page_id} not found in document {self.document_id}")
