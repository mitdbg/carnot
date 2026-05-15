import hashlib
import json
import os
from dataclasses import dataclass, field

import fitz
from pqdm.threads import pqdm

from skunk.retrieval.nodes.pdf_file import parse_pdf_pages

from .llm_wrapper import DEFAULT_LLM_WRAPPER
from .page_node import PageNode

DEFAULT_OCR_TEXT_DIR = "data/officeqa/treasury_bulletins_parsed/transformed"
DEFAULT_JSON_DIR = "data/officeqa/treasury_bulletins_parsed/jsons"
MAX_DOCUMENT_PROMPT_CHARS = 2000
PAGE_PROCESS_WORKERS = 32
PAGE_RENDER_DPI = 150


@dataclass
class DocumentNode:
    document_id: str
    filename: str
    document_title: str
    document_date: str
    num_pdf_pages: int
    page_nodes: list[PageNode] = field(default_factory=list)
    description: str = ""
    ocr_text_dir: str = DEFAULT_OCR_TEXT_DIR
    json_dir: str = DEFAULT_JSON_DIR
    page_process_workers: int = PAGE_PROCESS_WORKERS

    def __init__(
        self,
        filename: str,
        pdf_bytes: bytes,
        ocr_text_dir: str = DEFAULT_OCR_TEXT_DIR,
        json_dir: str = DEFAULT_JSON_DIR,
        page_process_workers: int = PAGE_PROCESS_WORKERS,
    ):

        sha256 = hashlib.sha256(pdf_bytes).hexdigest()
        self.document_id = f"doc:{sha256[:16]}"
        pdf_name = os.path.basename(filename)
        self.filename = pdf_name
        self.ocr_text_dir = ocr_text_dir
        self.json_dir = json_dir
        self.page_process_workers = page_process_workers

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
        page_images = parse_pdf_pages(
            filename,
            n_workers=self.page_process_workers,
            dpi=PAGE_RENDER_DPI,
        )
        page_images = [p for x in page_images for p in x]
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
                    f"Warning: element with content {e['content'][:100]} appears on multiple pages {page_ids}, assigning to first page"
                )
            # json is 1-indexed
            page_id = page_ids[0] - 1 if page_ids else None
            if page_id is not None:
                if page_id >= self.num_pdf_pages:
                    continue
                else:
                    page_elements[page_id].append(e)

        args = [
            (
                self.document_id,
                idx,
                page_elements.get(idx, []),
                page_images[idx],
            )
            for idx in range(self.num_pdf_pages)
            if len(page_elements[idx]) > 0
        ]
        self.page_nodes = [PageNode(*arg) for arg in args]

        self.page_nodes = pqdm(
            args,
            PageNode,
            n_jobs=self.page_process_workers,
            argument_type="args",
            desc=f"Processing page nodes in {self.filename}",
        )

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

        return DEFAULT_LLM_WRAPPER.call_llm(prompt)

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

        return DEFAULT_LLM_WRAPPER.call_llm(prompt)

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

        desc = DEFAULT_LLM_WRAPPER.call_llm(prompt)
        return desc
