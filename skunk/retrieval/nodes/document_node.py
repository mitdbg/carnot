import hashlib
import json
import numbers
import os
from dataclasses import dataclass, field
from io import StringIO

import fitz
import pandas as pd
import pymupdf
from markdownify import markdownify
from tqdm import tqdm

from .page_node import PageNode
from .tools import call_openrouter

DEFAULT_OCR_TEXT_DIR = "data/officeqa/treasury_bulletins_parsed/transformed"
MAX_DOCUMENT_PROMPT_CHARS = 2000

@dataclass
class DocumentNode:
    document_id: str
    filename: str
    document_title: str
    document_date: str
    page_count: int
    page_nodes: list[PageNode] = field(default_factory=list)
    description: str = ""
    ocr_text_dir: str = DEFAULT_OCR_TEXT_DIR

    def __init__(self, filename: str, content: bytes, ocr_text_dir: str = DEFAULT_OCR_TEXT_DIR):

        sha256 = hashlib.sha256(content).hexdigest()
        self.document_id = f"doc:{sha256[:16]}"
        pdf_name = os.path.basename(filename)
        self.filename = pdf_name
        self.ocr_text_dir = ocr_text_dir

        stem = os.path.splitext(pdf_name)[0]
        ocr_text_uri = os.path.join(self.ocr_text_dir, f"{stem}.txt")
        ocr_text = ""
        if os.path.exists(ocr_text_uri):
            with open(ocr_text_uri, errors="replace") as ocr_file:
                ocr_text = ocr_file.read()
        else:
            ocr_text_uri = None

        pdf = fitz.open(stream=content, filetype="pdf")
        self.page_count = len(pdf)
        self.document_title = self.extract_document_title(ocr_text)
        self.document_date = self.extract_document_date(ocr_text)
        self.description = self.describe_text(ocr_text)

        self.page_nodes = []
        page_texts, page_nums = self.match_page_text(pdf, ocr_text)
        for page_idx, page in enumerate(
            tqdm(
                pdf,
                total=len(pdf),
                desc=f"Processing pages in {self.filename}",
                unit="page",
            )
        ):
            page_image = page.get_pixmap(dpi=300).tobytes()  # render page to image bytes
            page_text = page_texts[page_idx]
            page_node = PageNode(
                document_id=self.document_id,
                page_pdf_number=page_idx,
                page_raw=page,
                page_text=page_text,
                page_image=page_image,
                page_number=page_nums.get(page_idx, ""),
            )
            if page_node is not None:
                self.page_nodes.append(page_node)

        pdf.close()

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

        return call_openrouter(prompt)

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

        return call_openrouter(prompt)

    def describe_text(self, text: str) -> str:
        fallback = f"{self.document_title} from {self.document_date} contains {self.page_count} pages."
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
Page count: {self.page_count}

Document OCR text:
{text[:MAX_DOCUMENT_PROMPT_CHARS]}
""".strip()

        desc = call_openrouter(prompt)
        return desc

    def source_json_path(self) -> str | None:
        stem = os.path.splitext(self.filename)[0]
        candidate_dirs = [
            os.path.join(os.path.dirname(self.ocr_text_dir), "jsons"),
            DEFAULT_OCR_TEXT_DIR.replace("/transformed", "/jsons"),
        ]
        for json_dir in candidate_dirs:
            json_path = os.path.join(json_dir, f"{stem}.json")
            if os.path.exists(json_path):
                return json_path
        return None

    def convert_json_content_to_text(self, content: str) -> str:
        if "<table" not in content.lower():
            if "<" in content and ">" in content:
                return markdownify(content).strip()
            return content

        converted_tables = []
        for dataframe in pd.read_html(StringIO(content)):
            if isinstance(dataframe.columns, pd.MultiIndex):
                headers = [
                    " > ".join(str(part) for part in column if str(part) != "nan")
                    for column in dataframe.columns
                ]
            else:
                headers = [str(column) for column in dataframe.columns]

            non_null_values = []
            for row in dataframe.itertuples(index=False, name=None):
                for value in row:
                    if not pd.isna(value):
                        non_null_values.append(value)

            force_float_values = (
                all(isinstance(column, int) for column in dataframe.columns)
                and dataframe.isna().any().any()
                and non_null_values
                and all(isinstance(value, numbers.Number) for value in non_null_values)
            )

            rows = []
            rows.append("| " + " | ".join(headers) + " |")
            rows.append("| " + " | ".join(["---"] * len(headers)) + " |")

            for row in dataframe.itertuples(index=False, name=None):
                cells = []
                for value in row:
                    if force_float_values and not pd.isna(value):
                        cells.append(f"{float(value):.1f}")
                    else:
                        cells.append(str(value))
                rows.append("| " + " | ".join(cells) + " |")

            converted_tables.append("\n".join(rows))

        return "\n\n".join(converted_tables)

    def match_page_text(self, pdf: pymupdf.Document, doc_text: str) -> tuple[dict[int, str], dict[int, str]]:
        source_json_path = self.source_json_path()
        if source_json_path is None:
            return {page_idx: "" for page_idx in range(len(pdf))}, {}

        with open(source_json_path, encoding="utf-8") as source_json_file:
            source_json = json.load(source_json_file)

        page_blocks = {page_idx: [] for page_idx in range(len(pdf))}
        page_nums = {}

        for element in tqdm(
            source_json["document"]["elements"],
            total=len(source_json["document"]["elements"]),
            desc=f"Parsing page text from JSON in {self.filename}",
            unit="element",
        ):
            content = element.get("content")
            if not content:
                continue

            converted_content = self.convert_json_content_to_text(content)
            page_ids = []
            for bbox in element.get("bbox") or []:
                page_id = bbox.get("page_id")
                if isinstance(page_id, int):
                    page_ids.append(page_id)

            for page_id in dict.fromkeys(page_ids):
                page_idx = page_id - 1
                if page_idx not in page_blocks:
                    continue
                page_blocks[page_idx].append(converted_content)
                if element.get("type") == "page_number" and page_idx not in page_nums:
                    page_nums[page_idx] = converted_content.strip()

        page_texts = {}
        for page_idx, blocks in page_blocks.items():
            if blocks:
                page_texts[page_idx] = "\n\n".join(blocks) + "\n\n"
            else:
                page_texts[page_idx] = ""

        return page_texts, page_nums
