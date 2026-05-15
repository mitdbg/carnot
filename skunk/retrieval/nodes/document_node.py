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

from skunk.retrieval.nodes.pdf_file import parse_pdf_pages

from .page_node import PageNode
from .llm_wrapper import DEFAULT_LLM_WRAPPER

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
    page_count: int
    page_nodes: list[PageNode] = field(default_factory=list)
    description: str = ""
    ocr_text_dir: str = DEFAULT_OCR_TEXT_DIR
    json_dir: str = DEFAULT_JSON_DIR

    def __init__(
        self,
        filename: str,
        content: bytes,
        ocr_text_dir: str = DEFAULT_OCR_TEXT_DIR,
        json_dir: str = DEFAULT_JSON_DIR,
    ):

        sha256 = hashlib.sha256(content).hexdigest()
        self.document_id = f"doc:{sha256[:16]}"
        pdf_name = os.path.basename(filename)
        self.filename = pdf_name
        self.ocr_text_dir = ocr_text_dir
        self.json_dir = json_dir

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

        page_images = [p for x in parse_pdf_pages(filename, n_workers=PAGE_PROCESS_WORKERS, dpi=PAGE_RENDER_DPI) for p in x]

        page_inputs = [
            {
                "page_image": page_images[page_idx],
                "page_idx": page_idx,
                "page_text": page_texts[page_idx],
                "page_number": page_nums.get(page_idx, ""),
            }
            for page_idx in range(len(pdf))
            if page_texts.get(page_idx, "").strip()
        ]
        pdf.close()

        text_batch_prompts = []
        text_batch_map = []
        vision_batch_requests = []
        page_outputs = [
            {
                "description": "",
                "text_blocks": '{"text_blocks": []}',
                "tables": '{"tables": []}',
                "plots": '{"plots": []}',
            }
            for _ in page_inputs
        ]

        for page_input_idx, page_input in enumerate(page_inputs):
            page_text = page_input["page_text"]
            text_batch_prompts.append(PageNode.description_prompt(page_text))
            text_batch_map.append((page_input_idx, "description"))
            text_batch_prompts.append(PageNode.text_blocks_prompt(page_text))
            text_batch_map.append((page_input_idx, "text_blocks"))
            text_batch_prompts.append(PageNode.tables_prompt(page_text))
            text_batch_map.append((page_input_idx, "tables"))
            vision_batch_requests.append((PageNode.plots_prompt(page_text), page_input["page_image"]))

        text_batch_outputs = DEFAULT_LLM_WRAPPER.batch_call_llm(
            text_batch_prompts,
            desc=f"Processing page text prompts in {self.filename}",
        )
        for response_idx, response_text in enumerate(text_batch_outputs):
            page_input_idx, output_kind = text_batch_map[response_idx]
            page_outputs[page_input_idx][output_kind] = response_text

        vision_batch_outputs = DEFAULT_LLM_WRAPPER.batch_call_llm_vision(
            vision_batch_requests,
            desc=f"Processing page vision prompts in {self.filename}",
        )
        for page_input_idx, response_text in enumerate(vision_batch_outputs):
            page_outputs[page_input_idx]["plots"] = response_text

        self.page_nodes = [
            PageNode.from_llm_outputs(
                document_id=self.document_id,
                page_pdf_number=page_input["page_idx"],
                page_image=page_input["page_image"],
                page_number=page_input["page_number"],
                description=page_outputs[page_input_idx]["description"],
                text_blocks_response=page_outputs[page_input_idx]["text_blocks"],
                tables_response=page_outputs[page_input_idx]["tables"],
                plots_response=page_outputs[page_input_idx]["plots"],
            )
            for page_input_idx, page_input in enumerate(page_inputs)
        ]

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

        desc = DEFAULT_LLM_WRAPPER.call_llm(prompt)
        return desc

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

        stem = os.path.splitext(self.filename)[0]
        source_json_path = os.path.join(self.json_dir, f"{stem}.json")

        if source_json_path is None:
            return {page_idx: "" for page_idx in range(len(pdf))}, {}

        with open(source_json_path, encoding="utf-8") as source_json_file:
            source_json = json.load(source_json_file)

        page_blocks = {page_idx: [] for page_idx in range(len(pdf))}
        page_nums = {}

        for element in tqdm(
            source_json["document"]["elements"],
            total=len(pdf),
            desc=f"Parsing page text from JSON in {self.filename}",
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
