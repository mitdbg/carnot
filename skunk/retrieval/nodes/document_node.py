from dataclasses import dataclass, field
import hashlib
import json
import os
import re
from typing import Dict, Tuple

import fitz
import pymupdf
from pymupdf.extra import page_count
from .page_node import PageNode
from .tools import call_openrouter, call_openrouter_vision

DEFAULT_OCR_TEXT_DIR = "data/officeqa/treasury_bulletins_parsed/transformed"
MAX_DOCUMENT_PROMPT_CHARS = 2000
OCR_MATCH_CHARS_PER_CHUNK = 4000

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

    def __init__(self, filename: str, content: bytes):

        sha256 = hashlib.sha256(content).hexdigest()
        self.document_id = f"doc:{sha256[:16]}"
        pdf_name = os.path.basename(filename)
        self.filename = pdf_name

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
        self.description=self.describe_text(ocr_text)

        self.page_nodes = []
        page_texts, page_nums = self.match_page_text(pdf, ocr_text)
        breakpoint()
        for page_idx, page in enumerate(pdf):
            page_image = page.get_pixmap(dpi=300).tobytes()  # render page to image bytes
            page_text = page_texts[page_idx]
            page_node = PageNode(document_id=self.document_id, page_pdf_number=page_idx, page_raw=page, page_text=page_text, page_image=page_image, page_num=page_nums.get(page_idx, ""))
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
    

    def match_page_text(self, pdf: pymupdf.Document, doc_text: str) -> Tuple[Dict[int, str], Dict[int, str]]:
        if not doc_text.strip():
            return {page_idx: "" for page_idx in range(len(pdf))}, {}

        page_texts = {}
        page_nums = {}
        last_matched_index = 0
        for page_idx, page in enumerate(pdf):
            blank_page_image = page.get_pixmap(dpi=35, alpha=False).tobytes("png")
            blank_prompt = """
The image is a very low-resolution render of a PDF page.

Determine whether the page is blank.

Rules:
- Return only valid JSON.
- Do not include markdown or explanation.
- A page is blank only if it has no meaningful visible content.
- Very faint scanner noise, small specks, or page shadows do not count as meaningful content.
- Text, tables, plots, page numbers, headers, footers, stamps, or images mean the page is not blank.

JSON shape:
{
  "blank": true
}
""".strip()

            raw_blank_match = call_openrouter_vision(blank_prompt, blank_page_image, model='google/gemini-2.5-flash-lite')
            raw_blank_match = raw_blank_match.strip()
            if raw_blank_match.startswith("```"):
                raw_blank_match = re.sub(r"^```(?:json)?\s*", "", raw_blank_match)
                raw_blank_match = re.sub(r"\s*```$", "", raw_blank_match)
            try:
                blank_match = json.loads(raw_blank_match)
            except json.JSONDecodeError:
                breakpoint()
                blank_match = {"blank": False}
            if blank_match["blank"]:
                page_texts[page_idx] = ""
                continue

            old_index = last_matched_index
            search_end = min(len(doc_text), old_index + OCR_MATCH_CHARS_PER_CHUNK)
            ocr_chunk = doc_text[old_index:search_end]

            page_image = page.get_pixmap(dpi=120, alpha=False).tobytes("png")
            prompt = f"""
The image is a rendered page from a PDF. The OCR chunk below starts exactly where the previous page match ended.

Find and match this page's content ends inside the OCR chunk.

Rules:
- Match the visible page image to the right OCR chunk text.
- Return only valid JSON.
- Do not include markdown or explanation.
- The matched page text will be OCR chunk characters that match the characters in the OCR text.
- Store the matched page text in the "page_text" field of the JSON.
- Confidence should be "high" if you are sure the match is correct, and "low" if you are uncertain.
- Detect the page number if it is visible and include it in the description, but do not rely on it for matching.

JSON shape:
{{
  "page_text": "the exact text from the OCR chunk that matches the page content",
  "page_number": str,
  "confidence": "high"
}}

OCR chunk:
{ocr_chunk}
""".strip()

            raw_match = call_openrouter_vision(prompt, page_image)
            raw_match = raw_match.strip()
            if raw_match.startswith("```"):
                raw_match = re.sub(r"^```(?:json)?\s*", "", raw_match)
                raw_match = re.sub(r"\s*```$", "", raw_match)

            match = json.loads(raw_match)
            if match.get("empty"):
                page_texts[page_idx] = ""
                continue

            page_texts[page_idx] = match.get("page_text", "")
            page_nums[page_idx] = match.get("page_number", "")
            # Find the next match starting point by looking for the matched text in the OCR chunk, and if not found, just move forward by the max chars per chunk
            if page_texts[page_idx]:
                found_index = ocr_chunk.find(page_texts[page_idx])
                if found_index != -1:
                    last_matched_index = old_index + found_index + len(page_texts[page_idx])
                else:
                    last_matched_index = max(last_matched_index, old_index + len(page_texts[page_idx]))
            else:
                last_matched_index = max(last_matched_index, old_index + len(page_texts[page_idx]))
        return page_texts, page_nums
