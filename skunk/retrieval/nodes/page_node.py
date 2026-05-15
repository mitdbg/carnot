from typing import Dict, List
from dataclasses import dataclass, field

from .llm_wrapper import parse_json_response
from .llm_wrapper import DEFAULT_LLM_WRAPPER

import pandas as pd
import pymupdf
import numbers
from io import StringIO
from markdownify import markdownify
from tqdm import tqdm

MAX_PAGE_PROMPT_CHARS = 2000
TABLE_FRACTION = 0.15


@dataclass
class PlotNode:
    plot_id: str
    plot_title: str
    # plot_data: bytes
    description: str


@dataclass
class TableNode:
    table_id: str
    table_title: str
    table_data: str
    table_date_start: str
    table_date_end: str
    description: str


@dataclass
class TextNode:
    text_id: str
    text: str
    description: str


@dataclass
class PageNode:
    page_id: str
    page_pdf_number: int
    page_number: str
    text_nodes: list[TextNode] = field(default_factory=list)
    table_nodes: list[TableNode] = field(default_factory=list)
    plot_nodes: list[PlotNode] = field(default_factory=list)
    description: str = ""

    def __init__(
        self,
        document_id: str,
        page_pdf_number: int,
        page_elements: List[Dict],
        page_image: bytes,
    ):
        self.page_id = f"{document_id}_page:{page_pdf_number}"
        self.page_pdf_number = page_pdf_number
        # TODO: KEEP IT IN MEMORY? NAH
        # self.page_image = page_image
        self.page_elements = page_elements

        page_text = ""
        candidate_nums = []
        metadata = []
        tables = []
        plots = []
        text_blocks = []
        for element in self.page_elements:
            content = element.get("content")
            if not content:
                continue

            page_text += "\n\n".join(content) + "\n\n"
            if element.get("type") == "page_number":
                candidate_nums.append(content.strip())
            elif element.get("type") in [
                "page_header",
                "page_footer",
                "section_header",
                "section_footer",
            ]:
                metadata.append(content.strip())
            elif element.get("type") == "table":
                tables.append(content.strip())
            else:
                text_blocks.append(content.strip())

        self.page_text = page_text 
        self.text_nodes = []
        self.table_nodes = []
        self.plot_nodes = []

        # Batching all prompts together
        textual_prompts = []
        textual_prompts.extend([PageNode.description_prompt(page_text, metadata)])
        textual_prompts.extend([PageNode.text_blocks_prompt(t) for t in text_blocks])
        textual_prompts.extend([PageNode.tables_prompt(t) for t in tables])

        responses = DEFAULT_LLM_WRAPPER.batch_call_llm(
            textual_prompts,
            desc=f"Processing textual prompts for page {self.page_pdf_number}",
        )

        self.description = responses[0]

        for idx, text_block in enumerate(text_blocks):
            desc = responses[idx+1]
            if not desc.strip():
                desc= "Blank or image-only page with no readable OCR text."

            self.text_nodes.append(TextNode(
                text_id=f"{self.page_id}_text:{len(self.text_nodes)}",
                text=text_block,
                description=desc
            ))

        for idx, table in enumerate(tables):
            response = responses[1 + len(text_blocks) + idx]
            parsed = parse_json_response(response)
            title = parsed.get("table_title", "")
            description = parsed['description']
            date_start = parsed['date_start']
            date_end = parsed['date_end']
            self.table_nodes.append(TableNode(
                table_id=f"{self.page_id}_table:{len(self.table_nodes)}",
                table_title=title,
                table_data=table,
                table_date_start=date_start,
                table_date_end=date_end,
                description=description
            ))
            
        visual_responses = DEFAULT_LLM_WRAPPER.call_llm_vision(
            PageNode.plots_prompt(page_text, page_image),
            page_image,
        )
        parsed = parse_json_response(visual_responses)
        for plot_idx, plot in enumerate(parsed.get("plots", [])):
            self.plot_nodes.append(PlotNode(
                plot_id=f"{self.page_id}_plot:{plot_idx}",
                plot_title=plot.get("plot_title", ""),
                description=plot.get("description", ""),
            ))


    @staticmethod
    def description_prompt(text: str, metadata: list) -> str:
        return f"""
Write a concise semantic index description for this page.

Rules:
- Return exactly one sentence.
- Maximum 180 characters.
- Describe the page's subject matter and the kinds of information it contains.
- Do not include labels, quotes, markdown, or explanation.
- Do not mention OCR or that you are reading extracted text.

Page metadata:
{"\n".join(metadata[:MAX_PAGE_PROMPT_CHARS])}
Page text:
{text[:MAX_PAGE_PROMPT_CHARS]}
""".strip()

    @staticmethod
    def text_blocks_prompt(text: str) -> str:
        return f"""
You are given a snippet of page text from a document page.
Write a concise description with important keyword that can serve to retrieve this text block in a semantic index.

Rules:
- Do not include markdown or explanation.
- Maximum 180 characters.
- Describe the text subject matter and the kinds of information it contains.
- Do not include labels, quotes, markdown, or explanation.
- Do not mention OCR or that you are reading extracted text.
Text block:
{text[:MAX_PAGE_PROMPT_CHARS]}
""".strip()

    @staticmethod
    def tables_prompt(table_string: str) -> str:
        table = pd.read_html(StringIO(table_string))[0]
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = [
                " ".join(str(part) for part in column if not pd.isna(part)).strip()
                for column in table.columns
            ]

        n_rows = len(table)
        sample_size = min(n_rows, max(3, int(n_rows * TABLE_FRACTION)))
        if n_rows > sample_size:
            edge_size = max(1, sample_size // 3)
            middle_size = max(1, sample_size - (2 * edge_size))
            middle_start = max(edge_size, (n_rows - middle_size) // 2)
            row_indices = (
                list(range(edge_size))
                + list(range(middle_start, min(n_rows, middle_start + middle_size)))
                + list(range(max(0, n_rows - edge_size), n_rows))
            )
            table = table.iloc[list(dict.fromkeys(row_indices))]

        table_rows = table.to_markdown(index=False)

        return f"""
Analyze this table extracted from a PDF page. You are only getting a sample of this table.
The sample contains the table headers and a few rows: some from the top of the table, some from the middle, and some from the end. Use this sample to understand what this table is about and what kinds of information it contains.
If the table contains date information, extract the earliest and latest dates mentioned in the table, preserving useful precision such as "June 2018" if no day appears or "2018" if no month is available. If no dates are visible, return empty strings for the date fields.

Rules:
- Return only valid JSON.
- Do not include markdown or explanation.
- Keep each description to one concise sentence.

JSON shape:
{{
    "table_title": "visible title or concise summary title",
    "date_start": "earliest date mentioned in the table, or empty string if no dates",
    "date_end": "latest date mentioned in the table, or empty string if no dates"
    "description": "concise semantic description of the table"
}}

Table sample:
{table_rows}
""".strip()

    @staticmethod
    def plots_prompt(page_text: str, page_image: bytes) -> str:
        return f"""
        Analyze this rendered PDF page and its extracted page text. Identify any plots, charts, graphs, or figure visualizations visible on the page.

        Rules:
        - Return only valid JSON.
        - Do not include markdown or explanation.
        - Consider only plots, charts, graphs, or figure visualizations, not tables or decorative images.
        - Keep each description to one concise sentence.
        - If there are no plots, return an empty "plots" list.

        JSON shape:
        {{
        "plots": [
            {{
            "plot_title": "visible title or concise invented title",
            "description": "concise semantic description of the plot"
            }}
        ]
        }}

Page text:
{page_text[:MAX_PAGE_PROMPT_CHARS]}
""".strip()
