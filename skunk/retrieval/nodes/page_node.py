import time
from dataclasses import dataclass, field
from io import StringIO
from typing import Dict, List

import pandas as pd
import json
from .llm_wrapper import get_llm_wrapper, parse_json_response

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
    text_nodes: Dict[str, TextNode] = field(default_factory=dict)
    table_nodes: Dict[str, TableNode] = field(default_factory=dict)
    plot_nodes: Dict[str, PlotNode] = field(default_factory=dict)
    description: str = ""

    def __init__(
        self,
        document_id: str,
        page_pdf_number: int,
        page_elements: list[dict],
        page_image: bytes | None,
        use_cache: bool = True,
    ):
        self.page_id = f"{document_id}_page:{page_pdf_number}"
        self.page_pdf_number = page_pdf_number
        self.use_cache = use_cache
        # TODO: KEEP IT IN MEMORY? NAH
        # self.page_image = page_image
        self.page_elements = page_elements

        page_text = ""
        candidate_nums = []
        metadata = []
        tables = []
        figures = False
        text_blocks = []
        for element in self.page_elements:
            content = element.get("content")
            if not content:
                continue
            if isinstance(content, list):
                content_text = "\n\n".join(
                    str(part) for part in content if part is not None
                )
            else:
                content_text = str(content)

            page_text += f"\n\n {content_text}"
            if element.get("type") == "page_number":
                candidate_nums.append(content_text.strip())
            elif element.get("type") in [
                "page_header",
                "page_footer",
                "section_header",
                "section_footer",
            ]:
                metadata.append(content_text.strip())
            elif element.get("type") == "table":
                tables.append(content_text.strip())
            elif element.get("type") == "figure":
                figures = True
            else:
                text_blocks.append(content_text.strip())

        self.page_text = page_text
        self.page_number = (
            candidate_nums[0] if candidate_nums else str(page_pdf_number + 1)
        )
        self.text_nodes = {}
        self.table_nodes = {}
        self.plot_nodes = {}

        start_time = time.time()
        llm_wrapper = get_llm_wrapper()
        text_prompt = PageNode.description_prompt(page_text)
        try:
            response = llm_wrapper.call_llm(text_prompt, use_cache=self.use_cache)
        except Exception as e:
            raise RuntimeError(f"{type(e).__name__}: {e}") from None
        self.description = response
        t1 = time.time()
        # print(f"PageNode description generated in {t1 - start_time:.2f} seconds")

        if len(text_blocks) > 0:
            prompts = PageNode.text_blocks_prompt(text_blocks)
            try:
                response = llm_wrapper.call_llm(prompts, use_cache=self.use_cache)
            except Exception as e:
                raise RuntimeError(f"{type(e).__name__}: {e}") from None
            parsed = parse_json_response(response)
            t2 = time.time()
            # print(
            #     f"PageNode text block descriptions generated in {t2 - t1:.2f} seconds"
            # )
            if isinstance(parsed, list):
                parsed = {"text_blocks": parsed}
            for idx, text_block in enumerate(parsed.get("text_blocks", [])):
                block_idx = int(text_block.get("block_idx", idx))
                if block_idx >= len(text_blocks):
                    print(
                        f"block_idx {block_idx} out of range for text_blocks with length {len(text_blocks)}"
                    )
                    block_idx = -1

                description = text_block.get("description", "")
                text_id = f"{self.page_id}_text:{block_idx}"
                self.text_nodes[text_id] = TextNode(
                    text_id=text_id,
                    text=text_blocks[block_idx],
                    description=description,
                )

        t3 = time.time()
        if len(tables) > 0:
            prompt = PageNode.tables_prompt(tables)
            try:
                response = llm_wrapper.call_llm(prompt, use_cache=self.use_cache)
            except Exception as e:
                raise RuntimeError(f"{type(e).__name__}: {e}") from None
            parsed = parse_json_response(response)
            t4 = time.time()
            # print(f"PageNode table descriptions generated in {t4 - t3:.2f} seconds")
            if isinstance(parsed, list):
                parsed = {"tables": parsed}

            for table_info in parsed.get("tables", []):
                title = table_info.get("table_title", "")
                table_idx = int(table_info.get("table_idx", -1))
                description = table_info.get("description", "")
                date_start = table_info.get("date_start", "")
                date_end = table_info.get("date_end", "")
                table_id = f"{self.page_id}_table:{table_idx}"
                self.table_nodes[table_id] = TableNode(
                        table_id=table_id,
                        table_title=title,
                        table_data=tables[table_idx],
                        table_date_start=date_start,
                        table_date_end=date_end,
                        description=description,
                    )
                

        if figures:
            t5 = time.time()
            try:
                visual_responses = llm_wrapper.call_llm_vision(
                    PageNode.plots_prompt(page_text, page_image),
                    page_image,
                    use_cache=self.use_cache,
                )
            except Exception as e:
                raise RuntimeError(f"{type(e).__name__}: {e}") from None
            parsed = parse_json_response(visual_responses)
            t6 = time.time()
            # print(f"PageNode plot descriptions generated in {t6 - t5:.2f} seconds")
            for plot_idx, plot in enumerate(parsed.get("plots", [])):
                plot_id = f"{self.page_id}_plot:{plot_idx}"
                self.plot_nodes[plot_id] = PlotNode(
                    plot_id=plot_id,
                    plot_title=plot.get("plot_title", ""),
                    description=plot.get("description", ""),
                )
        end_time = time.time()
        tot_time = end_time - start_time
        # print(f"Total PageNode processing time: {tot_time:.2f} seconds")
        # if tot_time > 30:
        # breakpoint()

    @staticmethod
    def description_prompt(text: str) -> str:
        return f"""
Write a concise semantic index description for this page.

Rules:
- Return exactly one sentence.
- Maximum 180 characters.
- Describe the page's subject matter and the kinds of information it contains.
- Do not include labels, quotes, markdown, or explanation.
- Do not mention OCR or that you are reading extracted text.

Page text:
{text[:MAX_PAGE_PROMPT_CHARS]}
""".strip()

    @staticmethod
    def text_blocks_prompt(text_blocks: List[str]) -> str:
        texts = json.dumps(
            [{"block_idx": idx, "text": block} for idx, block in enumerate(text_blocks)]
        )
        return f"""
You are given a list of snippets of page text from a document page.
Your task is to provide a concise semantic description for each text snippet, describing the subject matter and kinds of information contained in that text snippet.
Write a concise description with important keyword that can serve to retrieve this text block in a semantic index.
The output should be a JSON list of descriptions, one for each text block, in the same order as the input text blocks.
Output format:
json
{{
    'text_blocks':[{{
        'block_idx': 'index of the text block in the input list',
        'description': 'concise semantic description of the text block'
    }}]
}}
Rules:
- Do not include markdown or explanation.
- Maximum 180 characters.
- Describe the text subject matter and the kinds of information it contains.
- Do not include labels, quotes, markdown, or explanation.
- Do not mention OCR or that you are reading extracted text.
Text blocks:
{texts}
""".strip()

    @staticmethod
    def tables_prompt(tables: List[str]) -> str:
        inputs = []
        for table_string in tables:
            try:
                table = pd.read_html(StringIO(table_string))[0]
            except ValueError as e:
                continue
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

            inputs.append(table.to_markdown(index=False))

        str_input = json.dumps(
            [{"table_idx": idx, "table": table} for idx, table in enumerate(inputs)]
        )
        return f"""
Your task is to analyze the following table snippets extracted from a PDF page. You are only getting samples of these tables.
The samples contains the table headers and a few rows: some from the top of the table, some from the middle, and some from the end. Use these samples to understand what each table is about and what kinds of information it contains.
If the table contains date information, extract the earliest and latest dates mentioned in the table, preserving useful precision such as "June 2018" if no day appears or "2018" if no month is available. If no dates are visible, return empty strings for the date fields.

Rules:
- Return only valid JSON.
- Do not include markdown or explanation.
- Keep each description to one concise sentence.

JSON shape:
{{
    'tables':[{{
    "table_idx": "index of the table in the input list",
    "table_title": "visible title or concise summary title",
    "date_start": "earliest date mentioned in the table, or empty string if no dates",
    "date_end": "latest date mentioned in the table, or empty string if no dates"
    "description": "concise semantic description of the table"}}]
}}

Table samples:
{str_input}
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
