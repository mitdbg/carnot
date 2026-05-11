from dataclasses import dataclass, field

from .tools import call_openrouter, call_openrouter_vision, parse_json_response

MAX_PAGE_PROMPT_CHARS = 4000


@dataclass
class PlotNode:
    plot_id: str
    plot_title: str
    plot_data: bytes
    description: str


@dataclass
class TableNode:
    table_id: str
    table_title: str
    table_data: str
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
        page_raw: bytes,
        page_text: str,
        page_image: bytes,
        page_number: str = "",
    ):
        self.page_id = f"{document_id}_page:{page_pdf_number}"
        self.page_pdf_number = page_pdf_number
        self.page_image = page_image
        self.page_number = page_number
        self.description = self.describe_text(page_image, page_text)

        self.table_nodes = self.parse_tables(page_image, page_text)
        self.plot_nodes = self.parse_plots(page_image, page_text)
        self.text_nodes = self.parse_text_blocks(page_image, page_text)

    def describe_text(self, page_image: bytes, text: str) -> str:
        if not text.strip():
            return "Blank or image-only page with no readable OCR text."

        prompt = f"""
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

        return call_openrouter(prompt)

    def parse_text_blocks(self, page: bytes, page_text: str) -> list[TextNode]:
        if not page_text.strip():
            return []

        prompt = f"""
Split this page text into semantic text blocks for retrieval.

Rules:
- Return only valid JSON.
- Do not include markdown or explanation.
- Exclude tables, chart data, page numbers, repeated headers, and repeated footers when they are identifiable.
- Preserve the original wording of each text block.
- Keep each description to one concise sentence.
- If there are no narrative text blocks, return an empty "text_blocks" list.

JSON shape:
{{
  "text_blocks": [
    {{    
      "text": "verbatim text block from the page",
      "description": "concise semantic description of the block"
    }}
  ]
}}

Page text:
{page_text[:MAX_PAGE_PROMPT_CHARS]}
""".strip()

        parsed = parse_json_response(call_openrouter(prompt))
        return [
            TextNode(
                text_id=f"{self.page_id}_text:{block_idx}",
                text=block.get("text", ""),
                description=block.get("description", ""),
            )
            for block_idx, block in enumerate(parsed.get("text_blocks", []))
            if block.get("text", "").strip()
        ]

    def parse_tables(self, page: bytes, page_text: str) -> list[TableNode]:
        prompt = f"""
Analyze this rendered PDF page and its extracted page text. Extract any tables visible on the page.

Rules:
- Return only valid JSON.
- Do not include markdown or explanation.
- Include only real tables, not paragraphs, headings, or chart labels.
- Store table_data as CSV text when possible.
- Keep each description to one concise sentence.
- If there are no tables, return an empty "tables" list.

JSON shape:
{{
  "tables": [
    {{
      "table_title": "visible title or concise invented title",
      "table_data": "CSV representation of the table",
      "description": "concise semantic description of the table"
    }}
  ]
}}

Page text:
{page_text[:MAX_PAGE_PROMPT_CHARS]}
""".strip()

        parsed = parse_json_response(call_openrouter_vision(prompt, page))
        return [
            TableNode(
                table_id=f"{self.page_id}_table:{table_idx}",
                table_title=table.get("table_title", ""),
                table_data=table.get("table_data", ""),
                description=table.get("description", ""),
            )
            for table_idx, table in enumerate(parsed.get("tables", []))
            if table.get("table_data", "").strip()
        ]

    def parse_plots(self, page: bytes, page_text: str) -> list[PlotNode]:
        prompt = f"""
Analyze this rendered PDF page and its extracted page text. Identify any plots, charts, graphs, or figure visualizations visible on the page.

Rules:
- Return only valid JSON.
- Do not include markdown or explanation.
- Include only plots, charts, graphs, or figure visualizations, not tables or decorative images.
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

        parsed = parse_json_response(call_openrouter_vision(prompt, page))
        return [
            PlotNode(
                plot_id=f"{self.page_id}_plot:{plot_idx}",
                plot_title=plot.get("plot_title", ""),
                plot_data=page,
                description=plot.get("description", ""),
            )
            for plot_idx, plot in enumerate(parsed.get("plots", []))
            if plot.get("plot_title", "").strip() or plot.get("description", "").strip()
        ]
