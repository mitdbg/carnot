from dataclasses import dataclass, field


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
    page_doc_number: int
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
        page_num: str = "",
    ):
        self.page_id = f"{document_id}_page:{page_pdf_number}"
        self.page_pdf_number = page_pdf_number
        self.page_image = page_image
        self.page_num = page_num
        if self.is_page_empty(page_text):
            return None
        self.page_doc_number = self.extract_page_number(page_text)
        self.description = self.describe_text(page_text)

        self.table_nodes = self.parse_tables(page_raw, page_text)
        self.plot_nodes = self.parse_plots(page_raw, page_text)
        self.text_nodes = self.parse_text_blocks(page_raw, page_text)

    def is_page_empty(self, text: str) -> bool:
        # TODO implement a function that checks if the page is empty based on the OCR text or on an LLM analysis of the page content, to avoid creating page nodes for empty pages
        return len(text.strip()) == 0

    def extract_page_number(self, text: str) -> int:
        # TODO implement a function that extracts the page number from the OCR text, to be used as the page_doc_number for the page node
        # This could be implemented with an LLM that looks at the OCR text and extracts a page number if present, or returns None if no page number is found
        return -1

    def describe_text(self, text: str) -> str:
        # TODO implement a function that generates a concise description of the page content based on the OCR text, to be used as the page node description
        # This could be implemented with an LLM that takes the OCR text and generates a short summary or description of the page content
        return ""

    def parse_text_blocks(self, page: bytes, page_text: str) -> list[TextNode]:
        # TODO implement a function that parses the OCR text of the page and extracts text blocks to create TextNode objects, with the text_id being a combination of the page_id and a block index, the text being the block text, and the description being a concise summary of the block content
        return []

    def parse_tables(self, page: bytes, page_text: str) -> list[TableNode]:
        # TODO implement a function that parses the OCR text of the page and extracts tables to create TableNode objects, with the table_id being a combination of the page_id and a table index, the table_data being a string representation of the table content (e.g. in CSV format), and the description being a concise summary of the table content
        return []

    def parse_plots(self, page: bytes, page_text: str) -> list[PlotNode]:
        # TODO implement a function that analyzes the raw PDF page content and extracts plots to create PlotNode objects, with the plot_id being a combination of the page_id and a plot index, the plot_data being a bytes representation of the plot image (e.g. in PNG format), and the description being a concise summary of the plot content
        return []
