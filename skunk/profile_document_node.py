import json
import statistics
import sys
import time

import fitz

from skunk.retrieval.nodes.document_node import DocumentNode

DEFAULT_PDF_PATH = "data/officeqa/treasury_bulletin_pdfs/treasury_bulletin_1987_09.pdf"
PAGE_LIMIT = None
SLOWEST_PAGE_COUNT = 3

pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_PATH

with open(pdf_path, "rb") as pdf_file:
    pdf_bytes = pdf_file.read()

if PAGE_LIMIT:
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    pdf.delete_pages(range(PAGE_LIMIT, len(pdf)))
    pdf_bytes = pdf.tobytes()
    pdf.close()

document_start = time.perf_counter()
document_node = DocumentNode(pdf_path, pdf_bytes)
document_seconds = time.perf_counter() - document_start

page_nodes = document_node.page_nodes
page_times = [page_node.processing_seconds for page_node in page_nodes]
textual_times = [page_node.textual_processing_seconds for page_node in page_nodes]
visual_times = [page_node.visual_processing_seconds for page_node in page_nodes]


def timing_stats(values):
    if not values:
        return {
            "count": 0,
            "min_seconds": 0.0,
            "mean_seconds": 0.0,
            "median_seconds": 0.0,
            "max_seconds": 0.0,
        }

    return {
        "count": len(values),
        "min_seconds": min(values),
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "max_seconds": max(values),
    }


slowest_pages = sorted(
    [
        {
            "page_pdf_number": page_node.page_pdf_number,
            "processing_seconds": page_node.processing_seconds,
            "textual_processing_seconds": page_node.textual_processing_seconds,
            "visual_processing_seconds": page_node.visual_processing_seconds,
            "text_nodes": len(page_node.text_nodes),
            "table_nodes": len(page_node.table_nodes),
            "plot_nodes": len(page_node.plot_nodes),
        }
        for page_node in page_nodes
    ],
    key=lambda page: page["processing_seconds"],
    reverse=True,
)[:SLOWEST_PAGE_COUNT]

profile = {
    "document": {
        "pdf_path": pdf_path,
        "profiled_page_limit": PAGE_LIMIT,
        "document_id": document_node.document_id,
        "filename": document_node.filename,
        "total_pdf_pages": document_node.num_pdf_pages,
        "processed_pages": len(page_nodes),
        "processed_page_fraction": len(page_nodes) / document_node.num_pdf_pages
        if document_node.num_pdf_pages
        else 0.0,
        "document_seconds": document_seconds,
    },
    "page_processing": timing_stats(page_times),
    "textual_processing": timing_stats(textual_times),
    "visual_processing": timing_stats(visual_times),
    "visual_pages": {
        "processed_pages": len(visual_times),
        "total_pdf_pages": document_node.num_pdf_pages,
        "processed_page_fraction": len(visual_times) / document_node.num_pdf_pages
        if document_node.num_pdf_pages
        else 0.0,
    },
    "slowest_pages": slowest_pages,
}

print(json.dumps(profile, indent=2))
with open("profile_document_node.json", "w") as profile_file:
    json.dump(profile, profile_file, indent=2)
