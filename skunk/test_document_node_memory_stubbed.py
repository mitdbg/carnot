import os
import resource

os.environ["SKUNK_LLM_MODEL"] = "dummy"
os.environ["SKUNK_LLM_VISION_MODEL"] = "dummy"

from skunk.retrieval.nodes.document_node import DocumentNode

DATA_DIR = "/home/gerarvit/orcd/scratch/officeqa/"
PDF_PATH = f"{DATA_DIR}/treasury_bulletin_pdfs/treasury_bulletin_1954_04.pdf"
OCR_TEXT_DIR = f"{DATA_DIR}/treasury_bulletins_parsed/transformed"
JSON_DIR = f"{DATA_DIR}/treasury_bulletins_parsed/jsons"
PAGE_PROCESS_WORKERS = 1


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


print(f"Loading PDF: {PDF_PATH}")
print(f"RSS before reading PDF: {rss_mb():.2f} MB")

with open(PDF_PATH, "rb") as pdf_file:
    pdf_bytes = pdf_file.read()

print(f"PDF bytes: {len(pdf_bytes) / 1024 / 1024:.2f} MB")
print(f"RSS before DocumentNode: {rss_mb():.2f} MB")

document_node = DocumentNode(
    filename=PDF_PATH,
    pdf_bytes=pdf_bytes,
    ocr_text_dir=OCR_TEXT_DIR,
    json_dir=JSON_DIR,
    page_process_workers=PAGE_PROCESS_WORKERS,
    use_cache=False,
)

print(f"RSS after DocumentNode: {rss_mb():.2f} MB")
print(f"Document ID: {document_node.document_id}")
print(f"PDF pages: {document_node.num_pdf_pages}")
print(f"Page nodes: {len(document_node.page_nodes)}")

for page_id, page_node in sorted(
    document_node.page_nodes.items(), key=lambda item: item[1].page_pdf_number
):
    print(
        f"{page_id}: "
        f"text={len(page_node.text_nodes)} "
        f"tables={len(page_node.table_nodes)} "
        f"plots={len(page_node.plot_nodes)} "
        f"rss={rss_mb():.2f} MB"
    )
