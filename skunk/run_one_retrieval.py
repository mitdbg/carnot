from __future__ import annotations

import csv
import os
import sys

from dotenv import load_dotenv

REPO_ROOT = os.getenv("HOME") + "/carnot"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skunk.retrieval.retriever import Retriever
from skunk.retrieval.semantic_document_index import SemanticDocumentIndex

ENV_FILE = f"{REPO_ROOT}/.env"
DATASET = "officeqa"

QUERIES_CSV = f"{REPO_ROOT}/skunk/officeqa_pro.csv"
OFFICEQA_PATH = f"{REPO_ROOT}/skunk/officeqa_pro.csv"
PDF_DIR = f"{REPO_ROOT}/data/{DATASET}/treasury_bulletin_pdfs"
DOCS_DIR = f"{REPO_ROOT}/data/{DATASET}/treasury_bulletins_parsed/transformed"
RENDER_BINARY = True

load_dotenv(ENV_FILE)

K = 5
INDEX = "faiss"
RUN_AGENT = True

with open(QUERIES_CSV, newline="") as queries_file:
    queries = list(csv.DictReader(queries_file))

# 'What were the total expenditures (in millions of nominal dollars) for U.S national defense in the calendar year of 1940?'
# CALENDAR is important here, as the question is asking for the calendar year, not fiscal year.
# Source doc: treasury_bulletin_1941_01.pdf Page 15, doc page 5
# Answer = 2602
query = queries[0]
query = [q for q in queries if q["uid"] == "UID0031"][0]
print(query)

gold_files = [file for file in query["source_files"].splitlines()]
pdf_files = [fname for fname in os.listdir(PDF_DIR)]

input_files = ['treasury_bulletin_1941_01.pdf', 
                 'treasury_bulletin_1966_01.pdf', 
                 'treasury_bulletin_1942_01.pdf', 
                 'treasury_bulletin_2010_03.pdf', 
                 'treasury_bulletin_2021_12.pdf', 
                 ]

cutoff_years = (1987, 1994)
# take any document which has a date in the filename within the cutoff years, this is just to test that retrieval is working and not just memorizing the answer from the correct document
doc_years = [int(fname.split("_")[2].split(".")[0]) for fname in pdf_files]
input_files = [
    fname
    for fname, year in zip(pdf_files, doc_years)
    if cutoff_years[0] <= year <= cutoff_years[1]
]
print(f"input_files from {cutoff_years[0]} to {cutoff_years[1]}: {len(input_files)}")

input_files = sorted(input_files)
# indexed_files = pdf_files[:NUM_DOCUMENTS]
pdf_paths = []
for text_file in input_files:
    pdf_path = f"{PDF_DIR}/{os.path.splitext(text_file)[0]}.pdf"
    if os.path.exists(pdf_path):
        pdf_paths.append(pdf_path)

index = SemanticDocumentIndex(
    pdf_dir=PDF_DIR,
    ocr_text_dir=DOCS_DIR,
    render_binary=RENDER_BINARY,
)

document_ids = []
for pdf_path in pdf_paths:
    document_ids.append(index.add(pdf_path))

index.initialize()
retriever = Retriever(index)
documents = retriever.retrieve(query["question"])
counts = index.counts()

print(f"uid: {query['uid']}")
print(f"question: {query['question']}")
print(f"answer: {query['answer']}")
print(f"gold_source_files: {gold_files}")
print(f"retrieved results")
for doc in documents:
    print(f"Filename: {doc.filename}")
# print(f"index contents:\n{index}")
