from __future__ import annotations

import csv
import glob
import os
from dotenv import load_dotenv

import carnot
from carnot.data.dataset import Dataset
from carnot.operators.sem_topk import SemTopKOperator
from skunk.retrieval.semantic_document_index import SemanticDocumentIndex


REPO_ROOT = '/home/gerardo/carnot'

QUERIES_CSV = f"{REPO_ROOT}/skunk/officeqa_pro.csv"
DOCS_DIR = f"{REPO_ROOT}/data/officeqa/treasury_bulletins_parsed/transformed"
ENV_FILE = f"{REPO_ROOT}/.env"

OFFICEQA_PATH = f"{REPO_ROOT}/skunk/officeqa_pro.csv"
PDF_DIR = f"{REPO_ROOT}/data/officeqa/treasury_bulletin_pdfs"
RENDER_BINARY = True

load_dotenv(ENV_FILE)

ROW = 0
K = 5
INDEX = "faiss"
MAX_DOCS = None
RUN_AGENT = True
NUM_DOCUMENTS = 1

with open(QUERIES_CSV, newline="") as queries_file:
    queries = list(csv.DictReader(queries_file))

query = queries[ROW]

gold_files = [file for file in query["source_files"].splitlines()]
pdf_files = [fname for fname in os.listdir(PDF_DIR)]
 

indexed_files = pdf_files[:NUM_DOCUMENTS]
pdf_paths = []
for text_file in indexed_files:
    pdf_path = f"{PDF_DIR}/{os.path.splitext(text_file)[0]}.pdf"
    if os.path.exists(pdf_path):
        pdf_paths.append(pdf_path)

index = SemanticDocumentIndex(
    pdf_dir=PDF_DIR,
    render_binary=RENDER_BINARY,
)

document_ids = []
for pdf_path in pdf_paths:
    document_ids.append(index.add(pdf_path))

retrieved = index.retrieve(query["question"])
counts = index.counts()

print(f"uid: {query['uid']}")
print(f"question: {query['question']}")
print(f"answer: {query['answer']}")
print(f"gold_source_files: {gold_files}")
print(f"retrieved results: {retrieved}")

print(f"index contents:\n{index}")