from __future__ import annotations

import csv
import os
import sys
import time

from dotenv import load_dotenv
import pandas as pd

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
USE_CACHE = True

load_dotenv(ENV_FILE)

K = 5
INDEX = "faiss"
RUN_AGENT = True

with open(QUERIES_CSV, newline="") as queries_file:
    queries = list(csv.DictReader(queries_file))

input_files = [fname for fname in os.listdir(PDF_DIR)]
pdf_paths = []
for text_file in input_files:
    pdf_path = f"{PDF_DIR}/{os.path.splitext(text_file)[0]}.pdf"
    if os.path.exists(pdf_path):
        pdf_paths.append(pdf_path)

t0 = time.time()
print(f"Processing {len(pdf_paths)} PDFs...")
index = SemanticDocumentIndex(
    pdf_dir=PDF_DIR,
    ocr_text_dir=DOCS_DIR,
    render_binary=RENDER_BINARY,
    use_cache=USE_CACHE,
)
t1 = time.time()
index.add(list(reversed(pdf_paths)))
t2 = time.time()
if not index.initialized:
    index.initialize()

# retriever = Retriever(index, text_model="openrouter/google/gemini-3.1-pro-preview")
retriever = Retriever(index, text_model="openrouter/deepseek/deepseek-v4-flash")
rows = []
for query in queries:
    print(f"Running retrieval for query uid {query['uid']}")
    documents = retriever.retrieve(query["question"])
    counts = index.counts()

    gold_files = [
        file.replace("txt", "pdf") for file in query["source_files"].splitlines()
    ]

    print(f"uid: {query['uid']}")
    print(f"question: {query['question']}")
    print(f"answer: {query['answer']}")
    print(f"gold_source_files: {gold_files}")
    print(f"retrieved results")
    for doc in documents:
        print(f"Filename: {doc.filename}")
    tp = sum([1 for doc in documents if doc.filename in gold_files])
    fp = sum([1 for doc in documents if doc.filename not in gold_files])
    fn = sum(
        [1 for doc in gold_files if doc not in [doc.filename for doc in documents]]
    )
    rows.append(
        {
            "uid": query["uid"],
            "retrieved": documents,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp > 0 else 0,
            "recall": tp / (tp + fn) if tp + fn > 0 else 0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0,
        }
    )

df = pd.DataFrame(rows)
# Overall metrics
tp = df["tp"].sum()
fp = df["fp"].sum()
fn = df["fn"].sum()
p = tp / (tp + fp) if tp + fp > 0 else 0
r = tp / (tp + fn) if tp + fn > 0 else 0
f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0

print(f"Overall Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}")
# print(df)
df.to_csv(f"{REPO_ROOT}/skunk/retrieval_results.csv", index=False)
# print(f"index contents:\n{index}")
