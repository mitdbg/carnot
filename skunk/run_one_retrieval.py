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
DATA_DIR = "/home/gerarvit/orcd/scratch/officeqa/"
if not os.path.exists(DATA_DIR):
    DATA_DIR = f"{REPO_ROOT}/data"

QUERIES_CSV = f"{REPO_ROOT}/skunk/officeqa_pro.csv"
OFFICEQA_PATH = f"{REPO_ROOT}/skunk/officeqa_pro.csv"
PDF_DIR = f"{DATA_DIR}/treasury_bulletin_pdfs"
DOCS_DIR = f"{DATA_DIR}/treasury_bulletins_parsed/transformed"
JSON_DIR = f"{DATA_DIR}/treasury_bulletins_parsed/jsons"
RENDER_BINARY = True
USE_CACHE = True

load_dotenv(ENV_FILE)

RETRIEVAL_K_VALUES = [5, 10, 20, 50, 100]
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

pdf_paths = sorted(pdf_paths)
t0 = time.time()
index = SemanticDocumentIndex(
    pdf_dir=PDF_DIR,
    ocr_text_dir=DOCS_DIR,
    json_dir=JSON_DIR,
    render_binary=RENDER_BINARY,
    use_cache=USE_CACHE,
)
t1 = time.time()
print(f"Index creation time: {t1 - t0:.2f} seconds")
index.add(list(reversed(pdf_paths)))
t2 = time.time()
print(f"Documents add time: {t2 - t1:.2f} seconds")
if not index.initialized:
    index.initialize()
    t3 = time.time()
    print(f"Index initialization time: {t3 - t2:.2f} seconds")

# retriever = Retriever(index, text_model="openrouter/google/gemini-3.1-pro-preview")
retriever = Retriever(
    index,
    text_model="vertex_ai/gemini-embedding-001",
    final_documents_k=max(RETRIEVAL_K_VALUES),
)
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

    for k in RETRIEVAL_K_VALUES:
        documents_at_k = documents[:k]
        retrieved_files_at_k = [doc.filename for doc in documents_at_k]
        tp = sum([1 for filename in retrieved_files_at_k if filename in gold_files])
        fp = sum([1 for filename in retrieved_files_at_k if filename not in gold_files])
        fn = sum(
            [1 for filename in gold_files if filename not in retrieved_files_at_k]
        )
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0
        print(
            f"k={k}: Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}"
        )
        rows.append(
            {
                "uid": query["uid"],
                "k": k,
                "retrieved_files": retrieved_files_at_k,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

df = pd.DataFrame(rows)
summary_rows = []
for k in RETRIEVAL_K_VALUES:
    df_at_k = df[df["k"] == k]
    tp = df_at_k["tp"].sum()
    fp = df_at_k["fp"].sum()
    fn = df_at_k["fn"].sum()
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0
    print(f"Overall k={k} Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}")
    summary_rows.append(
        {
            "k": k,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
        }
    )
# print(df)
df.to_csv(f"{REPO_ROOT}/skunk/retrieval_results.csv", index=False)
pd.DataFrame(summary_rows).to_csv(
    f"{REPO_ROOT}/skunk/retrieval_results_summary.csv", index=False
)
# print(f"index contents:\n{index}")
