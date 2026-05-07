from __future__ import annotations

import csv
import glob
import os
from dotenv import load_dotenv

import carnot
from carnot.data.dataset import Dataset
from carnot.operators.sem_topk import SemTopKOperator


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

QUERIES_CSV = f"{REPO_ROOT}/data/officeqa/officeqa_pro.csv"
DOCS_DIR = f"{REPO_ROOT}/data/officeqa/treasury_bulletins_parsed/transformed"
ENV_FILE = f"{REPO_ROOT}/.env"

load_dotenv(ENV_FILE)

ROW = 0
K = 5
INDEX = "faiss"
MAX_DOCS = None

with open(QUERIES_CSV, newline="") as queries_file:
    queries = list(csv.DictReader(queries_file))

query = queries[ROW]

gold_files = [
    source_file.strip()
    for source_file in query["source_files"].splitlines()
    if source_file.strip()
]

document_paths = sorted(glob.glob(f"{DOCS_DIR}/*.txt"))
if MAX_DOCS is not None:
    gold_paths = [f"{DOCS_DIR}/{source_file}" for source_file in gold_files]
    document_paths = list(dict.fromkeys(gold_paths + document_paths))[:MAX_DOCS]
    document_paths = [path for path in document_paths if os.path.exists(path)]

if not document_paths:
    raise ValueError(f"No .txt documents found in {DOCS_DIR}")

items = []
for document_path in document_paths:
    with open(document_path, errors="replace") as document:
        items.append(
            {
                "uri": document_path,
                "source_file": os.path.basename(document_path),
                "contents": document.read(),
            }
        )

dataset = Dataset(
    name="OfficeQA Documents",
    annotation="Parsed Treasury Bulletin documents for OfficeQA retrieval.",
    items=items,
    dataset_id="officeqa_documents",
)


# execution = carnot.Execution(
#     query = "Find documents that are necessary to answer the question: " + query["question"],
#     datasets=[dataset.name],
#     llm_config={
#         "model": "gemini/gemini-embedding-2",
#         "api_key": os.getenv("GEMINI_API_KEY"),
#     }
# )

# operator = SemTopKOperator(
#     task=query["question"],
#     k=K,
#     dataset_id="RetrievedOfficeQADocuments",
#     max_workers=1,
#     index_name=INDEX,
#     model_id='openai/text-embedding-3-large',
#     llm_config={"GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
#                 "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")},
# )

# output_datasets, stats = operator(dataset.name, {dataset.name: dataset})
# retrieved = output_datasets["RetrievedOfficeQADocuments"].items
# predicted_files = [item.get("source_file") for item in retrieved]

print(f"uid: {query['uid']}")
print(f"question: {query['question']}")
print(f"index: {INDEX}")
print(f"corpus_size: {len(items)}")
print(f"k: {K}")
print(f"gold_source_files: {gold_files}")
print(f"predicted_source_files: {predicted_files}")
print(f"hit_any: {bool(set(gold_files) & set(predicted_files))}")
print(f"items_in: {stats.items_in}")
print(f"items_out: {stats.items_out}")
