"""Measure page- and document-level recall of a pure vector-search baseline
over the officeqa_pro questions across four ChromaDB collections
(qwen, qwen-strip-years, gemini, gemini-strip-years) and several values of k.

For each (collection, question) we embed the question with the appropriate
embedding model, retrieve the top max(K) results, and then compute -- for
each k in K -- the fraction of ground-truth pages / documents that appear
in the top-k results.  Recalls are averaged over all questions.

Run:
    python exp_vector_db_recall.py --output recall_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
import pandas as pd

from skunk.search_agent.prep.officeqa_eval import (
    metadata_to_doc_key,
    metadata_to_page_key,
    source_docs_to_page_keys,
    source_files_to_doc_keys,
)

K_VALUES = [1, 10, 100, 1000]
COLLECTIONS = ["qwen", "qwen-strip-years", "gemini", "gemini-strip-years"]

GEMINI_MODEL_ID = "gemini-embedding-2"
QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-8B"


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------
class GeminiEmbedder:
    def __init__(self, model_id: str = GEMINI_MODEL_ID, max_workers: int = 16):
        from google import genai  # local import so qwen-only runs don't need it

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_id = model_id
        self.max_workers = max_workers

    def embed(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types as genai_types

        def _one(t: str) -> list[float]:
            cfg = genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            res = self.client.models.embed_content(
                model=self.model_id, contents=t, config=cfg
            )
            return list(res.embeddings[0].values)  # type: ignore

        out: list[list[float] | None] = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = {pool.submit(_one, t): i for i, t in enumerate(texts)}
            for fut in as_completed(futs):
                out[futs[fut]] = fut.result()
        return out  # type: ignore[return-value]


class QwenEmbedder:
    def __init__(self, model_id: str = QWEN_MODEL_ID):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id)
        self.model_id = model_id

    def embed(self, texts: list[str]) -> list[list[float]]:
        # sentence-transformers Qwen3-Embedding supports a "query" prompt.
        kwargs: dict = {"normalize_embeddings": True, "show_progress_bar": True}
        try:
            embs = self.model.encode(texts, prompt_name="query", **kwargs)
        except (ValueError, KeyError):
            embs = self.model.encode(texts, **kwargs)
        return [list(map(float, row)) for row in embs]


def make_embedder(collection_name: str):
    if collection_name.startswith("gemini"):
        return GeminiEmbedder()
    if collection_name.startswith("qwen"):
        return QwenEmbedder()
    raise ValueError(f"Unknown collection: {collection_name}")


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------
def recall_at_k(retrieved: list[str], ground_truth: set[str], k: int) -> float | None:
    if not ground_truth:
        return None
    top = retrieved[:k]
    # Deduplicate while preserving order (we only care about set intersection).
    return len(set(top) & ground_truth) / len(ground_truth)


def evaluate_collection(
    collection_name: str,
    questions: list[str],
    gt_page_keys: list[list[str]],
    gt_doc_keys: list[list[str]],
    chroma_path: str,
    max_k: int,
) -> dict:
    print(f"\n=== Collection: {collection_name} ===", flush=True)
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(collection_name)

    embedder = make_embedder(collection_name)
    print(f"Embedding {len(questions)} questions with {embedder.model_id}...", flush=True)
    t0 = time.perf_counter()
    query_embeddings = embedder.embed(questions)
    print(f"  embedding time: {time.perf_counter() - t0:.1f}s", flush=True)

    # Aggregate per-k recall (only over questions with non-empty ground truth).
    page_recall_sums = {k: 0.0 for k in K_VALUES}
    page_recall_counts = {k: 0 for k in K_VALUES}
    doc_recall_sums = {k: 0.0 for k in K_VALUES}
    doc_recall_counts = {k: 0 for k in K_VALUES}

    per_question: list[dict] = []

    print(f"Querying collection (n_results={max_k})...", flush=True)
    t0 = time.perf_counter()
    for i, (qemb, gp, gd) in enumerate(zip(query_embeddings, gt_page_keys, gt_doc_keys, strict=True)):
        try:
            res = collection.query(
                query_embeddings=[qemb],
                n_results=max_k,
                include=["metadatas"],
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] query {i} failed: {e}", flush=True)
            continue
        metas = res["metadatas"][0]  # type: ignore[index]
        retrieved_pages = [metadata_to_page_key(m) for m in metas]  # type: ignore[assignment]
        retrieved_docs = [metadata_to_doc_key(m) for m in metas]  # type: ignore[assignment]

        gp_set = set(gp)
        gd_set = set(gd)
        q_record: dict = {"question_idx": i, "page": {}, "doc": {}}
        for k in K_VALUES:
            pr = recall_at_k(retrieved_pages, gp_set, k)
            dr = recall_at_k(retrieved_docs, gd_set, k)
            if pr is not None:
                page_recall_sums[k] += pr
                page_recall_counts[k] += 1
                q_record["page"][str(k)] = pr
            if dr is not None:
                doc_recall_sums[k] += dr
                doc_recall_counts[k] += 1
                q_record["doc"][str(k)] = dr
        per_question.append(q_record)

        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(questions)} questions", flush=True)
    print(f"  query time: {time.perf_counter() - t0:.1f}s", flush=True)

    summary = {
        "page_recall": {
            str(k): (page_recall_sums[k] / page_recall_counts[k]) if page_recall_counts[k] else None
            for k in K_VALUES
        },
        "doc_recall": {
            str(k): (doc_recall_sums[k] / doc_recall_counts[k]) if doc_recall_counts[k] else None
            for k in K_VALUES
        },
        "page_n": {str(k): page_recall_counts[k] for k in K_VALUES},
        "doc_n": {str(k): doc_recall_counts[k] for k in K_VALUES},
    }
    print(f"  page recall: {summary['page_recall']}", flush=True)
    print(f"  doc  recall: {summary['doc_recall']}", flush=True)
    return {"summary": summary, "per_question": per_question}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="officeqa_pro.csv")
    parser.add_argument("--chroma-path", default=".chromadb")
    parser.add_argument(
        "--collections",
        nargs="+",
        default=COLLECTIONS,
        help=f"Subset of collections to evaluate (default: {COLLECTIONS})",
    )
    parser.add_argument("--output", default="exp_vector_db_recall_results.json")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only evaluate the first N questions (for debugging).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.limit is not None:
        df = df.head(args.limit)
    print(f"Loaded {len(df)} questions from {args.csv}", flush=True)

    questions = df["question"].astype(str).tolist()
    gt_page_keys = [source_docs_to_page_keys(s) for s in df["source_docs"]]
    gt_doc_keys = [source_files_to_doc_keys(s) for s in df["source_files"]]

    n_with_pages = sum(1 for k in gt_page_keys if k)
    n_with_docs = sum(1 for k in gt_doc_keys if k)
    print(f"Questions with page ground truth: {n_with_pages}", flush=True)
    print(f"Questions with doc  ground truth: {n_with_docs}", flush=True)

    max_k = max(K_VALUES)
    all_results: dict = {
        "config": {
            "csv": args.csv,
            "chroma_path": args.chroma_path,
            "collections": args.collections,
            "k_values": K_VALUES,
            "num_questions": len(df),
        },
        "results": {},
    }

    for name in args.collections:
        try:
            all_results["results"][name] = evaluate_collection(
                collection_name=name,
                questions=questions,
                gt_page_keys=gt_page_keys,
                gt_doc_keys=gt_doc_keys,
                chroma_path=args.chroma_path,
                max_k=max_k,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[error] collection {name} failed: {e}", file=sys.stderr)
            all_results["results"][name] = {"error": str(e)}

        # Persist after each collection so partial progress isn't lost.
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Wrote intermediate results to {args.output}", flush=True)

    # Final pretty table.
    print("\n================ SUMMARY ================")
    header = f"{'collection':<22}" + "".join(f"  p@{k:<5}d@{k:<5}" for k in K_VALUES)
    print(header)
    for name, payload in all_results["results"].items():
        if "error" in payload:
            print(f"{name:<22}  ERROR: {payload['error']}")
            continue
        s = payload["summary"]
        row = f"{name:<22}"
        for k in K_VALUES:
            pr = s["page_recall"][str(k)]
            dr = s["doc_recall"][str(k)]
            row += f"  {pr:.3f} {dr:.3f}" if pr is not None and dr is not None else "   n/a   n/a"
        print(row)


if __name__ == "__main__":
    main()
