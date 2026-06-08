"""Evaluate the few-shot examples from query_planner.py against their
corresponding ChromaDB subset collections.

For each example the script:
1. Extracts the WHERE clause and identifies the target subset.
2. Queries the expanded ChromaDB collection with the filter.
3. Finds the matching QUEST query for ground-truth docs.
4. Computes filter-recall, recall, precision, MRR, and nDCG.
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quest_utils import prepare_quest_queries
from _internal.chroma_store import ChromaStore
from _internal.query_planner import FEW_SHOT_EXAMPLES

HERE = Path(__file__).resolve().parent

# Mapping of few-shot query text -> subset number
QUERY_SUBSET_MAP: Dict[str, int] = {
    "Grape family of flowering plants": 2,
    "American remakes of Danish films or about weddings in the United States": 1,
    "Non-fiction books from 1858, 1863, or 1859": 3,
    "Freshwater animals of Africa that are Endemic fauna of Angola": 2,
    "Malpighiales genera and Paleotropical flora but not Monotypic angiosperm genera": 2,
    "Fauna of the California chaparral, woodlands, Colorado Desert and other deserts": 1,
    "Novels set in Vietnam excluding Novels set in the 1970s": 1,
}


def recall_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def precision_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in relevant) / len(top)


def mrr_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    for i, d in enumerate(ranked[:k], 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    dcg = 0.0
    for i, d in enumerate(ranked[:k], 1):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 1)
    n_ideal = min(len(relevant), k)
    if n_ideal == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    output_path = HERE / "tmp" / "debug_few_shot_get_results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    persist_dir = str(HERE / "chroma_collections_text-embedding-3-large")

    # Load QUEST queries for ground-truth lookup
    quest_queries_by_subset: Dict[int, Dict[str, List[str]]] = {}
    for subset in [1, 2, 3]:
        qpath = HERE / f"tmp/subset_{subset}_quest_queries.jsonl"
        with open(qpath) as f:
            qs = [json.loads(l) for l in f if l.strip()]
        quest_queries_by_subset[subset] = {q["query"]: q["docs"] for q in qs}

    # Open stores (no embedding needed — we only use .get())
    stores: Dict[int, ChromaStore] = {}
    for subset in [1, 2, 3]:
        stores[subset] = ChromaStore(
            collection_name=f"quest_expanded_subset_{subset}",
            persist_directory=persist_dir,
            embed=False,
        )

    results = []
    for ex in FEW_SHOT_EXAMPLES:
        query_text = ex["question"]
        where_clause = ex["where"]
        template = ex.get("template", "")
        subset_id = QUERY_SUBSET_MAP.get(query_text)

        if subset_id is None:
            print(f"WARNING: No subset mapping for query: {query_text}")
            continue

        store = stores[subset_id]

        # Get all docs matching the filter
        filtered = store.get(where=where_clause)
        filtered_titles = [r["metadata"]["title"] for r in filtered if r.get("metadata")]
        unique_filtered = list(dict.fromkeys(filtered_titles))

        # Ground truth
        gt_map = quest_queries_by_subset.get(subset_id, {})
        gt_docs = gt_map.get(query_text, [])

        if not gt_docs:
            print(f"WARNING: No ground truth for query: {query_text}")

        gt_set = set(gt_docs)
        in_filter = [d for d in gt_docs if d in set(unique_filtered)]
        missing = [d for d in gt_docs if d not in set(unique_filtered)]

        filter_recall = len(in_filter) / len(gt_docs) if gt_docs else 0.0
        rec = recall_at_k(unique_filtered, gt_set, len(unique_filtered))
        prec = precision_at_k(unique_filtered, gt_set, len(unique_filtered))
        rr = mrr_at_k(unique_filtered, gt_set, len(unique_filtered))
        ndcg = ndcg_at_k(unique_filtered, gt_set, len(unique_filtered))

        entry = {
            "template": template,
            "query": query_text,
            "n_filter_chunks": len(filtered),
            "n_filter_unique_docs": len(unique_filtered),
            "n_ground_truth": len(gt_docs),
            "n_ground_truth_in_filter": len(in_filter),
            "ground_truth_in_filter": sorted(in_filter),
            "ground_truth_missing": sorted(missing),
            "filter_recall": filter_recall,
            "recall": rec,
            "precision": prec,
            "mrr": rr,
            "ndcg": ndcg,
        }
        results.append(entry)
        print(
            f"[{template}] {query_text}: "
            f"filter_recall={filter_recall:.4f}, "
            f"recall={rec:.4f}, "
            f"precision={prec:.4f}, "
            f"mrr={rr:.4f}, "
            f"ndcg={ndcg:.4f}"
        )

    # Write results
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # Summary
        n = len(results)
        if n > 0:
            summary = {
                "n_examples": n,
                "Average Filter Recall": sum(r["filter_recall"] for r in results) / n,
                "Average Recall": sum(r["recall"] for r in results) / n,
                "Average Precision": sum(r["precision"] for r in results) / n,
                "Average MRR": sum(r["mrr"] for r in results) / n,
                "Average nDCG": sum(r["ndcg"] for r in results) / n,
            }
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
            print(f"\nSummary: {json.dumps(summary, indent=2)}")

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
