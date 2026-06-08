from _internal.chroma_store import ChromaStore
import json
import pdb
import random
from pathlib import Path
from typing import Optional

# embedding_model_name = "Qwen/Qwen3-Embedding-4B"
embedding_model_name = "text-embedding-3-large"


def count_unique_templates(*paths: str | Path) -> int:
    templates = set()
    for path in paths:
        for entry in load_results_jsonl(path):
            t = entry.get("metadata", {}).get("template")
            if t is not None:
                templates.add(t)
    return sorted(templates)


def sample_one_per_template(*paths: str | Path, seed: Optional[int] = None) -> dict[str, tuple[str, str]]:
    by_template: dict[str, list[tuple[dict, str]]] = {}
    for path in paths:
        p = str(Path(path).name)
        for entry in load_results_jsonl(path):
            t = entry.get("metadata", {}).get("template")
            if t is not None:
                by_template.setdefault(t, []).append((entry, p))
    if seed is not None:
        random.seed(seed)
    result = {}
    for t, entries in sorted(by_template.items()):
        entry, p = random.choice(entries)
        result[t] = (entry["query"], p)
    return result


def load_results_jsonl(path: str | Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def compare_results_jsonl(
    path_a: str | Path,
    path_b: str | Path,
    metrics: Optional[list[str]] = None,
    verbose: bool = True,
) -> dict:
    """Compare two results JSONL files entry-by-entry up to the shorter file's length."""
    if metrics is None:
        metrics = ["recall@20", "precision@20", "mrr@20", "ndcg@20"]

    entries_a = load_results_jsonl(path_a)
    entries_b = load_results_jsonl(path_b)
    n_a, n_b = len(entries_a), len(entries_b)
    n = min(n_a, n_b)
    pairs = list(zip(entries_a[:n], entries_b[:n]))

    result = {
        "n_compared": n,
        "n_a": n_a,
        "n_b": n_b,
        "metrics": {},
    }

    for m in metrics:
        a_vals, b_vals = [], []
        a_wins_idx, b_wins_idx, ties_idx = [], [], []

        for i, (ea, eb) in enumerate(pairs):
            va = ea.get(m)
            vb = eb.get(m)
            if va is not None and vb is not None:
                a_vals.append(va)
                b_vals.append(vb)
                diff = va - vb
                if diff > 0:
                    a_wins_idx.append(i)
                elif diff < 0:
                    b_wins_idx.append(i)
                else:
                    ties_idx.append(i)

        result["metrics"][m] = {
            "a_avg": sum(a_vals) / len(a_vals) if a_vals else None,
            "b_avg": sum(b_vals) / len(b_vals) if b_vals else None,
            "diff_avg": (sum(a_vals) - sum(b_vals)) / len(a_vals) if a_vals else None,
            "a_wins_n": len(a_wins_idx),
            "b_wins_n": len(b_wins_idx),
            "ties_n": len(ties_idx),
            "a_wins": a_wins_idx,
            "b_wins": b_wins_idx,
            "ties": ties_idx,
        }

    if verbose:
        name_a, name_b = Path(path_a).name, Path(path_b).name
        print(f"Compared {result['n_compared']} entries (file A: {n_a}, file B: {n_b})")
        print(f"  A: {name_a}")
        print(f"  B: {name_b}")
        for m in metrics:
            r = result["metrics"][m]
            print(f"\n  {m}:")
            print(f"    A avg: {r['a_avg']:.4f}  B avg: {r['b_avg']:.4f}  diff (A-B): {r['diff_avg']:+.4f}")
            print(f"    A wins (n={len(r['a_wins'])}): {r['a_wins']}")
            print(f"    B wins (n={len(r['b_wins'])}): {r['b_wins']}")
            print(f"    Ties (n={len(r['ties'])}): {r['ties']}")

    return result


def compare_recall(path_a: str | Path, path_b: str | Path, verbose: bool = True) -> dict:
    return compare_results_jsonl(path_a, path_b, metrics=["recall@20"], verbose=verbose)

# unique_templates = count_unique_templates(
#     "tmp/subset_1_quest_queries.jsonl",
#     "tmp/subset_2_quest_queries.jsonl",
#     "tmp/subset_3_quest_queries.jsonl",
# )

# queries = sample_one_per_template(
#     "tmp/subset_1_quest_queries.jsonl",
#     "tmp/subset_2_quest_queries.jsonl",
#     "tmp/subset_3_quest_queries.jsonl",
#     seed=17,
# )

# pdb.set_trace()

# compare_results = compare_results_jsonl(
#     "results_Qwen/Qwen3-Embedding-4B/quest_eval_results_val_expanded_subset_3_1.jsonl",
#     "results_Qwen/Qwen3-Embedding-4B/quest_eval_results_val_expanded_subset_3.jsonl",
# )

# pdb.set_trace()

collection_suffix = "_subset_1"
store = ChromaStore(f"quest_expanded{collection_suffix}", f"./chroma_collections_{embedding_model_name}", embedding_model_name=None)
# store = ChromaStore(f"quest_base{collection_suffix}", f"./chroma_collections_{embedding_model_name}")


def get_doc_by_entity_id(entity_id: str):
    r = store.collection.get(where={"entity_id": entity_id}, include=["metadatas", "documents"], limit=1)
    if not r["ids"]:
        return None
    return {
        "id": r["ids"][0],
        "metadata": r["metadatas"][0],
        "text": r["documents"][0],
    }
    
def get_doc_by_title(title: str):
    r = store.collection.get(where={"title": title}, include=["metadatas", "documents"], limit=1)
    if not r["ids"]:
        return None
    return {
        "id": r["ids"][0],
        "metadata": r["metadatas"][0],
        "text": r["documents"][0],
    }

pdb.set_trace()
doc = get_doc_by_title("The Guilty (2021 film)")

doc = get_doc_by_entity_id("Serial_Killing_4_Dummys-046e750d")
print(len(doc["metadata"]))
