from __future__ import annotations

import math
from typing import Callable, Dict, List, Set


def recall_at_k(predicted: List[str], gold: Set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(predicted[:k]) & gold) / len(gold)


def precision_at_k(predicted: List[str], gold: Set[str], k: int) -> float:
    top = predicted[:k]
    if not top:
        return 0.0
    return len(set(top) & gold) / len(top)


def mrr_at_k(predicted: List[str], gold: Set[str], k: int) -> float:
    for i, doc in enumerate(predicted[:k]):
        if doc in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(predicted: List[str], relevance_fn: Callable[[str], float],
              gold_docs: List[str], k: int) -> float:
    def dcg(docs: List[str], k_: int) -> float:
        return sum(relevance_fn(docs[i]) / math.log2(i + 2)
                   for i in range(min(k_, len(docs))))
    actual = dcg(predicted, k)
    ideal = dcg(gold_docs, k)
    return actual / ideal if ideal > 0 else 0.0


def compute_metrics(predicted: List[str], gold_set: Set[str],
                    gold_list: List[str], k: int) -> Dict[str, float]:
    rel_fn = lambda d: 1.0 if d in gold_set else 0.0
    return {
        f"recall@{k}": recall_at_k(predicted, gold_set, k),
        f"precision@{k}": precision_at_k(predicted, gold_set, k),
        f"mrr@{k}": mrr_at_k(predicted, gold_set, k),
        f"ndcg@{k}": ndcg_at_k(predicted, rel_fn, gold_list, k),
    }
