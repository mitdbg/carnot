"""Modular retrieval evaluation runner.

Run from auto_retrieval/:
  python -m retrieval_eval.runner --setup dense --subsets 1,2,3 --k 20
  python -m retrieval_eval.runner --setup meta_rerank --subsets 1 --k 20

Setups (3.1-3.6):
  dense          query -> Dense (text-embedding-3-large) -> top-K
  meta_dense     query -> metadata filter -> Dense -> top-K
  splade         query -> Splade v2 -> top-K
  meta_splade    query -> metadata filter -> Splade v2 -> top-K
  colbert        query -> ColBERT Zero -> top-K
  meta_colbert   query -> metadata filter -> ColBERT Zero -> top-K

Reranker setups (4.1-4.4):
  dense_rerank   query -> Dense -> reranker -> top-K
  splade_rerank  query -> Splade v2 -> reranker -> top-K
  colbert_rerank query -> ColBERT Zero -> reranker -> top-K
  meta_rerank    query -> metadata filter -> reranker -> top-K

For meta_* setups, filtered titles are loaded from:
  results_text-embedding-3-large/quest_eval_results_val_expanded_subset_N.jsonl
  (produced by run_quest_eval.py with filter_only=True).

For *_rerank setups, the retriever returns top-K, then the reranker
reorders them. meta_rerank reranks all filtered titles directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent.parent
load_dotenv(HERE / ".env")
sys.path.insert(0, str(HERE))

from quest_utils import prepare_quest_queries, read_jsonl, QuestQuery
from retrieval_eval.metrics import compute_metrics
from retrieval_eval.retrievers import (
    DenseRetriever,
    SpladeRetriever,
    ColBERTZeroRetriever,
)
from retrieval_eval.reranker import CrossEncoderReranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SETUPS = [
    "dense", "meta_dense", "splade", "meta_splade",
    "colbert", "meta_colbert",
    "dense_rerank", "splade_rerank", "colbert_rerank", "meta_rerank",
]

SETUP_CONFIG = {
    "dense":          {"retriever": "dense",   "filter": False, "rerank": False},
    "meta_dense":     {"retriever": "dense",   "filter": True,  "rerank": False},
    "splade":         {"retriever": "splade",  "filter": False, "rerank": False},
    "meta_splade":    {"retriever": "splade",  "filter": True,  "rerank": False},
    "colbert":        {"retriever": "colbert", "filter": False, "rerank": False},
    "meta_colbert":   {"retriever": "colbert", "filter": True,  "rerank": False},
    "dense_rerank":   {"retriever": "dense",   "filter": False, "rerank": True},
    "splade_rerank":  {"retriever": "splade",  "filter": False, "rerank": True},
    "colbert_rerank": {"retriever": "colbert", "filter": False, "rerank": True},
    "meta_rerank":    {"retriever": None,      "filter": True,  "rerank": True},
}


# -- Data loading ----------------------------------------------------------

def load_corpus(subset_n: int) -> Dict[str, str]:
    path = HERE / f"tmp/subset_{subset_n}_documents.jsonl"
    corpus: Dict[str, str] = {}
    for row in read_jsonl(str(path)):
        title = (row.get("title") or "").strip()
        text = (row.get("text") or row.get("description") or "").strip()
        if title:
            corpus[title] = text
    logger.info("Loaded corpus: %d docs (subset %d)", len(corpus), subset_n)
    return corpus


def load_queries(subset_n: int) -> List[QuestQuery]:
    return prepare_quest_queries(
        str(HERE / f"tmp/subset_{subset_n}_quest_queries.jsonl"))


def load_expanded_results(subset_n: int) -> Dict[int, List[str]]:
    """Load per-query filtered_titles from expanded eval results."""
    path = (HERE / "results_text-embedding-3-large"
            / f"quest_eval_results_val_expanded_subset_{subset_n}.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"Expanded results not found: {path}\n"
            "Run run_quest_eval.py with filter_only=True first.")
    results: Dict[int, List[str]] = {}
    for line in open(path, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        qi = entry.get("query_index")
        if qi is None:
            continue
        titles = entry.get("filtered_titles")
        if titles is None:
            raise ValueError(
                f"Expanded results for subset {subset_n} missing 'filtered_titles' field. "
                "Re-run run_quest_eval.py to regenerate with filtered_titles.")
        results[qi] = titles
    logger.info("Loaded expanded filter results: %d queries (subset %d)",
                len(results), subset_n)
    return results


# -- Singleton retriever/reranker factories --------------------------------

_retrievers: Dict[str, Any] = {}
_reranker: Optional[CrossEncoderReranker] = None


def get_retriever(name: str):
    if name not in _retrievers:
        if name == "dense":
            _retrievers[name] = DenseRetriever()
        elif name == "splade":
            _retrievers[name] = SpladeRetriever(device="cuda")
        elif name == "colbert":
            _retrievers[name] = ColBERTZeroRetriever()
        else:
            raise ValueError(f"Unknown retriever: {name}")
    return _retrievers[name]


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(device="cuda")
    return _reranker


# -- Core evaluation -------------------------------------------------------

def evaluate_setup(
    setup: str,
    subset_n: int,
    k: int = 20,
    output_dir: Optional[Path] = None,
) -> Dict[str, float]:
    cfg = SETUP_CONFIG[setup]
    corpus = load_corpus(subset_n)
    queries = load_queries(subset_n)

    expanded: Dict[int, List[str]] = {}
    if cfg["filter"]:
        expanded = load_expanded_results(subset_n)

    retriever = None
    if cfg["retriever"]:
        retriever = get_retriever(cfg["retriever"])
        if hasattr(retriever, "encode_corpus"):
            logger.info("Pre-encoding corpus for %s ...", cfg["retriever"])
            retriever.encode_corpus(corpus)

    reranker = get_reranker() if cfg["rerank"] else None

    if output_dir is None:
        output_dir = HERE / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{setup}_subset_{subset_n}.jsonl"
    f_out = open(out_path, "w", encoding="utf-8")

    accum: Dict[str, float] = {}
    n_queries = len(queries)

    try:
        for qi, q in enumerate(queries):
            gold_set = set(q.docs)
            gold_list = q.docs

            # Determine document pool
            if cfg["filter"]:
                ftitles = expanded.get(qi, [])
                docs = {t: corpus[t] for t in ftitles if t in corpus}
            else:
                docs = corpus

            # Execute retrieval pipeline
            if retriever and not cfg["rerank"]:
                results = retriever.search(q.query, docs, k)
                predicted = [did for did, _ in results]

            elif retriever and cfg["rerank"]:
                initial = retriever.search(q.query, docs, k)
                reranked = reranker.rerank(q.query, initial, docs, k)
                predicted = [did for did, _ in reranked]

            elif cfg["rerank"] and not retriever:
                # meta_rerank: rerank all filtered docs directly
                pairs = [(t, 0.0) for t in docs]
                reranked = reranker.rerank(q.query, pairs, docs, k)
                predicted = [did for did, _ in reranked]

            else:
                raise ValueError(f"Invalid config for setup: {setup}")

            metrics = compute_metrics(predicted, gold_set, gold_list, k)
            for mkey, val in metrics.items():
                accum[mkey] = accum.get(mkey, 0.0) + val

            entry = {"query_index": qi, "query": q.query}
            if cfg["filter"]:
                entry["n_filtered"] = len(docs)
            entry.update(metrics)
            entry["predicted_top"] = predicted[:k]
            f_out.write(json.dumps(entry) + "\n")
            f_out.flush()

            if (qi + 1) % 10 == 0 or qi == 0:
                logger.info("[%s subset_%d] %d/%d", setup, subset_n,
                            qi + 1, n_queries)

    finally:
        averages = {m: v / n_queries for m, v in accum.items()} if n_queries else {}
        summary = {"_summary": True, "n_queries": n_queries}
        summary.update(averages)
        f_out.write(json.dumps(summary) + "\n")
        f_out.close()

    logger.info("Saved %s", out_path)
    return averages


def print_averages(setup: str, subset_n: int, avgs: Dict[str, float]) -> None:
    if not avgs:
        print(f"  {setup} subset_{subset_n}: NO RESULTS")
        return
    parts = [f"{m}: {v:.4f}" for m, v in sorted(avgs.items())]
    print(f"  {setup} subset_{subset_n}: {', '.join(parts)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup", required=True, choices=SETUPS)
    parser.add_argument("--subsets", default="1,2,3")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    subsets = [int(s.strip()) for s in args.subsets.split(",")]
    output_dir = Path(args.output_dir) if args.output_dir else None

    print(f"\n{'='*60}")
    print(f"  Setup: {args.setup}  |  K={args.k}  |  Subsets: {subsets}")
    print(f"{'='*60}\n")

    all_avgs: Dict[str, float] = {}
    n_subsets = 0

    for subset_n in subsets:
        t0 = time.time()
        avgs = evaluate_setup(
            setup=args.setup, subset_n=subset_n,
            k=args.k, output_dir=output_dir,
        )
        elapsed = time.time() - t0
        print_averages(args.setup, subset_n, avgs)
        logger.info("Subset %d done in %.1fs", subset_n, elapsed)
        if avgs:
            n_subsets += 1
            for m, v in avgs.items():
                all_avgs[m] = all_avgs.get(m, 0.0) + v

    if n_subsets > 1:
        overall = {m: v / n_subsets for m, v in all_avgs.items()}
        print(f"\n  OVERALL ({n_subsets} subsets):")
        parts = [f"{m}: {v:.4f}" for m, v in sorted(overall.items())]
        print(f"    {', '.join(parts)}")


if __name__ == "__main__":
    main()
