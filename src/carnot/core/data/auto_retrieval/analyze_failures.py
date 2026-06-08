"""Analyze retrieval evaluation results: failure overlap and metadata potential."""

import json
import os
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).parent / "eval_results"
SUBSETS = [1, 2, 3]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path):
    """Return (list_of_per_query_dicts, summary_dict)."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    summary = records[-1] if records[-1].get("_summary") else None
    per_query = [r for r in records if not r.get("_summary")]
    return per_query, summary


def avg(vals):
    return sum(vals) / len(vals) if vals else 0.0


# ── 1. Aggregate average scores across subsets ───────────────────────────────

print("=" * 80)
print("PART 1: Average scores across all subsets (averaged over subsets 1-3)")
print("=" * 80)

setups = [
    "dense", "meta_dense",
    "splade", "meta_splade",
    "colbert", "meta_colbert",
    "dense_rerank", "splade_rerank", "colbert_rerank", "meta_rerank",
]

metrics = ["recall@20", "precision@20", "mrr@20", "ndcg@20"]

setup_avg = {}
for setup in setups:
    subset_summaries = []
    for s in SUBSETS:
        path = EVAL_DIR / f"{setup}_subset_{s}.jsonl"
        _, summary = load_jsonl(path)
        subset_summaries.append(summary)
    avg_metrics = {}
    for m in metrics:
        avg_metrics[m] = avg([ss[m] for ss in subset_summaries])
    setup_avg[setup] = avg_metrics

# Print table
header = f"{'Setup':<20}" + "".join(f"{m:>14}" for m in metrics)
print(header)
print("-" * len(header))
for setup in setups:
    row = f"{setup:<20}"
    for m in metrics:
        row += f"{setup_avg[setup][m]:>14.4f}"
    print(row)

# ── 2. Meta-prefilter analysis ───────────────────────────────────────────────

print("\n" + "=" * 80)
print("PART 2: Metadata as prefilter — delta (meta_X - X)")
print("=" * 80)

for base, meta in [("dense", "meta_dense"), ("splade", "meta_splade"), ("colbert", "meta_colbert")]:
    deltas = {m: setup_avg[meta][m] - setup_avg[base][m] for m in metrics}
    row = f"{meta} - {base:<12}"
    for m in metrics:
        row += f"  {m}: {deltas[m]:+.4f}"
    print(row)

# ── 3. Reranking analysis ────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("PART 3: Reranking delta (X_rerank - X)")
print("=" * 80)

for base, reranked in [("dense", "dense_rerank"), ("splade", "splade_rerank"), ("colbert", "colbert_rerank")]:
    deltas = {m: setup_avg[reranked][m] - setup_avg[base][m] for m in metrics}
    row = f"{reranked} - {base:<12}"
    for m in metrics:
        row += f"  {m}: {deltas[m]:+.4f}"
    print(row)

# ── 4. Per-query failure analysis for base retrievers ────────────────────────

print("\n" + "=" * 80)
print("PART 4: Per-query failure analysis (dense, splade, colbert)")
print("=" * 80)

RECALL_THRESHOLD = 0.5  # query "fails" if recall@20 < threshold

# Load per-query data for base retrievers
base_retrievers = ["dense", "splade", "colbert"]
per_query_data = {}  # (setup, subset) -> list of per-query dicts

for setup in base_retrievers:
    for s in SUBSETS:
        path = EVAL_DIR / f"{setup}_subset_{s}.jsonl"
        pq, _ = load_jsonl(path)
        per_query_data[(setup, s)] = pq

# Also load meta variants for comparison
meta_setups = ["meta_dense", "meta_splade", "meta_colbert", "meta_rerank"]
for setup in meta_setups:
    for s in SUBSETS:
        path = EVAL_DIR / f"{setup}_subset_{s}.jsonl"
        pq, _ = load_jsonl(path)
        per_query_data[(setup, s)] = pq

# Also load rerank variants
rerank_setups = ["dense_rerank", "splade_rerank", "colbert_rerank"]
for setup in rerank_setups:
    for s in SUBSETS:
        path = EVAL_DIR / f"{setup}_subset_{s}.jsonl"
        pq, _ = load_jsonl(path)
        per_query_data[(setup, s)] = pq

# Identify failures per retriever per subset
failures = {}  # (setup, subset) -> set of query_indices that failed
for setup in base_retrievers:
    for s in SUBSETS:
        failed = set()
        for q in per_query_data[(setup, s)]:
            if q["recall@20"] < RECALL_THRESHOLD:
                failed.add(q["query_index"])
        failures[(setup, s)] = failed

# Overlap analysis per subset
print(f"\nFailure threshold: recall@20 < {RECALL_THRESHOLD}")
for s in SUBSETS:
    print(f"\n--- Subset {s} ---")
    for setup in base_retrievers:
        print(f"  {setup}: {len(failures[(setup, s)])} failed queries out of 100")

    d_fail = failures[("dense", s)]
    sp_fail = failures[("splade", s)]
    c_fail = failures[("colbert", s)]

    all_fail = d_fail & sp_fail & c_fail
    any_fail = d_fail | sp_fail | c_fail
    d_sp = d_fail & sp_fail
    d_c = d_fail & c_fail
    sp_c = sp_fail & c_fail

    print(f"  All three fail:         {len(all_fail)}")
    print(f"  Dense ∩ Splade fail:    {len(d_sp)}")
    print(f"  Dense ∩ Colbert fail:   {len(d_c)}")
    print(f"  Splade ∩ Colbert fail:  {len(sp_c)}")
    print(f"  At least one fails:     {len(any_fail)}")
    print(f"  Only Dense fails:       {len(d_fail - sp_fail - c_fail)}")
    print(f"  Only Splade fails:      {len(sp_fail - d_fail - c_fail)}")
    print(f"  Only Colbert fails:     {len(c_fail - d_fail - sp_fail)}")

# ── 5. Can metadata help on failures? ────────────────────────────────────────

print("\n" + "=" * 80)
print("PART 5: Can metadata help on failed queries?")
print("  For queries where colbert fails, does meta_colbert / meta_rerank do better?")
print("=" * 80)

for s in SUBSETS:
    print(f"\n--- Subset {s} ---")
    c_fail_idx = failures[("colbert", s)]
    
    if not c_fail_idx:
        print("  No colbert failures!")
        continue
    
    # For each failed colbert query, check meta_colbert and meta_rerank scores
    colbert_data = {q["query_index"]: q for q in per_query_data[("colbert", s)]}
    meta_colbert_data = {q["query_index"]: q for q in per_query_data[("meta_colbert", s)]}
    meta_rerank_data = {q["query_index"]: q for q in per_query_data[("meta_rerank", s)]}
    
    meta_colbert_helps = 0
    meta_rerank_helps = 0
    meta_any_helps = 0
    
    colbert_recall_on_fail = []
    meta_colbert_recall_on_fail = []
    meta_rerank_recall_on_fail = []
    
    for qi in sorted(c_fail_idx):
        c_recall = colbert_data[qi]["recall@20"]
        mc_recall = meta_colbert_data[qi]["recall@20"]
        mr_recall = meta_rerank_data[qi]["recall@20"]
        
        colbert_recall_on_fail.append(c_recall)
        meta_colbert_recall_on_fail.append(mc_recall)
        meta_rerank_recall_on_fail.append(mr_recall)
        
        if mc_recall > c_recall + 0.05:
            meta_colbert_helps += 1
        if mr_recall > c_recall + 0.05:
            meta_rerank_helps += 1
        if mc_recall > c_recall + 0.05 or mr_recall > c_recall + 0.05:
            meta_any_helps += 1
    
    n_fail = len(c_fail_idx)
    print(f"  Colbert failures: {n_fail}")
    print(f"  Avg recall on failures — colbert: {avg(colbert_recall_on_fail):.4f}")
    print(f"  Avg recall on failures — meta_colbert: {avg(meta_colbert_recall_on_fail):.4f}")
    print(f"  Avg recall on failures — meta_rerank: {avg(meta_rerank_recall_on_fail):.4f}")
    print(f"  meta_colbert improves (>5% recall): {meta_colbert_helps}/{n_fail} ({100*meta_colbert_helps/n_fail:.1f}%)")
    print(f"  meta_rerank improves (>5% recall):  {meta_rerank_helps}/{n_fail} ({100*meta_rerank_helps/n_fail:.1f}%)")
    print(f"  Any meta improves (>5% recall):     {meta_any_helps}/{n_fail} ({100*meta_any_helps/n_fail:.1f}%)")


# ── 6. Oracle router analysis ────────────────────────────────────────────────

print("\n" + "=" * 80)
print("PART 6: Oracle router — pick best of colbert vs meta_colbert per query")
print("=" * 80)

for s in SUBSETS:
    print(f"\n--- Subset {s} ---")
    colbert_data = {q["query_index"]: q for q in per_query_data[("colbert", s)]}
    meta_colbert_data = {q["query_index"]: q for q in per_query_data[("meta_colbert", s)]}
    meta_rerank_data = {q["query_index"]: q for q in per_query_data[("meta_rerank", s)]}
    
    n = len(colbert_data)
    
    # Oracle: best of colbert vs meta_colbert
    oracle_recall_cm = []
    oracle_ndcg_cm = []
    # Oracle: best of colbert vs meta_rerank
    oracle_recall_cr = []
    oracle_ndcg_cr = []
    # Oracle: best of colbert, meta_colbert, meta_rerank
    oracle_recall_all = []
    oracle_ndcg_all = []
    
    colbert_chosen = 0
    meta_colbert_chosen = 0
    meta_rerank_chosen = 0
    
    for qi in range(n):
        c = colbert_data[qi]
        mc = meta_colbert_data[qi]
        mr = meta_rerank_data[qi]
        
        # Best of colbert vs meta_colbert (by recall)
        oracle_recall_cm.append(max(c["recall@20"], mc["recall@20"]))
        oracle_ndcg_cm.append(max(c["ndcg@20"], mc["ndcg@20"]))
        
        # Best of colbert vs meta_rerank
        oracle_recall_cr.append(max(c["recall@20"], mr["recall@20"]))
        oracle_ndcg_cr.append(max(c["ndcg@20"], mr["ndcg@20"]))
        
        # Best of all three
        best_recall = max(c["recall@20"], mc["recall@20"], mr["recall@20"])
        oracle_recall_all.append(best_recall)
        oracle_ndcg_all.append(max(c["ndcg@20"], mc["ndcg@20"], mr["ndcg@20"]))
        
        # Which is chosen?
        best = max(c["recall@20"], mc["recall@20"], mr["recall@20"])
        if c["recall@20"] == best:
            colbert_chosen += 1
        elif mc["recall@20"] == best:
            meta_colbert_chosen += 1
        else:
            meta_rerank_chosen += 1
    
    col_avg = setup_avg["colbert"]
    print(f"  Colbert alone:                  recall={avg([colbert_data[i]['recall@20'] for i in range(n)]):.4f}  ndcg={avg([colbert_data[i]['ndcg@20'] for i in range(n)]):.4f}")
    print(f"  Oracle(colbert, meta_colbert):   recall={avg(oracle_recall_cm):.4f}  ndcg={avg(oracle_ndcg_cm):.4f}")
    print(f"  Oracle(colbert, meta_rerank):    recall={avg(oracle_recall_cr):.4f}  ndcg={avg(oracle_ndcg_cr):.4f}")
    print(f"  Oracle(colbert, mc, mr):         recall={avg(oracle_recall_all):.4f}  ndcg={avg(oracle_ndcg_all):.4f}")
    print(f"  Oracle choice dist: colbert={colbert_chosen}, meta_colbert={meta_colbert_chosen}, meta_rerank={meta_rerank_chosen}")


# ── 7. Detailed failure examples ─────────────────────────────────────────────

print("\n" + "=" * 80)
print("PART 7: Example queries where metadata helps colbert failures")
print("=" * 80)

for s in SUBSETS:
    colbert_data = {q["query_index"]: q for q in per_query_data[("colbert", s)]}
    meta_colbert_data = {q["query_index"]: q for q in per_query_data[("meta_colbert", s)]}
    meta_rerank_data = {q["query_index"]: q for q in per_query_data[("meta_rerank", s)]}
    
    c_fail_idx = failures[("colbert", s)]
    
    examples = []
    for qi in sorted(c_fail_idx):
        c = colbert_data[qi]
        mc = meta_colbert_data[qi]
        mr = meta_rerank_data[qi]
        
        improvement = max(mc["recall@20"], mr["recall@20"]) - c["recall@20"]
        if improvement > 0.05:
            examples.append((qi, c, mc, mr, improvement))
    
    examples.sort(key=lambda x: -x[4])
    
    print(f"\n--- Subset {s}: {len(examples)} improved queries (out of {len(c_fail_idx)} colbert failures) ---")
    for qi, c, mc, mr, imp in examples[:5]:
        print(f"\n  Query {qi}: \"{c['query'][:100]}\"")
        print(f"    colbert recall:      {c['recall@20']:.4f}")
        print(f"    meta_colbert recall: {mc['recall@20']:.4f}")
        print(f"    meta_rerank recall:  {mr['recall@20']:.4f}")
        if "n_filtered" in mc:
            print(f"    n_filtered (meta_colbert): {mc['n_filtered']}")

# ── 8. Queries where ALL three base retrievers fail ──────────────────────────

print("\n" + "=" * 80)
print("PART 8: Queries where ALL three retrievers fail — can metadata help?")
print("=" * 80)

for s in SUBSETS:
    all_fail = failures[("dense", s)] & failures[("splade", s)] & failures[("colbert", s)]
    
    dense_data = {q["query_index"]: q for q in per_query_data[("dense", s)]}
    splade_data = {q["query_index"]: q for q in per_query_data[("splade", s)]}
    colbert_data = {q["query_index"]: q for q in per_query_data[("colbert", s)]}
    meta_colbert_data = {q["query_index"]: q for q in per_query_data[("meta_colbert", s)]}
    meta_rerank_data = {q["query_index"]: q for q in per_query_data[("meta_rerank", s)]}
    
    meta_helps_count = 0
    for qi in sorted(all_fail):
        c = colbert_data[qi]
        mc = meta_colbert_data[qi]
        mr = meta_rerank_data[qi]
        best_base = max(dense_data[qi]["recall@20"], splade_data[qi]["recall@20"], c["recall@20"])
        best_meta = max(mc["recall@20"], mr["recall@20"])
        if best_meta > best_base + 0.05:
            meta_helps_count += 1
    
    print(f"\n--- Subset {s}: {len(all_fail)} queries where all 3 fail ---")
    print(f"  Metadata improves over best base: {meta_helps_count}/{len(all_fail)}")
    
    # Show examples
    examples = []
    for qi in sorted(all_fail):
        d = dense_data[qi]
        sp = splade_data[qi]
        c = colbert_data[qi]
        mc = meta_colbert_data[qi]
        mr = meta_rerank_data[qi]
        best_base = max(d["recall@20"], sp["recall@20"], c["recall@20"])
        best_meta = max(mc["recall@20"], mr["recall@20"])
        examples.append((qi, d, sp, c, mc, mr, best_meta - best_base))
    
    examples.sort(key=lambda x: -x[6])
    for qi, d, sp, c, mc, mr, delta in examples[:5]:
        print(f"\n  Query {qi}: \"{c['query'][:100]}\"")
        print(f"    dense: {d['recall@20']:.4f}  splade: {sp['recall@20']:.4f}  colbert: {c['recall@20']:.4f}")
        print(f"    meta_colbert: {mc['recall@20']:.4f}  meta_rerank: {mr['recall@20']:.4f}  (delta={delta:+.4f})")


# ── 9. Distribution analysis: recall buckets ─────────────────────────────────

print("\n" + "=" * 80)
print("PART 9: Recall distribution (all subsets combined)")
print("=" * 80)

buckets = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
for setup in ["dense", "splade", "colbert", "meta_colbert", "meta_rerank"]:
    all_recalls = []
    for s in SUBSETS:
        for q in per_query_data[(setup, s)]:
            all_recalls.append(q["recall@20"])
    
    bucket_counts = []
    for lo, hi in buckets:
        count = sum(1 for r in all_recalls if lo <= r < hi)
        bucket_counts.append(count)
    
    print(f"  {setup:<16}", end="")
    for (lo, hi), c in zip(buckets, bucket_counts):
        print(f"  [{lo:.1f},{hi:.1f}):{c:>4}", end="")
    print()
