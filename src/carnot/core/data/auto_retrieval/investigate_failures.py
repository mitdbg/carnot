"""
Investigate hard failure queries:
1. All-three-fail queries
2. Colbert-only failures
3. Truly hard queries (ALL approaches fail, including metadata)
4. Root cause analysis: query patterns, ground-truth sizes, template types
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).parent / "eval_results"
TMP_DIR = Path(__file__).parent / "tmp"
META_DIR = Path(__file__).parent / "results_text-embedding-3-large"
SUBSETS = [1, 2, 3]
RECALL_THRESH = 0.5

# ── Loaders ──────────────────────────────────────────────────────────────────

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    per_query = [r for r in records if not r.get("_summary")]
    return per_query


def load_queries(subset):
    path = TMP_DIR / f"subset_{subset}_quest_queries.jsonl"
    queries = []
    with open(path) as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def load_meta_results(subset):
    """Load expanded filter results (where_clause + filtered_titles)."""
    path = META_DIR / f"quest_eval_results_val_expanded_subset_{subset}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def avg(vals):
    return sum(vals) / len(vals) if vals else 0.0


# ── Load everything ──────────────────────────────────────────────────────────

ALL_SETUPS = [
    "dense", "splade", "colbert",
    "meta_dense", "meta_splade", "meta_colbert",
    "dense_rerank", "splade_rerank", "colbert_rerank",
    "meta_rerank",
]
BASE = ["dense", "splade", "colbert"]

# per_query[(setup, subset)] = list of dicts
per_query = {}
for setup in ALL_SETUPS:
    for s in SUBSETS:
        per_query[(setup, s)] = load_jsonl(EVAL_DIR / f"{setup}_subset_{s}.jsonl")

# queries[subset] = list of QuestQuery dicts
queries = {s: load_queries(s) for s in SUBSETS}

# meta_results[subset] = list of expanded filter result dicts
meta_results = {s: load_meta_results(s) for s in SUBSETS}


# ── Helper: get best recall across all setups for a query ────────────────────

def best_recall_all_setups(subset, qi):
    return max(per_query[(setup, subset)][qi]["recall@20"] for setup in ALL_SETUPS)


def best_recall_base(subset, qi):
    return max(per_query[(setup, subset)][qi]["recall@20"] for setup in BASE)


# ── SECTION 1: Categorize all failures ───────────────────────────────────────

print("=" * 90)
print("SECTION 1: Failure categorization per subset")
print("=" * 90)

all_three_fail_queries = {}  # subset -> list of (qi, query_dict)
colbert_only_fail_queries = {}
truly_hard_queries = {}  # ALL 10 setups fail

for s in SUBSETS:
    n = len(per_query[("dense", s)])
    d_fail = {i for i in range(n) if per_query[("dense", s)][i]["recall@20"] < RECALL_THRESH}
    sp_fail = {i for i in range(n) if per_query[("splade", s)][i]["recall@20"] < RECALL_THRESH}
    c_fail = {i for i in range(n) if per_query[("colbert", s)][i]["recall@20"] < RECALL_THRESH}

    all3 = sorted(d_fail & sp_fail & c_fail)
    c_only = sorted(c_fail - d_fail - sp_fail)

    # Truly hard: best recall across ALL 10 setups < threshold
    truly_hard = sorted(qi for qi in range(n) if best_recall_all_setups(s, qi) < RECALL_THRESH)

    all_three_fail_queries[s] = all3
    colbert_only_fail_queries[s] = c_only
    truly_hard_queries[s] = truly_hard

    print(f"\n--- Subset {s} ---")
    print(f"  All-three-fail:     {len(all3)}")
    print(f"  Colbert-only-fail:  {len(c_only)}")
    print(f"  Truly hard (all 10 setups fail): {len(truly_hard)}")


# ── SECTION 2: List ALL "all-three-fail" queries ─────────────────────────────

print("\n" + "=" * 90)
print("SECTION 2: All queries where dense, splade, AND colbert all fail")
print("=" * 90)

for s in SUBSETS:
    print(f"\n{'='*40} Subset {s} {'='*40}")
    for qi in all_three_fail_queries[s]:
        q = queries[s][qi]
        d_r = per_query[("dense", s)][qi]["recall@20"]
        sp_r = per_query[("splade", s)][qi]["recall@20"]
        c_r = per_query[("colbert", s)][qi]["recall@20"]
        mc_r = per_query[("meta_colbert", s)][qi]["recall@20"]
        mr_r = per_query[("meta_rerank", s)][qi]["recall@20"]
        best_all = best_recall_all_setups(s, qi)

        truly_hard = " *** TRULY HARD ***" if best_all < RECALL_THRESH else ""
        print(f"\n  Query {qi}: \"{q['query']}\"{truly_hard}")
        print(f"    original: \"{q['original_query']}\"")
        print(f"    template: \"{q['metadata'].get('template', '?')}\"")
        print(f"    domain:   {q['metadata'].get('domain', '?')}")
        print(f"    #gold docs: {len(q['docs'])}")
        print(f"    recalls — dense: {d_r:.3f}  splade: {sp_r:.3f}  colbert: {c_r:.3f}  meta_colbert: {mc_r:.3f}  meta_rerank: {mr_r:.3f}  best_all: {best_all:.3f}")

        # Show what metadata filter produced
        mr = meta_results[s][qi]
        wc = mr.get("where_clause", {})
        n_filt = mr.get("n_filtered_docs", "?")
        print(f"    metadata filter → {n_filt} docs; where_clause: {json.dumps(wc, default=str)[:200]}")


# ── SECTION 3: Colbert-only failures ─────────────────────────────────────────

print("\n" + "=" * 90)
print("SECTION 3: Queries where ONLY colbert fails (dense & splade succeed)")
print("=" * 90)

for s in SUBSETS:
    print(f"\n{'='*40} Subset {s} {'='*40}")
    if not colbert_only_fail_queries[s]:
        print("  (none)")
        continue
    for qi in colbert_only_fail_queries[s]:
        q = queries[s][qi]
        d_r = per_query[("dense", s)][qi]["recall@20"]
        sp_r = per_query[("splade", s)][qi]["recall@20"]
        c_r = per_query[("colbert", s)][qi]["recall@20"]
        mc_r = per_query[("meta_colbert", s)][qi]["recall@20"]
        print(f"\n  Query {qi}: \"{q['query']}\"")
        print(f"    original: \"{q['original_query']}\"")
        print(f"    template: \"{q['metadata'].get('template', '?')}\"")
        print(f"    domain:   {q['metadata'].get('domain', '?')}")
        print(f"    #gold docs: {len(q['docs'])}")
        print(f"    recalls — dense: {d_r:.3f}  splade: {sp_r:.3f}  colbert: {c_r:.3f}  meta_colbert: {mc_r:.3f}")

        # Show gold docs and colbert predictions
        pred_top = per_query[("colbert", s)][qi].get("predicted_top", [])
        gold = set(q['docs'])
        missed = gold - set(pred_top[:20])
        print(f"    gold docs: {q['docs'][:8]}{'...' if len(q['docs']) > 8 else ''}")
        print(f"    colbert missed: {sorted(missed)[:8]}{'...' if len(missed) > 8 else ''}")


# ── SECTION 4: Truly hard queries — deep analysis ────────────────────────────

print("\n" + "=" * 90)
print("SECTION 4: TRULY HARD queries (ALL 10 setups fail) — deep analysis")
print("=" * 90)

truly_hard_all = []  # collect for pattern analysis

for s in SUBSETS:
    print(f"\n{'='*40} Subset {s} {'='*40}")
    for qi in truly_hard_queries[s]:
        q = queries[s][qi]
        d_r = per_query[("dense", s)][qi]["recall@20"]
        sp_r = per_query[("splade", s)][qi]["recall@20"]
        c_r = per_query[("colbert", s)][qi]["recall@20"]
        best_all = best_recall_all_setups(s, qi)

        # Collect all setup recalls
        all_recalls = {setup: per_query[(setup, s)][qi]["recall@20"] for setup in ALL_SETUPS}
        best_setup = max(all_recalls, key=all_recalls.get)

        mr = meta_results[s][qi]
        wc = mr.get("where_clause", {})
        n_filt = mr.get("n_filtered_docs", "?")
        filtered_titles = mr.get("filtered_titles", [])

        gold = set(q['docs'])
        gold_in_filter = gold & set(filtered_titles)

        # Colbert predicted
        c_pred = per_query[("colbert", s)][qi].get("predicted_top", [])
        c_hits = set(c_pred[:20]) & gold
        c_missed = gold - set(c_pred[:20])

        truly_hard_all.append({
            "subset": s,
            "qi": qi,
            "query": q["query"],
            "original_query": q["original_query"],
            "template": q["metadata"].get("template", "?"),
            "domain": q["metadata"].get("domain", "?"),
            "n_gold": len(q["docs"]),
            "best_recall": best_all,
            "best_setup": best_setup,
            "n_filtered": n_filt,
            "gold_in_filter": len(gold_in_filter),
            "gold_docs": q["docs"],
            "where_clause": wc,
        })

        print(f"\n  Query {qi}: \"{q['query']}\"")
        print(f"    original:     \"{q['original_query']}\"")
        print(f"    template:     \"{q['metadata'].get('template', '?')}\"")
        print(f"    domain:       {q['metadata'].get('domain', '?')}")
        print(f"    #gold docs:   {len(q['docs'])}")
        print(f"    best recall:  {best_all:.3f} (from {best_setup})")
        print(f"    all recalls:  d={d_r:.3f} sp={sp_r:.3f} c={c_r:.3f}")
        print(f"      mc={all_recalls['meta_colbert']:.3f} mr={all_recalls['meta_rerank']:.3f}")
        print(f"      dr={all_recalls['dense_rerank']:.3f} spr={all_recalls['splade_rerank']:.3f} cr={all_recalls['colbert_rerank']:.3f}")
        print(f"    metadata filter → {n_filt} docs")
        print(f"    gold docs in filter: {len(gold_in_filter)}/{len(gold)}  ({100*len(gold_in_filter)/len(gold) if gold else 0:.0f}%)")
        print(f"    where_clause: {json.dumps(wc, default=str)[:300]}")
        print(f"    gold docs:    {q['docs'][:6]}{'...' if len(q['docs']) > 6 else ''}")
        print(f"    colbert hits: {sorted(c_hits)[:6]}")
        print(f"    colbert missed: {sorted(c_missed)[:6]}{'...' if len(c_missed) > 6 else ''}")


# ── SECTION 5: Pattern analysis of truly hard queries ─────────────────────────

print("\n" + "=" * 90)
print("SECTION 5: Pattern analysis of truly hard queries")
print("=" * 90)

n_hard = len(truly_hard_all)
print(f"\nTotal truly hard queries across all subsets: {n_hard}")

# Template distribution
templates = Counter(q["template"] for q in truly_hard_all)
print(f"\nTemplate distribution:")
for t, count in templates.most_common():
    print(f"  \"{t}\": {count} ({100*count/n_hard:.0f}%)")

# Domain distribution
domains = Counter(q["domain"] for q in truly_hard_all)
print(f"\nDomain distribution:")
for d, count in domains.most_common():
    print(f"  {d}: {count} ({100*count/n_hard:.0f}%)")

# Gold set size distribution
gold_sizes = [q["n_gold"] for q in truly_hard_all]
print(f"\nGold set size: min={min(gold_sizes)}, max={max(gold_sizes)}, median={sorted(gold_sizes)[len(gold_sizes)//2]}, mean={avg(gold_sizes):.1f}")

# Metadata filter quality: how many gold docs are captured by the filter?
filter_capture = [q["gold_in_filter"] / q["n_gold"] if q["n_gold"] > 0 else 0 for q in truly_hard_all]
print(f"\nMetadata filter gold capture rate: mean={avg(filter_capture):.2%}, min={min(filter_capture):.2%}, max={max(filter_capture):.2%}")

low_capture = [q for q in truly_hard_all if q["gold_in_filter"] / q["n_gold"] < 0.5]
print(f"  Queries where filter captures <50% gold: {len(low_capture)}/{n_hard}")
high_capture = [q for q in truly_hard_all if q["gold_in_filter"] / q["n_gold"] >= 0.8]
print(f"  Queries where filter captures >=80% gold: {len(high_capture)}/{n_hard}")

# Filter result size
filter_sizes = [q["n_filtered"] for q in truly_hard_all if isinstance(q["n_filtered"], int)]
if filter_sizes:
    print(f"\nFilter result size: min={min(filter_sizes)}, max={max(filter_sizes)}, median={sorted(filter_sizes)[len(filter_sizes)//2]}, mean={avg(filter_sizes):.0f}")

# Best recall distribution
best_recalls = [q["best_recall"] for q in truly_hard_all]
print(f"\nBest recall (across all 10 setups): mean={avg(best_recalls):.3f}, min={min(best_recalls):.3f}, max={max(best_recalls):.3f}")

# Bucket: near-zero vs low
near_zero = [q for q in truly_hard_all if q["best_recall"] < 0.2]
low = [q for q in truly_hard_all if 0.2 <= q["best_recall"] < 0.5]
print(f"  Near-zero (<0.2 best recall): {len(near_zero)}")
print(f"  Low (0.2-0.5 best recall):    {len(low)}")


# ── SECTION 6: Hypothesize failure modes ─────────────────────────────────────

print("\n" + "=" * 90)
print("SECTION 6: Failure mode categorization")
print("=" * 90)

# Categorize each truly hard query
categories = defaultdict(list)

for q in truly_hard_all:
    template = q["template"]
    n_gold = q["n_gold"]
    filter_rate = q["gold_in_filter"] / n_gold if n_gold > 0 else 0
    best = q["best_recall"]
    query_text = q["query"].lower()
    orig = q["original_query"].lower()

    reasons = []

    # 1. Large gold set → recall limited by k=20
    if n_gold > 20:
        reasons.append("LARGE_GOLD_SET")

    # 2. Complex set operations (not, excluding, intersection)
    if "not" in orig or "excluding" in orig or "that are not" in query_text or "excluding" in query_text:
        reasons.append("NEGATION/EXCLUSION")

    if template and ("not" in template or "excluding" in template):
        reasons.append("NEGATION_TEMPLATE")

    # 3. Multi-constraint intersection
    marks = orig.count("<mark>")
    if marks >= 3:
        reasons.append("MULTI_CONSTRAINT(3+_marks)")
    elif marks == 2 and ("and" in query_text or "also" in query_text or "that are" in query_text):
        reasons.append("INTERSECTION")

    # 4. Metadata filter fails to capture gold
    if filter_rate < 0.5:
        reasons.append(f"FILTER_MISSES_GOLD({filter_rate:.0%})")

    # 5. Overly broad filter (too many results)
    if isinstance(q["n_filtered"], int) and q["n_filtered"] > 500:
        reasons.append(f"FILTER_TOO_BROAD({q['n_filtered']})")

    # 6. Near-zero recall everywhere
    if best < 0.15:
        reasons.append("NEAR_ZERO_ALL")

    if not reasons:
        reasons.append("UNKNOWN")

    for r in reasons:
        categories[r].append(q)

    q["reasons"] = reasons

print(f"\nFailure mode frequency (queries can have multiple modes):")
for cat, qs in sorted(categories.items(), key=lambda x: -len(x[1])):
    print(f"\n  {cat}: {len(qs)} queries")
    for q in qs[:3]:
        print(f"    - [{q['subset']}:{q['qi']}] \"{q['query'][:80]}\" (gold={q['n_gold']}, best={q['best_recall']:.3f})")


# ── SECTION 7: Compare hard queries vs easy queries ──────────────────────────

print("\n" + "=" * 90)
print("SECTION 7: Hard vs Easy query characteristics")
print("=" * 90)

easy_queries = []
for s in SUBSETS:
    n = len(per_query[("colbert", s)])
    for qi in range(n):
        c_r = per_query[("colbert", s)][qi]["recall@20"]
        if c_r >= 0.8:
            q = queries[s][qi]
            easy_queries.append({
                "subset": s, "qi": qi,
                "query": q["query"],
                "template": q["metadata"].get("template", "?"),
                "domain": q["metadata"].get("domain", "?"),
                "n_gold": len(q["docs"]),
                "recall": c_r,
            })

print(f"\nEasy queries (colbert recall >= 0.8): {len(easy_queries)}")
print(f"Truly hard queries: {n_hard}")

# Template comparison
easy_templates = Counter(q["template"] for q in easy_queries)
print(f"\nTemplate distribution — Easy:")
for t, count in easy_templates.most_common(5):
    print(f"  \"{t}\": {count} ({100*count/len(easy_queries):.0f}%)")
print(f"Template distribution — Hard:")
for t, count in templates.most_common(5):
    print(f"  \"{t}\": {count} ({100*count/n_hard:.0f}%)")

# Domain comparison
easy_domains = Counter(q["domain"] for q in easy_queries)
print(f"\nDomain distribution — Easy:")
for d, count in easy_domains.most_common():
    print(f"  {d}: {count} ({100*count/len(easy_queries):.0f}%)")
print(f"Domain distribution — Hard:")
for d, count in domains.most_common():
    print(f"  {d}: {count} ({100*count/n_hard:.0f}%)")

# Gold set size
easy_gold = [q["n_gold"] for q in easy_queries]
print(f"\nGold set size — Easy: mean={avg(easy_gold):.1f}, median={sorted(easy_gold)[len(easy_gold)//2]}")
print(f"Gold set size — Hard: mean={avg(gold_sizes):.1f}, median={sorted(gold_sizes)[len(gold_sizes)//2]}")

# Query length
easy_qlen = [len(q["query"].split()) for q in easy_queries]
hard_qlen = [len(q["query"].split()) for q in truly_hard_all]
print(f"\nQuery word count — Easy: mean={avg(easy_qlen):.1f}")
print(f"Query word count — Hard: mean={avg(hard_qlen):.1f}")


# ── SECTION 8: Unique query analysis ─────────────────────────────────────────
# Some queries repeat across subsets. Let's deduplicate.

print("\n" + "=" * 90)
print("SECTION 8: Unique truly hard queries (deduplicated across subsets)")
print("=" * 90)

seen = set()
unique_hard = []
for q in truly_hard_all:
    key = q["original_query"]
    if key not in seen:
        seen.add(key)
        unique_hard.append(q)

print(f"\nTotal: {n_hard}, Unique: {len(unique_hard)}")
print(f"\nAll unique truly hard queries:\n")

for i, q in enumerate(unique_hard):
    print(f"  {i+1:2d}. [{q['subset']}:{q['qi']}] \"{q['query']}\"")
    print(f"      original: \"{q['original_query']}\"")
    print(f"      template: \"{q['template']}\" | domain: {q['domain']} | #gold: {q['n_gold']} | best_recall: {q['best_recall']:.3f}")
    print(f"      failure modes: {', '.join(q['reasons'])}")
    print(f"      filter: {q['n_filtered']} docs, captures {q['gold_in_filter']}/{q['n_gold']} gold")
    print()
