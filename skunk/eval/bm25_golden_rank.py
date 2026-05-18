"""For each retrieve branch, rank year-filtered candidates by BM25 and report
the rank (and normalized rank) of every golden page that lives in that branch's
candidate set.

This isolates the BM25 ordering quality from any K-cap policy: if goldens
cluster near the top, BM25 is informative; if they spread uniformly, BM25
ordering carries no signal.

Tie-breaking: ties broken by stable sort (descending score). Goldens with
score = 0 are appended after every scored page in arbitrary order — for
ranking we put them at rank = (N_scored + 1) and mark them as `unrankable`.

Usage:
    SKUNK_USE_DIRECT_GEMINI=1 uv run python eval/bm25_golden_rank.py
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from skunk.common import LLMClient, load_env_file  # noqa: E402
load_env_file(_REPO_ROOT / ".env")
from skunk.config import SkunkConfig  # noqa: E402
from skunk.plan import Plan, RetrieveBranch  # noqa: E402
from skunk.page_index import default_profile  # noqa: E402
from skunk.page_index.retrieve_probe import (  # noqa: E402
    load_catalog, load_concept_tree, one_shot_parent_chapter_retrieve,
)
from skunk.page_index.bm25 import tokenize  # noqa: E402
from skunk.page_index.bm25_runtime import build_chapter_index  # noqa: E402

_PERIOD = default_profile().period_parser

_MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
           "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}


def _parse_source_doc(url: str):
    m = re.search(r"/([a-z]+)-(\d{4})-\d+\?page=(\d+)", url)
    if not m:
        return None
    mn = _MONTHS.get(m.group(1).lower())
    if mn is None:
        return None
    return (f"{m.group(2)}-{mn:02d}", int(m.group(3)))


def main() -> int:
    cfg = SkunkConfig.from_env()
    llm = LLMClient(cfg)

    tree = load_concept_tree(_REPO_ROOT / "data/page_index/concept_tree.json")
    cat_rows = load_catalog(_REPO_ROOT / "cache/page_index_v4/catalog")
    catalog = {(r.bulletin, r.page): r for r in cat_rows}

    test_uids = set(json.loads((_REPO_ROOT / "eval/test_set_uids.json").read_text())["uids"])

    benchmark = []
    with (_REPO_ROOT / "data/officeqa_pro.csv").open() as f:
        for row in csv.DictReader(f):
            if row["uid"] in test_uids:
                continue
            goldens = []
            for url in (row.get("source_docs") or "").splitlines():
                ref = _parse_source_doc(url.strip())
                if ref:
                    goldens.append(ref)
            benchmark.append({"uid": row["uid"], "question": row["question"], "goldens": goldens})

    plan_cache_path = _REPO_ROOT / "cache/retrieve_bench_plans.jsonl"
    plans = {json.loads(l)["uid"]: json.loads(l)["plan_json"] for l in plan_cache_path.open()}

    bm25_cache = {}
    def get_idx(chap):
        if chap not in bm25_cache:
            bm25_cache[chap] = build_chapter_index(tree["chapters"][chap]["pages"], catalog)
        return bm25_cache[chap]

    # Per-branch work: return list of dicts, one per (branch, golden_in_branch).
    def work(rec):
        uid = rec["uid"]
        goldens = set(rec["goldens"])
        if uid not in plans:
            return []
        try:
            plan_obj = Plan.model_validate_json(plans[uid])
        except Exception:
            return []
        branches = [b for b in plan_obj.branches if isinstance(b, RetrieveBranch)]
        out = []
        for bi, b in enumerate(branches):
            chap_top, trace = one_shot_parent_chapter_retrieve(
                tree, question=rec["question"], concept=b.key, period=b.period,
                llm=llm, catalog_index=catalog, uid=uid, retrieve_idx=bi,
            )
            win = _PERIOD.year_window(b.period)
            if win is None:
                filt_pages = [(c["bulletin"], int(c["page"])) for c in chap_top]
            else:
                lo, hi = win
                filt_pages = []
                for c in chap_top:
                    p = (c["bulletin"], int(c["page"]))
                    row = catalog.get(p)
                    if row is None or row.min_year is None or row.max_year is None:
                        filt_pages.append(p)
                    elif row.max_year >= lo and row.min_year <= hi:
                        filt_pages.append(p)
            if not trace.picked_chapters or not filt_pages:
                continue
            chap = trace.picked_chapters[0]
            idx = get_idx(chap)
            q_toks = tokenize(rec["question"]) + 2 * tokenize(b.key)
            scores_all = idx.score(q_toks)
            # Restrict to the year-filtered candidate set
            filt_set = set(filt_pages)
            scored = [(p, scores_all.get(p, 0.0)) for p in filt_pages]
            # Stable sort by descending score; tie-broken by insertion order
            scored.sort(key=lambda ps: -ps[1])
            rank_of = {p: i + 1 for i, (p, _) in enumerate(scored)}  # 1-based
            score_of = {p: s for p, s in scored}
            N = len(scored)
            for g in goldens & filt_set:
                rank = rank_of[g]
                sc = score_of[g]
                # Count tied scores to disambiguate stable-sort rank from
                # canonical rank (= number of pages with STRICTLY higher score + 1)
                strictly_higher = sum(1 for _, s in scored if s > sc)
                canonical_rank = strictly_higher + 1
                out.append({
                    "uid": uid, "branch": bi, "chapter": chap, "period": b.period,
                    "golden": list(g),
                    "stable_rank": rank, "canonical_rank": canonical_rank,
                    "N_candidates": N, "score": sc,
                    "normalized_rank": canonical_rank / N,
                    "unrankable": sc == 0.0,
                })
            # Also note goldens not in this branch's candidate set (out of scope here)
        return out

    print(f"Processing {len(benchmark)} UIDs...", flush=True)
    t0 = time.monotonic()
    all_rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, r) for r in benchmark]
        for i, f in enumerate(as_completed(futs)):
            all_rows.extend(f.result())
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(benchmark)}  elapsed {time.monotonic()-t0:.0f}s", flush=True)
    print(f"Done in {time.monotonic()-t0:.0f}s. {len(all_rows)} (branch, golden) pairs.\n")

    if not all_rows:
        print("No data."); return 1

    ranks = [r["canonical_rank"] for r in all_rows]
    norms = [r["normalized_rank"] for r in all_rows]
    Ns    = [r["N_candidates"] for r in all_rows]
    unrnk = sum(1 for r in all_rows if r["unrankable"])

    def pct(xs, p):
        xs = sorted(xs)
        if not xs: return 0
        i = max(0, min(len(xs)-1, int(round(p * (len(xs)-1)))))
        return xs[i]

    print("Branch-level golden-rank distribution (lower = better)")
    print(f"  pairs:                 {len(all_rows)}")
    print(f"  unrankable (score=0):  {unrnk} ({unrnk/len(all_rows):.1%})")
    print()
    print(f"  canonical_rank — mean / median: {statistics.mean(ranks):>8.1f} / {statistics.median(ranks):>6.1f}")
    print(f"  candidate set N — mean / median: {statistics.mean(Ns):>8.0f} / {statistics.median(Ns):>6.0f}")
    print(f"  normalized rank — mean / median: {statistics.mean(norms):>8.3f} / {statistics.median(norms):>6.3f}")
    print()
    print("  rank quantiles (raw):")
    for p in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
        print(f"     p{int(p*100):>2}: rank {pct(ranks, p):>8}  norm {pct(norms, p):>5.3f}")
    print()

    # Recall@K curve at the BRANCH level (per-branch top K hits the golden)
    print("Recall@K (per (branch, golden) pair — fraction whose canonical_rank ≤ K):")
    for K in (10, 20, 50, 100, 200, 500, 1000, 2000):
        hits = sum(1 for r in all_rows if r["canonical_rank"] <= K and not r["unrankable"])
        print(f"  K={K:<6} {hits/len(all_rows)*100:>5.1f} %    ({hits}/{len(all_rows)})")
    print()

    # Per-branch normalized-rank buckets
    print("Normalized-rank bucket distribution (canonical_rank / N_candidates):")
    buckets = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0001]
    for lo, hi in zip(buckets, buckets[1:]):
        n = sum(1 for r in all_rows if lo <= r["normalized_rank"] < hi)
        bar = "█" * int(50 * n / len(all_rows))
        print(f"  [{lo:>5.2f}, {hi:>5.2f})  {n:>4}  {n/len(all_rows)*100:>5.1f} %  {bar}")
    print()

    # Worst cases — goldens that are far down the ranking
    worst = sorted(all_rows, key=lambda r: -r["canonical_rank"])[:15]
    print("Worst 15 (golden, canonical_rank, N, score):")
    for r in worst:
        print(f"  {r['uid']:<10} br{r['branch']}  golden {tuple(r['golden'])}  "
              f"rank {r['canonical_rank']:>6}/{r['N_candidates']:>6}  "
              f"score {r['score']:>7.2f}  chap='{r['chapter']}'")

    out = _REPO_ROOT / "cache/bm25_golden_rank.json"
    out.write_text(json.dumps({"rows": all_rows}, indent=1))
    print(f"\nSaved {len(all_rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
