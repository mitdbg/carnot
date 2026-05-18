"""Compare per-branch top-K recall: BM25 ordering vs the current chronological
ordering (what extract sees today). Same L1 + year-filter input set, two
different orderings.

This is the relevant A/B for the question 'should we hand extract a BM25-
ordered page list': how often is the golden in the top K of each ordering?

Reuses `cache/bm25_golden_rank.json` for the BM25 ranks and re-runs L1+
year-filter once per branch to recover the chronological position.

Usage:
    SKUNK_USE_DIRECT_GEMINI=1 uv run python eval/bm25_vs_chronological.py
"""

from __future__ import annotations

import csv
import json
import re
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
            gs = []
            for url in (row.get("source_docs") or "").splitlines():
                ref = _parse_source_doc(url.strip())
                if ref:
                    gs.append(ref)
            benchmark.append({"uid": row["uid"], "question": row["question"], "goldens": gs})

    plans = {json.loads(l)["uid"]: json.loads(l)["plan_json"]
             for l in (_REPO_ROOT / "cache/retrieve_bench_plans.jsonl").open()}

    # Pull BM25 ranks from earlier dump for matching (uid, branch, golden) tuples.
    bm25_rank = {}
    bm25_data = json.load(open(_REPO_ROOT / "cache/bm25_golden_rank.json"))
    for r in bm25_data["rows"]:
        bm25_rank[(r["uid"], r["branch"], tuple(r["golden"]))] = {
            "rank": r["canonical_rank"], "N": r["N_candidates"],
            "unrankable": r["unrankable"],
        }

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
                filt = [(c["bulletin"], int(c["page"])) for c in chap_top]
            else:
                lo, hi = win
                filt = []
                for c in chap_top:
                    p = (c["bulletin"], int(c["page"]))
                    row = catalog.get(p)
                    if row is None or row.min_year is None or row.max_year is None:
                        filt.append(p)
                    elif row.max_year >= lo and row.min_year <= hi:
                        filt.append(p)
            N = len(filt)
            rank_of = {p: i + 1 for i, p in enumerate(filt)}    # 1-based chronological rank
            for g in goldens:
                if g not in rank_of:
                    continue
                chrono_rank = rank_of[g]
                bm = bm25_rank.get((uid, bi, g))   # may be missing if branch differed run-to-run
                out.append({
                    "uid": uid, "branch": bi, "golden": list(g),
                    "N_candidates": N,
                    "chrono_rank": chrono_rank,
                    "bm25_rank": (bm["rank"] if bm else None),
                    "bm25_unrankable": (bm["unrankable"] if bm else None),
                })
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

    # Filter to rows that have both chronological AND BM25 ranks for apples-to-apples
    matched = [r for r in all_rows if r["bm25_rank"] is not None]
    print(f"Pairs with both rankings (apples-to-apples): {len(matched)} / {len(all_rows)}\n")

    print(f"{'K':<6}{'chrono recall@K':<22}{'BM25 recall@K':<22}{'delta':<10}")
    print('-' * 60)
    for K in (5, 10, 20, 50, 100, 200, 500, 1000):
        c_hit = sum(1 for r in matched if r["chrono_rank"] <= K)
        b_hit = sum(1 for r in matched if r["bm25_rank"] <= K and not r["bm25_unrankable"])
        n = len(matched)
        c_pct = c_hit / n * 100
        b_pct = b_hit / n * 100
        delta = b_pct - c_pct
        sign = '+' if delta >= 0 else ''
        print(f'K={K:<5}{c_hit}/{n} = {c_pct:>5.1f} %      {b_hit}/{n} = {b_pct:>5.1f} %      {sign}{delta:>+5.1f} pp')

    # How many ranks does BM25 typically shave off?
    import statistics
    deltas = [r["chrono_rank"] - r["bm25_rank"] for r in matched if not r["bm25_unrankable"]]
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    print()
    print(f"Per-(branch, golden) rank delta (chronological_rank - bm25_rank, positive = BM25 wins):")
    print(f"  BM25 better: {wins}/{len(deltas)} ({wins/len(deltas):.1%})")
    print(f"  BM25 worse:  {losses}/{len(deltas)} ({losses/len(deltas):.1%})")
    print(f"  Tied:        {ties}/{len(deltas)}")
    print(f"  mean delta:  {statistics.mean(deltas):+.0f} ranks (positive = BM25 ranks closer to top)")
    print(f"  median delta: {statistics.median(deltas):+.0f}")

    # Distribution of how-much-better BM25 is when it wins
    win_deltas = sorted([d for d in deltas if d > 0])
    if win_deltas:
        print(f"\n  On BM25-wins: median improvement = {statistics.median(win_deltas):.0f} ranks;  "
              f"p90 = {win_deltas[int(0.9*len(win_deltas))]:.0f};  max = {max(win_deltas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
