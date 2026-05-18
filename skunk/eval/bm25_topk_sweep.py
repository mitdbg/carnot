"""Recall@K sweep for BM25-as-hard-cap retriever.

For each dev UID, run L1 chapter pick ONCE (LLM), year-filter (deterministic),
then BM25-score the survivors. Sweep K post-hoc — no extra LLM cost per K.

Reports per-K: micro-recall, mean pages/UID, fraction of goldens that BM25
*could not* rank (score=0, i.e. lexically invisible to the question).

Usage:
    SKUNK_USE_DIRECT_GEMINI=1 uv run python eval/bm25_topk_sweep.py
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

from skunk.common import HarnessContext, LLMClient, load_env_file  # noqa: E402
load_env_file(_REPO_ROOT / ".env")
from skunk.config import SkunkConfig  # noqa: E402
from skunk.plan import Plan, RetrieveBranch  # noqa: E402
from skunk.page_index import default_profile  # noqa: E402
from skunk.page_index.retrieve_probe import (  # noqa: E402
    load_catalog, load_concept_tree, one_shot_parent_chapter_retrieve,
)
from skunk.page_index.bm25 import tokenize  # noqa: E402
from skunk.page_index.bm25_runtime import build_chapter_index  # noqa: E402

KS = [100, 250, 500, 1000]
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
    print(f"Tree: {sum(c.get('n_pages',0) for c in tree['chapters'].values())} pages   "
          f"Catalog: {len(catalog)} rows")

    # Test-set filter (CLAUDE.md)
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
    print(f"Dev UIDs: {len(benchmark)} (test-set excluded)")

    # Load plans from cache
    plan_cache_path = _REPO_ROOT / "cache/retrieve_bench_plans.jsonl"
    if not plan_cache_path.exists():
        print(f"FAIL: need plan cache at {plan_cache_path}; "
              f"run eval/eval_retrieve.py first.", file=sys.stderr)
        return 2
    plans = {}
    for line in plan_cache_path.open():
        d = json.loads(line)
        plans[d["uid"]] = d["plan_json"]
    print(f"Plan cache: {len(plans)} entries")

    # Per-chapter BM25 index cache
    bm25_cache = {}
    def get_idx(chap):
        if chap not in bm25_cache:
            pages = tree["chapters"][chap]["pages"]
            bm25_cache[chap] = build_chapter_index(pages, catalog)
        return bm25_cache[chap]

    # Per-branch work: do LLM L1 pick, year-filter, BM25-score. Record rank
    # of every golden page (if in_filt and scored).
    def work(rec):
        uid = rec["uid"]
        goldens = set(rec["goldens"])
        if uid not in plans:
            return None
        try:
            plan_obj = Plan.model_validate_json(plans[uid])
        except Exception:
            return None
        branches = [b for b in plan_obj.branches if isinstance(b, RetrieveBranch)]
        all_scored = []          # list[(page, score)] across branches
        all_unscored = set()     # pages with score=0 across branches
        baseline_set = set()     # union of year-filtered survivors across branches
        for bi, b in enumerate(branches):
            chap_top, trace = one_shot_parent_chapter_retrieve(
                tree, question=rec["question"], concept=b.key, period=b.period,
                llm=llm, catalog_index=catalog, uid=uid, retrieve_idx=bi,
            )
            win = _PERIOD.year_window(b.period)
            if win is None:
                filt = chap_top
            else:
                lo, hi = win
                filt = []
                for c in chap_top:
                    row = catalog.get((c["bulletin"], int(c["page"])))
                    if row is None or row.min_year is None or row.max_year is None:
                        filt.append(c)
                    elif row.max_year >= lo and row.min_year <= hi:
                        filt.append(c)
            if not trace.picked_chapters or not filt:
                continue
            idx = get_idx(trace.picked_chapters[0])
            q_toks = tokenize(rec["question"]) + 2 * tokenize(b.key)
            scores = idx.score(q_toks)
            for c in filt:
                bp = (c["bulletin"], int(c["page"]))
                baseline_set.add(bp)
                s = scores.get(bp, 0.0)
                if s > 0:
                    all_scored.append((bp, s))
                else:
                    all_unscored.add(bp)
        # Per-K recall against UNION of branch top-Ks
        # Use a stable sort by descending score; ties broken arbitrarily (Python sort is stable).
        all_scored.sort(key=lambda sp: -sp[1])
        return {
            "uid": uid,
            "goldens": goldens,
            "baseline_set": baseline_set,
            "scored_sorted": all_scored,
            "unscored_set": all_unscored,
        }

    print(f"\nProcessing {len(benchmark)} UIDs (LLM L1 pick per branch)...")
    t0 = time.monotonic()
    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, rec) for rec in benchmark]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if r is not None:
                rows.append(r)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(benchmark)}  "
                      f"(elapsed {time.monotonic()-t0:.0f}s)", flush=True)
    print(f"Processed {len(rows)} UIDs in {time.monotonic()-t0:.0f}s\n")

    # Compute recall@K per UID for each K (per-branch top-K union)
    # Actually we collapsed across branches; cleaner: take top-K of the
    # combined scored list. For per-branch top-K, would need per-branch
    # results. For now use combined.
    def recall_at_k(rows, K):
        hits = 0
        goldens = 0
        pred_pages = 0
        per_uid_recalls = []
        for r in rows:
            preds = {bp for bp, _ in r["scored_sorted"][:K]}
            h = len(r["goldens"] & preds)
            hits += h
            goldens += len(r["goldens"])
            pred_pages += len(preds)
            per_uid_recalls.append(h / max(1, len(r["goldens"])))
        return {
            "K": K,
            "micro_recall": hits / max(1, goldens),
            "macro_recall": sum(per_uid_recalls) / max(1, len(per_uid_recalls)),
            "mean_pred": pred_pages / max(1, len(rows)),
            "hits": hits,
            "goldens": goldens,
        }

    # Also: baseline recall (full year-filtered set, no BM25 cap)
    def baseline_recall(rows):
        hits = 0; goldens = 0; pred_pages = 0
        for r in rows:
            h = len(r["goldens"] & r["baseline_set"])
            hits += h
            goldens += len(r["goldens"])
            pred_pages += len(r["baseline_set"])
        return {"K": "all", "micro_recall": hits / max(1, goldens),
                "mean_pred": pred_pages / max(1, len(rows)),
                "hits": hits, "goldens": goldens}

    # BM25 ceiling: goldens that have nonzero score in at least one branch
    def bm25_ceiling(rows):
        hits = 0; goldens = 0
        scored_misses = []   # goldens IN baseline but score=0
        for r in rows:
            scored_pages = {bp for bp, _ in r["scored_sorted"]}
            for g in r["goldens"]:
                if g in scored_pages:
                    hits += 1
                elif g in r["baseline_set"]:
                    scored_misses.append((r["uid"], g))
                goldens += 1
        return {"micro_recall": hits / max(1, goldens),
                "hits": hits, "goldens": goldens,
                "n_goldens_bm25_invisible": len(scored_misses),
                "examples": scored_misses[:8]}

    print("=" * 70)
    print(f"{'cut':<8}{'micro_recall':<16}{'macro_recall':<16}{'mean_pred':<12}{'hits/goldens'}")
    print("-" * 70)
    base = baseline_recall(rows)
    print(f"{'all':<8}{base['micro_recall']*100:>13.1f} %  {'-':<16}"
          f"{base['mean_pred']:<12.0f}{base['hits']}/{base['goldens']}")
    ceil = bm25_ceiling(rows)
    print(f"{'BM25∞':<8}{ceil['micro_recall']*100:>13.1f} %  {'-':<16}{'-':<12}"
          f"{ceil['hits']}/{ceil['goldens']}    "
          f"({ceil['n_goldens_bm25_invisible']} goldens unrankable: score=0)")
    for K in KS:
        m = recall_at_k(rows, K)
        print(f"K={K:<6}{m['micro_recall']*100:>13.1f} %"
              f"  {m['macro_recall']*100:>13.1f} %"
              f"  {m['mean_pred']:<12.1f}{m['hits']}/{m['goldens']}")
    print("=" * 70)

    if ceil["examples"]:
        print("\nGoldens unrankable by BM25 (score=0, sample):")
        for uid, g in ceil["examples"]:
            print(f"  {uid}: {g}")

    # Save for reuse
    out_path = _REPO_ROOT / "cache/bm25_topk_sweep.json"
    payload = {
        "rows": [{"uid": r["uid"],
                  "n_goldens": len(r["goldens"]),
                  "n_baseline": len(r["baseline_set"]),
                  "n_scored": len(r["scored_sorted"]),
                  "goldens": list(r["goldens"]),
                  "scored_top200": [(list(bp), s) for bp, s in r["scored_sorted"][:200]],
                  } for r in rows],
        "summary": {
            "baseline": base, "bm25_ceiling": {k: v for k, v in ceil.items() if k != "examples"},
            "per_k": [recall_at_k(rows, K) for K in KS],
        },
    }
    out_path.write_text(json.dumps(payload, default=str, indent=1))
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
