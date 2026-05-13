"""Run the noisy leg of a 10-UID golden-vs-noisy comparison.

The golden leg is already done — current 3-flash-preview traces live in
`eval/traces_golden/` and are scored in `eval/golden32_final.csv`. This script
builds the noisy leg by:

  1. Sampling 10 UIDs from the 32-set (3 fixed for continuity with the prior
     comparison + 7 random).
  2. For each UID, finding the smallest seed shift such that
     `make_noisy_pages(...)` injects at least one confounder. Stored per-UID
     so every UID in the set actually receives noise (the prior run had only
     4 of 10 receiving any).
  3. Calling `skunk.run.run_question(..., golden_pages=<noisy refs>)` for each
     UID, writing traces under `eval/traces_comparison10_noisy/` and a CSV
     report at `eval/comparison10_noisy_report.csv`. Also dumps a
     `eval/comparison10_selection.json` for reproducibility.

Usage:

    SKUNK_GEMINI_MODEL=gemini-3-flash-preview SKUNK_GEMINI_RPM=500 \\
        python -m eval.run_comparison
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_load_env(REPO_ROOT / ".env")

import pandas as pd  # noqa: E402

from eval.eval_e2e import _parse_source_docs  # noqa: E402
from eval.noise import load_noise_pool, make_noisy_pages  # noqa: E402
from skunk.run import load_plan_cache, run_question  # noqa: E402

NOISE_PROB = 0.7
BASE_SEED = 42
SAMPLE_SEED = 1
SEED_RETRY_CAP = 100
FIXED_UIDS = ["UID0010", "UID0013", "UID0111"]
N_TOTAL = 10

TRACE_DIR = REPO_ROOT / "eval" / "traces_comparison10_noisy"
REPORT_CSV = REPO_ROOT / "eval" / "comparison10_noisy_report.csv"
SELECTION_JSON = REPO_ROOT / "eval" / "comparison10_selection.json"
GOLDEN_TRACES = REPO_ROOT / "eval" / "traces_golden"
BENCH_CSV = REPO_ROOT / "data" / "officeqa_pro.csv"
POOL_PATH = REPO_ROOT / "eval" / "noise_pool.json"


def pick_uids() -> list[str]:
    golden_uids = sorted(p.stem for p in GOLDEN_TRACES.glob("UID*.txt"))
    pool_uids = [u for u in golden_uids if u not in FIXED_UIDS]
    rng = random.Random(SAMPLE_SEED)
    sampled = rng.sample(pool_uids, N_TOTAL - len(FIXED_UIDS))
    return FIXED_UIDS + sampled


def find_seed(uid: str, golden_refs, pdf_dir: str, pool) -> tuple[int, list]:
    """Return (seed, noisy_refs) where noisy_refs has >0 confounders."""
    for k in range(SEED_RETRY_CAP):
        seed = BASE_SEED + k
        noisy = make_noisy_pages(
            uid, golden_refs,
            noise_prob=NOISE_PROB, seed=seed,
            pdf_dir=pdf_dir, pool=pool,
        )
        if len(noisy) > len(golden_refs):
            return seed, noisy
    raise RuntimeError(
        f"{uid}: no seed in [{BASE_SEED}, {BASE_SEED + SEED_RETRY_CAP}) "
        f"produced any confounder at noise_prob={NOISE_PROB}"
    )


def main() -> None:
    pdf_dir = os.environ.get(
        "OFFICEQA_PDF_DIR",
        str(Path.home() / "Desktop/officeqa/treasury_bulletin_pdfs"),
    )
    pool = load_noise_pool(str(POOL_PATH))
    bench = pd.read_csv(BENCH_CSV).set_index("uid")

    uids = pick_uids()
    print(f"[comparison] picked {len(uids)} UIDs:", ", ".join(uids))

    plan_cache_csv = "data/dsl_planning_pass.csv"
    plan_cache = load_plan_cache(plan_cache_csv)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    selection: dict = {
        "sample_seed": SAMPLE_SEED,
        "base_noise_seed": BASE_SEED,
        "noise_prob": NOISE_PROB,
        "fixed_uids": FIXED_UIDS,
        "per_uid": {},
    }
    rows: list[dict] = []

    for uid in uids:
        if uid not in bench.index:
            print(f"[comparison] WARNING: {uid} not in bench CSV; skipping", file=sys.stderr)
            continue
        question = str(bench.loc[uid, "question"])
        gold = bench.loc[uid, "answer"]
        gold = "" if pd.isna(gold) else str(gold)
        golden_refs = _parse_source_docs(str(bench.loc[uid, "source_docs"] or ""))
        if not golden_refs:
            print(f"[comparison] WARNING: {uid} has no golden pages; skipping", file=sys.stderr)
            continue

        seed, noisy_refs = find_seed(uid, golden_refs, pdf_dir, pool)
        n_conf = len(noisy_refs) - len(golden_refs)
        confounders = [
            {"month": r.month, "page": r.page}
            for r in noisy_refs[len(golden_refs):]
        ]
        selection["per_uid"][uid] = {
            "noise_seed": seed,
            "n_confounders": n_conf,
            "n_golden": len(golden_refs),
            "confounders": confounders,
        }
        print(f"[comparison] {uid}: seed={seed} → {n_conf} confounder(s)")

        cached_plan = plan_cache.get(uid)
        trace_path = str(TRACE_DIR / f"{uid}.txt")
        try:
            result = run_question(
                question=question, verbose=False, golden_pages=noisy_refs,
                cached_plan_text=cached_plan, uid=uid,
                plan_cache_csv=plan_cache_csv, trace_path=trace_path,
            )
        except Exception as e:
            result = {"question": question, "answer": None, "failed": True,
                      "reason": f"harness crash: {type(e).__name__}: {e}", "n_steps": 0}

        ans = result["answer"] if not result["failed"] else ""
        tag = "FAIL" if result["failed"] else f"OK ans={ans}"
        print(f"[comparison] {uid}: {tag}")

        rows.append({
            "uid": uid,
            "question": question,
            "predicted": ans,
            "gold_answer": gold,
            "failed": result["failed"],
            "reason": result.get("reason") or "",
            "n_steps": result.get("n_steps", 0),
            "n_confounders": n_conf,
            "noise_seed": seed,
        })

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["uid", "question", "predicted", "gold_answer", "failed",
              "reason", "n_steps", "n_confounders", "noise_seed"]
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    SELECTION_JSON.write_text(json.dumps(selection, indent=2))
    n_total = len(rows)
    n_failed = sum(1 for r in rows if r["failed"])
    print(f"\n[comparison] wrote {REPORT_CSV} ({n_total} rows)")
    print(f"[comparison] wrote {SELECTION_JSON}")
    print(f"[comparison] {n_total - n_failed}/{n_total} produced an answer")


if __name__ == "__main__":
    main()
