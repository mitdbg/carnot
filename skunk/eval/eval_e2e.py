"""End-to-end eval. Runs the full pipeline (planner → orchestrator → 6 subagents)
per question and writes predicted vs gold answers to a CSV report.

Scoring is intentionally out of scope here — the report is the artifact a downstream
scorer consumes. The gap between this report and eval_extraction quantifies retrieval
cost; the gap to the gold answer floor quantifies extraction cost.

Usage
-----
  # Smoke set (4 UIDs) — default
  python -m eval.eval_e2e --csv data/officeqa_pro.csv --report eval/e2e_report.csv

  # All UIDs in the CSV
  python -m eval.eval_e2e --csv data/officeqa_pro.csv --report eval/e2e_report.csv --all

  # Inject golden pages (skip retrieve subagent) — recommended for isolating retrieval cost
  python -m eval.eval_e2e --csv data/officeqa_pro.csv --report eval/e2e_report.csv --golden

  # Run only specific UIDs
  python -m eval.eval_e2e --csv data/officeqa_pro.csv --report eval/e2e_report.csv \\
      --uids UID0001,UID0030 --golden --verbose

Golden mode (`--golden`)
------------------------
Bypasses the retrieve subagent. For each question, `source_docs?page=N` URLs in
data/officeqa_pro.csv are parsed (eval/golden.py) and injected as a DocHandle
directly into the operator chain. extract/compute/etc. then run on exactly the
pages the benchmark deems relevant. This is the right mode for measuring the
extract+compute ceiling — any failure here is a downstream-of-retrieval bug.

Use the default (live retrieve) mode to see the end-to-end number including
retrieval cost. The delta between the two reports is the retrieval contribution.

Traces (`--trace-dir`, default `eval/traces`)
---------------------------------------------
Every question writes a `{uid}.txt` trace file containing, per operator step:
op name, args, input/output (full repr), elapsed_s (which naturally absorbs
Gemini retry wait time), and any error. Per-step subagent events (tier dispatch
in extract, codegen attempts in compute, verifier responses, etc.) are grouped
under each step. Use these for post-hoc auditability of every run.

Pass `--trace-dir ''` to disable. Pass `--verbose` to additionally live-stream
the events to stdout during the run.

Gemini transient failures
-------------------------
`LLMClient` (src/skunk/common/llm.py) retries 429/5xx errors with a
fixed delay. Tune via env: `SKUNK_GEMINI_RETRY_DELAY` (default 30s),
`SKUNK_GEMINI_MAX_RETRIES` (default 5). After exhaustion the step fails
cleanly and the trace records the final exception.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd

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

from eval.golden import load_golden  # noqa: E402
from skunk.config import SkunkConfig  # noqa: E402
from skunk.run import SMOKE_UIDS, load_plan_cache, run_question  # noqa: E402

REPORT_FIELDS = ["uid", "question", "predicted", "gold_answer", "failed", "reason", "n_steps"]


def _pick_uids(df: pd.DataFrame, run_all: bool, uids_arg: str | None) -> list[str]:
    if uids_arg:
        return [u.strip() for u in uids_arg.split(",") if u.strip()]
    if run_all:
        return [str(u) for u in df["uid"].tolist()]
    return list(SMOKE_UIDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end OfficeQA eval (smoke by default)")
    parser.add_argument("--csv", required=True, help="Path to officeqa_pro.csv")
    parser.add_argument("--report", required=True, help="Output CSV report path")
    parser.add_argument("--plan-cache-csv", default=SkunkConfig.from_env().plan_cache_csv,
                        help="Plan cache CSV (default: %(default)s)")
    parser.add_argument("--all", action="store_true", help="Run every UID in --csv (default: smoke set only)")
    parser.add_argument("--uids", help="Comma-separated UIDs (overrides --smoke / --all)")
    parser.add_argument("--golden", action="store_true",
                        help="Inject golden pages from --csv instead of running retrieve")
    parser.add_argument("--trace-dir", default="eval/traces",
                        help="Per-question debug trace directory (default: %(default)s; '' to disable)")
    parser.add_argument("--verbose", action="store_true",
                        help="Live-print orchestrator + subagent events to stdout per question")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df_by_uid = df.set_index("uid")

    uids = _pick_uids(df, args.all, args.uids)
    print(f"[e2e] Running {len(uids)} UID(s){' (smoke)' if not args.all else ''}")

    golden_lookup = load_golden(args.csv) if args.golden else None
    plan_cache = load_plan_cache(args.plan_cache_csv)
    if not plan_cache:
        print(f"[e2e] WARNING: plan cache {args.plan_cache_csv!r} is empty or missing — "
              f"will fall back to live LLM planner per question", file=sys.stderr)

    rows: list[dict] = []
    for uid in uids:
        if uid not in df_by_uid.index:
            print(f"[e2e] WARNING: {uid!r} not found in {args.csv}", file=sys.stderr)
            continue
        row = df_by_uid.loc[uid]
        question = str(row["question"])
        gold_answer = row.get("answer")
        gold_answer = "" if pd.isna(gold_answer) else str(gold_answer)

        print(f"\n{'='*60}\nUID: {uid}\nQ: {question}")

        golden_pages = None
        if golden_lookup is not None:
            golden_pages = golden_lookup.get(uid, [])
            if not golden_pages:
                print(f"[e2e] WARNING: no golden pages for {uid!r}")

        cached_plan_text = plan_cache.get(uid)
        if cached_plan_text is None:
            print(f"[e2e] WARNING: no cached plan for {uid!r}, falling back to LLM planner")

        trace_path = None
        if args.trace_dir:
            trace_path = str(Path(args.trace_dir) / f"{uid}.txt")

        try:
            result = run_question(
                question=question,
                verbose=args.verbose,
                golden_pages=golden_pages,
                cached_plan_text=cached_plan_text,
                uid=uid,
                plan_cache_csv=args.plan_cache_csv,
                trace_path=trace_path,
            )
        except Exception as e:
            result = {"question": question, "answer": None, "failed": True,
                      "reason": f"harness crash: {type(e).__name__}: {e}", "n_steps": 0}

        if result["failed"]:
            print(f"FAILED: {result['reason']}")
        else:
            print(f"Answer: {result['answer']}")

        rows.append({
            "uid": uid,
            "question": question,
            "predicted": result["answer"] if not result["failed"] else "",
            "gold_answer": gold_answer,
            "failed": result["failed"],
            "reason": result["reason"] or "",
            "n_steps": result.get("n_steps", 0),
        })

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_total = len(rows)
    n_failed = sum(1 for r in rows if r["failed"])
    print(f"\n[e2e] Wrote {out} ({n_total} rows)")
    print(f"[e2e] Summary: {n_total - n_failed}/{n_total} produced an answer "
          f"(no scoring — see {out} for predicted vs gold_answer)")


if __name__ == "__main__":
    main()
