"""End-to-end eval. Runs the full pipeline (planner → orchestrator → 6 subagents)
per question and writes predicted vs gold answers to a CSV report.

Scoring is intentionally out of scope here — the report is the artifact a downstream
scorer consumes. The gap between this report and eval_extraction quantifies retrieval
cost; the gap to the gold answer floor quantifies extraction cost.

Usage
-----
  # All UIDs in the CSV (default)
  python -m eval.eval_e2e --csv data/officeqa_pro.csv --report eval/e2e_report.csv

  # Sample 10 random UIDs
  python -m eval.eval_e2e --csv data/officeqa_pro.csv --report eval/e2e_report.csv --sample 10

  # Sample 10% of UIDs
  python -m eval.eval_e2e --csv data/officeqa_pro.csv --report eval/e2e_report.csv --sample 10%

  # Run only specific UIDs
  python -m eval.eval_e2e --csv data/officeqa_pro.csv --report eval/e2e_report.csv \\
      --uids UID0001,UID0030 --golden

  # Suppress verbose output
  python -m eval.eval_e2e --csv data/officeqa_pro.csv --report eval/e2e_report.csv --quiet

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

Pass `--trace-dir ''` to disable. Pass `--quiet` to suppress live stdout streaming.

Gemini rate limiting + retries
------------------------------
`LLMClient` (src/skunk/common/llm.py) paces all Gemini calls through a
process-wide token bucket sized by `SKUNK_GEMINI_RPM` (default 1000 to match
the Flash 2.5 paid-tier quota). On any error from the SDK, the call retries
with exponential backoff (start 50ms, doubling, capped at 1s, up to 10 retries)
and logs each failure to stderr. If you see 429s persistently, lower the rpm.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
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

from skunk.config import SkunkConfig  # noqa: E402
from skunk.dsl import PageRef  # noqa: E402
from skunk.run import load_plan_cache, run_question  # noqa: E402

# ---------------------------------------------------------------------------
# Golden page parsing (source_docs URLs → PageRefs)
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}

_URL_RE = re.compile(
    r"/(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)"
    r"-(?P<year>\d{4})[^?]*\?page=(?P<page>\d+)",
    re.IGNORECASE,
)


def _parse_source_docs(source_docs: str) -> list[PageRef]:
    """Extract every (year, month, page) tuple from a source_docs cell."""
    out: list[PageRef] = []
    if not isinstance(source_docs, str):
        return out
    for m in _URL_RE.finditer(source_docs):
        month_mm = _MONTH_MAP[m.group("month").lower()]
        out.append(PageRef(month=f"{m.group('year')}-{month_mm}", page=int(m.group("page"))))
    return out


def load_golden(csv_path: str | Path) -> dict[str, list[PageRef]]:
    """Map uid → list[PageRef] for every question in the benchmark CSV."""
    df = pd.read_csv(csv_path)
    return {row["uid"]: _parse_source_docs(str(row.get("source_docs", "")))
            for _, row in df.iterrows()}

REPORT_FIELDS = ["uid", "question", "predicted", "gold_answer", "failed", "reason", "n_steps"]


def _pick_uids(df: pd.DataFrame, sample: str | None, uids_arg: str | None) -> list[str]:
    if uids_arg:
        return [u.strip() for u in uids_arg.split(",") if u.strip()]
    all_uids = [str(u) for u in df["uid"].tolist()]
    if sample is None:
        return all_uids
    if sample.endswith("%"):
        n = max(1, round(len(all_uids) * float(sample[:-1]) / 100))
    else:
        val = float(sample)
        n = max(1, round(len(all_uids) * val)) if val < 1.0 else int(val)
    return random.sample(all_uids, min(n, len(all_uids)))


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end OfficeQA eval (all UIDs by default)")
    parser.add_argument("--csv", required=True, help="Path to officeqa_pro.csv")
    parser.add_argument("--report", required=True, help="Output CSV report path")
    parser.add_argument("--plan-cache-csv", default=SkunkConfig.from_env().plan_cache_csv,
                        help="Plan cache CSV (default: %(default)s)")
    parser.add_argument("--sample",
                        help="Run a random subset: integer count (e.g. '10') or percentage (e.g. '10%%')")
    parser.add_argument("--uids", help="Comma-separated UIDs (overrides --sample)")
    parser.add_argument("--golden", action="store_true",
                        help="Inject golden pages from --csv instead of running retrieve")
    parser.add_argument("--golden-noisy", action="store_true",
                        help="Like --golden, but per-page Bernoulli(noise_prob) appends one confounder PageRef "
                             "(same-bulletin drift or cross-year keyword match from --noise-pool)")
    parser.add_argument("--noise-prob", type=float, default=0.5,
                        help="Per-golden-page probability of appending a confounder (default: %(default)s)")
    parser.add_argument("--noise-seed", type=int, default=42,
                        help="Seed for deterministic confounder selection (default: %(default)s)")
    parser.add_argument("--noise-pool", default="eval/noise_pool.json",
                        help="Path to precomputed cross-year confounder pool (default: %(default)s)")
    parser.add_argument("--trace-dir", default="eval/traces",
                        help="Per-question debug trace directory (default: %(default)s; '' to disable)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress live orchestrator + subagent events (verbose is on by default)")
    args = parser.parse_args()

    if args.golden and args.golden_noisy:
        parser.error("--golden and --golden-noisy are mutually exclusive")

    df = pd.read_csv(args.csv)
    df_by_uid = df.set_index("uid")

    uids = _pick_uids(df, args.sample, args.uids)
    sample_note = f" (sample: {args.sample})" if args.sample and not args.uids else ""
    print(f"[e2e] Running {len(uids)} UID(s){sample_note}")

    golden_lookup = load_golden(args.csv) if (args.golden or args.golden_noisy) else None
    noise_pool = None
    if args.golden_noisy:
        from eval.noise import load_noise_pool
        noise_pool = load_noise_pool(args.noise_pool)
    plan_cache = load_plan_cache(args.plan_cache_csv)
    if not plan_cache:
        print(f"[e2e] WARNING: plan cache {args.plan_cache_csv!r} is empty or missing — "
              f"will fall back to live LLM planner per question", file=sys.stderr)

    verbose = not args.quiet
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
            elif args.golden_noisy:
                from eval.noise import make_noisy_pages
                pdf_dir = os.environ.get(
                    "OFFICEQA_PDF_DIR",
                    str(Path.home() / "Desktop/officeqa/treasury_bulletin_pdfs"),
                )
                golden_pages = make_noisy_pages(
                    uid, golden_pages,
                    noise_prob=args.noise_prob,
                    seed=args.noise_seed,
                    pdf_dir=pdf_dir,
                    pool=noise_pool,
                )

        cached_plan_text = plan_cache.get(uid)
        if cached_plan_text is None:
            print(f"[e2e] WARNING: no cached plan for {uid!r}, falling back to LLM planner")

        trace_path = None
        if args.trace_dir:
            trace_path = str(Path(args.trace_dir) / f"{uid}.txt")

        try:
            result = run_question(
                question=question,
                verbose=verbose,
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
