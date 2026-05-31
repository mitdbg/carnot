"""End-to-end eval. Runs the full pipeline (planner → orchestrator → 4 operators)
per question and writes predicted vs gold answers to a CSV report.

Scoring is intentionally out of scope here — the report is the artifact a
downstream scorer consumes.

Usage
-----
  # All UIDs in the CSV (default)
  python -m eval.eval_e2e --csv data/officeqa_pro.csv --report eval/e2e_report.csv

  # Sample 10 random UIDs
  python -m eval.eval_e2e --csv data/officeqa_pro.csv --report eval/e2e_report.csv --sample 10

  # Run only specific UIDs, bypassing retrieve with golden pages
  python -m eval.eval_e2e --csv data/officeqa_pro.csv --report eval/e2e_report.csv \\
      --uids UID0001,UID0030 --golden

`--golden` parses `source_docs?page=N` URLs from --csv and injects them as
PageRefs, so extract/compute run on exactly the pages the benchmark deems
relevant. Use it to measure the extract+compute ceiling without retrieval cost.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


# .env must be loaded before importing skunk so LLMClient sees the API keys.
_load_env(REPO_ROOT / ".env")

from skunk import (  # noqa: E402
    HarnessContext,
    MissingData,
    Orchestrator,
    PageRef,
    SkunkConfig,
    load_prompt_overrides,
)

from eval.util import dump_trace  # noqa: E402

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


# ---------------------------------------------------------------------------
# Per-question runner
# ---------------------------------------------------------------------------

def _run_one_question(
    question: str,
    *,
    verbose: bool,
    uid: str | None = None,
    golden_pages: list[PageRef] | None = None,
    trace_path: str | None = None,
) -> dict:
    """Run one question through the Orchestrator and collect a result dict."""
    if golden_pages:
        # Normalize to fresh PageRef instances regardless of input shape.
        golden_pages = [PageRef(month=g.month, page=g.page) for g in golden_pages]
        if verbose:
            print(f"[e2e] Golden pages: {len(golden_pages)} → {[str(r) for r in golden_pages]}")

    config = SkunkConfig.from_env()
    config.golden_pages = golden_pages

    overrides_path = Path(config.prompt_overrides_path)
    prompt_overrides = load_prompt_overrides(overrides_path) if overrides_path.exists() else ()

    ctx = HarnessContext(
        question=question,
        verbose=verbose,
        config=config,
        prompt_overrides=prompt_overrides,
    )

    if verbose:
        print(f"\n[e2e] Planning: {question[:80]}...")

    t0 = time.perf_counter()
    orch = Orchestrator(ctx)
    missing: MissingData | None = None
    try:
        orch.execute()
    except MissingData as e:
        # compute reported insufficient data after the recovery budget; record
        # as a failed row and keep the batch going. Anything else propagates.
        missing = e
    wall_s = time.perf_counter() - t0
    trace = orch.trace
    plan_obj = orch.current_plan

    if verbose:
        if plan_obj is not None:
            print(f"[e2e] Plan: {plan_obj.model_dump_json()}")
        print(trace.pretty())
        print(f"[e2e] wall_s={wall_s:.3f}")

    if trace_path is not None:
        plan_dump = plan_obj.model_dump_json() if plan_obj else "(unavailable)"
        dump_trace(trace_path, uid=uid, question=question, plan_text=plan_dump,
                   golden_pages=golden_pages, trace=trace, events=ctx.events,
                   model=config.llm_model)

    if missing is not None:
        return {"question": question, "answer": None, "failed": True,
                "reason": f"MissingData: {missing.reason}", "n_steps": len(trace.steps)}
    return {
        "question": question,
        "answer": trace.answer,
        "failed": trace.failed,
        "reason": trace.failure_reason,
        "n_steps": len(trace.steps),
    }


REPORT_FIELDS = ["uid", "question", "predicted", "gold_answer", "failed", "reason", "n_steps"]

# Held-out test set — see CLAUDE.md. Loaded lazily so the file is optional.
_TEST_SET_PATH = REPO_ROOT / "eval" / "test_set_uids.json"


def _load_test_set() -> set[str]:
    if not _TEST_SET_PATH.exists():
        return set()
    import json
    with _TEST_SET_PATH.open() as f:
        return set(json.load(f).get("uids", []))


def _pick_uids(
    df: pd.DataFrame,
    sample: int | None,
    uids_arg: str | None,
    exclude: set[str] | None = None,
) -> list[str]:
    if uids_arg:
        return [u.strip() for u in uids_arg.split(",") if u.strip()]
    all_uids = [str(u) for u in df["uid"].tolist()]
    if exclude:
        all_uids = [u for u in all_uids if u not in exclude]
    if sample is None:
        return all_uids
    return random.sample(all_uids, min(sample, len(all_uids)))


def main() -> None:
    # Force line-buffered stdout/stderr so live operator/LLM events stream to
    # logs and `tail -f` in real time. Without this, Python block-buffers when
    # stdout is redirected to a file (~4–8KB chunks), making the harness look
    # stalled mid-question even though it's working.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="End-to-end OfficeQA eval (all UIDs by default)")
    parser.add_argument("--csv", required=True, help="Path to officeqa_pro.csv")
    parser.add_argument("--report", required=True, help="Output CSV report path")
    parser.add_argument("--sample", type=int,
                        help="Run a random subset of N UIDs")
    parser.add_argument("--uids", help="Comma-separated UIDs (overrides --sample)")
    parser.add_argument("--golden", action="store_true",
                        help="Inject golden pages from --csv instead of running retrieve")
    parser.add_argument("--trace-dir", default="eval/traces",
                        help="Per-question debug trace directory (default: %(default)s; '' to disable)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress live orchestrator + operator events (verbose is on by default)")
    parser.add_argument("--include-test-set", action="store_true",
                        help="Include the held-out test-set UIDs (see CLAUDE.md). Default is to exclude "
                             "them — only opt in for a deliberate final-number measurement.")
    parser.add_argument("--workers", type=int, default=32,
                        help="UID-level concurrency (default: %(default)s). Each worker runs one "
                             "_run_one_question independently; the process-level LLM rate limiter "
                             "throttles cross-worker traffic to config.llm_rpm.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df_by_uid = df.set_index("uid")

    test_set = _load_test_set()

    # ABORT (not warn) when --uids names test UIDs without --include-test-set.
    # Per CLAUDE.md: there is no legitimate reason for an explicit UID list
    # from the command line to hit the test set unless --include-test-set is
    # set. Random samples are filtered silently below (the user didn't ask
    # for those specific UIDs).
    if args.uids and test_set and not args.include_test_set:
        requested = {u.strip() for u in args.uids.split(",") if u.strip()}
        collision = sorted(requested & test_set)
        if collision:
            print(
                f"[e2e] ABORT: --uids names {len(collision)} held-out test-set UID(s): "
                f"{', '.join(collision)}. "
                f"Pass --include-test-set to override (see CLAUDE.md).",
                file=sys.stderr,
            )
            sys.exit(2)

    # Exclude test UIDs from the pool BEFORE sampling so --sample N returns N
    # dev UIDs (rather than N minus however many test UIDs happened to be drawn).
    exclude = test_set if (test_set and not args.include_test_set) else None
    if exclude and not args.uids:
        print(f"[e2e] Excluded {len(exclude)} held-out test-set UID(s) from the pool. "
              f"Pass --include-test-set to override (see CLAUDE.md).", file=sys.stderr)
    uids = _pick_uids(df, args.sample, args.uids, exclude=exclude)

    if test_set and args.include_test_set:
        print(f"[e2e] WARNING: --include-test-set is on; the held-out {len(test_set)}-UID test set "
              f"is in play. Only use this for final-number measurement, not iterative tuning.",
              file=sys.stderr)

    sample_note = f" (sample: {args.sample})" if args.sample and not args.uids else ""
    print(f"[e2e] Running {len(uids)} UID(s){sample_note} with --workers {args.workers}")

    golden_lookup = load_golden(args.csv) if args.golden else None

    verbose = not args.quiet

    def process_uid(uid: str) -> dict | None:
        """Run one UID end-to-end. Returns a row dict, or None if the UID
        is missing from the CSV (skip-with-warning, not fatal). Catches
        and records uncaught exceptions so one runaway UID doesn't kill
        the batch."""
        if uid not in df_by_uid.index:
            print(f"[e2e] WARNING: {uid!r} not found in {args.csv}", file=sys.stderr)
            return None
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

        trace_path = None
        if args.trace_dir:
            trace_path = str(Path(args.trace_dir) / f"{uid}.txt")

        try:
            result = _run_one_question(
                question=question,
                verbose=verbose,
                golden_pages=golden_pages,
                uid=uid,
                trace_path=trace_path,
            )
        except Exception as e:
            import traceback as _tb
            tb_str = _tb.format_exc()
            print(f"[e2e] ABORTED UID {uid}: {type(e).__name__}: {e}", file=sys.stderr)
            if trace_path is not None:
                try:
                    Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(trace_path + ".failed").write_text(
                        f"UID: {uid}\nQ: {question}\n\nUncaught exception:\n{tb_str}\n"
                    )
                except Exception:
                    pass
            result = {"question": question, "answer": None, "failed": True,
                     "reason": f"Uncaught: {type(e).__name__}: {e}", "n_steps": 0}

        if result["failed"]:
            print(f"[e2e] {uid} FAILED: {result['reason']}")
        else:
            print(f"[e2e] {uid} Answer: {result['answer']}")

        return {
            "uid": uid,
            "question": question,
            "predicted": result["answer"] if not result["failed"] else "",
            "gold_answer": gold_answer,
            "failed": result["failed"],
            "reason": result["reason"] or "",
            "n_steps": result.get("n_steps", 0),
        }

    # Pre-allocate slots so the output CSV preserves the input UID order
    # regardless of completion order under --workers > 1. The process-level
    # rate limiter inside LLMClient throttles cross-worker traffic.
    results: list[dict | None] = [None] * len(uids)
    if args.workers <= 1:
        for i, uid in enumerate(uids):
            results[i] = process_uid(uid)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_uid, uid): i for i, uid in enumerate(uids)}
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()

    rows = [r for r in results if r is not None]

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
