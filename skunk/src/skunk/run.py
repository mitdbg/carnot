"""CLI entrypoint for the OfficeQA agentic harness.

Usage examples:

  # Run a single question (calls LLM planner, updates plan cache)
  python -m skunk.run \\
      --question "Total US National Defense expenditure in CY 1940, millions nominal USD" \\
      --verbose

  # Run by UID from the annotated CSV (calls planner, updates cache)
  python -m skunk.run \\
      --uid UID0001 \\
      --csv data/officeqa_pro.csv

  # Run by UID using cached plan (skips LLM planner call)
  python -m skunk.run \\
      --uid UID0001 \\
      --csv data/officeqa_pro.csv \\
      --cached-plan --verbose

  # Use golden pages (skips retrieve subagent)
  python -m skunk.run \\
      --uid UID0001 \\
      --csv data/officeqa_pro.csv \\
      --golden --verbose

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SMOKE_UIDS = ["UID0001", "UID0030", "UID0010", "UID0022"]
DEFAULT_PLAN_CSV = "data/dsl_planning_pass.csv"


# ---------------------------------------------------------------------------
# Plan cache helpers
# ---------------------------------------------------------------------------

def load_plan_cache(plan_csv: str) -> dict[str, str]:
    """Return {uid: plan_text} from the plan cache CSV."""
    p = Path(plan_csv)
    if not p.exists():
        return {}
    with p.open(newline="", encoding="utf-8") as f:
        return {row["uid"]: row["plan_text"] for row in csv.DictReader(f) if row.get("plan_text")}


def save_plan_to_cache(uid: str, question: str, plan_text: str, plan_csv: str) -> None:
    """Upsert (uid, plan_text) into the plan cache CSV."""
    p = Path(plan_csv)
    p.parent.mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict] = {}
    if p.exists():
        with p.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row["uid"]] = row

    rows[uid] = {"uid": uid, "question": question, "plan_text": plan_text}

    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "question", "plan_text"])
        writer.writeheader()
        writer.writerows(rows.values())


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_question(
    question: str,
    manifest_path: str | None,
    cache_dir: str,
    verbose: bool,
    golden_pages: list | None = None,
    cached_plan_text: str | None = None,
    uid: str | None = None,
    plan_csv: str = DEFAULT_PLAN_CSV,
) -> dict:
    from skunk.subagents.base import HarnessContext
    from skunk.planner import plan
    from skunk.orchestrator import execute

    golden_handle = None
    if golden_pages:
        from skunk.dsl import DocHandle, PageRef
        refs = [PageRef(year=g.year, month=g.bulletin_id, page=g.page) for g in golden_pages]
        golden_handle = DocHandle(refs=refs, desc=f"golden ({len(refs)} pages)")
        if verbose:
            print(f"[run] Golden handle: {len(refs)} pages → {[str(r) for r in refs]}")

    ctx = HarnessContext(
        question=question,
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        golden_handle=golden_handle,
    )

    if cached_plan_text is not None:
        if verbose:
            print(f"\n[run] Using cached plan...")
        try:
            from skunk.dsl import parse, serialize, validate
            chain = parse(cached_plan_text)
            result = validate(chain)
            if not result.ok:
                raise ValueError(f"Cached plan failed validation: {result.errors}")
            if verbose:
                print(f"[run] Plan: {serialize(chain)}")
        except Exception as e:
            return {"question": question, "answer": None, "failed": True, "reason": f"cached plan error: {e}"}
    else:
        if verbose:
            print(f"\n[run] Planning: {question[:80]}...")
        try:
            from skunk.dsl import serialize
            chain = plan(question, ctx)
            plan_text = serialize(chain)
            if verbose:
                print(f"[run] Plan: {plan_text}")
            if uid is not None:
                save_plan_to_cache(uid, question, plan_text, plan_csv)
                if verbose:
                    print(f"[run] Plan cached for {uid}")
        except Exception as e:
            return {"question": question, "answer": None, "failed": True, "reason": f"planning: {e}"}

    trace = execute(chain, ctx)

    if verbose:
        print(trace.pretty())

    return {
        "question": question,
        "answer": trace.answer,
        "failed": trace.failed,
        "reason": trace.failure_reason,
        "n_steps": len(trace.steps),
    }


def load_question_from_csv(csv_path: str, uid: str) -> str:
    import pandas as pd
    df = pd.read_csv(csv_path)
    rows = df[df["uid"] == uid]
    if rows.empty:
        raise ValueError(f"UID {uid!r} not found in {csv_path}")
    return str(rows.iloc[0]["question"])


def main() -> None:
    parser = argparse.ArgumentParser(description="OfficeQA agentic harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question", help="Raw question text")
    group.add_argument("--uid", help="UID from annotated CSV")
    group.add_argument("--uids", help="Comma-separated UIDs")
    group.add_argument("--smoke", action="store_true", help=f"Run smoke UIDs: {SMOKE_UIDS}")

    parser.add_argument("--csv", help="Path to annotated OfficeQA CSV (needed for --uid/--uids/--smoke)")
    parser.add_argument("--manifest", help="Path to manifest.csv", default=None)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", help="Write results to JSON file")
    parser.add_argument(
        "--golden", action="store_true",
        help="Inject golden pages from --csv instead of running retrieve subagent. "
             "Requires --uid/--uids/--smoke and --csv."
    )
    parser.add_argument(
        "--cached-plan", action="store_true",
        help="Use cached plan from --plan-csv instead of calling the LLM planner."
    )
    parser.add_argument(
        "--plan-csv", default=DEFAULT_PLAN_CSV,
        help=f"Path to plan cache CSV (default: {DEFAULT_PLAN_CSV})."
    )
    args = parser.parse_args()

    if args.golden and args.question:
        parser.error("--golden requires --uid/--uids/--smoke (cannot use with --question)")
    if args.golden and not args.csv:
        parser.error("--golden requires --csv to load golden pages")

    golden_lookup: dict | None = None
    if args.golden:
        try:
            from eval.golden import load_golden
            golden_lookup = load_golden(args.csv)
        except Exception as e:
            print(f"[run] ERROR loading golden pages: {e}", file=sys.stderr)
            sys.exit(1)

    plan_cache: dict[str, str] = {}
    if args.cached_plan:
        plan_cache = load_plan_cache(args.plan_csv)
        if not plan_cache:
            print(f"[run] WARNING: plan cache {args.plan_csv!r} is empty or missing", file=sys.stderr)

    # Build question list
    questions: list[tuple[str | None, str]] = []  # (uid, question_text)

    if args.question:
        questions = [(None, args.question)]
    elif args.smoke or args.uid or args.uids:
        if not args.csv:
            parser.error("--csv is required for --uid / --uids / --smoke")
        uids = SMOKE_UIDS if args.smoke else (
            [args.uid] if args.uid else [u.strip() for u in args.uids.split(",") if u.strip()]
        )
        for uid in uids:
            try:
                q = load_question_from_csv(args.csv, uid)
                questions.append((uid, q))
            except Exception as e:
                print(f"[run] WARNING: {e}", file=sys.stderr)

    results = []
    for uid, question in questions:
        print(f"\n{'='*60}")
        if uid:
            print(f"UID: {uid}")
        print(f"Q: {question}")

        golden_pages = None
        if golden_lookup is not None and uid is not None:
            golden_pages = golden_lookup.get(uid, [])
            if not golden_pages:
                print(f"[run] WARNING: no golden pages found for {uid!r}")

        cached_plan_text = None
        if args.cached_plan and uid is not None:
            cached_plan_text = plan_cache.get(uid)
            if cached_plan_text is None:
                print(f"[run] WARNING: no cached plan for {uid!r}, falling back to LLM planner")

        result = run_question(
            question=question,
            manifest_path=args.manifest,
            cache_dir=args.cache_dir,
            verbose=args.verbose,
            golden_pages=golden_pages,
            cached_plan_text=cached_plan_text,
            uid=uid,
            plan_csv=args.plan_csv,
        )
        if uid:
            result["uid"] = uid

        if result["failed"]:
            print(f"FAILED: {result['reason']}")
        else:
            print(f"Answer: {result['answer']}")

        results.append(result)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n[run] Results written to {args.output}")

    n_total = len(results)
    n_failed = sum(1 for r in results if r["failed"])
    print(f"\n[run] Summary: {n_total - n_failed}/{n_total} succeeded")


if __name__ == "__main__":
    main()
