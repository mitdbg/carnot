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

  # Use golden pages (skips retrieve operator)
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
import time
from pathlib import Path
from typing import TYPE_CHECKING

from skunk.config import SkunkConfig

if TYPE_CHECKING:
    from skunk.plan import PageRef

SMOKE_UIDS = ["UID0001", "UID0030", "UID0010", "UID0022"]


# ---------------------------------------------------------------------------
# Plan cache helpers
# ---------------------------------------------------------------------------

def load_plan_cache(plan_cache_csv: str) -> dict[str, str]:
    """Return {uid: plan_json_string} from the plan cache CSV."""
    p = Path(plan_cache_csv)
    if not p.exists():
        return {}
    with p.open(newline="", encoding="utf-8") as f:
        return {row["uid"]: row["plan_json"] for row in csv.DictReader(f) if row.get("plan_json")}


def save_plan_to_cache(uid: str, question: str, plan_json: str, plan_cache_csv: str) -> None:
    """Upsert (uid, plan_json) into the plan cache CSV."""
    p = Path(plan_cache_csv)
    p.parent.mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict] = {}
    if p.exists():
        with p.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row["uid"]] = row

    rows[uid] = {"uid": uid, "question": question, "plan_json": plan_json}

    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "question", "plan_json"])
        writer.writeheader()
        writer.writerows(rows.values())


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_question(
    question: str,
    verbose: bool,
    golden_pages: list["PageRef"] | None = None,
    cached_plan_json: str | None = None,
    uid: str | None = None,
    plan_cache_csv: str | None = None,
    trace_path: str | None = None,
) -> dict:
    from skunk.common import HarnessContext
    from skunk.orchestrator import execute
    from skunk.plan import PlannerExecutor, from_json, to_json, validate

    if golden_pages:
        from skunk.plan import PageRef
        # Re-construct PageRef instances so callers can pass plain (month, page)
        # tuple-shaped values without depending on the plan module themselves.
        golden_pages = [PageRef(month=g.month, page=g.page) for g in golden_pages]
        if verbose:
            print(f"[run] Golden pages: {len(golden_pages)} → {[str(r) for r in golden_pages]}")

    config = SkunkConfig.from_env()
    config.golden_pages = golden_pages
    csv_path = plan_cache_csv or config.plan_cache_csv

    from pathlib import Path as _Path

    from skunk.prompt_overrides import load_prompt_overrides
    overrides_path = _Path(config.prompt_overrides_path)
    prompt_overrides = load_prompt_overrides(overrides_path) if overrides_path.exists() else ()

    ctx = HarnessContext(
        question=question,
        verbose=verbose,
        config=config,
        prompt_overrides=prompt_overrides,
    )

    t0 = time.perf_counter()

    if cached_plan_json is not None:
        if verbose:
            print("\n[run] Using cached plan...")
        try:
            plan_obj = from_json(cached_plan_json)
            result = validate(plan_obj)
            if not result.ok:
                return {"question": question, "answer": None, "failed": True,
                        "reason": f"cached plan validation failed: {result.errors}"}
            if verbose:
                print(f"[run] Plan: {to_json(plan_obj)}")
        except Exception as e:
            return {"question": question, "answer": None, "failed": True, "reason": f"cached plan error: {e}"}
    else:
        if verbose:
            print(f"\n[run] Planning: {question[:80]}...")
        try:
            plan_obj = PlannerExecutor().plan(question, ctx)
            plan_json = to_json(plan_obj)
            if verbose:
                print(f"[run] Plan: {plan_json}")
            if uid is not None:
                save_plan_to_cache(uid, question, plan_json, csv_path)
                if verbose:
                    print(f"[run] Plan cached for {uid}")
        except Exception as e:
            return {"question": question, "answer": None, "failed": True, "reason": f"planning: {e}"}

    trace = execute(plan_obj, ctx)
    wall_s = time.perf_counter() - t0
    ctx.emit("run", "question done", wall_s=round(wall_s, 3))

    if verbose:
        print(trace.pretty())

    if trace_path is not None:
        try:
            plan_dump = to_json(plan_obj)
        except Exception:
            plan_dump = cached_plan_json or "(unavailable)"
        _dump_trace(trace_path, uid=uid, question=question, plan_text=plan_dump,
                    golden_pages=golden_pages, trace=trace, events=ctx.events,
                    model=config.llm_model)

    return {
        "question": question,
        "answer": trace.answer,
        "failed": trace.failed,
        "reason": trace.failure_reason,
        "n_steps": len(trace.steps),
    }


def _dump_trace(path: str, *, uid: str | None, question: str, plan_text: str,
                golden_pages: list["PageRef"] | None, trace, events: list[dict],
                model: str) -> None:
    """Write a comprehensive per-question debug trace to `path`."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"UID: {uid or '(none)'}")
    lines.append("=" * 80)
    lines.append(f"Question: {question}")
    lines.append(f"Model: {model}")
    lines.append(f"Plan: {plan_text}")
    if golden_pages:
        lines.append(f"Golden pages ({len(golden_pages)}):")
        for g in golden_pages:
            lines.append(f"  - month={g.month} page={g.page}")
    else:
        lines.append("Golden pages: (none)")
    lines.append("")
    lines.append(f"Outcome: {'FAILED — ' + (trace.failure_reason or '') if trace.failed else 'OK'}")
    lines.append(f"Final answer: {trace.answer!r}")
    lines.append("")

    events_per_step: dict[int, list[dict]] = {s.step_idx: [] for s in trace.steps}
    cur_idx: int | None = None
    for ev in events:
        if ev.get("source") == "_step" and ev.get("message") == "begin":
            cur_idx = int(ev.get("step_idx", 0))
            continue
        if cur_idx is not None and cur_idx in events_per_step:
            events_per_step[cur_idx].append(ev)

    for step in trace.steps:
        lines.append("-" * 80)
        lines.append(f"Step {step.step_idx}: {step.op}  args={step.args}  ({step.elapsed_s:.2f}s)")
        lines.append(f"  in:  {step.input_full}")
        if step.error:
            lines.append(f"  ERROR: {step.error}")
        else:
            lines.append(f"  out: {step.output_full}")
        evs = events_per_step.get(step.step_idx, [])
        if evs:
            lines.append("  events:")
            for ev in evs:
                src = ev.get("source", "")
                msg = ev.get("message", "")
                extras = {k: v for k, v in ev.items() if k not in {"source", "message"}}
                lines.append(f"    [{src}] {msg}")
                for k, v in extras.items():
                    s = repr(v)
                    # Keep full LLM I/O for performance/cost analysis; truncate everything else.
                    if not (src == "llm" and k in ("input_text", "output_text")) and len(s) > 800:
                        s = s[:800] + "...(truncated)"
                    lines.append(f"      {k}: {s}")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


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
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", help="Write results to JSON file")
    parser.add_argument(
        "--golden", action="store_true",
        help="Inject golden pages from --csv instead of running retrieve operator. "
             "Requires --uid/--uids/--smoke and --csv."
    )
    parser.add_argument(
        "--cached-plan", action="store_true",
        help="Use cached plan from --plan-cache-csv instead of calling the LLM planner."
    )
    _default_plan_cache_csv = SkunkConfig.from_env().plan_cache_csv
    parser.add_argument(
        "--plan-cache-csv", default=_default_plan_cache_csv,
        help=f"Path to plan cache CSV (default: {_default_plan_cache_csv})."
    )
    args = parser.parse_args()

    if args.golden and args.question:
        parser.error("--golden requires --uid/--uids/--smoke (cannot use with --question)")
    if args.golden and not args.csv:
        parser.error("--golden requires --csv to load golden pages")

    golden_lookup: dict | None = None
    if args.golden:
        try:
            from eval.eval_e2e import load_golden
            golden_lookup = load_golden(args.csv)
        except Exception as e:
            print(f"[run] ERROR loading golden pages: {e}", file=sys.stderr)
            sys.exit(1)

    plan_cache: dict[str, str] = {}
    if args.cached_plan:
        plan_cache = load_plan_cache(args.plan_cache_csv)
        if not plan_cache:
            print(f"[run] WARNING: plan cache {args.plan_cache_csv!r} is empty or missing", file=sys.stderr)

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

        cached_plan_json = None
        if args.cached_plan and uid is not None:
            cached_plan_json = plan_cache.get(uid)
            if cached_plan_json is None:
                print(f"[run] WARNING: no cached plan for {uid!r}, falling back to LLM planner")

        result = run_question(
            question=question,
            verbose=args.verbose,
            golden_pages=golden_pages,
            cached_plan_json=cached_plan_json,
            uid=uid,
            plan_cache_csv=args.plan_cache_csv,
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
