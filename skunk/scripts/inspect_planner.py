"""Run the planner on N random dev UIDs and dump the resulting plans for
manual quality inspection. Excludes the held-out test set by default
(see CLAUDE.md).

Usage:
    python scripts/inspect_planner.py --n 16 --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from skunk.common import HarnessContext  # noqa: E402
from skunk.config import SkunkConfig  # noqa: E402
from skunk.dsl import to_json  # noqa: E402
from skunk.planner import PlannerOperator  # noqa: E402
from skunk.prompt_overrides import load_prompt_overrides  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=REPO_ROOT / "data" / "officeqa_pro.csv")
    ap.add_argument("--test-set", type=Path, default=REPO_ROOT / "eval" / "test_set_uids.json")
    ap.add_argument("--n", type=int, default=16, help="Number of dev UIDs to sample")
    ap.add_argument("--seed", type=int, default=1, help="Random sample seed")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Parallel planner LLM calls (default: %(default)s)")
    ap.add_argument("--uids", default=None,
                    help="Comma-separated UIDs to inspect (overrides --n / --seed sampling)")
    args = ap.parse_args()

    # Load + filter UIDs.
    df = pd.read_csv(args.csv).set_index("uid")
    all_uids = list(df.index.astype(str))
    test_uids: set[str] = set(json.loads(args.test_set.read_text()).get("uids", []))
    dev_uids = [u for u in all_uids if u not in test_uids]
    print(f"# {len(all_uids)} total UIDs, {len(test_uids)} held-out test, {len(dev_uids)} dev",
          file=sys.stderr)
    if any(u in test_uids for u in dev_uids):
        print("ERROR: dev set still contains test UIDs", file=sys.stderr)
        return 1

    if args.uids:
        picks = [u.strip() for u in args.uids.split(",") if u.strip()]
        bad = [u for u in picks if u in test_uids]
        if bad:
            print(f"ERROR: requested UIDs include held-out test set: {bad}", file=sys.stderr)
            return 1
        missing = [u for u in picks if u not in df.index]
        if missing:
            print(f"ERROR: UIDs not in CSV: {missing}", file=sys.stderr)
            return 1
        print(f"# Explicit UIDs ({len(picks)}): {picks}\n", file=sys.stderr)
    else:
        rng = random.Random(args.seed)
        picks = rng.sample(dev_uids, k=min(args.n, len(dev_uids)))
        picks.sort()
        print(f"# Sampled {len(picks)} UIDs (seed={args.seed}): {picks}\n", file=sys.stderr)

    # Set up planner context (mirrors run_question's setup, minus golden injection).
    config = SkunkConfig.from_env()
    overrides_path = Path(config.prompt_overrides_path)
    prompt_overrides = load_prompt_overrides(overrides_path) if overrides_path.exists() else ()
    planner = PlannerOperator()

    def _plan_one(uid: str) -> tuple[str, str, str, str | None, str | None]:
        row = df.loc[uid]
        question = str(row["question"])
        gold = str(row.get("answer", ""))
        ctx = HarnessContext(
            question=question, verbose=False, config=config,
            prompt_overrides=prompt_overrides,
        )
        try:
            plan = planner.plan(question, ctx)
            return (uid, question, gold, to_json(plan, indent=2), None)
        except Exception as e:
            return (uid, question, gold, None, str(e))

    import time
    n_ok = n_fail = 0
    results: dict[str, tuple] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(_plan_one, uid): uid for uid in picks}
        for fut in as_completed(futures):
            uid_, question, gold, plan_json, err = fut.result()
            results[uid_] = (question, gold, plan_json, err)
            status = "OK" if err is None else f"FAIL: {err[:80]}"
            elapsed = time.time() - t0
            print(f"[done t={elapsed:5.1f}s {len(results):2}/{len(picks)}] {uid_}: {status}",
                  file=sys.stderr, flush=True)
            if err is None:
                n_ok += 1
            else:
                n_fail += 1

    # Print in deterministic UID order regardless of completion order.
    for uid in picks:
        question, gold, plan_json, err = results[uid]
        print("=" * 80)
        print(uid)
        print(f"Q: {question}")
        print(f"GOLD: {gold[:200]}")
        if plan_json is not None:
            print("PLAN:")
            print(plan_json)
        else:
            print(f"PLAN: FAILED — {err}")
        print()

    print("=" * 80, file=sys.stderr)
    print(f"Done. ok={n_ok} fail={n_fail}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
