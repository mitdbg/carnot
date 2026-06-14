"""Build a practice-harness questions JSON from the OfficeQA benchmark.

The local Cup practice server (`cup_kit.practice_server`) serves a
`{"rounds":[{"round_num","questions":[{question_id,prompt,canonical_answer}]}]}` file and
scores submissions against `canonical_answer`. This script samples N **dev** UIDs from
`data/officeqa_pro.csv` (question→prompt, answer→canonical_answer, uid→question_id) and writes
that file for a single round.

🚨 Held-out test set: the 32 UIDs in `eval/test_set_uids.json` are EXCLUDED from sampling, and
an explicit `--uids` list that intersects the test set ABORTS (see skunk/CLAUDE.md). This keeps a
casual "give it a spin" run off the held-out set.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKUNK_ROOT = HERE.parent
BENCH_CSV = SKUNK_ROOT / "data" / "officeqa_pro.csv"
TEST_SET_PATH = SKUNK_ROOT / "eval" / "test_set_uids.json"


def _load_test_set() -> set[str]:
    if not TEST_SET_PATH.exists():
        return set()
    return set(json.loads(TEST_SET_PATH.read_text()).get("uids", []))


def _load_rows() -> dict[str, dict[str, str]]:
    with BENCH_CSV.open() as handle:
        return {row["uid"]: row for row in csv.DictReader(handle)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num", type=int, default=4, help="how many dev UIDs to sample")
    parser.add_argument("--uids", default="", help="explicit comma-separated UIDs (overrides --num)")
    parser.add_argument("--seed", type=int, default=None, help="sampling seed (default: nondeterministic)")
    parser.add_argument("--round-num", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True, help="output questions JSON path")
    args = parser.parse_args(argv)

    rows = _load_rows()
    test_set = _load_test_set()

    if args.uids:
        uids = [u.strip() for u in args.uids.split(",") if u.strip()]
        unknown = [u for u in uids if u not in rows]
        if unknown:
            print(f"ERROR: unknown UIDs not in benchmark: {unknown}", file=sys.stderr)
            return 2
        # Mirror eval_e2e: an explicit list touching the held-out set ABORTS (not warns).
        contaminated = sorted(set(uids) & test_set)
        if contaminated:
            print(
                f"ERROR: refusing to run — these UIDs are in the held-out test set: {contaminated}\n"
                f"See skunk/CLAUDE.md. Sample from dev only, or pick different UIDs.",
                file=sys.stderr,
            )
            return 2
    else:
        dev_uids = sorted(set(rows) - test_set)
        if args.num > len(dev_uids):
            print(f"ERROR: asked for {args.num} but only {len(dev_uids)} dev UIDs exist", file=sys.stderr)
            return 2
        rng = random.Random(args.seed)
        uids = sorted(rng.sample(dev_uids, args.num))

    questions = [
        {"question_id": uid, "prompt": rows[uid]["question"], "canonical_answer": rows[uid]["answer"]}
        for uid in uids
    ]
    payload = {"rounds": [{"round_num": args.round_num, "questions": questions}]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"wrote {len(questions)} questions → {args.out}")
    for q in questions:
        print(f"  {q['question_id']}: {q['prompt'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
