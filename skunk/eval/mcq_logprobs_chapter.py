"""MCQ self-consistency chapter pick experiment.

Gemini via OpenRouter doesn't expose token logprobs (logprobs.content is
None), so we approximate model confidence via **self-consistency
sampling**: run the same MCQ prompt N times at temperature > 0 and use
vote frequencies as P(chapter).

For each retrieve branch in the 32-UID dev set, prompts the LLM with a
multiple-choice menu (letters A..H mapped to the 8 real parent chapters)
N times. Reports:

  - top chapter
  - top-1 probability
  - top-2 margin
  - entropy
  - gold's chapter
  - whether the gold's chapter is in top-K

Specifically watch UID0042 to see if its known wrong-chapter pick
corresponds to a low-confidence (low margin / high entropy) call.

Usage:
    SKUNK_LLM_MODEL=google/gemini-3-flash-preview \\
    PYTHONPATH=src python3 eval/mcq_logprobs_chapter.py \\
        --n 32 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_REPO_ROOT = Path(__file__).resolve().parent.parent
_load_env(_REPO_ROOT / ".env")
sys.path.insert(0, str(_REPO_ROOT / "src"))

from openai import OpenAI  # noqa: E402

from skunk.config import SkunkConfig  # noqa: E402
from skunk.dsl import RetrieveBranch, parse as dsl_parse  # noqa: E402
from skunk.page_index.concept_tree import load_catalog  # noqa: E402
from skunk.page_index.retrieve_probe import (  # noqa: E402
    _PARENT_CHAPTER_MAP, load_concept_tree, parent_chapter_index,
)

_MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
           "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
           "november": 11, "december": 12}


def _parse_source_doc(url: str) -> tuple[str, int] | None:
    m = re.search(r"/([a-z]+)-(\d{4})-\d+\?page=(\d+)", url)
    if not m:
        return None
    mn = _MONTHS.get(m.group(1).lower())
    if mn is None:
        return None
    return (f"{m.group(2)}-{mn:02d}", int(m.group(3)))


def _load_benchmark(csv_path: Path) -> list[dict]:
    out: list[dict] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            goldens: list[tuple[str, int]] = []
            for url in (row.get("source_docs") or "").splitlines():
                ref = _parse_source_doc(url.strip())
                if ref:
                    goldens.append(ref)
            out.append({"uid": row["uid"], "question": row["question"],
                        "goldens": goldens})
    return out


def _load_plan_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[d["uid"]] = d["plan_text"]
    return out


_LETTERS = list("ABCDEFGH")


_SYSTEM = """You pick the SINGLE best Treasury Bulletin parent chapter for
a question. Answer with exactly one letter (A, B, C, ...) and nothing else."""


def _build_user_prompt(question: str, concept: str, period: str,
                        chapter_listing: list[tuple[str, int]]) -> str:
    menu = "\n".join(f"  {_LETTERS[i]}. {ch}  ({n} pages)"
                     for i, (ch, n) in enumerate(chapter_listing))
    return (
        f"Question: {question}\n\n"
        f"Concept: {concept}\n"
        f"Period:  {period}\n\n"
        f"Pick one chapter:\n{menu}\n\nAnswer: "
    )


def _gold_chapter(gold: tuple[str, int],
                  catalog_index: dict[tuple[str, int], dict]) -> str | None:
    row = catalog_index.get(gold)
    if row is None:
        return None
    raw = (row.section or "").strip()
    if not raw:
        return None
    # Tree's banner_rewrite + parent map
    return None  # placeholder; set up below


def main() -> int:
    ap = argparse.ArgumentParser(
        description="MCQ-with-logprobs chapter pick experiment.")
    ap.add_argument("--benchmark", type=Path,
                    default=_REPO_ROOT / "data/officeqa_pro.csv")
    ap.add_argument("--catalog-dir", type=Path,
                    default=_REPO_ROOT / "cache/page_index")
    ap.add_argument("--plan-cache", type=Path,
                    default=_REPO_ROOT / "cache/retrieve_bench_plans.jsonl")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--uids", type=str, default=None,
                    help="Comma-separated UIDs to run instead of sampling.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=8,
                    help="Number of MCQ samples per branch (self-consistency).")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "cache/mcq_logprobs_results.jsonl")
    args = ap.parse_args()

    cfg = SkunkConfig.from_env()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2
    client = OpenAI(api_key=api_key,
                    base_url="https://openrouter.ai/api/v1")
    model = cfg.llm_model
    print(f"Model: {model}    samples/branch: {args.n_samples}    "
          f"temperature: {args.temperature}")

    tree = load_concept_tree(args.catalog_dir / "concept_tree.json")
    chapters = parent_chapter_index(tree)
    listing = sorted(
        ((ch, sum(tree["sections"].get(s, {}).get("n_pages_in_section", 0)
                  for s in subs))
         for ch, subs in chapters.items() if ch != "Unsectioned"),
        key=lambda x: -x[1],
    )
    listing = listing[:8]  # cap at 8 chapters (A..H)
    letter_to_chapter = {_LETTERS[i]: ch for i, (ch, _) in enumerate(listing)}
    chapter_to_letter = {ch: L for L, ch in letter_to_chapter.items()}
    print(f"Chapters (A..{_LETTERS[len(listing) - 1]}):")
    for letter, (ch, n) in zip(_LETTERS, listing):
        print(f"  {letter}. {ch}  ({n} pages)")

    catalog = load_catalog(args.catalog_dir)
    catalog_index = {(r.bulletin, r.page): r for r in catalog}
    rewrite = tree.get("meta", {}).get("banner_rewrite") or {}

    def gold_chapter(b: str, p: int) -> str | None:
        row = catalog_index.get((b, p))
        if row is None:
            return None
        raw = (row.section or "").strip()
        if raw:
            section = rewrite.get(raw.lower(), raw)
        else:
            section = "Unsectioned"
        return _PARENT_CHAPTER_MAP.get(section, section)

    # Sample UIDs.
    benchmark = _load_benchmark(args.benchmark)
    if args.uids:
        wanted = {u.strip() for u in args.uids.split(",") if u.strip()}
        sampled = [r for r in benchmark if r["uid"] in wanted]
    else:
        rng = random.Random(args.seed)
        sampled = rng.sample(benchmark, min(args.n, len(benchmark)))
    sampled.sort(key=lambda r: r["uid"])
    print(f"\nRunning on {len(sampled)} UIDs\n")

    plan_cache = _load_plan_cache(args.plan_cache)

    def _one_sample(user: str) -> str | None:
        """One MCQ sample. Returns the picked letter (or None on bad output)."""
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=8,
            temperature=args.temperature,
            extra_body={"reasoning": {"enabled": False}},
        )
        text = (resp.choices[0].message.content or "").strip()
        for c in text:
            if c in letter_to_chapter:
                return c
        return None

    def _one_branch(uid: str, question: str, branch_idx: int,
                    b: RetrieveBranch, golds: list[tuple[str, int]]):
        user = _build_user_prompt(question, b.concept, b.period, listing)
        # Run N self-consistency samples; vote frequencies → probability.
        votes: Counter = Counter()
        for _ in range(args.n_samples):
            letter = _one_sample(user)
            if letter:
                votes[letter] += 1
        total_valid = sum(votes.values())

        normalized_probs: dict[str, float] = {}
        if total_valid > 0:
            normalized_probs = {L: c / total_valid for L, c in votes.items()}

        items = sorted(normalized_probs.items(), key=lambda x: -x[1])
        picked_letter = items[0][0] if items else None
        top_p = items[0][1] if items else None
        top2_margin = (items[0][1] - items[1][1]) if len(items) > 1 else (
            top_p if top_p is not None else None
        )
        if normalized_probs:
            entropy = -sum(p * math.log(max(p, 1e-12))
                           for p in normalized_probs.values())
        else:
            entropy = None

        picked_chapter = letter_to_chapter.get(picked_letter) if picked_letter else None
        gold_chapters = sorted({c for c in (gold_chapter(g[0], g[1])
                                            for g in golds) if c})
        hit = picked_chapter in gold_chapters if picked_chapter else False

        return {
            "uid": uid, "branch_idx": branch_idx,
            "concept": b.concept, "period": b.period,
            "picked_letter": picked_letter,
            "picked_chapter": picked_chapter,
            "gold_chapters": gold_chapters,
            "hit": hit,
            "top_p": top_p,
            "top2_margin": top2_margin,
            "entropy": entropy,
            "votes": {letter_to_chapter[L]: c for L, c in votes.items()},
            "all_probs": {letter_to_chapter[L]: p
                          for L, p in normalized_probs.items()},
        }

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for row in sampled:
            uid = row["uid"]
            if uid not in plan_cache:
                print(f"[skip] {uid}: no cached plan", file=sys.stderr)
                continue
            try:
                plan_obj = dsl_parse(plan_cache[uid])
            except Exception as e:
                print(f"[skip] {uid}: plan parse error {e}", file=sys.stderr)
                continue
            for compute in plan_obj.computes:
                for i, b in enumerate(compute.branches):
                    if isinstance(b, RetrieveBranch):
                        futs.append(ex.submit(_one_branch, uid,
                                              row["question"], i, b,
                                              row["goldens"]))
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            tp = r['top_p'] if r['top_p'] is not None else float('nan')
            m = r['top2_margin'] if r['top2_margin'] is not None else float('nan')
            ent = r['entropy'] if r['entropy'] is not None else float('nan')
            print(f"  {r['uid']:<10}  branch={r['branch_idx']}  "
                  f"picked={r['picked_chapter']!r:<60}  "
                  f"top_p={tp:.3f}  margin={m:.3f}  entropy={ent:.2f}  "
                  f"gold={r['gold_chapters']}  HIT={r['hit']}")

    results.sort(key=lambda r: (r["uid"], r["branch_idx"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary.
    print()
    n = len(results)
    n_hit = sum(1 for r in results if r["hit"])
    print(f"Branches: {n}  picks-matching-gold-chapter: {n_hit} ({n_hit/n:.1%})")
    margin_hit = [r["top2_margin"] for r in results if r["hit"]
                  and r["top2_margin"] is not None]
    margin_miss = [r["top2_margin"] for r in results if not r["hit"]
                   and r["top2_margin"] is not None]
    if margin_hit:
        print(f"  hits  margin: median {sorted(margin_hit)[len(margin_hit)//2]:.3f}  "
              f"min {min(margin_hit):.3f}  max {max(margin_hit):.3f}")
    if margin_miss:
        print(f"  miss  margin: median {sorted(margin_miss)[len(margin_miss)//2]:.3f}  "
              f"min {min(margin_miss):.3f}  max {max(margin_miss):.3f}")
    print(f"\nResults written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
