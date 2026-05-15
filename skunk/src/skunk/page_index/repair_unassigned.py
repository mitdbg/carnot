"""LLM-classify the catalog pages currently unreachable through the tree.

A page is "unassigned" when build_tree drops it for one of two reasons:

  1. `section=None` (no harvest_toc hit for that bulletin) — falls into
     the `Unsectioned` synthetic chapter, hard to reach via section_pick.
  2. `section` is set but none of the page's `keywords` got a cluster
     assignment — usually a parser fragment ("and Off-budget Outlays by
     Agency") or an OCR'd keyword ("Yielda", "Perioda") that the LLM
     skipped during keyword clustering.

This job classifies each such page into one of the 9 canonical parent
chapters from `_PARENT_CHAPTER_MAP`. Output: `unassigned_repair.jsonl`,
one entry per page with the classified chapter. Downstream `build_tree`
augmentation can add these pages under a synthetic `_repaired` cluster
in the picked chapter's primary section so they become reachable.

Usage:
    SKUNK_LLM_MODEL=google/gemini-2.5-flash \\
    PYTHONPATH=src python3 -m skunk.page_index.repair_unassigned \\
        --catalog-dir cache/page_index \\
        --tree cache/page_index/concept_tree.json \\
        --out cache/page_index/unassigned_repair.jsonl \\
        --workers 16 --batch-size 30

The cheap gemini-2.5-flash + thinking_budget=0 (default) is plenty for
this typo-clean / chapter-assignment task; cost is ~$0.10 for the full
corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from skunk.common import LLMClient
from skunk.config import SkunkConfig

from .concept_tree import load_catalog
from .retrieve_probe import _PARENT_CHAPTER_MAP, load_concept_tree
from .schema import PageCatalogRow


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_REPO_ROOT = Path(__file__).resolve().parents[3]
_load_env(_REPO_ROOT / ".env")


_PARENT_CHAPTERS = sorted({c for c in _PARENT_CHAPTER_MAP.values()
                           if c != "Unsectioned"})


_REPAIR_SYSTEM = """You classify U.S. Treasury Bulletin pages into one of
these canonical parent chapters:

{chapters}

You will see a batch of pages, each with:
  - bulletin month (YYYY-MM)
  - raw section banner (may be empty)
  - table title (may be OCR-corrupted)
  - keywords (short noun phrases extracted from the page; may be
    parser fragments like "and ... by Agency" or OCR errors like
    "Yielda")

For each page, pick the SINGLE chapter that best matches its content.
Use the table_title as the primary signal; keywords are supplementary.

Rules:
  - Output a SINGLE JSON object (no prose, no fences):
      {{"assignments": [
        {{"id": <0-based page index in batch>, "chapter": "<chapter>"}},
        ...
      ]}}
  - You MUST emit one entry per input page. Use the EXACT chapter label
    as shown.
  - Every page MUST be assigned to one of the chapters above. "Unsectioned"
    is NOT a valid output — even for ambiguous pages, pick the single
    best-matching chapter. When uncertain, prefer Federal Fiscal
    Operations for budget/agency content, Federal Debt for any debt /
    Treasury security topic, and Profile of the Economy for narrative
    articles.
"""


def _classify_batch(
    batch: list[dict[str, Any]], llm: LLMClient, system_prompt: str,
) -> list[dict[str, Any]]:
    """One LLM call per batch. Returns [{bulletin, page, chapter}, ...]."""
    items = [
        {"id": i,
         "bulletin": p["bulletin"],
         "raw_section": p.get("section") or "",
         "table_title": (p.get("table_title") or "")[:300],
         "keywords": p.get("keywords", [])[:6]}
        for i, p in enumerate(batch)
    ]
    user = (
        f"Classify {len(items)} pages. Output exactly {len(items)} entries.\n\n"
        + json.dumps(items, ensure_ascii=False, indent=1)
    )
    resp = llm.call(system=system_prompt, user=user,
                    temperature=0.0, thinking_budget=0)
    text = resp.text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        return []

    valid = {c.lower(): c for c in _PARENT_CHAPTERS}
    out: list[dict[str, Any]] = []
    for entry in (obj or {}).get("assignments", []) or []:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry["id"])
            chapter_str = str(entry["chapter"]).strip()
        except (KeyError, ValueError, TypeError):
            continue
        if not (0 <= idx < len(batch)):
            continue
        canonical = valid.get(chapter_str.lower())
        if canonical is None:
            continue
        p = batch[idx]
        out.append({"bulletin": p["bulletin"], "page": p["page"],
                    "chapter": canonical})
    return out


def _collect_unassigned(
    catalog: list[PageCatalogRow], tree: dict[str, Any],
) -> list[dict[str, Any]]:
    """Find catalog pages that don't appear under any cluster in the tree."""
    reachable: set[tuple[str, int]] = set()
    for sd in tree.get("sections", {}).values():
        for cd in sd.get("clusters", {}).values():
            for posts in cd.get("keywords", {}).values():
                for p in posts:
                    reachable.add((p["bulletin"], p["page"]))

    out: list[dict[str, Any]] = []
    for r in catalog:
        if r.page_kind not in ("table", "chart", "prose"):
            continue
        if (r.bulletin, r.page) in reachable:
            continue
        out.append({
            "bulletin": r.bulletin,
            "page": r.page,
            "section": r.section,
            "table_title": r.table_title,
            "keywords": list(r.keywords),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="LLM-classify unassigned catalog pages into parent chapters.",
    )
    ap.add_argument("--catalog-dir", type=Path,
                    default=Path("cache/page_index"))
    ap.add_argument("--tree", type=Path,
                    default=Path("cache/page_index/concept_tree.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("cache/page_index/unassigned_repair.jsonl"))
    ap.add_argument("--batch-size", type=int, default=30)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug: cap total pages classified.")
    ap.add_argument("--seed-file", type=Path, default=None,
                    help="If set, load pages-to-classify from this jsonl "
                         "(one {bulletin, page, section, table_title, "
                         "keywords} per line). Skips the catalog scan.")
    args = ap.parse_args()

    if args.seed_file is not None:
        unassigned = [json.loads(line) for line in args.seed_file.open()]
    else:
        catalog = load_catalog(args.catalog_dir)
        tree = load_concept_tree(args.tree)
        unassigned = _collect_unassigned(catalog, tree)
    if args.limit:
        unassigned = unassigned[:args.limit]
    print(f"Unassigned pages: {len(unassigned)}", flush=True)
    if not unassigned:
        args.out.write_text("")
        return 0

    cfg = SkunkConfig.from_env()
    llm = LLMClient(cfg)
    print(f"Model: {cfg.llm_model}    workers: {args.workers}    "
          f"batch_size: {args.batch_size}", flush=True)

    system_prompt = _REPAIR_SYSTEM.format(
        chapters=json.dumps(_PARENT_CHAPTERS, ensure_ascii=False, indent=1),
    )

    batches = [unassigned[i:i + args.batch_size]
               for i in range(0, len(unassigned), args.batch_size)]
    n_batches = len(batches)
    print(f"Running {n_batches} batches…", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    written = 0
    done = 0
    with args.out.open("w") as fh, ThreadPoolExecutor(
        max_workers=args.workers,
    ) as ex:
        futs = {ex.submit(_classify_batch, b, llm, system_prompt): i
                for i, b in enumerate(batches)}
        for f in as_completed(futs):
            done += 1
            try:
                results = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"[batch error] {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                continue
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False))
                fh.write("\n")
                written += 1
            if done % 10 == 0 or done == n_batches:
                elapsed = time.monotonic() - t0
                print(f"  [{done}/{n_batches}] wrote {written} "
                      f"({elapsed:.1f}s)", flush=True)

    print(f"\nWrote {written}/{len(unassigned)} assignments → {args.out} "
          f"in {time.monotonic() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
