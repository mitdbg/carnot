"""Render-corpus worker: rasterize one static shard of the corpus's PDF pages into the
shared `renders/` PNG cache. Launched in bulk by `render_corpus` — ONE OS process per shard
so PyMuPDF rasterization runs truly in parallel (the GIL serializes threads, not processes).
This is a standalone cache-warming prep step, not a stage of the page-index build pipeline.

The shard is a stride over a deterministic flat page list (every page of every bulletin, in
sorted order): shard `i` of `n` renders the global page indices where `idx % n == i`. The
launcher and every worker rebuild that same list from the shared manifest, so no per-page
work list ever crosses the process boundary — each worker just recomputes its own slice.
`render_to_cache` is cache-first (an existing PNG is skipped), so a re-run resumes cheaply.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from officeqa.page_index.store import render_to_cache


def _flat_pages(bulletins: list[tuple[str, int]]) -> Iterator[tuple[str, int]]:
    """The deterministic global page order both launcher and worker partition over: every
    1-based page of every bulletin, in the manifest's (sorted) bulletin order."""
    for month, n_pages in bulletins:
        for page in range(1, n_pages + 1):
            yield month, page


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Render one static shard of corpus pages into the renders/ PNG cache."
    )
    ap.add_argument("--manifest", required=True, help="path to the shared render manifest JSON")
    ap.add_argument("--shard", type=int, required=True, help="this worker's shard index (0-based)")
    ap.add_argument("--num-shards", type=int, required=True, help="total number of shards")
    ap.add_argument("--renders-dir", required=True, help="target renders/ cache directory")
    ap.add_argument("--progress-every", type=int, default=200, help="pages between stderr progress lines")
    args = ap.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text())
    pdf_dir = manifest["pdf_dir"]
    bulletins: list[tuple[str, int]] = manifest["bulletins"]

    counts: dict[str, int] = {"rendered": 0, "cached": 0, "skipped": 0}
    done = 0
    t0 = time.time()
    for idx, (month, page) in enumerate(_flat_pages(bulletins)):
        if idx % args.num_shards != args.shard:
            continue
        status = render_to_cache(month, page, pdf_dir, args.renders_dir)
        counts[status] = counts.get(status, 0) + 1
        done += 1
        if done % args.progress_every == 0:
            rate = done / max(time.time() - t0, 1e-9)
            print(
                f"[shard {args.shard}/{args.num_shards}] {done} pages | {counts} | {rate:.1f} pg/s",
                file=sys.stderr,
                flush=True,
            )

    # Final summary on stdout — the launcher captures this line to aggregate across shards.
    summary = {**counts, "shard": args.shard, "elapsed_s": round(time.time() - t0, 1)}
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
