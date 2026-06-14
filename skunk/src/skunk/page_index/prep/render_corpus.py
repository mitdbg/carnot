"""Render the ENTIRE Treasury Bulletin corpus to the page-index `renders/` PNG cache, in
parallel across OS processes.

Why a standalone tool, not a pipeline stage: rasterizing all ~89k corpus pages is a big,
embarrassingly parallel, LLM-free job, and PyMuPDF holds the GIL while it rasterizes — so
threads buy nothing and only separate processes give real parallelism. This launcher
statically partitions every page of every bulletin across `--workers` child processes
(`render_worker`), each rendering a fixed stride of the page list. Run it once to warm the
cache that `PageStore` (query-time vision) and the build's `vision_rescan` stage both read.

  python3 -m skunk.page_index.prep.render_corpus                       # all bulletins, one proc/core
  python3 -m skunk.page_index.prep.render_corpus --workers 8
  python3 -m skunk.page_index.prep.render_corpus --only 1990-09,1992-03  # smoke test

Cache-first and resumable: a page whose PNG already exists is skipped, so a re-run only
fills gaps. Target defaults to `$SKUNK_PAGE_INDEX_DIR/renders`; source PDFs to
`$OFFICEQA_PDF_DIR` (else the repo default corpus path).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Mirrors `common.pdf_path_for`: treasury_bulletin_YYYY_MM.pdf <-> "YYYY-MM". Kept local so
# this prep tool stays decoupled from the build pipeline module (and its LLM imports).
_FNAME = re.compile(r"^treasury_bulletin_(\d{4})_(\d{2})\.pdf$")
_DEFAULT_PDF_DIR = Path.home() / "Desktop/officeqa/treasury_bulletin_pdfs"


def _discover(pdf_dir: Path, only: set[str] | None) -> list[str]:
    """Corpus bulletin ids (sorted YYYY-MM) found under `pdf_dir`, optionally restricted."""
    out: list[str] = []
    for p in sorted(pdf_dir.iterdir()):
        m = _FNAME.match(p.name)
        if not m:
            continue
        month = f"{m.group(1)}-{m.group(2)}"
        if only and month not in only:
            continue
        out.append(month)
    return out


def _page_count(pdf_path: Path) -> int:
    import fitz

    with fitz.open(pdf_path) as doc:
        return doc.page_count


def _build_manifest(pdf_dir: Path, months: list[str]) -> list[list[object]]:
    """[[month, n_pages], ...] in bulletin order — the page universe the workers stride over.
    A PDF that can't be opened is dropped (n_pages 0) so it never enters the page list."""
    bulletins: list[list[object]] = []
    for month in months:
        pdf = pdf_dir / f"treasury_bulletin_{month[:4]}_{month[5:]}.pdf"
        try:
            n = _page_count(pdf)
        except Exception:  # noqa: BLE001 — corrupt/unreadable PDF: skip, don't abort the build
            n = 0
        if n:
            bulletins.append([month, n])
    return bulletins


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8, help="parallel render processes")
    ap.add_argument(
        "--pdf-dir",
        default=os.environ.get("OFFICEQA_PDF_DIR") or str(_DEFAULT_PDF_DIR),
        help="corpus PDF directory",
    )
    ap.add_argument("--renders-dir", default=None, help="target cache dir (default $SKUNK_PAGE_INDEX_DIR/renders)")
    ap.add_argument("--only", default=None, help="comma-separated YYYY-MM bulletins (smoke test)")
    args = ap.parse_args(argv)

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_dir():
        ap.error(f"--pdf-dir not found: {pdf_dir}")

    renders_dir = args.renders_dir
    if renders_dir is None:
        root = os.environ.get("SKUNK_PAGE_INDEX_DIR")
        if not root:
            ap.error("pass --renders-dir or set SKUNK_PAGE_INDEX_DIR")
        renders_dir = str(Path(root) / "renders")
    Path(renders_dir).mkdir(parents=True, exist_ok=True)

    only = {s.strip() for s in args.only.split(",")} if args.only else None
    months = _discover(pdf_dir, only)
    if not months:
        ap.error(f"no treasury_bulletin_*.pdf found under {pdf_dir}")

    bulletins = _build_manifest(pdf_dir, months)
    total_pages = sum(n for _, n in bulletins)  # type: ignore[misc]
    workers = max(1, min(args.workers, total_pages))

    manifest_path = Path(renders_dir) / ".render_manifest.json"
    manifest_path.write_text(json.dumps({"pdf_dir": str(pdf_dir), "bulletins": bulletins}))

    print(f"[render_corpus] {len(bulletins)} bulletins, {total_pages} pages -> {renders_dir}")
    print(f"[render_corpus] launching {workers} worker process(es)")

    t0 = time.time()
    procs: list[subprocess.Popen] = []
    for i in range(workers):
        cmd = [
            sys.executable,
            "-m",
            "skunk.page_index.prep.render_worker",
            "--manifest", str(manifest_path),
            "--shard", str(i),
            "--num-shards", str(workers),
            "--renders-dir", str(renders_dir),
        ]
        # stderr inherited → live per-shard progress streams to this terminal; stdout piped →
        # the worker's final summary line is captured below for aggregation.
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=os.environ.copy()))

    agg = {"rendered": 0, "cached": 0, "skipped": 0}
    failed = 0
    for p in procs:
        out, _ = p.communicate()
        if p.returncode != 0:
            failed += 1
            print(f"[render_corpus] worker pid {p.pid} exited {p.returncode}", file=sys.stderr)
            continue
        lines = [ln for ln in (out or "").splitlines() if ln.strip()]
        if not lines:
            continue
        try:
            res = json.loads(lines[-1])
        except json.JSONDecodeError:
            continue
        for k in agg:
            agg[k] += res.get(k, 0)

    elapsed = time.time() - t0
    print(f"[render_corpus] done in {elapsed / 60:.1f} min | {agg} | {failed} worker(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
