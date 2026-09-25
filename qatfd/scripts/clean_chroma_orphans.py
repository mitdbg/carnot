#!/usr/bin/env python3
"""Remove orphaned HNSW segment directories from a ChromaDB persistent store.

Chroma (1.5.x) deletes a collection's rows from `chroma.sqlite3` but leaves the vector segment
directory (`<store>/<segment-uuid>/`) on disk. Runs that create and drop many working-set
collections therefore leak several GB per run. This script deletes every directory in the store
that (a) is not referenced by `segments.id` in the sqlite catalog, (b) contains only the known HNSW
files, and (c) was last modified at least `--min-age-min` minutes ago (so a segment that is being
created right now is never touched). Directories still held open by a running server free their
space when the server restarts.

Usage:
    clean_chroma_orphans.py [--store DIR] [--min-age-min N] [--dry-run] [--quiet]
    clean_chroma_orphans.py --watch 600      # loop forever, sweeping every 600 s
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import time

_HNSW_FILES = {"data_level0.bin", "header.bin", "length.bin", "link_lists.bin", "index_metadata.pickle"}
_DEFAULT_STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "officeqa", "chromadb")


def _live_segments(store: str) -> set[str]:
    con = sqlite3.connect(f"file:{os.path.join(store, 'chroma.sqlite3')}?mode=ro", uri=True)
    try:
        return {row[0] for row in con.execute("select id from segments")}
    finally:
        con.close()


def _dir_size(path: str) -> int:
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


def _is_hnsw_only(path: str) -> bool:
    entries = os.listdir(path)
    return bool(entries) and all(e in _HNSW_FILES and os.path.isfile(os.path.join(path, e)) for e in entries)


def sweep(store: str, min_age_s: float, dry_run: bool, quiet: bool) -> tuple[int, int, int]:
    """Returns (removed, bytes_freed, skipped_young)."""
    live = _live_segments(store)
    now = time.time()
    removed = freed = young = 0
    for name in sorted(os.listdir(store)):
        path = os.path.join(store, name)
        if not os.path.isdir(path) or name in live or len(name) != 36:
            continue
        if not _is_hnsw_only(path):
            if not quiet:
                print(f"[clean] skip {name}: unexpected contents {sorted(os.listdir(path))[:5]}", file=sys.stderr)
            continue
        if now - os.path.getmtime(path) < min_age_s:
            young += 1
            continue
        size = _dir_size(path)
        if not dry_run:
            shutil.rmtree(path)
        removed += 1
        freed += size
    if not quiet or removed:
        verb = "would remove" if dry_run else "removed"
        print(f"[clean] {time.strftime('%Y-%m-%d %H:%M:%S')} {verb} {removed} orphan dir(s), "
              f"{freed / 1e9:.2f} GB; {len(live)} live segment(s); {young} too recent", flush=True)
    return removed, freed, young


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=os.path.normpath(_DEFAULT_STORE))
    ap.add_argument("--min-age-min", type=float, default=10.0, help="ignore dirs modified within the last N minutes")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="print only when something was removed")
    ap.add_argument("--watch", type=float, default=None, metavar="SECONDS", help="repeat every SECONDS until killed")
    args = ap.parse_args()
    if not os.path.isfile(os.path.join(args.store, "chroma.sqlite3")):
        sys.exit(f"[clean] no chroma.sqlite3 under {args.store}")
    while True:
        sweep(args.store, args.min_age_min * 60.0, args.dry_run, args.quiet)
        if args.watch is None:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
