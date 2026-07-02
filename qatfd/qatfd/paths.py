"""Filesystem anchors. qatfd lives at <repo>/qatfd and imports the sibling
`skunk` package at <repo>/skunk (SKUNK_DIR is the anchor for skunk's `.env`).

ALL benchmark data — vector indices, questions, corpora, split files, and prompt-override
YAMLs — lives under <repo>/qatfd/benchmarks/<benchmark>/; relative benchmark paths resolve
there via `resolve_under_benchmarks`."""

from __future__ import annotations

import os
from pathlib import Path

# <repo>/qatfd/qatfd/paths.py -> parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
SKUNK_DIR = REPO_ROOT / "skunk"
# Root of the benchmark data + indices. Defaults to <repo>/qatfd/benchmarks; override with
# QATFD_BENCHMARKS_DIR on a box that mounts the corpora elsewhere.
BENCHMARKS_DIR = Path(os.environ.get("QATFD_BENCHMARKS_DIR") or (REPO_ROOT / "qatfd" / "benchmarks"))


def resolve_under_benchmarks(path: str | Path) -> Path:
    """Resolve `path` to an absolute path, treating relative paths as relative to the
    benchmark-data root (qatfd/benchmarks/). Every benchmark's corpus, index, questions,
    and split file live under `{BENCHMARKS_DIR}/<benchmark>/...`."""
    p = Path(path)
    return p if p.is_absolute() else (BENCHMARKS_DIR / p)
