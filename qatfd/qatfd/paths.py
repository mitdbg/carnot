"""Filesystem anchors. qatfd lives at <repo>/qatfd and imports the sibling
`skunk` package at <repo>/skunk; benchmark data + indices live under skunk/.
Relative paths from skunk's config/.env are resolved against `SKUNK_DIR`."""

from __future__ import annotations

from pathlib import Path

# <repo>/qatfd/qatfd/paths.py -> parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
SKUNK_DIR = REPO_ROOT / "skunk"


def resolve_under_skunk(path: str | Path) -> Path:
    """Resolve `path` to an absolute path, treating relative paths as relative to
    the skunk/ directory (where skunk's .env / config paths are anchored)."""
    p = Path(path)
    return p if p.is_absolute() else (SKUNK_DIR / p)
