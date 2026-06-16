"""DAIS render geometry, path helpers, and PNG render/load functions.

Single home for the rasterization logic carved out of ``clean_corpus.py``. Two consumers:

* ``render_corpus.py`` (the standalone pre-render CLI) renders every PDF page and every
  table crop to disk once, offline.
* ``clean_corpus.py`` *loads* those PNGs as the LLM's full-page context + table crop,
  falling back to an inline render only when a PNG is absent.

Paths are keyed by the same ``doc_id`` / ``chunk_id`` the SearchAgent returns
(:func:`skunk.dais.dais_common.doc_id_of` / :func:`chunk_id_of`), so any consumer can build
the file path from a retrieved id with no extra lookup:

  * page  -> ``<page_renders_dir>/<file_id>_<page_id>.png``
  * table -> ``<table_renders_dir>/<file_id>_<page_id>_<element_id>.png``

Geometry (crop padding, DPIs except the page DPI) is reused from the OfficeQA table
corrector so the two pipelines stay calibrated identically.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from skunk.common import B64Image
from skunk.dais.dais_common import chunk_id_of, doc_id_of
# Reuse the corpus-agnostic crop geometry from the OfficeQA table corrector.
from skunk.search_agent.prep.table_corrector import CROP_DPI, CROP_PAD_PTS

# Page PNG render DPI: the single source for both the harness UI (display) and
# clean_corpus's full-page LLM context. Matches the Treasury PageStore convention.
PAGE_DPI = 200


def _matrix(dpi: int):
    import fitz

    return fitz.Matrix(dpi / 72, dpi / 72)


# --------------------------------------------------------------------------- paths
def page_render_path(page_renders_dir: str | Path, file_id: str, page_id: int | str) -> Path:
    """``<page_renders_dir>/<doc_id>.png`` — keyed by the SearchAgent ``doc_id``."""
    return Path(page_renders_dir) / f"{doc_id_of(file_id, page_id)}.png"


def table_render_path(
    table_renders_dir: str | Path, file_id: str, page_id: int | str, element_id: int | str
) -> Path:
    """``<table_renders_dir>/<chunk_id>.png`` — keyed by the SearchAgent ``chunk_id``."""
    return Path(table_renders_dir) / f"{chunk_id_of(file_id, page_id, element_id)}.png"


# --------------------------------------------------------------------------- elements
def group_by_page(doc: dict) -> dict[int, list[dict]]:
    """Map page_id -> elements on that page, preserving parser order."""
    by_page: dict[int, list[dict]] = {}
    for elt in doc["document"]["elements"]:
        page_id = elt["bbox"][0]["page_id"]
        by_page.setdefault(page_id, []).append(elt)
    return by_page


# --------------------------------------------------------------------------- render
def render_page_png_bytes(pg) -> bytes:
    """Full-page PNG bytes at :data:`PAGE_DPI` (shared UI display + LLM context image)."""
    return pg.get_pixmap(matrix=_matrix(PAGE_DPI)).tobytes("png")


def render_table_png_bytes(pg, coord, scale: float) -> bytes | None:
    """Tight-crop PNG bytes of a table's bbox (parser px -> PDF points via ``scale``),
    padded by ``CROP_PAD_PTS`` and clamped to the page. None if the bbox is degenerate."""
    import fitz

    r = pg.rect
    x0, y0, x1, y1 = coord
    clip = fitz.Rect(
        max(0.0, x0 / scale - CROP_PAD_PTS),
        max(0.0, y0 / scale - CROP_PAD_PTS),
        min(r.width, x1 / scale + CROP_PAD_PTS),
        min(r.height, y1 / scale + CROP_PAD_PTS),
    )
    if clip.width <= 2 or clip.height <= 2:
        return None
    pix = pg.get_pixmap(matrix=_matrix(CROP_DPI), clip=clip)
    if pix.width <= 0 or pix.height <= 0:
        return None
    return pix.tobytes("png")


# --------------------------------------------------------------------------- load
def png_b64image(data: bytes) -> B64Image:
    """Wrap raw PNG bytes as a ``B64Image`` (the shape the LLM consumes)."""
    return B64Image(mime="image/png", data=base64.standard_b64encode(data).decode())


def _read_png(path: Path) -> B64Image | None:
    """Load a rendered PNG into a ``B64Image``, or None if it isn't on disk."""
    if not path.exists():
        return None
    return png_b64image(path.read_bytes())


def load_page_image(page_renders_dir: str | Path, file_id: str, page_id: int | str) -> B64Image | None:
    """Pre-rendered full-page PNG as a ``B64Image``, or None if not yet rendered."""
    return _read_png(page_render_path(page_renders_dir, file_id, page_id))


def load_table_image(
    table_renders_dir: str | Path, file_id: str, page_id: int | str, element_id: int | str
) -> B64Image | None:
    """Pre-rendered table-crop PNG as a ``B64Image``, or None if not yet rendered."""
    return _read_png(table_render_path(table_renders_dir, file_id, page_id, element_id))


# --------------------------------------------------------------------------- write
def atomic_write_png(path: str | Path, data: bytes) -> None:
    """Write PNG bytes via a tmp file + rename (no partial files; safe under concurrency)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.rename(path)
