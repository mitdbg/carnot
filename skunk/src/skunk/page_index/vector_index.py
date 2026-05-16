"""Dense retrieval index over the PageIndex catalog.

Builds one Matryoshka-truncated `gemini-embedding-001` vector per catalog row
using the same blob text the hierarchical leaf-rank LLM trusts
(`_build_leaf_blob` in `retrieve_probe.py`). Persists as a single `.npz` matrix
plus a parallel JSONL of `{bulletin, page}` keys.

Query time is brute-force masked cosine over an L2-normalized float32 matrix.
At 89k pages × 768 dims (~270 MB) the matmul is faster than the setup cost of
an ANN structure and returns exact neighbors, so we stay numpy-only — no FAISS
or chroma. If the corpus grows ~10× we can drop in `faiss-cpu` with the same
API surface.

Workflow:
  >>> from skunk.page_index.vector_index import (
  ...     build_vector_index, load_vector_index, period_mask, search,
  ... )
  >>> # Offline (one-shot per catalog rebuild):
  >>> build_vector_index(catalog_dir, out_dir, llm)
  >>> # Query time:
  >>> idx = load_vector_index(out_dir)
  >>> mask = period_mask(idx.keys, period="CY1940", catalog_index=catalog)
  >>> hits = search(idx, query_vec, mask, top_k=50)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skunk.common import LLMClient
from skunk.config import SkunkConfig

from .period import intervals_overlap, period_to_intervals
from .retrieve_probe import build_vector_blob, load_catalog
from .schema import PageCatalogRow


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class VectorIndex:
    """Loaded vector index. `vectors` is L2-normalized float32 of shape
    [N, dim]; `keys[i]` is the catalog identity for row i."""
    vectors: np.ndarray                       # shape [N, dim], float32, L2-normalized
    keys: list[tuple[str, int]]               # (bulletin, page) per row
    dim: int


# ---------------------------------------------------------------------------
# Index build (offline, one-shot)
# ---------------------------------------------------------------------------

def _row_is_indexable(row: PageCatalogRow) -> bool:
    """Skip pages that carry no useful text — blank/TOC pages would just
    inject noise into the matmul. A row is indexable when it has a
    table/chart block with metadata, a prose block backed by keywords,
    or — for block-less rows — at least one keyword or date string.
    """
    if not row.content_blocks:
        return bool(row.keywords or row.dates)
    has_prose = False
    for b in row.content_blocks:
        if b.kind == "prose":
            has_prose = True
            continue
        if b.title or b.column_headers or b.row_headers_sample:
            return True
    return has_prose and bool(row.keywords)


def build_vector_index(
    catalog_dir: Path,
    out_dir: Path,
    llm: LLMClient,
    *,
    dim: int = 768,
    workers: int = 4,
    batch_size: int = 100,
    model: str = "gemini-embedding-001",
) -> VectorIndex:
    """Build and persist `vectors.npz` + `vectors_keys.jsonl` under `out_dir`.

    Reads every `*.jsonl` under `catalog_dir`, filters to indexable pages,
    builds the same blob text the L3 LLM sees plus a corpus-framing
    preamble (so the embedding model knows the page is a Treasury
    Bulletin entry), batches embed calls and runs `workers` in parallel.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[vector_index] loading catalog from {catalog_dir}", flush=True)
    rows = load_catalog(catalog_dir)
    rows = [r for r in rows if _row_is_indexable(r)]
    print(f"[vector_index] {len(rows)} indexable rows  model={model}", flush=True)

    # Minimal blob: title + keywords only. Drops column_headers,
    # row_headers_sample, and dates — those fields are dominated by
    # generic axis labels that dilute the pooled embedding without
    # adding discriminating signal (see `build_vector_blob` docstring).
    blobs: list[str] = [build_vector_blob(r) for r in rows]
    keys: list[tuple[str, int]] = [(r.bulletin, r.page) for r in rows]

    # Split into batches and embed in parallel.
    chunks: list[tuple[int, list[str]]] = []
    for start in range(0, len(blobs), batch_size):
        chunks.append((start, blobs[start:start + batch_size]))

    vectors = np.zeros((len(blobs), dim), dtype=np.float32)
    t0 = time.monotonic()

    def _run(start: int, chunk: list[str]) -> tuple[int, list[list[float]]]:
        vecs = llm.embed(
            chunk,
            task_type="RETRIEVAL_DOCUMENT",
            dim=dim,
            model=model,
            batch_size=batch_size,
        )
        return start, vecs

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_run, start, chunk) for start, chunk in chunks]
        for f in as_completed(futs):
            start, vecs = f.result()
            arr = np.asarray(vecs, dtype=np.float32)
            vectors[start:start + arr.shape[0]] = arr
            done += arr.shape[0]
            if done % (batch_size * 20) < batch_size:
                elapsed = time.monotonic() - t0
                rate = done / max(elapsed, 1e-6)
                print(
                    f"[vector_index] embedded {done}/{len(blobs)} "
                    f"({rate:.0f}/s, {elapsed:.0f}s elapsed)",
                    flush=True,
                )

    # L2-normalize so cosine collapses to a dot product at query time.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    vectors = vectors.astype(np.float32, copy=False)

    npz_path = out_dir / "vectors.npz"
    keys_path = out_dir / "vectors_keys.jsonl"
    np.savez_compressed(npz_path, vectors=vectors)
    with keys_path.open("w") as fh:
        for bulletin, page in keys:
            fh.write(json.dumps(
                {"bulletin": bulletin, "page": page},
                ensure_ascii=False,
            ))
            fh.write("\n")
    print(
        f"[vector_index] wrote {len(keys)} vectors (dim={dim}) → "
        f"{npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB), "
        f"{keys_path} in {time.monotonic() - t0:.1f}s",
        flush=True,
    )
    return VectorIndex(vectors=vectors, keys=keys, dim=dim)


# ---------------------------------------------------------------------------
# Index load
# ---------------------------------------------------------------------------

def load_vector_index(out_dir: Path) -> VectorIndex:
    """Load `vectors.npz` + `vectors_keys.jsonl` produced by
    `build_vector_index`. Vectors are mmap'd; keys are parsed into a list."""
    npz_path = out_dir / "vectors.npz"
    keys_path = out_dir / "vectors_keys.jsonl"
    if not npz_path.exists():
        raise FileNotFoundError(f"vector index missing: {npz_path}")
    if not keys_path.exists():
        raise FileNotFoundError(f"vector index keys missing: {keys_path}")
    npz = np.load(npz_path, mmap_mode="r")
    vectors = npz["vectors"]
    keys: list[tuple[str, int]] = []
    with keys_path.open() as fh:
        for line in fh:
            d = json.loads(line)
            keys.append((d["bulletin"], int(d["page"])))
    if vectors.shape[0] != len(keys):
        raise ValueError(
            f"vector index shape mismatch: {vectors.shape[0]} rows vs {len(keys)} keys"
        )
    return VectorIndex(vectors=vectors, keys=keys, dim=int(vectors.shape[1]))


# ---------------------------------------------------------------------------
# Symbolic period prefilter
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b")
_PUBLISH_LAG_MONTHS = 12  # retrospective bulletins land up to a year later


def _bulletin_to_iso(bulletin: str) -> str | None:
    """`'YYYY-MM'` → ISO `'YYYY-MM-15'` for interval comparison; None if malformed."""
    try:
        y, m = bulletin.split("-")
        return f"{int(y):04d}-{int(m):02d}-15"
    except (ValueError, AttributeError):
        return None


def _add_months(iso: str, months: int) -> str:
    y, m, d = int(iso[:4]), int(iso[5:7]), int(iso[8:10])
    total = y * 12 + (m - 1) + months
    ny, nm = divmod(total, 12)
    return f"{ny:04d}-{nm + 1:02d}-{d:02d}"


def period_mask(
    keys: list[tuple[str, int]],
    period: str,
    *,
    catalog_index: dict[tuple[str, int], PageCatalogRow] | None = None,
) -> np.ndarray:
    """Boolean mask over `keys` selecting rows likely to overlap `period`.

    Two passes (a row passes if EITHER hits):
      1. bulletin month within `[period_start, period_end + PUBLISH_LAG_MONTHS]`,
         to capture retrospectives published after a period closes;
      2. when `catalog_index` is given, scan `dates` for any 4-digit year
         inside the period.

    Without `catalog_index`, only the bulletin-month check fires — still a
    big reduction vs. brute force on 89k rows.
    """
    intervals = period_to_intervals(period)
    bulletin_windows = [
        (start, _add_months(end, _PUBLISH_LAG_MONTHS)) for (start, end) in intervals
    ]

    period_years: set[int] = set()
    for start, end in intervals:
        try:
            period_years.update(range(int(start[:4]), int(end[:4]) + 1))
        except ValueError:
            continue

    mask = np.zeros(len(keys), dtype=bool)
    for i, (bulletin, page) in enumerate(keys):
        b_iso = _bulletin_to_iso(bulletin)
        if b_iso is not None:
            for w_start, w_end in bulletin_windows:
                if intervals_overlap(b_iso, b_iso, w_start, w_end):
                    mask[i] = True
                    break
        if mask[i]:
            continue
        if catalog_index is not None and period_years:
            row = catalog_index.get((bulletin, page))
            if row is not None and row.dates:
                for s in row.dates:
                    for m in _YEAR_RE.findall(s):
                        if int(m) in period_years:
                            mask[i] = True
                            break
                    if mask[i]:
                        break
    return mask


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    index: VectorIndex,
    query_vec: np.ndarray,
    mask: np.ndarray | None,
    top_k: int,
) -> list[tuple[float, tuple[str, int]]]:
    """Masked cosine top-K against `index.vectors`. Assumes `index.vectors`
    is L2-normalized; `query_vec` is L2-normalized inside. Returns
    `[(score, key), ...]` best-first."""
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    nq = np.linalg.norm(q)
    if nq == 0:
        return []
    q = q / nq

    if mask is None:
        scores = index.vectors @ q
        candidates = np.arange(scores.shape[0])
    else:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        scores = index.vectors[idx] @ q
        candidates = idx

    if top_k >= scores.shape[0]:
        order = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, top_k - 1)[:top_k]
        order = part[np.argsort(-scores[part])]

    out: list[tuple[float, tuple[str, int]]] = []
    for j in order:
        gi = int(candidates[j])
        out.append((float(scores[j]), index.keys[gi]))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    from skunk.common import load_env_file
    repo_root = Path(__file__).resolve().parents[3]
    load_env_file(repo_root / ".env")

    ap = argparse.ArgumentParser(
        description="Build a dense vector index over a PageIndex catalog.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build vectors.npz + vectors_keys.jsonl.")
    b.add_argument("--catalog-dir", type=Path,
                   default=repo_root / "cache/page_index")
    b.add_argument("--out-dir", type=Path,
                   default=repo_root / "cache/page_index")
    b.add_argument("--dim", type=int, default=768)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--batch-size", type=int, default=100)
    b.add_argument("--model", type=str, default="gemini-embedding-001")

    args = ap.parse_args(argv)

    if args.cmd == "build":
        cfg = SkunkConfig.from_env()
        llm = LLMClient(cfg)
        build_vector_index(
            args.catalog_dir, args.out_dir, llm,
            dim=args.dim, workers=args.workers,
            batch_size=args.batch_size, model=args.model,
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
