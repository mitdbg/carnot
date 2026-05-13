"""Offline concept-tree builder.

Reads the per-page catalog (one JSONL per bulletin in `cache/page_index/`),
groups distinct keyword phrases by section, embeds them, clusters per section
via pure-numpy k-means, and emits a single `concept_tree.json` shaped:

    {
      "sections": {
        "<section label>": {
          "clusters": {
            "C00": {
              "label": "<verbatim phrase closest to centroid>",
              "central_terms": ["...", "...", "..."],
              "terms": {
                "<verbatim keyword>": [
                  {"bulletin": "YYYY-MM", "page": int,
                   "description": "<page table_title verbatim>"},
                  ...
                ],
                ...
              }
            }, ...
          }
        }, ...
      },
      "meta": { "model": "gemini-embedding-001", "n_sections": N, ... }
    }

Per-section namespacing keeps a "Marketable" mention under "Statutory debt
limitation" distinct from "Marketable" in "Capital movements" — same word,
different concept space. (User decision.)

The page → cluster mapping is INTENTIONALLY many-to-many at the term level:
one page appears under every keyword it lifts, so a multi-concept page is
reachable through any of its facets.

CLI:
    python -m skunk.page_index.concept_tree \\
        --catalog-dir cache/page_index \\
        --out cache/page_index/concept_tree.json \\
        --embed-model gemini-embedding-001
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .retrieve_probe import load_catalog
from .schema import PageCatalogRow


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# Embedding via google.genai — small wrapper with batching + simple retry.
# ---------------------------------------------------------------------------

def embed_terms(
    terms: list[str],
    model: str = "gemini-embedding-001",
    batch_size: int = 64,
    verbose: bool = False,
) -> np.ndarray:
    """Return an (N, D) ndarray of unit-normalized embeddings for `terms`.

    Batches the API to keep request size sane and retries each batch on
    transient errors (mirrors the LLMClient retry shape in common.py).
    """
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)

    vectors: list[np.ndarray] = []
    n = len(terms)
    done = 0
    t0 = time.monotonic()
    for start in range(0, n, batch_size):
        batch = terms[start:start + batch_size]
        delay = 0.1
        for attempt in range(10):
            try:
                resp = client.models.embed_content(model=model, contents=batch)
                arrs = [np.array(e.values, dtype=np.float32) for e in resp.embeddings]
                vectors.extend(arrs)
                done += len(batch)
                break
            except Exception as e:
                if attempt == 9:
                    raise
                print(f"[embed] attempt {attempt+1}/10 failed "
                      f"({type(e).__name__}): {e}; sleeping {delay:.1f}s",
                      file=sys.stderr, flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 5.0)
        if verbose and done % 200 == 0:
            print(f"  [embed] {done}/{n} terms in {time.monotonic()-t0:.1f}s",
                  flush=True)

    if verbose:
        print(f"  [embed] {n}/{n} terms in {time.monotonic()-t0:.1f}s", flush=True)

    arr = np.vstack(vectors)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# ---------------------------------------------------------------------------
# Pure-numpy spherical k-means (cosine, on unit-normalized vectors).
# ---------------------------------------------------------------------------

def _kmeans_cosine(X: np.ndarray, k: int, *, n_iter: int = 25,
                   seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """k-means with cosine distance on unit-normalized X (shape N×D).
    Returns (assignments, centroids). Lloyd's algorithm; small k is fine.
    """
    n, d = X.shape
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    # k-means++ light: pick first center random, subsequent by farthest from existing.
    idx0 = int(rng.integers(0, n))
    centers = X[idx0:idx0 + 1].copy()
    for _ in range(1, k):
        sims = X @ centers.T  # (N, current_k)
        max_sim = sims.max(axis=1)
        dist = np.clip(1.0 - max_sim, 0.0, None)  # guard fp slop
        total = dist.sum()
        if total <= 1e-12:
            nxt = int(rng.integers(0, n))
        else:
            probs = dist / total
            nxt = int(rng.choice(n, p=probs))
        centers = np.vstack([centers, X[nxt:nxt + 1]])

    assign = np.zeros(n, dtype=np.int32)
    for _ in range(n_iter):
        sims = X @ centers.T            # (N, k)
        new_assign = sims.argmax(axis=1)
        if np.array_equal(new_assign, assign):
            assign = new_assign
            break
        assign = new_assign
        for j in range(k):
            mask = assign == j
            if mask.any():
                centers[j] = X[mask].mean(axis=0)
                norm = np.linalg.norm(centers[j])
                if norm > 0:
                    centers[j] = centers[j] / norm
    return assign, centers


# ---------------------------------------------------------------------------
# Section → cluster → term → postings assembly.
# ---------------------------------------------------------------------------

def _terms_in_section(rows: list[PageCatalogRow]) -> dict[str, list[tuple[str, int, str]]]:
    """Return {term → [(bulletin, page, description), ...]} for one section."""
    postings: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for r in rows:
        if r.page_kind not in ("table", "chart"):
            continue
        if not r.keywords:
            continue
        desc = r.table_title or ""
        for kw in r.keywords:
            term = kw.strip()
            if not term:
                continue
            postings[term].append((r.bulletin, r.page, desc))
    return postings


def _pick_k(n_terms: int) -> int:
    """k ≈ n/8, bounded to [3, 20]. Small sections still get a few clusters."""
    if n_terms <= 4:
        return min(n_terms, 2)
    return max(3, min(20, n_terms // 8))


def build_concept_tree(
    catalog: list[PageCatalogRow],
    *,
    embed_model: str = "gemini-embedding-001",
    verbose: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    by_section: dict[str | None, list[PageCatalogRow]] = defaultdict(list)
    for r in catalog:
        by_section[r.section].append(r)

    sections_out: dict[str, Any] = {}
    all_section_keys = sorted(
        [s for s in by_section if s],
        key=lambda s: -len(by_section[s]),
    )
    if verbose:
        print(f"[concept_tree] {len(all_section_keys)} sections "
              f"({sum(len(by_section[s]) for s in all_section_keys)} pages, "
              f"plus {len(by_section.get(None, []))} unsectioned)", flush=True)

    for section in all_section_keys:
        rows = by_section[section]
        postings = _terms_in_section(rows)
        terms = sorted(postings.keys())
        if not terms:
            continue
        if verbose:
            print(f"\n[section] {section!r}: {len(rows)} pages, "
                  f"{len(terms)} distinct terms", flush=True)

        embs = embed_terms(terms, model=embed_model, verbose=verbose)
        k = _pick_k(len(terms))
        assign, centers = _kmeans_cosine(embs, k, seed=seed)
        if verbose:
            print(f"  [cluster] k={k}", flush=True)

        clusters_out: dict[str, Any] = {}
        for j in range(k):
            mask = assign == j
            if not mask.any():
                continue
            cluster_terms_idx = np.where(mask)[0]
            # Order terms in this cluster by cosine to centroid.
            sims = (embs[cluster_terms_idx] @ centers[j]).reshape(-1)
            order = np.argsort(-sims)
            ordered_term_ids = cluster_terms_idx[order]
            ordered_terms = [terms[i] for i in ordered_term_ids]

            label = ordered_terms[0]
            central = ordered_terms[: min(3, len(ordered_terms))]

            terms_obj: dict[str, list[dict[str, Any]]] = {}
            for t in ordered_terms:
                terms_obj[t] = [
                    {"bulletin": b, "page": p, "description": d}
                    for (b, p, d) in postings[t]
                ]

            cid = f"C{j:02d}"
            clusters_out[cid] = {
                "label": label,
                "central_terms": central,
                "terms": terms_obj,
                "n_terms": len(ordered_terms),
                "n_pages": sum(len(v) for v in terms_obj.values()),
            }

        sections_out[section] = {
            "n_pages_in_section": len(rows),
            "n_terms": len(terms),
            "n_clusters": len(clusters_out),
            "clusters": clusters_out,
        }

    return {
        "meta": {
            "embed_model": embed_model,
            "n_sections": len(sections_out),
            "seed": seed,
        },
        "sections": sections_out,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    _load_env(Path(__file__).resolve().parents[3] / ".env")

    ap = argparse.ArgumentParser(description="Build concept tree from page-index catalog.")
    ap.add_argument("--catalog-dir", type=Path, default=Path("cache/page_index"))
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON path (default: <catalog-dir>/concept_tree.json)")
    ap.add_argument("--embed-model", default="gemini-embedding-001")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    catalog = load_catalog(args.catalog_dir)
    if not catalog:
        print(f"No catalog rows found under {args.catalog_dir}", file=sys.stderr)
        return 2
    print(f"Loaded {len(catalog)} catalog rows from {args.catalog_dir}", flush=True)

    tree = build_concept_tree(
        catalog,
        embed_model=args.embed_model,
        verbose=args.verbose,
        seed=args.seed,
    )

    out_path = args.out or (args.catalog_dir / "concept_tree.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tree, ensure_ascii=False, indent=2))

    n_terms = sum(s["n_terms"] for s in tree["sections"].values())
    n_clusters = sum(s["n_clusters"] for s in tree["sections"].values())
    print(f"\nWrote concept tree → {out_path}")
    print(f"  sections: {len(tree['sections'])}  "
          f"clusters: {n_clusters}  "
          f"distinct terms: {n_terms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
