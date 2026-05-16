"""End-to-end pipeline: parsed-JSON corpus → flat indexed Treasury Bulletin tree.

A single CLI runs every stage. Outputs land under `--output-dir`:

    <output-dir>/
      catalog/{YYYY-MM}.jsonl × N        per-bulletin catalog rows
      l1/{YYYY-MM}.json × N              per-bulletin L1 chapter spans (Phase 1)
      concept_tree.json                   final flat tree (Phase 3)
      manifest.json                       build metadata

Stages (each idempotent):

  build_catalog   per-bulletin parse → catalog/*.jsonl with page metadata.
                  NO LLM at this stage; banner_self and printed_page
                  populate from the parsed JSON deterministically.

  extract_l1      Phase 1. Per-bulletin strict-L1 ToC harvest. One LLM
                  call per bulletin returns the top-level chapter spans
                  (printed_page → chapter name). Written to l1/*.json.

  place_pages     Phase 2. Per-page placement cascade. For each content
                  page: Method A (printed-page span lookup against the
                  bulletin's L1), Method B (banner_self exact match),
                  fallback (a1/a2) typo correction, (b) neighbor
                  inheritance, (c) LLM prediction. Mutates catalog/*.jsonl
                  in place to set `l1_local` on every row.

  merge_chapters  Phase 3. Cross-bulletin merge of L1 names into a flat
                  global chapter set. Deterministic normalize → one LLM
                  clustering call → flat tree written to concept_tree.json.

  manifest        Build metadata.

Typical invocation:

    python -m skunk.page_index.pipeline \\
        --output-dir cache/page_index_v3/ \\
        --workers 16

Estimated cost ~$0.55 (extract_l1 $0.40 + place_pages ~$0.10 + merge ~$0.05).
Wall-clock ~10–15 min with 16 workers.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from skunk.common import LLMClient, load_env_file
from skunk.config import SkunkConfig

from .build import (
    _filter_window, _list_corpus_pdfs, _parse_window,
    _persist_bulletin, _process_one_bulletin,
)
from .classify import looks_like_toc
from .merge import build_tree as merge_build_tree
from .pdf import (
    parse_bulletin_filename, parsed_json_dir, pdf_path_for, read_pdf_pages,
)
from .place import place_all
from .schema import PageCatalogRow
from .toc import SectionSpan, harvest_toc


_REPO_ROOT = Path(__file__).resolve().parents[3]
load_env_file(_REPO_ROOT / ".env")


STAGES = ("build_catalog", "extract_l1", "place_pages",
          "merge_chapters", "manifest")


def _config_with_model(model: str | None) -> SkunkConfig:
    """Return a SkunkConfig with `llm_model` optionally overridden."""
    cfg = SkunkConfig.from_env()
    if model and model != cfg.llm_model:
        cfg = dataclasses.replace(cfg, llm_model=model)
    return cfg


# ---------------------------------------------------------------------------
# Stage: build_catalog
# ---------------------------------------------------------------------------

def stage_build_catalog(args: argparse.Namespace) -> None:
    """Catalog every bulletin into <out>/catalog/. NO LLM at this stage."""
    print("\n=== Stage 1: build_catalog ===", flush=True)
    pdf_dir: Path = args.pdf_dir
    if not pdf_dir.is_dir():
        sys.exit(f"PDF dir does not exist: {pdf_dir}")
    parsed_dir = args.parsed_json_dir or parsed_json_dir()
    if not parsed_dir.is_dir():
        sys.exit(f"Parsed-JSON dir does not exist: {parsed_dir}")

    all_pdfs = _list_corpus_pdfs(pdf_dir)
    chosen = _filter_window(all_pdfs, args.window)
    if args.bulletins:
        wanted = {b.strip() for b in args.bulletins.split(",") if b.strip()}
        chosen = [p for p in chosen if parse_bulletin_filename(p) in wanted]
        missing = wanted - {parse_bulletin_filename(p) for p in chosen}
        if missing:
            sys.exit(f"Requested bulletins not in window: {sorted(missing)}")
    if not chosen:
        sys.exit(f"No bulletins matched window {args.window} / --bulletins")

    out_dir = args.output_dir / "catalog"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {len(chosen)} bulletins → {out_dir}")
    print(f"  parsed-JSON: {parsed_dir}")
    print(f"  workers: {args.workers}    (no LLM at this stage)\n", flush=True)

    t0 = time.monotonic()
    n_ok = n_err = 0

    def _do_one(pdf_path: Path) -> tuple[Path, Path | None, str | None]:
        try:
            rows = _process_one_bulletin(
                pdf_path=pdf_path, verbose=args.verbose,
                parsed_dir=parsed_dir, llm=None,  # ToC harvest moved to Phase 1
            )
            out_path = _persist_bulletin(rows, out_dir)
            return pdf_path, out_path, None
        except Exception as e:  # noqa: BLE001
            return pdf_path, None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_do_one, p) for p in chosen]
        for f in as_completed(futs):
            pdf_path, out_path, err = f.result()
            bulletin = parse_bulletin_filename(pdf_path)
            if err:
                n_err += 1
                print(f"  [err] {bulletin}: {err}", file=sys.stderr, flush=True)
            else:
                n_ok += 1
                if args.verbose:
                    print(f"  [ok]  {bulletin} → {out_path}", flush=True)
    print(f"  built {n_ok} bulletins, {n_err} errors "
          f"in {time.monotonic() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Stage: extract_l1 (Phase 1)
# ---------------------------------------------------------------------------

def _save_l1(path: Path, spans: list[SectionSpan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [dataclasses.asdict(sp) for sp in spans]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_l1(path: Path) -> list[SectionSpan]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [SectionSpan(**item) for item in raw]


def _l1_from_banners(
    catalog_path: Path, *, min_pages: int = 3, max_l1: int = 25,
) -> list[SectionSpan]:
    """Fallback L1 derivation when harvest_toc returns nothing.

    Some bulletins (notably 1949–1964) lack a usable ToC page in the
    parsed JSON. Their body pages still print clean section banners in
    [page_header] / [title] elements, so we count banner_self occurrences
    and treat any banner appearing on >= min_pages as an L1 entry.

    Returns SectionSpans with empty `start_page_printed` / `end_page_printed`.
    The span-lookup placement Method (A) in place.py skips these
    (non-numeric printed-page label), but Method B (banner_self exact
    match) handles them directly.
    """
    if not catalog_path.exists():
        return []
    counts: Counter[str] = Counter()
    for line in catalog_path.open():
        line = line.strip()
        if not line:
            continue
        row = PageCatalogRow.from_json(line)
        if not row.content_blocks:
            continue
        b = (row.banner_self or "").strip()
        if b:
            counts[b] += 1
    spans: list[SectionSpan] = []
    for banner, n in counts.most_common(max_l1):
        if n < min_pages:
            break
        spans.append(SectionSpan(
            section=banner, start_page_printed="", end_page_printed="",
        ))
    return spans


def stage_extract_l1(args: argparse.Namespace) -> None:
    """Phase 1: per-bulletin strict-L1 ToC harvest."""
    print("\n=== Stage 2: extract_l1 (Phase 1) ===", flush=True)
    catalog_dir = args.output_dir / "catalog"
    if not catalog_dir.is_dir():
        sys.exit(f"No catalog at {catalog_dir} — run build_catalog first.")
    parsed_dir = args.parsed_json_dir or parsed_json_dir()
    l1_dir = args.output_dir / "l1"
    l1_dir.mkdir(parents=True, exist_ok=True)

    bulletins = sorted(p.stem for p in catalog_dir.glob("*.jsonl"))
    print(f"  {len(bulletins)} bulletins → {l1_dir}", flush=True)

    cfg = _config_with_model(args.l1_model)
    llm = LLMClient(cfg)
    print(f"  model: {cfg.llm_model}\n", flush=True)
    t0 = time.monotonic()

    def _do_one(bulletin: str) -> tuple[str, int, str, str | None]:
        try:
            pdf_path = pdf_path_for(bulletin, args.pdf_dir)
            pages = read_pdf_pages(pdf_path, parsed_dir=parsed_dir)
            toc_candidates = [
                (idx, text) for idx, text in pages.items()
                if looks_like_toc(text, idx)
            ]
            source = "toc"
            spans = harvest_toc(
                bulletin_month=bulletin,
                toc_candidate_pages=toc_candidates,
                total_pdf_pages=len(pages),
                llm=llm,
            ) if toc_candidates else []
            if not spans:
                # Fallback: derive L1 vocabulary from the bulletin's own
                # per-page banner frequency.
                catalog_path = catalog_dir / f"{bulletin}.jsonl"
                spans = _l1_from_banners(catalog_path)
                source = "banners" if spans else "empty"
            _save_l1(l1_dir / f"{bulletin}.json", spans)
            return bulletin, len(spans), source, None
        except Exception as e:  # noqa: BLE001
            return bulletin, 0, "error", f"{type(e).__name__}: {e}"

    n_ok = n_err = 0
    n_empty = 0
    src_counts: Counter[str] = Counter()
    span_dist: list[int] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_do_one, b) for b in bulletins]
        for f in as_completed(futs):
            bulletin, n_spans, source, err = f.result()
            if err:
                n_err += 1
                print(f"  [err] {bulletin}: {err}", file=sys.stderr, flush=True)
            else:
                n_ok += 1
                span_dist.append(n_spans)
                src_counts[source] += 1
                if n_spans == 0:
                    n_empty += 1
                if args.verbose:
                    print(f"  [ok]  {bulletin}: {n_spans} L1 spans "
                          f"(source={source})", flush=True)

    if span_dist:
        span_dist.sort()
        mid = span_dist[len(span_dist) // 2]
        src_summary = ", ".join(f"{s}={n}" for s, n in src_counts.most_common())
        print(f"  done {n_ok}/{len(bulletins)} bulletins, {n_err} errors, "
              f"{n_empty} with 0 spans, median {mid} spans  ({src_summary}) "
              f"in {time.monotonic() - t0:.1f}s", flush=True)
    else:
        print(f"  done {n_ok}/{len(bulletins)}, {n_err} errors", flush=True)


# ---------------------------------------------------------------------------
# Stage: place_pages (Phase 2)
# ---------------------------------------------------------------------------

def _load_catalog_by_bulletin(catalog_dir: Path) -> dict[str, list[PageCatalogRow]]:
    out: dict[str, list[PageCatalogRow]] = {}
    for p in sorted(catalog_dir.glob("*.jsonl")):
        rows: list[PageCatalogRow] = []
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            rows.append(PageCatalogRow.from_json(line))
        if rows:
            out[rows[0].bulletin] = rows
    return out


def _persist_catalog_by_bulletin(
    catalog_by_bulletin: dict[str, list[PageCatalogRow]], catalog_dir: Path,
) -> None:
    """Final on-disk write at the end of Phase 2. Strips `banner_self`
    before serializing — that field is internal-only (build → extract_l1
    → place_pages) and has no external consumer."""
    for bulletin, rows in catalog_by_bulletin.items():
        path = catalog_dir / f"{bulletin}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(r.to_json(drop_banner_self=True))
                f.write("\n")


def stage_place_pages(args: argparse.Namespace) -> None:
    """Phase 2: per-page placement cascade. Mutates catalog/*.jsonl."""
    print("\n=== Stage 3: place_pages (Phase 2) ===", flush=True)
    catalog_dir = args.output_dir / "catalog"
    l1_dir = args.output_dir / "l1"
    if not catalog_dir.is_dir() or not l1_dir.is_dir():
        sys.exit("catalog/ or l1/ missing — run earlier stages first.")

    catalog_by_bulletin = _load_catalog_by_bulletin(catalog_dir)
    l1_by_bulletin: dict[str, list[SectionSpan]] = {}
    for bulletin in catalog_by_bulletin:
        l1_by_bulletin[bulletin] = _load_l1(l1_dir / f"{bulletin}.json")

    print(f"  {len(catalog_by_bulletin)} bulletins loaded; "
          f"sum L1 spans = {sum(len(v) for v in l1_by_bulletin.values())}",
          flush=True)
    cfg = _config_with_model(args.place_model)
    llm = LLMClient(cfg)
    print(f"  cascade model: {cfg.llm_model}", flush=True)

    place_all(
        catalog_by_bulletin, l1_by_bulletin, llm,
        workers=args.workers, verbose=True,
    )
    _persist_catalog_by_bulletin(catalog_by_bulletin, catalog_dir)
    print(f"  persisted updated catalog → {catalog_dir}", flush=True)


# ---------------------------------------------------------------------------
# Stage: merge_chapters (Phase 3)
# ---------------------------------------------------------------------------

def stage_merge_chapters(args: argparse.Namespace) -> None:
    """Phase 3: cross-bulletin merge → flat concept_tree.json."""
    print("\n=== Stage 4: merge_chapters (Phase 3) ===", flush=True)
    catalog_dir = args.output_dir / "catalog"
    if not catalog_dir.is_dir():
        sys.exit("catalog/ missing — run earlier stages first.")

    catalog: list[PageCatalogRow] = []
    for p in sorted(catalog_dir.glob("*.jsonl")):
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            catalog.append(PageCatalogRow.from_json(line))
    print(f"  {len(catalog)} catalog rows loaded", flush=True)

    cfg = _config_with_model(args.merge_model)
    llm = LLMClient(cfg)
    print(f"  merge model: {cfg.llm_model}", flush=True)

    tree = merge_build_tree(catalog, llm, drop_unfiled=True, verbose=True)
    out_path = args.output_dir / "concept_tree.json"
    out_path.write_text(json.dumps(tree, ensure_ascii=False, indent=2))
    print(f"  wrote tree → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Stage: manifest
# ---------------------------------------------------------------------------

def stage_manifest(args: argparse.Namespace) -> None:
    """Emit build metadata: page counts, file hashes, git sha, models."""
    print("\n=== Stage 5: manifest ===", flush=True)
    out = args.output_dir
    tree = json.loads((out / "concept_tree.json").read_text())
    chapter_pages = {ch: data["n_pages"]
                     for ch, data in tree["chapters"].items()}
    total_pages = sum(chapter_pages.values())

    catalog_rows = 0
    for p in (out / "catalog").glob("*.jsonl"):
        catalog_rows += sum(1 for _ in p.open())
    n_l1_files = len(list((out / "l1").glob("*.json")))

    tree_hash = hashlib.sha256(
        (out / "concept_tree.json").read_bytes()
    ).hexdigest()[:16]
    git_sha = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    cfg = SkunkConfig.from_env()
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": git_sha[:12] if git_sha else None,
        "pdf_dir": str(args.pdf_dir),
        "corpus": {
            "pdf_dir": str(args.pdf_dir),
            "bulletins": len(list((out / "catalog").glob("*.jsonl"))),
            "window": list(args.window),
        },
        "tree": {
            "path": "concept_tree.json",
            "sha256_first16": tree_hash,
            "n_chapters": len(chapter_pages),
            "n_pages_indexed": total_pages,
            "chapter_page_counts": chapter_pages,
        },
        "catalog": {
            "path": "catalog/",
            "n_files": len(list((out / "catalog").glob("*.jsonl"))),
            "n_rows": catalog_rows,
        },
        "l1": {
            "path": "l1/",
            "n_files": n_l1_files,
        },
        "models": {
            "extract_l1": args.l1_model or cfg.llm_model,
            "place_pages": args.place_model or cfg.llm_model,
            "merge_chapters": args.merge_model or cfg.llm_model,
            "retriever_runtime": cfg.llm_model,
        },
    }
    out_path = out / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  wrote manifest → {out_path}", flush=True)
    print(f"  chapters: {len(chapter_pages)}    pages: {total_pages}    "
          f"catalog rows: {catalog_rows}    L1 files: {n_l1_files}",
          flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_STAGE_FUNCS = {
    "build_catalog": stage_build_catalog,
    "extract_l1": stage_extract_l1,
    "place_pages": stage_place_pages,
    "merge_chapters": stage_merge_chapters,
    "manifest": stage_manifest,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="End-to-end three-phase page-index pipeline.")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("cache/page_index"),
                    help="Target directory for all stage outputs.")
    ap.add_argument("--window", type=_parse_window, default=(1939, 2025),
                    help="Inclusive year window (default 1939-2025).")
    ap.add_argument("--bulletins", type=str, default=None,
                    help="Comma-separated YYYY-MM to override (debug).")
    ap.add_argument("--pdf-dir", type=Path,
                    default=Path(os.environ.get(
                        "OFFICEQA_PDF_DIR",
                        os.path.expanduser("~/Desktop/officeqa/treasury_bulletin_pdfs"))))
    ap.add_argument("--parsed-json-dir", type=Path, default=None,
                    help="Parsed-JSON corpus dir; defaults to env/parsed_json_dir().")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--l1-model", type=str,
                    default="google/gemini-3-flash-preview",
                    help="LLM model for Phase 1 strict-L1 ToC harvest.")
    ap.add_argument("--place-model", type=str,
                    default="google/gemini-3-flash-preview",
                    help="LLM model for Phase 2 typo / prediction.")
    ap.add_argument("--merge-model", type=str,
                    default="google/gemini-3-flash-preview",
                    help="LLM model for Phase 3 chapter clustering.")
    ap.add_argument("--start-from", choices=STAGES, default=None,
                    help="Skip stages prior to this one (their outputs must exist).")
    for s in STAGES:
        ap.add_argument(f"--skip-{s.replace('_', '-')}", action="store_true",
                        dest=f"skip_{s}")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.start_from:
        start_idx = STAGES.index(args.start_from)
    else:
        start_idx = 0
    to_run = [s for s in STAGES[start_idx:]
              if not getattr(args, f"skip_{s}", False)]

    print(f"Pipeline → {args.output_dir}", flush=True)
    print(f"  window: {args.window}    workers: {args.workers}", flush=True)
    print(f"  stages: {to_run}", flush=True)

    t0 = time.monotonic()
    for stage in to_run:
        _STAGE_FUNCS[stage](args)
    print(f"\nPipeline complete in {time.monotonic() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
