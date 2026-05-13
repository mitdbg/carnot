"""Build pipeline — sample bulletins → harvest TOC → per-page extract → persist.

CLI:
    python -m skunk.page_index.build \\
        --window 1950-1954 \\
        --n 20 \\
        --seed 42 \\
        --out cache/page_index

Output:
    cache/page_index/{YYYY-MM}.jsonl   one row per page (PageCatalogRow.to_json)
    cache/page_index/manifest.json     metadata about the build run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _load_env(path: Path) -> None:
    """Mirror of eval_e2e._load_env — picks up GEMINI_API_KEY etc. from .env."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_REPO_ROOT = Path(__file__).resolve().parents[3]
_load_env(_REPO_ROOT / ".env")

from skunk.common import LLMClient  # noqa: E402
from skunk.config import SkunkConfig  # noqa: E402

from .classify import char_metrics, cheap_classify
from .extract_fields import extract_page_fields
from .pdf import (
    parse_bulletin_filename, parsed_json_dir, read_page_sections, read_pdf_pages,
)
from .schema import PageCatalogRow
from .toc import SectionSpan, harvest_toc, section_for_page


_FILENAME_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})\.pdf$")


def _list_corpus_pdfs(pdf_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(pdf_dir.iterdir()):
        if _FILENAME_RE.search(p.name):
            out.append(p)
    return out


def _filter_window(pdfs: list[Path], window: tuple[int, int]) -> list[Path]:
    y_lo, y_hi = window
    keep: list[Path] = []
    for p in pdfs:
        m = _FILENAME_RE.search(p.name)
        if not m:
            continue
        year = int(m.group(1))
        if y_lo <= year <= y_hi:
            keep.append(p)
    return keep


def _parse_window(s: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{4})-(\d{4})", s)
    if not m:
        raise argparse.ArgumentTypeError(f"--window must be YYYY-YYYY, got {s!r}")
    lo, hi = int(m.group(1)), int(m.group(2))
    if hi < lo:
        raise argparse.ArgumentTypeError(f"--window end < start: {s}")
    return lo, hi


def _bulletin_month_str(bulletin: str) -> str:
    """'1953-06' is already month-form; return as-is. Single source of truth."""
    return bulletin


def _is_retrospective(periods_covered, bulletin_month: str) -> bool:
    """A page is 'retrospective' if every period it reports on ends before its
    publication month (i.e. the page is summarizing already-completed periods,
    not the bulletin's own publication month)."""
    if not periods_covered:
        return False
    bulletin_end_iso = f"{bulletin_month}-28"  # conservative end-of-month proxy
    return all(p.end < bulletin_end_iso for p in periods_covered)


def _process_one_bulletin(
    pdf_path: Path,
    llm: LLMClient,
    n_workers: int,
    verbose: bool,
    parsed_dir: Path | None = None,
) -> list[PageCatalogRow]:
    bulletin_month = parse_bulletin_filename(pdf_path)
    pages = read_pdf_pages(pdf_path, parsed_dir=parsed_dir)
    page_sections = read_page_sections(pdf_path, parsed_dir=parsed_dir)
    n_pages = len(pages)

    # Stage 1: cheap classification.
    cheap_kinds: dict[int, str | None] = {}
    metrics: dict[int, tuple[int, float]] = {}
    for pdf_idx, text in pages.items():
        cc, dr = char_metrics(text)
        metrics[pdf_idx] = (cc, dr)
        cheap_kinds[pdf_idx] = cheap_classify(text, pdf_idx)

    toc_candidates = [(idx, pages[idx]) for idx, k in cheap_kinds.items() if k == "toc"]

    # Stage 2: TOC harvest (one LLM call per bulletin).
    t0 = time.monotonic()
    spans: list[SectionSpan] = harvest_toc(
        bulletin_month=bulletin_month,
        toc_candidate_pages=toc_candidates,
        total_pdf_pages=n_pages,
        llm=llm,
    )
    if verbose:
        print(
            f"  [toc] {bulletin_month}: {len(spans)} sections in "
            f"{time.monotonic() - t0:.1f}s ({len(toc_candidates)} TOC candidate pages)"
        )

    # Stage 3: per-page LLM extract for pages that fell through the heuristic.
    targets = [idx for idx, k in cheap_kinds.items() if k is None]

    extracted_fields: dict[int, dict] = {}

    def _worker(idx: int) -> tuple[int, dict]:
        return idx, extract_page_fields(
            bulletin_month=bulletin_month,
            pdf_page=idx,
            page_text=pages[idx],
            llm=llm,
        )

    if targets:
        t1 = time.monotonic()
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_worker, idx) for idx in targets]
            done = 0
            for f in as_completed(futs):
                idx, fields = f.result()
                extracted_fields[idx] = fields
                done += 1
                if verbose and done % 25 == 0:
                    print(
                        f"  [extract] {bulletin_month}: {done}/{len(targets)} "
                        f"pages in {time.monotonic() - t1:.1f}s"
                    )
        if verbose:
            kinds_count: dict[str, int] = {}
            for f in extracted_fields.values():
                k = f.get("page_kind", "?")
                kinds_count[k] = kinds_count.get(k, 0) + 1
            print(
                f"  [extract] {bulletin_month}: done {len(targets)} pages in "
                f"{time.monotonic() - t1:.1f}s | kinds={kinds_count}"
            )

    # Stage 4: assemble PageCatalogRows.
    rows: list[PageCatalogRow] = []
    for pdf_idx, text in pages.items():
        cc, dr = metrics[pdf_idx]
        row = PageCatalogRow(
            bulletin=bulletin_month,
            page=pdf_idx,
            file_path=str(pdf_path.resolve()),
            char_count=cc,
            digit_ratio=round(dr, 3),
        )
        # Section: prefer per-page banner from parsed JSON (ground truth);
        # fall back to TOC-inferred span if the page has no banner.
        row.section = page_sections.get(pdf_idx) or section_for_page(spans, pdf_idx)

        cheap_k = cheap_kinds[pdf_idx]
        if cheap_k is not None:
            row.page_kind = cheap_k  # "blank" or "toc"
            rows.append(row)
            continue

        fields = extracted_fields.get(pdf_idx) or {"page_kind": "text"}
        row.page_kind = fields["page_kind"]
        if row.page_kind in ("table", "chart"):
            row.table_title = fields.get("table_title")
            row.column_headers = fields.get("column_headers", [])
            row.row_headers_sample = fields.get("row_headers_sample", [])
            row.keywords = fields.get("keywords", [])
            row.periods_covered = fields.get("periods_covered", [])
            row.granularity = fields.get("granularity", "unknown")
            row.is_retrospective = _is_retrospective(row.periods_covered, bulletin_month)
        rows.append(row)

    return rows


def _persist_bulletin(rows: list[PageCatalogRow], out_dir: Path) -> Path:
    if not rows:
        raise ValueError("no rows to persist")
    bulletin = rows[0].bulletin
    path = out_dir / f"{bulletin}.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(r.to_json())
            f.write("\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Build page-index catalog for a sample of bulletins.")
    ap.add_argument("--window", type=_parse_window, required=True,
                    help="Inclusive year window, e.g. 1950-1954.")
    ap.add_argument("--n", type=int, default=20,
                    help="Number of bulletins to sample from the window.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the sample.")
    ap.add_argument("--pdf-dir", type=Path,
                    default=Path(os.environ.get(
                        "OFFICEQA_PDF_DIR",
                        os.path.expanduser("~/Desktop/officeqa/treasury_bulletin_pdfs"))),
                    help="Directory holding treasury_bulletin_*.pdf files.")
    ap.add_argument("--parsed-json-dir", type=Path, default=None,
                    help="Directory holding the parsed-JSON cache "
                         "(default: $OFFICEQA_PARSED_JSON_DIR, then "
                         "~/Desktop/officeqa/treasury_bulletins_parsed/jsons/).")
    ap.add_argument("--out", type=Path, default=Path("cache/page_index"),
                    help="Output directory for the JSONL catalog.")
    ap.add_argument("--bulletins", type=str, default=None,
                    help="Comma-separated list of YYYY-MM to override the random sample (debug).")
    ap.add_argument("--workers", type=int, default=8,
                    help="Threadpool size per bulletin (extract stage).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    pdf_dir: Path = args.pdf_dir
    if not pdf_dir.is_dir():
        print(f"PDF dir does not exist: {pdf_dir}", file=sys.stderr)
        return 2

    all_pdfs = _list_corpus_pdfs(pdf_dir)
    windowed = _filter_window(all_pdfs, args.window)
    if not windowed:
        print(f"No bulletins found in window {args.window[0]}-{args.window[1]} "
              f"under {pdf_dir}", file=sys.stderr)
        return 2

    if args.bulletins:
        wanted = {b.strip() for b in args.bulletins.split(",") if b.strip()}
        chosen = [p for p in windowed if parse_bulletin_filename(p) in wanted]
        missing = wanted - {parse_bulletin_filename(p) for p in chosen}
        if missing:
            print(f"Requested bulletins not in window: {sorted(missing)}", file=sys.stderr)
            return 2
    else:
        rng = random.Random(args.seed)
        n = min(args.n, len(windowed))
        chosen = sorted(rng.sample(windowed, n), key=lambda p: p.name)

    args.out.mkdir(parents=True, exist_ok=True)

    parsed_dir: Path = (args.parsed_json_dir
                        if args.parsed_json_dir is not None
                        else parsed_json_dir())
    if not parsed_dir.is_dir():
        print(f"Parsed-JSON dir does not exist: {parsed_dir}", file=sys.stderr)
        return 2

    cfg = SkunkConfig.from_env()
    llm = LLMClient(cfg)

    print(f"Building page index for {len(chosen)} bulletins → {args.out}")
    print(f"  page text source: parsed-JSON @ {parsed_dir}")
    for p in chosen:
        print(f"  - {parse_bulletin_filename(p)}  ({p.name})")
    print()

    started = time.monotonic()
    summary: list[dict] = []
    for pdf_path in chosen:
        bulletin = parse_bulletin_filename(pdf_path)
        t0 = time.monotonic()
        try:
            rows = _process_one_bulletin(
                pdf_path=pdf_path,
                llm=llm,
                n_workers=args.workers,
                verbose=args.verbose,
                parsed_dir=parsed_dir,
            )
        except Exception as e:
            print(f"[error] {bulletin}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            summary.append({"bulletin": bulletin, "status": "error", "error": str(e)})
            continue
        out_path = _persist_bulletin(rows, args.out)
        kinds_count: dict[str, int] = {}
        for r in rows:
            kinds_count[r.page_kind] = kinds_count.get(r.page_kind, 0) + 1
        dt = time.monotonic() - t0
        print(f"[done] {bulletin}: {len(rows)} pages in {dt:.1f}s "
              f"kinds={kinds_count} → {out_path}")
        summary.append({
            "bulletin": bulletin,
            "status": "ok",
            "n_pages": len(rows),
            "kinds": kinds_count,
            "duration_s": round(dt, 2),
            "out_path": str(out_path),
        })

    total_dt = time.monotonic() - started
    manifest = {
        "window": list(args.window),
        "n_requested": args.n,
        "n_built": sum(1 for s in summary if s["status"] == "ok"),
        "seed": args.seed,
        "bulletins": summary,
        "duration_s": round(total_dt, 2),
        "gemini_model": cfg.gemini_model,
    }
    manifest_path = args.out / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest → {manifest_path}  (total {total_dt:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
