"""v0.4 build pipeline — fully deterministic per-page extract, no LLM here.

Sample bulletins (default: ALL in the window) → per-page parse → persist.

CLI:
    python -m skunk.page_index.build \\
        --window 1950-1954 \\
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

from .classify import (
    char_metrics, cheap_classify, has_prose_content, has_visual_elements,
)  # noqa: E402
from .extract_fields import parse_page_fields  # noqa: E402
from .pdf import (  # noqa: E402
    parse_bulletin_filename, parsed_json_dir, read_page_elements,
    read_pdf_pages,
)
from .schema import PageCatalogRow  # noqa: E402
from .toc import harvest_toc, section_for_printed_page  # noqa: E402


def _extract_printed_page(elements: list[dict]) -> str | None:
    """Pull the printed-page footer label from a page's parsed-JSON
    elements (`type == "page_number"`). Returns None when no such
    element exists (front-matter / divider / blank pages typically).
    """
    for el in elements:
        if el.get("type") == "page_number":
            content = el.get("content")
            if content is not None:
                txt = str(content).strip()
                if txt:
                    return txt
    return None


_FILENAME_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})\.pdf$")

_CONT_SUFFIX_RE = re.compile(
    r"[,\s\-–—]+(?:con|cont|continued)\s*\.?\s*\)?\s*$", re.IGNORECASE,
)


def _merge_continuation_pages(rows: list[PageCatalogRow]) -> int:
    """Forward-fill table identity across continuation pages.

    A table/chart page is a continuation when EITHER:
      (a) its section banner or table_title ends with `con / cont / continued`, OR
      (b) page_kind ∈ {table, chart} but table_title is None AND the previous
          table/chart page has a non-empty table_title (caption-less continuation,
          ~9% of table pages — the layout parser drops the caption).

    For matched pages, UNION the parent's keywords onto the current row and
    replace `table_title` / `section` with the parent's. The page's own
    column_headers / row_headers_sample / dates are preserved.

    Returns the number of pages merged (for build-time stats).
    """
    rows.sort(key=lambda r: r.page)
    parent: PageCatalogRow | None = None
    merged = 0
    for r in rows:
        if r.page_kind not in ("table", "chart"):
            parent = None
            continue
        is_explicit_cont = bool(
            (r.table_title and _CONT_SUFFIX_RE.search(r.table_title)) or
            (r.section and _CONT_SUFFIX_RE.search(r.section))
        )
        is_implicit_cont = (
            r.table_title is None
            and parent is not None
            and parent.table_title is not None
        )
        if (is_explicit_cont or is_implicit_cont) and parent is not None:
            r.table_title = parent.table_title
            if parent.section:
                r.section = parent.section
            seen = {k.lower() for k in r.keywords}
            for k in parent.keywords:
                if k.lower() not in seen:
                    r.keywords.append(k)
                    seen.add(k.lower())
            merged += 1
            # Continuation pages also act as parents for further continuations
            # (e.g. a 3-page table). The parent identity is unchanged; we just
            # let the loop continue.
        else:
            # This page establishes a new identity if it has a table_title.
            if r.table_title:
                parent = r
            # else keep existing parent — a captionless table page that fails
            # the implicit-cont check (because there's no captioned parent yet)
            # stays as-is and doesn't reset the chain.
    return merged


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


def _process_one_bulletin(
    pdf_path: Path,
    verbose: bool,
    parsed_dir: Path | None = None,
    llm: LLMClient | None = None,
) -> list[PageCatalogRow]:
    bulletin_month = parse_bulletin_filename(pdf_path)
    pages = read_pdf_pages(pdf_path, parsed_dir=parsed_dir)
    page_elements = read_page_elements(pdf_path, parsed_dir=parsed_dir)

    # Stage 1: cheap classification (blank/toc) + char metrics.
    cheap_kinds: dict[int, str | None] = {}
    metrics: dict[int, tuple[int, float]] = {}
    for pdf_idx, text in pages.items():
        cc, dr = char_metrics(text)
        metrics[pdf_idx] = (cc, dr)
        cheap_kinds[pdf_idx] = cheap_classify(text, pdf_idx)

    # Stage 1.5: ToC harvest. One LLM call per bulletin → section spans in
    # the bulletin's own printed-page numbering. Each page's `printed_page`
    # is matched against these spans; pages outside every span (including
    # bulletins where harvest_toc returns nothing) land in 'Unsectioned' at
    # tree-build time. No per-page-banner fallback — the banner vocabulary
    # was the dominant L1 noise source.
    spans = []
    if llm is not None:
        toc_candidates = [
            (idx, pages[idx]) for idx, k in cheap_kinds.items() if k == "toc"
        ]
        if toc_candidates:
            spans = harvest_toc(
                bulletin_month=bulletin_month,
                toc_candidate_pages=toc_candidates,
                total_pdf_pages=len(pages),
                llm=llm,
            )

    # Stage 2: assemble rows. For every page that fell through cheap classify
    # AND has a visual element (table/figure/image), run the deterministic
    # parser. Pure-text pages get a row with page_kind="text" and no fields.
    rows: list[PageCatalogRow] = []
    n_table_or_chart = 0
    n_text = 0
    n_unsectioned = 0
    for pdf_idx, text in pages.items():
        cc, dr = metrics[pdf_idx]
        row = PageCatalogRow(
            bulletin=bulletin_month,
            page=pdf_idx,
            file_path=str(pdf_path.resolve()),
            char_count=cc,
            digit_ratio=round(dr, 3),
        )
        elements = page_elements.get(pdf_idx, [])
        row.printed_page = _extract_printed_page(elements)
        row.section = (section_for_printed_page(spans, row.printed_page)
                       if spans else None)
        if row.section is None:
            n_unsectioned += 1

        cheap_k = cheap_kinds[pdf_idx]
        if cheap_k is not None:
            row.page_kind = cheap_k  # "blank" or "toc"
            rows.append(row)
            continue

        if not (has_visual_elements(elements) or has_prose_content(elements)):
            row.page_kind = "text"
            n_text += 1
            rows.append(row)
            continue

        fields = parse_page_fields(elements)
        row.page_kind = fields["page_kind"]
        if row.page_kind in ("table", "chart", "prose"):
            row.table_title = fields.get("table_title")
            row.column_headers = fields.get("column_headers", [])
            row.row_headers_sample = fields.get("row_headers_sample", [])
            row.keywords = fields.get("keywords", [])
            row.dates = fields.get("dates", [])
            n_table_or_chart += 1
        else:
            n_text += 1
        rows.append(row)

    # Forward-fill continuation pages so the parent table's keywords also
    # point to its continuation pages (multi-page tables). This is a
    # preprocessing-time fix; the merge changes are local to this bulletin.
    n_merged = _merge_continuation_pages(rows)

    if verbose:
        print(f"  [{bulletin_month}] {n_table_or_chart} table/chart pages, "
              f"{n_text} text pages, {len(rows)} total, "
              f"{n_merged} continuation merges, {len(spans)} ToC spans, "
              f"{n_unsectioned} unsectioned")
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
    ap = argparse.ArgumentParser(
        description="v0.4 build page-index catalog for bulletins in a window.")
    ap.add_argument("--window", type=_parse_window, required=True,
                    help="Inclusive year window, e.g. 1950-1954.")
    ap.add_argument("--n", type=int, default=None,
                    help="If set, sample this many bulletins from the window "
                         "(debug). Default: index ALL bulletins in window.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the --n sample (only used if --n set).")
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
                    help="Comma-separated list of YYYY-MM to override (debug).")
    ap.add_argument("--workers", type=int, default=8,
                    help="Bulletin-level parallelism (each bulletin is processed independently).")
    ap.add_argument("--no-toc", action="store_true",
                    help="Skip the per-bulletin harvest_toc LLM call. "
                         "Falls back to per-page banner sections only.")
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
    elif args.n is not None:
        rng = random.Random(args.seed)
        n = min(args.n, len(windowed))
        chosen = sorted(rng.sample(windowed, n), key=lambda p: p.name)
    else:
        chosen = list(windowed)

    args.out.mkdir(parents=True, exist_ok=True)

    parsed_dir: Path = (args.parsed_json_dir
                        if args.parsed_json_dir is not None
                        else parsed_json_dir())
    if not parsed_dir.is_dir():
        print(f"Parsed-JSON dir does not exist: {parsed_dir}", file=sys.stderr)
        return 2

    llm: LLMClient | None = None
    if not args.no_toc:
        cfg = SkunkConfig.from_env()
        llm = LLMClient(cfg)
        print(f"Building page index for {len(chosen)} bulletins → {args.out}")
        print(f"  page text source: parsed-JSON @ {parsed_dir}")
        print(f"  workers: {args.workers}    harvest_toc: on ({cfg.llm_model})\n")
    else:
        print(f"Building page index for {len(chosen)} bulletins → {args.out}")
        print(f"  page text source: parsed-JSON @ {parsed_dir}")
        print(f"  workers: {args.workers}    harvest_toc: off (--no-toc)\n")

    started = time.monotonic()
    summary: list[dict] = []

    def _do_one(pdf_path: Path) -> tuple[Path, list[PageCatalogRow] | None, float, str | None]:
        bulletin = parse_bulletin_filename(pdf_path)
        t0 = time.monotonic()
        try:
            rows = _process_one_bulletin(
                pdf_path=pdf_path, verbose=args.verbose, parsed_dir=parsed_dir,
                llm=llm,
            )
        except Exception as e:  # noqa: BLE001
            return pdf_path, None, time.monotonic() - t0, f"{type(e).__name__}: {e}"
        return pdf_path, rows, time.monotonic() - t0, None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_do_one, p): p for p in chosen}
        for f in as_completed(futs):
            pdf_path, rows, dt, err = f.result()
            bulletin = parse_bulletin_filename(pdf_path)
            if err is not None or rows is None:
                print(f"[error] {bulletin}: {err}", file=sys.stderr, flush=True)
                summary.append({"bulletin": bulletin, "status": "error", "error": err})
                continue
            out_path = _persist_bulletin(rows, args.out)
            kinds_count: dict[str, int] = {}
            for r in rows:
                kinds_count[r.page_kind] = kinds_count.get(r.page_kind, 0) + 1
            print(f"[done] {bulletin}: {len(rows)} pages in {dt:.1f}s "
                  f"kinds={kinds_count} → {out_path}")
            summary.append({
                "bulletin": bulletin, "status": "ok", "n_pages": len(rows),
                "kinds": kinds_count, "duration_s": round(dt, 2),
                "out_path": str(out_path),
            })

    total_dt = time.monotonic() - started
    manifest = {
        "window": list(args.window),
        "n_requested": args.n,
        "n_built": sum(1 for s in summary if s["status"] == "ok"),
        "bulletins": sorted(summary, key=lambda s: s["bulletin"]),
        "duration_s": round(total_dt, 2),
        "extract_mode": "deterministic_v0.4",
    }
    manifest_path = args.out / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest → {manifest_path}  (total {total_dt:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
