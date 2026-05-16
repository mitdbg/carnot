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


from skunk.common import load_env_file  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
load_env_file(_REPO_ROOT / ".env")

from .classify import (
    char_metrics, cheap_classify, has_prose_content, has_visual_elements,
)  # noqa: E402
from .extract_fields import parse_page_fields  # noqa: E402
from .pdf import (  # noqa: E402
    _FILENAME_RE, parse_bulletin_filename, parsed_json_dir,
    read_page_elements, read_pdf_pages,
)
from .schema import PageCatalogRow  # noqa: E402


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


# Banners that are obviously not section labels: month-year cover headers
# ("May 1972"), bare years, short fragments, and the recurring publication
# masthead ("Treasury Bulletin") that appears as [page_header] on every
# page. These would pollute the canonicalization pool if treated as
# section banners.
_NOISE_BANNER_RE = re.compile(
    r"^\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec|"
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{4}\s*\.?\s*$",
    re.IGNORECASE,
)
_BARE_YEAR_RE = re.compile(r"^\s*\d{4}\s*\.?\s*$")
_MASTHEAD_RE = re.compile(
    r"^\s*(?:treasury\s+bulletin|u\.?\s*s\.?\s+treasury(?:\s+department)?|"
    r"department\s+of\s+the\s+treasury)\s*\.?\s*$",
    re.IGNORECASE,
)


def _is_section_banner(content: str) -> bool:
    s = (content or "").strip()
    if len(s) < 6:
        return False
    if _NOISE_BANNER_RE.match(s):
        return False
    if _BARE_YEAR_RE.match(s):
        return False
    if _MASTHEAD_RE.match(s):
        return False
    return True


def _extract_page_banner(elements: list[dict]) -> str | None:
    """The page's own section banner.

    Walks every `[title]` and `[page_header]` element in document order
    and returns the first one that passes the section-banner shape filter
    (drops the bulletin masthead, month-year cover headers, bare years,
    and very short fragments). Treasury bulletins place the section name
    in either a title element (section-start pages) or as a second
    page_header below the masthead, so checking both in order is correct.

    This signal recovers section-affinity for pages whose bulletin-level
    ToC harvest returned nothing (1949–1964 broken-ToC bulletins).
    """
    for el in elements:
        if el.get("type") not in ("title", "page_header"):
            continue
        content = (el.get("content") or "").strip()
        if content and _is_section_banner(content):
            return content
    return None


_CONT_SUFFIX_RE = re.compile(
    r"[,\s\-–—]+(?:con|cont|continued)\s*\.?\s*\)?\s*$", re.IGNORECASE,
)


def _first_titled_block(r: PageCatalogRow):
    """First non-prose ContentBlock that carries a title, or None."""
    for b in r.content_blocks:
        if b.kind == "prose":
            continue
        if b.title:
            return b
    return None


def _merge_continuation_pages(rows: list[PageCatalogRow]) -> int:
    """Forward-fill table identity across continuation pages.

    A content page is a continuation when EITHER:
      (a) its first table/chart block's title ends with `con / cont / continued`, OR
      (b) the page has table/chart blocks with no title AND the previous content
          page has a non-empty title (caption-less continuation, ~9% of table
          pages — the layout parser drops the caption).

    For matched pages, UNION the parent's keywords onto the current row and
    copy the parent's first-block title onto the current row's first
    table/chart block.

    Returns the number of pages merged (for build-time stats).
    """
    rows.sort(key=lambda r: r.page)
    parent: PageCatalogRow | None = None
    merged = 0
    for r in rows:
        has_visual = any(b.kind != "prose" for b in r.content_blocks)
        if not has_visual:
            parent = None
            continue

        first_block = next((b for b in r.content_blocks if b.kind != "prose"), None)
        is_explicit_cont = bool(
            first_block and first_block.title
            and _CONT_SUFFIX_RE.search(first_block.title)
        )
        parent_titled = _first_titled_block(parent) if parent else None
        is_implicit_cont = (
            first_block is not None and not first_block.title
            and parent_titled is not None
        )
        if (is_explicit_cont or is_implicit_cont) and parent is not None and parent_titled is not None:
            if first_block is not None:
                first_block.title = parent_titled.title
            seen = {k.lower() for k in r.keywords}
            for k in parent.keywords:
                if k.lower() not in seen:
                    r.keywords.append(k)
                    seen.add(k.lower())
            merged += 1
            # Continuation pages also act as parents for further continuations
            # (e.g. a 3-page table). The parent identity is unchanged.
        else:
            # This page establishes a new identity if any block has a title.
            if _first_titled_block(r):
                parent = r
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

    # Stage 2: assemble rows. Pages flagged blank/toc by cheap_classify
    # skip the parser and stay block-less. Pages without any visual or
    # prose anchors also skip the parser. Everything else runs the
    # deterministic parser; rows with non-empty content_blocks are the
    # retrievable set.
    rows: list[PageCatalogRow] = []
    n_content = 0
    for pdf_idx, text in pages.items():
        cc, dr = metrics[pdf_idx]
        row = PageCatalogRow(
            bulletin=bulletin_month,
            page=pdf_idx,
            char_count=cc,
            digit_ratio=round(dr, 3),
        )
        elements = page_elements.get(pdf_idx, [])
        row.printed_page = _extract_printed_page(elements)
        row.banner_self = _extract_page_banner(elements)

        if cheap_kinds[pdf_idx] is not None:
            # blank or toc — skip the parser, leave content_blocks empty.
            rows.append(row)
            continue

        if not (has_visual_elements(elements) or has_prose_content(elements)):
            rows.append(row)
            continue

        fields = parse_page_fields(elements)
        row.content_blocks = fields.get("content_blocks", [])
        row.keywords = fields.get("keywords", [])
        row.dates = fields.get("dates", [])
        if row.content_blocks:
            n_content += 1
        rows.append(row)

    # Forward-fill continuation pages so the parent table's keywords also
    # point to its continuation pages (multi-page tables). This is a
    # preprocessing-time fix; the merge changes are local to this bulletin.
    n_merged = _merge_continuation_pages(rows)

    if verbose:
        print(f"  [{bulletin_month}] {n_content} content pages, "
              f"{len(rows) - n_content} non-content pages, "
              f"{len(rows)} total, {n_merged} continuation merges")
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

    print(f"Building page index for {len(chosen)} bulletins → {args.out}")
    print(f"  page text source: parsed-JSON @ {parsed_dir}")
    print(f"  workers: {args.workers}    (no LLM at this stage; ToC harvest "
          f"runs in pipeline.extract_l1)\n")

    started = time.monotonic()
    summary: list[dict] = []

    def _do_one(pdf_path: Path) -> tuple[Path, list[PageCatalogRow] | None, float, str | None]:
        bulletin = parse_bulletin_filename(pdf_path)
        t0 = time.monotonic()
        try:
            rows = _process_one_bulletin(
                pdf_path=pdf_path, verbose=args.verbose, parsed_dir=parsed_dir,
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
            n_content = sum(1 for r in rows if r.content_blocks)
            print(f"[done] {bulletin}: {len(rows)} pages ({n_content} content) "
                  f"in {dt:.1f}s → {out_path}")
            summary.append({
                "bulletin": bulletin, "status": "ok", "n_pages": len(rows),
                "n_content": n_content, "duration_s": round(dt, 2),
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
