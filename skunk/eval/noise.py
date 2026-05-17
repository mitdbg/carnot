"""Confounder generation for `--golden-noisy` eval mode.

Two stages share this module.

1. **Offline pool builder** (CLI: `python -m eval.noise build ...`)
   For every (uid, golden_page_idx) in the benchmark CSV: extract distinctive
   keywords from the golden page via an LLM, then text-search adjacent-year
   bulletins (±1, ±2 years × same month) for pages with high keyword overlap.
   Dump the ranked candidates to a JSON pool file.

   Per-bulletin page-text is cached on disk under `cache/pages_text/` so reruns
   are free after the first pass.

2. **Runtime sampler** (`make_noisy_pages`)
   Per golden PageRef, keep tossing a Bernoulli(noise_prob) coin: each `True`
   appends one confounder PageRef, stops on first `False` (geometric). Expected
   confounders per golden page = p / (1 - p). Each appended confounder is drawn
   from either:
     - 30% same-bulletin page drift (±1..±5 pages, computed inline)
     - 70% cross-year keyword match (sampled from the precomputed pool,
       weighted by score; falls back to drift when the pool has no entry).
   A hard cap (`_MAX_CONFOUNDERS_PER_GOLDEN`) prevents runaway when
   p is set very close to 1.0.
   Confounders are appended to a copy of `golden` with no flag and no label;
   they are indistinguishable from golden refs downstream.

Determinism: confounder selection is a pure function of (uid, golden_idx, seed)
via `hashlib.sha256` → `random.Random`. The pool itself is content-determined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import fitz

# ---------------------------------------------------------------------------
# Lazy, process-local caches
# ---------------------------------------------------------------------------

_PAGE_COUNT_CACHE: dict[str, int] = {}
_UNIVERSE_CACHE: set[tuple[int, int]] | None = None
_PAGE_TEXT_CACHE: dict[str, dict[int, str]] = {}

_MAX_CONFOUNDERS_PER_GOLDEN = 4  # cap for geometric loop

_FILENAME_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})\.pdf$")


def _file_path(pdf_dir: str, year: int, month: int) -> str:
    return f"{pdf_dir}/treasury_bulletin_{year:04d}_{month:02d}.pdf"


def _n_pages(file_path: str) -> int:
    if file_path not in _PAGE_COUNT_CACHE:
        with fitz.open(file_path) as doc:
            _PAGE_COUNT_CACHE[file_path] = len(doc)
    return _PAGE_COUNT_CACHE[file_path]


def _universe(pdf_dir: str) -> set[tuple[int, int]]:
    global _UNIVERSE_CACHE
    if _UNIVERSE_CACHE is None:
        out: set[tuple[int, int]] = set()
        for p in Path(pdf_dir).glob("treasury_bulletin_*.pdf"):
            m = _FILENAME_RE.search(p.name)
            if m:
                out.add((int(m.group(1)), int(m.group(2))))
        _UNIVERSE_CACHE = out
    return _UNIVERSE_CACHE


def _load_page_texts(file_path: str, cache_dir: Path | None = None) -> dict[int, str]:
    """Return {1-based page index: text} for every page of the PDF.

    Persists to `cache_dir/<stem>.json` when provided so the keyword search
    can scan many adjacent-year bulletins without re-running fitz each time.
    """
    if file_path in _PAGE_TEXT_CACHE:
        return _PAGE_TEXT_CACHE[file_path]

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = cache_dir / f"{Path(file_path).stem}.json"
        if cache_path.exists():
            try:
                raw = json.loads(cache_path.read_text())
                texts = {int(k): v for k, v in raw.items()}
                _PAGE_TEXT_CACHE[file_path] = texts
                return texts
            except Exception:
                pass

    texts: dict[int, str] = {}
    with fitz.open(file_path) as doc:
        for i in range(len(doc)):
            texts[i + 1] = doc[i].get_text()
    _PAGE_TEXT_CACHE[file_path] = texts

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(texts, ensure_ascii=False))

    return texts


# ---------------------------------------------------------------------------
# Stage A — Offline pool build
# ---------------------------------------------------------------------------

_KEYWORD_SYSTEM = (
    "You extract distinctive search keywords from a U.S. Treasury Bulletin page so "
    "that another page covering the SAME TOPIC in a different-year bulletin can be "
    "located by substring search.\n\n"
    "Return 8-12 multi-word phrases (2-5 words each). Each must:\n"
    "  - appear VERBATIM on the page;\n"
    "  - be a table title, series name, agency name, or specific category label\n"
    "    that would plausibly recur in adjacent years' bulletins covering the same\n"
    "    series;\n"
    "  - be distinctive (avoid generic words like 'total', 'year', 'amount',\n"
    "    'millions of dollars');\n"
    "  - be lowercased.\n\n"
    "Output ONLY a JSON array of strings. No fences, no commentary."
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.MULTILINE)


def _extract_keywords(page_text: str, llm) -> list[str]:
    """One LLM call → list of lowercased multi-word phrases. Empty on failure."""
    if not page_text.strip():
        return []
    user = f"Page text:\n\n{page_text[:8000]}\n\nReturn the JSON array."
    resp = llm.call(system=_KEYWORD_SYSTEM, user=user, temperature=0.0, thinking_budget=0)
    raw = _FENCE_RE.sub("", resp.text.strip()).strip()
    try:
        kws = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(kws, list):
        return []
    return [
        str(k).strip().lower()
        for k in kws
        if isinstance(k, str) and len(str(k).strip()) >= 4
    ]


def _score_page(text: str, keywords: list[str]) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def _adjacent_year_bulletins(
    year: int, month: int, universe: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    out = []
    for dy in (-2, -1, 1, 2):
        cand = (year + dy, month)
        if cand in universe:
            out.append(cand)
    return out


def build_pool(
    csv_path: str,
    out_path: str,
    pdf_dir: str,
    cache_dir: Path,
    limit_uids: set[str] | None = None,
) -> None:
    import pandas as pd

    from skunk.common import LLMClient
    from skunk.config import SkunkConfig

    from eval.eval_e2e import _parse_source_docs

    llm = LLMClient(SkunkConfig.from_env())
    universe = _universe(pdf_dir)

    df = pd.read_csv(csv_path)
    pool: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for _, row in df.iterrows():
        uid = str(row["uid"])
        if limit_uids is not None and uid not in limit_uids:
            continue
        golden = _parse_source_docs(str(row.get("source_docs", "")))
        if not golden:
            continue

        per_uid: dict[str, list[dict[str, Any]]] = {}
        for idx, gref in enumerate(golden):
            year, mon = int(gref.month[:4]), int(gref.month[5:7])
            gfile = _file_path(pdf_dir, year, mon)
            if not Path(gfile).exists():
                print(f"[noise.build] {uid}[{idx}]: golden PDF missing: {gfile}", file=sys.stderr)
                continue
            try:
                texts = _load_page_texts(gfile, cache_dir=cache_dir)
            except Exception as e:
                print(f"[noise.build] {uid}[{idx}]: cannot read {gfile}: {e}", file=sys.stderr)
                continue
            page_text = texts.get(gref.page, "")
            keywords = _extract_keywords(page_text, llm)
            if not keywords:
                print(f"[noise.build] {uid}[{idx}]: no keywords extracted", file=sys.stderr)
                continue

            candidates: list[tuple[int, str, int]] = []
            for nyr, nmo in _adjacent_year_bulletins(year, mon, universe):
                nfile = _file_path(pdf_dir, nyr, nmo)
                try:
                    ntexts = _load_page_texts(nfile, cache_dir=cache_dir)
                except Exception as e:
                    print(f"[noise.build] {uid}[{idx}]: skip {nfile}: {e}", file=sys.stderr)
                    continue
                month_str = f"{nyr:04d}-{nmo:02d}"
                for pg, txt in ntexts.items():
                    s = _score_page(txt, keywords)
                    if s >= 2:
                        candidates.append((s, month_str, pg))

            candidates.sort(reverse=True)
            top = [{"month": m, "page": p, "score": s} for s, m, p in candidates[:5]]
            print(
                f"[noise.build] {uid}[{idx}] m={gref.month} pg={gref.page} -> "
                f"{len(keywords)} keywords, {len(top)} candidates",
                file=sys.stderr,
            )
            if top:
                per_uid[str(idx)] = top

        if per_uid:
            pool[uid] = per_uid

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pool, indent=2))
    print(f"[noise.build] wrote {out} ({len(pool)} uids)", file=sys.stderr)


# ---------------------------------------------------------------------------
# Stage B — Runtime sampler
# ---------------------------------------------------------------------------

def load_noise_pool(path: str) -> dict[str, dict[str, list[dict[str, Any]]]]:
    p = Path(path)
    if not p.exists():
        print(
            f"[noise] WARNING: noise pool {path!r} missing — all rolls will fall back to drift",
            file=sys.stderr,
        )
        return {}
    return json.loads(p.read_text())


def _rng_for(seed: int, uid: str, idx: int) -> random.Random:
    h = hashlib.sha256(f"{seed}|{uid}|{idx}".encode()).hexdigest()[:16]
    return random.Random(int(h, 16))


def _drift_confounder(rng: random.Random, golden_ref, pdf_dir: str):
    from skunk.plan import PageRef

    year, mon = int(golden_ref.month[:4]), int(golden_ref.month[5:7])
    fp = _file_path(pdf_dir, year, mon)
    try:
        n = _n_pages(fp)
    except Exception:
        return None
    offsets = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
    rng.shuffle(offsets)
    for off in offsets:
        new_page = golden_ref.page + off
        if 1 <= new_page <= n and new_page != golden_ref.page:
            return PageRef(month=golden_ref.month, page=new_page, file_path=fp)
    return None


def _pool_confounder(rng: random.Random, uid: str, idx: int, pool, pdf_dir: str):
    from skunk.plan import PageRef

    entries = pool.get(uid, {}).get(str(idx))
    if not entries:
        return None
    weights = [max(1, int(e.get("score", 1))) for e in entries]
    pick = rng.choices(entries, weights=weights, k=1)[0]
    month = pick["month"]
    year, mon = int(month[:4]), int(month[5:7])
    return PageRef(
        month=month,
        page=int(pick["page"]),
        file_path=_file_path(pdf_dir, year, mon),
    )


def make_noisy_pages(
    uid: str,
    golden: list,
    *,
    noise_prob: float,
    seed: int,
    pdf_dir: str,
    pool: dict,
) -> list:
    """Return list[PageRef] = golden + per-page-rolled confounders (no label).

    Each golden page tosses a Bernoulli(noise_prob) coin in a loop; every `True`
    appends one confounder, the loop ends on the first `False`. Capped at
    `_MAX_CONFOUNDERS_PER_GOLDEN` to keep runaway behavior bounded when p ≈ 1.
    """
    out = list(golden)
    for idx, gref in enumerate(golden):
        rng = _rng_for(seed, uid, idx)
        added = 0
        while added < _MAX_CONFOUNDERS_PER_GOLDEN and rng.random() < noise_prob:
            bucket = rng.choices(["drift", "pool"], weights=[30, 70])[0]
            if bucket == "pool":
                cand = _pool_confounder(rng, uid, idx, pool, pdf_dir)
                if cand is None:
                    cand = _drift_confounder(rng, gref, pdf_dir)
            else:
                cand = _drift_confounder(rng, gref, pdf_dir)
                if cand is None:
                    cand = _pool_confounder(rng, uid, idx, pool, pdf_dir)
            if cand is None:
                break  # no source can produce a confounder; abandon this golden page
            out.append(cand)
            added += 1
    return out


# ---------------------------------------------------------------------------
# CLI: `python -m eval.noise build ...`
# ---------------------------------------------------------------------------

def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def _main() -> None:
    parser = argparse.ArgumentParser(description="Confounder pool tools for --golden-noisy")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build the cross-year keyword-matched pool")
    b.add_argument("--csv", required=True, help="Path to officeqa_pro.csv")
    b.add_argument("--out", required=True, help="Output pool JSON path")
    b.add_argument(
        "--pdf-dir",
        default=os.environ.get(
            "OFFICEQA_PDF_DIR",
            str(Path.home() / "Desktop/officeqa/treasury_bulletin_pdfs"),
        ),
    )
    b.add_argument("--text-cache-dir", default="cache/pages_text")
    b.add_argument("--limit-uids", default=None, help="Comma-separated UIDs for incremental builds")

    args = parser.parse_args()
    if args.cmd == "build":
        _load_env(Path(__file__).resolve().parents[1] / ".env")
        limit = (
            {u.strip() for u in args.limit_uids.split(",") if u.strip()}
            if args.limit_uids
            else None
        )
        build_pool(args.csv, args.out, args.pdf_dir, Path(args.text_cache_dir), limit_uids=limit)


if __name__ == "__main__":
    _main()
