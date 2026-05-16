"""Phase 2: per-page placement under the bulletin's own L1 chapter list.

For each catalog page with at least one ContentBlock (table / chart /
prose), find the bulletin's L1 chapter that owns it. The L1 list is
the strict-L1 output of Phase 1 (`extract_l1`) — typically 5-15
entries per bulletin.

Cascade order (cheapest first):

  Method A — span lookup
      `printed_page` falls within an L1 span from Phase 1 → assign.

  Method B — banner exact match
      Page's filtered `banner_self` (normalized + lowercased)
      matches one of the bulletin's L1 names → assign.

  Fallback (a1) — deterministic typo correction
      difflib.SequenceMatcher between banner / title and the bulletin's
      L1 list. Accept the top match iff ratio >= 0.85 AND it beats the
      runner-up by >= 0.05. Rescues OCR variants like "FEERAL DEBT"
      against the bulletin's "FEDERAL DEBT" without an LLM.

  Fallback (a2) — LLM typo correction (ambiguous batch only)
      For pages where (a1) found a close-but-not-clear winner, batch and
      ask the LLM "given this bulletin's chapter list and the page's
      banner/title, which L1 fits — or null". Small candidate set so
      the prompt is cheap and decisive.

  Fallback (b) — adjacent-page inheritance
      Iterate to fixed point: a page inherits the chapter of its ±2
      same-bulletin neighbors if at least two of them agree
      unanimously. Investigation showed 56.8% of unassigned pages have
      unanimous ±2 neighbors and they cluster in runs (mean 8.2).

  Fallback (c) — LLM prediction (last resort)
      Residual pages get one strict LLM call with their full metadata
      (table_title, captions, top keywords) and the bulletin's L1 list.
      Forced to pick one.

  Bucket (d) — _Unfiled
      Pages with no metadata that survived all cascades. Bucketed
      under a per-bulletin "_Unfiled" placeholder; Phase 3 may merge or
      drop them.

The module is pure orchestration over a single bulletin's catalog rows
and L1 spans; concurrency across bulletins lives in `pipeline.py`.
"""

from __future__ import annotations

import difflib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from skunk.common import LLMClient, parse_json_response

from .schema import PageCatalogRow
from .text_norm import normalize_label, norm_key
from .toc import SectionSpan, section_for_printed_page

UNFILED = "_Unfiled"

_TYPO_RATIO_MIN = 0.85
_TYPO_MARGIN_MIN = 0.05


# ---------------------------------------------------------------------------
# Deterministic typo match (Fallback a1)
# ---------------------------------------------------------------------------

def _typo_match(
    label: str | None, l1_names_lower: dict[str, str],
) -> str | None:
    """Best-fit match for `label` among `l1_names_lower` values.

    Returns the L1 name when one candidate clears `_TYPO_RATIO_MIN` AND
    beats the runner-up by `_TYPO_MARGIN_MIN`. Otherwise None — the
    caller routes those to the LLM typo step where the small candidate
    set is cheap to disambiguate.
    """
    if not label or not l1_names_lower:
        return None
    normed = normalize_label(label).lower().strip()
    if not normed:
        return None
    scored: list[tuple[float, str]] = [
        (difflib.SequenceMatcher(a=normed, b=lower_key, autojunk=False).ratio(), orig)
        for lower_key, orig in l1_names_lower.items()
    ]
    scored.sort(reverse=True)
    top_r, top_name = scored[0]
    runner_r = scored[1][0] if len(scored) > 1 else 0.0
    if top_r >= _TYPO_RATIO_MIN and (top_r - runner_r) >= _TYPO_MARGIN_MIN:
        return top_name
    return None


# ---------------------------------------------------------------------------
# LLM typo correction (Fallback a2) and LLM prediction (Fallback c)
# ---------------------------------------------------------------------------

_TYPO_SYSTEM = """You match Treasury Bulletin pages to one of the bulletin's
own chapter headings by their page banner.

You will receive:
  - The bulletin's chapter list — a small set of top-level (L1) chapter
    headings, exactly as printed in the bulletin's Table of Contents.
  - A batch of pages, each with a `banner` (the page's own
    [page_header]/[title] string, possibly OCR-corrupted) and a `title`
    (the page's table caption, often longer and more descriptive).

For each page, decide which chapter from the bulletin's list it belongs
to. Treat OCR variants ("FEERAL DEBT" → "FEDERAL DEBT"), missing
whitespace ("TRUSTFUNDS" → "TRUST FUNDS"), and synonym phrasings ("Public
debt operations" vs. "Debt operations") as matches. If no chapter is a
reasonable match, output null — don't force one.

Output a SINGLE JSON object (no prose, no fences):
  {"assignments": [
    {"id": <0-based index>, "chapter": "<exact L1 name>" | null},
    ...
  ]}

You MUST emit one entry per input page. Use the chapter name EXACTLY as
shown in the bulletin's chapter list.
"""


_PREDICT_SYSTEM = """You place Treasury Bulletin pages into one of the
bulletin's chapters using the page's full metadata.

You will receive:
  - The bulletin's chapter list — top-level (L1) headings from that
    bulletin's Table of Contents.
  - A batch of pages, each with:
      - `banner`: the page's [page_header]/[title] (may be empty or OCR'd)
      - `table_title`: verbatim caption, often the strongest signal
      - `column_headers`: column header strings (truncated)
      - `keywords`: top noun phrases extracted from the page
      - `dates`: dated periods on the page (e.g. "December 31, 1949")

For each page, pick the SINGLE best-matching chapter from the bulletin's
list. You MUST pick one — every page lands somewhere. Use the
table_title and column_headers as the primary signal; banner is
supplementary (it may be missing or noisy).

Output a SINGLE JSON object (no prose, no fences):
  {"assignments": [
    {"id": <0-based index>, "chapter": "<exact L1 name>"},
    ...
  ]}

You MUST emit one entry per input page. Use the chapter name EXACTLY as
shown.
"""


def _llm_classify_batch(
    pages: list[PageCatalogRow],
    l1_names: list[str],
    llm: LLMClient,
    *,
    system_prompt: str,
    allow_null: bool,
    include_metadata: bool,
) -> list[str | None]:
    """One LLM call over `pages`. Returns chapter (or None) per input page,
    aligned by index."""
    items: list[dict[str, Any]] = []
    for i, p in enumerate(pages):
        item: dict[str, Any] = {"id": i, "banner": p.banner_self or ""}
        item["title"] = (p.primary_title or "")[:300]
        if include_metadata:
            item["column_headers"] = p.all_column_headers()[:8]
            item["keywords"] = p.keywords[:6]
            item["dates"] = p.dates[:3]
        items.append(item)

    user = (
        f"Bulletin chapter list ({len(l1_names)}):\n"
        f"{json.dumps(l1_names, ensure_ascii=False, indent=1)}\n\n"
        f"Pages to classify ({len(items)}):\n"
        f"{json.dumps(items, ensure_ascii=False, indent=1)}\n"
    )
    resp = llm.call(system=system_prompt, user=user,
                    temperature=0.0, thinking_budget=0)
    obj = parse_json_response(resp.text)

    valid = {norm_key(name): name for name in l1_names}
    out: list[str | None] = [None] * len(pages)
    for entry in (obj or {}).get("assignments", []) or []:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry["id"])
        except (KeyError, ValueError, TypeError):
            continue
        if not (0 <= idx < len(pages)):
            continue
        ch = entry.get("chapter")
        if ch is None:
            if allow_null:
                out[idx] = None
            continue
        canonical = valid.get(norm_key(str(ch)))
        if canonical is not None:
            out[idx] = canonical
    return out


def _llm_typo_correct(
    pool: list[PageCatalogRow], l1_names: list[str], llm: LLMClient,
) -> list[str | None]:
    return _llm_classify_batch(
        pool, l1_names, llm,
        system_prompt=_TYPO_SYSTEM,
        allow_null=True, include_metadata=False,
    )


def _llm_predict(
    pool: list[PageCatalogRow], l1_names: list[str], llm: LLMClient,
) -> list[str | None]:
    return _llm_classify_batch(
        pool, l1_names, llm,
        system_prompt=_PREDICT_SYSTEM,
        allow_null=False, include_metadata=True,
    )


# ---------------------------------------------------------------------------
# Per-bulletin placement
# ---------------------------------------------------------------------------

def place_bulletin(
    rows: list[PageCatalogRow],
    l1_spans: list[SectionSpan],
    llm: LLMClient | None,
) -> dict[str, int]:
    """Mutate `rows` in place — set `r.l1_local` on every content page.

    `l1_spans` is the bulletin's Phase-1 L1 list (typically 5-15 entries).
    Returns a Counter dict of per-method placement counts.
    """
    stats: Counter[str] = Counter()

    l1_names = [sp.section for sp in l1_spans]
    if not l1_names:
        # No L1 vocabulary at all — every content page is unfilable.
        for r in rows:
            if r.content_blocks:
                r.l1_local = UNFILED
                stats["d_unfiled_no_l1"] += 1
        return dict(stats)

    l1_names_lower = {norm_key(normalize_label(n)): n for n in l1_names}

    content_rows: list[PageCatalogRow] = [
        r for r in rows if r.content_blocks
    ]
    if not content_rows:
        return dict(stats)

    # ── Method A: printed-page span lookup ─────────────────────────────
    pool_after_ab: list[PageCatalogRow] = []
    for r in content_rows:
        if r.printed_page and l1_spans:
            sec = section_for_printed_page(l1_spans, r.printed_page)
            if sec:
                r.l1_local = sec
                stats["A_span"] += 1
                continue

        # ── Method B: banner_self exact match ──────────────────────────
        if r.banner_self:
            key = norm_key(normalize_label(r.banner_self))
            if key in l1_names_lower:
                r.l1_local = l1_names_lower[key]
                stats["B_banner_exact"] += 1
                continue

        pool_after_ab.append(r)

    # ── Fallback (a1): deterministic typo ─────────────────────────────
    pool_a2: list[PageCatalogRow] = []
    for r in pool_after_ab:
        match = _typo_match(r.banner_self or r.primary_title, l1_names_lower)
        if match is not None:
            r.l1_local = match
            stats["a1_typo_det"] += 1
        else:
            pool_a2.append(r)

    # ── Fallback (a2): LLM typo correction (allow null) ───────────────
    pool_b: list[PageCatalogRow] = []
    if pool_a2 and llm is not None:
        results = _llm_typo_correct(pool_a2, l1_names, llm)
        for r, picked in zip(pool_a2, results):
            if picked is not None:
                r.l1_local = picked
                stats["a2_typo_llm"] += 1
            else:
                pool_b.append(r)
    else:
        pool_b = list(pool_a2)

    # ── Fallback (b): neighbor inheritance, iterate to fixed point ────
    placed_by_page: dict[int, str] = {
        r.page: r.l1_local for r in rows if r.l1_local
    }
    pool_c = pool_b
    while True:
        new_inh: list[PageCatalogRow] = []
        still: list[PageCatalogRow] = []
        for r in pool_c:
            votes: list[str] = []
            for off in (-2, -1, 1, 2):
                v = placed_by_page.get(r.page + off)
                if v and v != UNFILED:
                    votes.append(v)
            if len(votes) >= 2 and len(set(votes)) == 1:
                r.l1_local = votes[0]
                placed_by_page[r.page] = votes[0]
                new_inh.append(r)
                stats["b_neighbor"] += 1
            else:
                still.append(r)
        if not new_inh:
            break
        pool_c = still

    # ── Fallback (c): LLM prediction (forced pick) ────────────────────
    if pool_c and llm is not None:
        results = _llm_predict(pool_c, l1_names, llm)
        for r, picked in zip(pool_c, results):
            if picked is not None:
                r.l1_local = picked
                placed_by_page[r.page] = picked
                stats["c_llm_predict"] += 1
            else:
                # LLM didn't pick — shouldn't happen with strict prompt
                # but guard anyway.
                r.l1_local = UNFILED
                stats["d_unfiled_llm_skip"] += 1
    else:
        for r in pool_c:
            r.l1_local = UNFILED
            stats["d_unfiled_no_llm"] += 1

    return dict(stats)


# ---------------------------------------------------------------------------
# Cross-bulletin driver (parallel)
# ---------------------------------------------------------------------------

def place_all(
    catalog_by_bulletin: dict[str, list[PageCatalogRow]],
    l1_by_bulletin: dict[str, list[SectionSpan]],
    llm: LLMClient | None,
    *,
    workers: int = 16,
    verbose: bool = True,
) -> dict[str, dict[str, int]]:
    """Run `place_bulletin` over every bulletin in parallel.

    Returns `{bulletin: per_method_stats}`. Mutates the catalog rows in
    place — caller is expected to persist them after placement.
    """
    items = sorted(catalog_by_bulletin.items())
    out: dict[str, dict[str, int]] = {}
    started = time.monotonic()

    def _run(bulletin: str) -> tuple[str, dict[str, int]]:
        rows = catalog_by_bulletin[bulletin]
        spans = l1_by_bulletin.get(bulletin, [])
        return bulletin, place_bulletin(rows, spans, llm)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run, b): b for b, _ in items}
        done = 0
        for f in as_completed(futs):
            bulletin, stats = f.result()
            out[bulletin] = stats
            done += 1
            if verbose and (done % 50 == 0 or done == len(items)):
                print(f"  placed {done}/{len(items)} bulletins "
                      f"({time.monotonic() - started:.1f}s)", flush=True)

    if verbose:
        totals: Counter[str] = Counter()
        for s in out.values():
            for k, n in s.items():
                totals[k] += n
        print("  placement totals:", flush=True)
        for k in sorted(totals.keys()):
            print(f"    {totals[k]:>7}  {k}", flush=True)
    return out


