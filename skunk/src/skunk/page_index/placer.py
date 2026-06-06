"""Build phase 3b — per-page placement.

Assign each content page a level-1 `section` (one of the bulletin's L1 chapters
from the harvester) via a waterfall: ToC-span lookup, banner-exact, deterministic
typo match, LLM typo/predict, neighbor inheritance. Unfilable pages get `UNFILED`."""

from __future__ import annotations

import difflib
import json
from collections import Counter
from typing import Any

from skunk.common import LLMClient, parse_json_response

from .data_model import BuildPage
from .l1_harvest import SectionSpan, section_for_printed_page
from .text_norm import normalize_label, norm_key

# Sentinel chapter for content pages that couldn't be filed under any L1 section.
UNFILED = "_Unfiled"


# Empirical thresholds for the deterministic difflib typo matcher, tuned on the
# 1939-2025 Treasury Bulletin OCR corpus. The 0.85 floor catches OCR variants
# ("FEERAL DEBT" vs "FEDERAL DEBT") without accepting spurious matches across
# unrelated chapter names; the 0.05 margin keeps difflib from picking between two
# near-tie matches (defers to the LLM typo path instead).
_TYPO_RATIO_MIN = 0.85
_TYPO_MARGIN_MIN = 0.05


def _typo_match(label: str | None, l1_names_lower: dict[str, str]) -> str | None:
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


def _llm_classify_batch(
    pages: list[BuildPage],
    l1_names: list[str],
    llm: LLMClient,
    *,
    system_prompt: str,
    allow_null: bool,
    include_metadata: bool,
) -> list[str | None]:
    """One LLM call over `pages`. Returns chapter (or None) per input
    page, aligned by index."""
    items: list[dict[str, Any]] = []
    for i, p in enumerate(pages):
        item: dict[str, Any] = {"id": i, "banner": p.banner_self or ""}
        item["title"] = (p.primary_title or "")[:300]
        if include_metadata:
            item["column_headers"] = p.all_column_headers()[:8]
            item["keywords"] = p.keywords[:6]
        items.append(item)

    user = (
        f"Bulletin chapter list ({len(l1_names)}):\n"
        f"{json.dumps(l1_names, ensure_ascii=False, indent=1)}\n\n"
        f"Pages to classify ({len(items)}):\n"
        f"{json.dumps(items, ensure_ascii=False, indent=1)}\n"
    )
    resp = llm.call(system=system_prompt, user=user,
                    temperature=0.0, effort="off")
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


class TreasuryPagePlacer:
    """`PagePlacer` impl for Treasury Bulletin pages."""

    def place_bulletin(
        self,
        rows: list[BuildPage],
        spans: list[SectionSpan],
        llm: LLMClient | None,
    ) -> dict[str, int]:
        stats: Counter[str] = Counter()

        l1_names = [sp.section for sp in spans]
        if not l1_names:
            for r in rows:
                if r.content_blocks:
                    r.section = UNFILED
                    stats["d_unfiled_no_l1"] += 1
            return dict(stats)

        l1_names_lower = {norm_key(normalize_label(n)): n for n in l1_names}
        content_rows = [r for r in rows if r.content_blocks]
        if not content_rows:
            return dict(stats)

        # ── Methods A + B ────────────────────────────────────────────
        pool_after_ab: list[BuildPage] = []
        for r in content_rows:
            if r.printed_page and spans:
                sec = section_for_printed_page(spans, r.printed_page)
                if sec:
                    r.section = sec
                    stats["A_span"] += 1
                    continue

            if r.banner_self:
                key = norm_key(normalize_label(r.banner_self))
                if key in l1_names_lower:
                    r.section = l1_names_lower[key]
                    stats["B_banner_exact"] += 1
                    continue

            pool_after_ab.append(r)

        # ── Fallback a1: deterministic typo ──────────────────────────
        pool_a2: list[BuildPage] = []
        for r in pool_after_ab:
            match = _typo_match(
                r.banner_self or r.primary_title,
                l1_names_lower,
            )
            if match is not None:
                r.section = match
                stats["a1_typo_det"] += 1
            else:
                pool_a2.append(r)

        # ── Fallback a2: LLM typo correction (allow null) ────────────
        pool_b: list[BuildPage] = []
        if pool_a2 and llm is not None:
            results = _llm_classify_batch(
                pool_a2, l1_names, llm,
                system_prompt=PLACER_TYPO_SYSTEM,
                allow_null=True, include_metadata=False,
            )
            for r, picked in zip(pool_a2, results):
                if picked is not None:
                    r.section = picked
                    stats["a2_typo_llm"] += 1
                else:
                    pool_b.append(r)
        else:
            pool_b = list(pool_a2)

        # ── Fallback b: neighbor inheritance ─────────────────────────
        placed_by_page: dict[int, str] = {
            r.page: r.section for r in rows if r.section
        }
        pool_c = pool_b
        while True:
            new_inh: list[BuildPage] = []
            still: list[BuildPage] = []
            for r in pool_c:
                votes: list[str] = []
                for off in (-2, -1, 1, 2):
                    v = placed_by_page.get(r.page + off)
                    if v and v != UNFILED:
                        votes.append(v)
                if len(votes) >= 2 and len(set(votes)) == 1:
                    r.section = votes[0]
                    placed_by_page[r.page] = votes[0]
                    new_inh.append(r)
                    stats["b_neighbor"] += 1
                else:
                    still.append(r)
            if not new_inh:
                break
            pool_c = still

        # ── Fallback c: LLM strict prediction ────────────────────────
        if pool_c and llm is not None:
            results = _llm_classify_batch(
                pool_c, l1_names, llm,
                system_prompt=PLACER_PREDICT_SYSTEM,
                allow_null=False, include_metadata=True,
            )
            for r, picked in zip(pool_c, results):
                if picked is not None:
                    r.section = picked
                    placed_by_page[r.page] = picked
                    stats["c_llm_predict"] += 1
                else:
                    r.section = UNFILED
                    stats["d_unfiled_llm_skip"] += 1
        else:
            for r in pool_c:
                r.section = UNFILED
                stats["d_unfiled_no_llm"] += 1

        return dict(stats)


# ---------------------------------------------------------------------------
# Pass 1: deterministic normalization
# ---------------------------------------------------------------------------

PLACER_TYPO_SYSTEM = """You match Treasury Bulletin pages to one of the bulletin's own chapter
headings by their page banner.

You receive the bulletin's L1 chapter list (verbatim from its TOC) and a batch of pages, each
with a `banner` (the page's own [page_header]/[title], possibly OCR-corrupted) and a `title`
(the table caption, often longer and more descriptive).

For each page, pick which chapter it belongs to. Treat OCR variants ("FEERAL DEBT" → "FEDERAL
DEBT"), missing whitespace ("TRUSTFUNDS" → "TRUST FUNDS"), and synonym phrasings ("Public debt
operations" vs "Debt operations") as matches. Output null if none is a reasonable match —
don't force one.

Output a SINGLE JSON object, no prose or fences:
  {"assignments": [{"id": <0-based index>, "chapter": "<exact L1 name>" | null}, ...]}

One entry per input page; use the chapter name EXACTLY as shown.
"""


PLACER_PREDICT_SYSTEM = """You place Treasury Bulletin pages into one of the bulletin's chapters
using the page's full metadata.

You receive the bulletin's L1 chapter list and a batch of pages, each with:
  - `banner`: the page's [page_header]/[title] (may be empty or OCR'd)
  - `title`: verbatim table caption — often the strongest signal
  - `column_headers`: column header strings (truncated)
  - `keywords`: top noun phrases from the page

For each page pick the SINGLE best-matching chapter — every page lands somewhere, so you MUST
pick one. Use `title` and `column_headers` as the primary signal; `banner` is supplementary
(may be missing or noisy).

Output a SINGLE JSON object, no prose or fences:
  {"assignments": [{"id": <0-based index>, "chapter": "<exact L1 name>"}, ...]}

One entry per input page; use the chapter name EXACTLY as shown.
"""


