"""Treasury Bulletin page placer — `PagePlacer` impl.

Six-method cascade per content page (cheapest first):

  A — printed-page span lookup (deterministic)
  B — banner_self exact match (deterministic)
  a1 — difflib-based typo correction (deterministic)
  a2 — LLM typo correction (allow null) on the ambiguous residual
  b — neighbor inheritance (deterministic, iterate to fixed point)
  c — LLM strict-pick prediction (forced pick) on the final residual
  d — `_Unfiled` bucket (everything that survived)

Mutates `rows` in place; returns per-method counters for visibility.
"""

from __future__ import annotations

import difflib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from skunk.common import LLMClient, parse_json_response

from ...schema import PageCatalogRow
from ...stages.l1_harvest import SectionSpan, section_for_printed_page
from ...stages.placer import UNFILED
from .prompts import PLACER_PREDICT_SYSTEM, PLACER_TYPO_SYSTEM
from .text_norm import norm_key, normalize_label


@dataclass(frozen=True)
class TypoMatchConfig:
    """Empirical thresholds for the deterministic difflib typo matcher.

    Tuned on the 1939-2025 Treasury Bulletin OCR corpus. The 0.85 floor
    catches OCR variants ("FEERAL DEBT" vs "FEDERAL DEBT") without
    accepting spurious matches across unrelated chapter names; the 0.05
    margin keeps difflib from picking between two near-tie matches
    (defers to the LLM typo path instead).
    """
    ratio_min: float = 0.85
    margin_min: float = 0.05


_DEFAULT_TYPO = TypoMatchConfig()


def _typo_match(
    label: str | None,
    l1_names_lower: dict[str, str],
    cfg: TypoMatchConfig = _DEFAULT_TYPO,
) -> str | None:
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
    if top_r >= cfg.ratio_min and (top_r - runner_r) >= cfg.margin_min:
        return top_name
    return None


def _llm_classify_batch(
    pages: list[PageCatalogRow],
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

    def __init__(self, typo_cfg: TypoMatchConfig = _DEFAULT_TYPO) -> None:
        self.typo_cfg = typo_cfg

    def place_bulletin(
        self,
        rows: list[PageCatalogRow],
        spans: list[SectionSpan],
        llm: LLMClient | None,
    ) -> dict[str, int]:
        stats: Counter[str] = Counter()

        l1_names = [sp.section for sp in spans]
        if not l1_names:
            for r in rows:
                if r.content_blocks:
                    r.l1_local = UNFILED
                    stats["d_unfiled_no_l1"] += 1
            return dict(stats)

        l1_names_lower = {norm_key(normalize_label(n)): n for n in l1_names}
        content_rows = [r for r in rows if r.content_blocks]
        if not content_rows:
            return dict(stats)

        # ── Methods A + B ────────────────────────────────────────────
        pool_after_ab: list[PageCatalogRow] = []
        for r in content_rows:
            if r.printed_page and spans:
                sec = section_for_printed_page(spans, r.printed_page)
                if sec:
                    r.l1_local = sec
                    stats["A_span"] += 1
                    continue

            if r.banner_self:
                key = norm_key(normalize_label(r.banner_self))
                if key in l1_names_lower:
                    r.l1_local = l1_names_lower[key]
                    stats["B_banner_exact"] += 1
                    continue

            pool_after_ab.append(r)

        # ── Fallback a1: deterministic typo ──────────────────────────
        pool_a2: list[PageCatalogRow] = []
        for r in pool_after_ab:
            match = _typo_match(
                r.banner_self or r.primary_title,
                l1_names_lower,
                self.typo_cfg,
            )
            if match is not None:
                r.l1_local = match
                stats["a1_typo_det"] += 1
            else:
                pool_a2.append(r)

        # ── Fallback a2: LLM typo correction (allow null) ────────────
        pool_b: list[PageCatalogRow] = []
        if pool_a2 and llm is not None:
            results = _llm_classify_batch(
                pool_a2, l1_names, llm,
                system_prompt=PLACER_TYPO_SYSTEM,
                allow_null=True, include_metadata=False,
            )
            for r, picked in zip(pool_a2, results):
                if picked is not None:
                    r.l1_local = picked
                    stats["a2_typo_llm"] += 1
                else:
                    pool_b.append(r)
        else:
            pool_b = list(pool_a2)

        # ── Fallback b: neighbor inheritance ─────────────────────────
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

        # ── Fallback c: LLM strict prediction ────────────────────────
        if pool_c and llm is not None:
            results = _llm_classify_batch(
                pool_c, l1_names, llm,
                system_prompt=PLACER_PREDICT_SYSTEM,
                allow_null=False, include_metadata=True,
            )
            for r, picked in zip(pool_c, results):
                if picked is not None:
                    r.l1_local = picked
                    placed_by_page[r.page] = picked
                    stats["c_llm_predict"] += 1
                else:
                    r.l1_local = UNFILED
                    stats["d_unfiled_llm_skip"] += 1
        else:
            for r in pool_c:
                r.l1_local = UNFILED
                stats["d_unfiled_no_llm"] += 1

        return dict(stats)
