"""Treasury Bulletin chapter merger — `ChapterMerger` impl.

Four-pass cross-bulletin merge of per-bulletin L1 vocabularies into the
flat global canonical chapter set:

  Pass 1 — Deterministic normalization (`normalize_label`, group by
           lowercased form). Collapses OCR variants, case-only diffs,
           trailing punctuation.
  Pass 2 — LLM clustering across era-drift synonyms.
  Pass 3 — LLM consolidation: roll Pass-2 canonicals up into the eight
           recurring modern chapters (+ rare era-specific overflow).
  Pass 4 — LLM description + examples per final chapter.

Output: the flat tree dict that becomes shipped `concept_tree.json`.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from typing import Any

from skunk.common import LLMClient, parse_json_response

from ...schema import PageCatalogRow
from ...stages.placer import UNFILED
from .prompts import (
    MERGE_CLUSTER_SYSTEM, MERGE_CONSOLIDATE_SYSTEM, MERGE_DESCRIBE_SYSTEM,
)
from .text_norm import norm_key, normalize_label

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pass 1: deterministic normalization
# ---------------------------------------------------------------------------

def _normalize_groups(
    l1_counts: dict[str, int],
) -> tuple[dict[str, list[str]], dict[str, int], dict[str, str]]:
    """Group L1 names by their normalized lowercased form.

    Returns:
      groups:        {norm_key → [raw_l1_name, ...]}
      group_pages:   {norm_key → total page count}
      group_display: {norm_key → preferred display form}
    """
    groups: dict[str, list[str]] = {}
    pages: dict[str, int] = {}
    for raw, n in l1_counts.items():
        normed = normalize_label(raw)
        if not normed:
            continue
        k = norm_key(normed)
        groups.setdefault(k, []).append(raw)
        pages[k] = pages.get(k, 0) + n

    display: dict[str, str] = {}
    for k, members in groups.items():
        norms = [normalize_label(m) for m in members]
        title_cased = [n for n in norms
                       if n and not n.isupper() and not n.islower()]
        if title_cased:
            display[k] = Counter(title_cased).most_common(1)[0][0]
        else:
            display[k] = max(norms, key=len) if norms else members[0]
    return groups, pages, display


# ---------------------------------------------------------------------------
# LLM passes
# ---------------------------------------------------------------------------

def _llm_cluster(
    display_names: list[str], llm: LLMClient, *, verbose: bool = False,
) -> dict[str, str]:
    """Pass 2 — one LLM call clusters display names into canonical
    chapters. Returns `{display_name → canonical}`. Names omitted from
    the response fall back to their own canonical (one-member).
    """
    if not display_names:
        return {}

    user = (
        f"Distinct chapter names to cluster ({len(display_names)}):\n"
        f"{json.dumps(display_names, ensure_ascii=False, indent=1)}\n"
    )
    t0 = time.monotonic()
    resp = llm.call(system=MERGE_CLUSTER_SYSTEM, user=user,
                    temperature=0.0, effort="off")
    if verbose:
        log.info(f"[merge] LLM clustering in {time.monotonic() - t0:.1f}s")
    obj = parse_json_response(resp.text)
    clusters = (obj or {}).get("clusters", []) if isinstance(obj, dict) else []

    mapping: dict[str, str] = {}
    for c in clusters:
        if not isinstance(c, dict):
            continue
        canonical = str(c.get("canonical") or "").strip()
        members = c.get("members") or []
        if not canonical or not isinstance(members, list):
            continue
        for m in members:
            key = str(m).strip()
            if key:
                mapping[key] = canonical

    for name in display_names:
        mapping.setdefault(name, name)
    return mapping


def _llm_consolidate(
    canonical_chapters: list[dict[str, Any]], llm: LLMClient,
    *, verbose: bool = False,
) -> dict[str, str]:
    """Pass 3 — roll up Pass-2 canonicals into broader parents.

    `canonical_chapters` is a list of dicts:
        {"canonical": str, "n_pages": int, "members": list[str]}

    Returns `{pass2_canonical → final_parent}`. Omissions fall back to
    self-parent (no rollup).
    """
    if not canonical_chapters:
        return {}
    user = (
        f"Pass-2 canonical chapters ({len(canonical_chapters)}):\n"
        f"{json.dumps(canonical_chapters, ensure_ascii=False, indent=1)}\n"
    )
    t0 = time.monotonic()
    resp = llm.call(system=MERGE_CONSOLIDATE_SYSTEM, user=user,
                    temperature=0.0, effort="off")
    if verbose:
        log.info(f"[merge] Pass-3 consolidation LLM in "
                 f"{time.monotonic() - t0:.1f}s")
    obj = parse_json_response(resp.text)
    consolidations = (obj or {}).get("consolidations", []) if isinstance(obj, dict) else []

    mapping: dict[str, str] = {}
    for c in consolidations:
        if not isinstance(c, dict):
            continue
        parent = str(c.get("parent") or "").strip()
        members = c.get("members") or []
        if not parent or not isinstance(members, list):
            continue
        for m in members:
            key = str(m).strip()
            if key:
                mapping[key] = parent

    for c in canonical_chapters:
        name = c["canonical"]
        mapping.setdefault(name, name)
    return mapping


def _llm_describe_chapters(
    chapters_with_members: dict[str, list[str]], llm: LLMClient,
    *, verbose: bool = False,
) -> dict[str, dict[str, Any]]:
    """Pass 4 — one LLM call → {chapter → {description, examples}}."""
    if not chapters_with_members:
        return {}
    payload = [
        {"chapter": ch, "members": members}
        for ch, members in chapters_with_members.items()
    ]
    user = (
        f"Chapters to describe ({len(payload)}):\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=1)}\n"
    )
    t0 = time.monotonic()
    resp = llm.call(system=MERGE_DESCRIBE_SYSTEM, user=user,
                    temperature=0.0, effort="off")
    if verbose:
        log.info(f"[merge] Pass-4 description LLM in "
                 f"{time.monotonic() - t0:.1f}s")
    obj = parse_json_response(resp.text)
    raw = (obj or {}).get("chapters", {}) if isinstance(obj, dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for ch in chapters_with_members:
        entry = raw.get(ch) if isinstance(raw, dict) else None
        if not isinstance(entry, dict):
            out[ch] = {"description": "", "examples": []}
            continue
        desc = str(entry.get("description") or "").strip()
        examples_raw = entry.get("examples") or []
        examples = [str(e).strip() for e in examples_raw
                    if isinstance(e, str) and e.strip()]
        out[ch] = {"description": desc, "examples": examples}
    return out


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

def _posting_for_row(r: PageCatalogRow) -> dict[str, Any]:
    return {
        "bulletin": r.bulletin,
        "page": r.page,
        "key_phrases": list(r.keywords),
    }


class TreasuryChapterMerger:
    """`ChapterMerger` impl for Treasury Bulletin chapter consolidation."""

    def build_tree(
        self,
        catalog: list[PageCatalogRow],
        llm: LLMClient,
        *,
        drop_unfiled: bool = True,
        verbose: bool = True,
    ) -> dict[str, Any]:
        content_rows: list[PageCatalogRow] = []
        counts: Counter[str] = Counter()
        for r in catalog:
            if not r.content_blocks:
                continue
            l1 = (r.l1_local or "").strip()
            if not l1 or (l1 == UNFILED and drop_unfiled):
                continue
            counts[l1] += 1
            content_rows.append(r)

        if verbose:
            log.info(f"[merge] {len(counts)} distinct l1_local names "
                     f"covering {sum(counts.values())} pages")

        # ── Pass 1: deterministic normalization ────────────────────────
        groups, group_pages, group_display = _normalize_groups(dict(counts))
        if verbose:
            log.info(f"[merge] Pass 1 collapse: {len(counts)} → "
                     f"{len(groups)} normalized groups")

        raw_to_display: dict[str, str] = {}
        for k, raws in groups.items():
            for raw in raws:
                raw_to_display[raw] = group_display[k]

        # ── Pass 2: LLM clustering ────────────────────────────────────
        display_names = sorted(
            set(group_display.values()),
            key=lambda x: -group_pages[norm_key(normalize_label(x))],
        )
        display_to_canonical = _llm_cluster(display_names, llm, verbose=verbose)

        if verbose:
            canonicals = set(display_to_canonical.values())
            log.info(f"[merge] Pass 2 clustering: {len(display_names)} → "
                     f"{len(canonicals)} canonical chapters")

        raw_to_pass2: dict[str, str] = {
            raw: display_to_canonical.get(display, display)
            for raw, display in raw_to_display.items()
        }

        # ── Pass 3: consolidation ──────────────────────────────────────
        pass2_to_members: dict[str, set[str]] = {}
        pass2_pages: Counter[str] = Counter()
        for raw, p2 in raw_to_pass2.items():
            pass2_to_members.setdefault(p2, set()).add(raw)
            pass2_pages[p2] += counts.get(raw, 0)
        pass2_canonicals = [
            {
                "canonical": p2,
                "n_pages": pass2_pages[p2],
                "members": sorted(pass2_to_members[p2]),
            }
            for p2 in sorted(pass2_to_members, key=lambda c: -pass2_pages[c])
        ]
        pass2_to_final = _llm_consolidate(pass2_canonicals, llm, verbose=verbose)

        if verbose:
            finals = set(pass2_to_final.values())
            log.info(f"[merge] Pass 3 consolidation: {len(pass2_canonicals)} → "
                     f"{len(finals)} final canonical chapters")

        raw_to_canonical: dict[str, str] = {
            raw: pass2_to_final.get(p2, p2) for raw, p2 in raw_to_pass2.items()
        }

        # ── Pass 4: description ────────────────────────────────────────
        chapter_to_members: dict[str, list[str]] = {}
        for raw, canonical in raw_to_canonical.items():
            chapter_to_members.setdefault(canonical, []).append(raw)
        for canonical in chapter_to_members:
            chapter_to_members[canonical] = sorted(chapter_to_members[canonical])
        descriptions = _llm_describe_chapters(
            chapter_to_members, llm, verbose=verbose,
        )

        # ── Assemble tree ──────────────────────────────────────────────
        chapters: dict[str, dict[str, Any]] = {}
        for r in content_rows:
            l1 = (r.l1_local or "").strip()
            canonical = raw_to_canonical.get(l1, l1)
            meta = descriptions.get(canonical, {"description": "", "examples": []})
            bucket = chapters.setdefault(canonical, {
                "n_pages": 0,
                "description": meta["description"],
                "examples": list(meta["examples"]),
                "pages": [],
            })
            bucket["pages"].append(_posting_for_row(r))
            bucket["n_pages"] += 1

        tree = {"chapters": chapters}

        if verbose:
            log.info(f"[merge] final tree: {len(chapters)} chapters, "
                     f"{sum(c['n_pages'] for c in chapters.values())} pages")
            for chapter, data in sorted(chapters.items(),
                                        key=lambda kv: -kv[1]["n_pages"]):
                log.info(f"  {data['n_pages']:>6}  {chapter}")
        return tree
