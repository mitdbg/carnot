"""Phase 3: cross-bulletin merge of per-bulletin L1 vocabularies.

After `extract_l1` (Phase 1) and `place_pages` (Phase 2), every content
page in the catalog has an `l1_local` — the chapter name from its own
bulletin's ToC. This module merges those per-bulletin L1 names into a
clean global canonical chapter set; postings flow with each merge.

Two passes:

  Pass 1 — Deterministic normalization
      Apply `normalize_label` to every L1 name observed in the catalog.
      Group names by their lowercased normalized form. This collapses
      OCR variants ("FEERAL DEBT"), case-only differences ("Federal
      Debt" vs "FEDERAL DEBT"), and trailing-punctuation noise.

  Pass 2 — LLM clustering (one call)
      The deterministic groups still leave era-drift synonyms apart
      ("PUBLIC DEBT OPERATIONS" vs "FEDERAL DEBT", or a 1942 "Receipts
      and expenditures" against the 1985 "FEDERAL FISCAL OPERATIONS").
      One LLM call clusters these into canonical chapters. The prompt
      explicitly does NOT force a target count — the number of
      canonical chapters emerges from the data.

Output: a flat tree

    {
      "chapters": {
        "<canonical>": {
          "n_pages": int,
          "members": ["<l1_name>", ...],
          "pages": [{bulletin, page, key_phrases}, ...]
        },
        ...
      },
      "meta": {
        "n_chapters": int,
        "merge_log": [...],
        "starting_l1_distinct": int,
      }
    }
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from typing import Any

from skunk.common import LLMClient, parse_json_response

from .place import UNFILED
from .schema import PageCatalogRow
from .text_norm import norm_key, normalize_label


# ---------------------------------------------------------------------------
# Pass 1: deterministic normalization
# ---------------------------------------------------------------------------

def _normalize_groups(
    l1_counts: dict[str, int],
) -> tuple[dict[str, list[str]], dict[str, int], dict[str, str]]:
    """Group L1 names by their normalized lowercased form.

    Returns:
      groups: {norm_key → [raw_l1_name, ...]}
      group_pages: {norm_key → total page count}
      group_display: {norm_key → preferred display form}
    """
    groups: dict[str, list[str]] = defaultdict(list)
    pages: dict[str, int] = defaultdict(int)
    for raw, n in l1_counts.items():
        normed = normalize_label(raw)
        if not normed:
            continue
        k = norm_key(normed)
        groups[k].append(raw)
        pages[k] += n

    display: dict[str, str] = {}
    for k, members in groups.items():
        # Prefer a title-cased / mixed-case display form over all-caps;
        # fall back to the longest member if all are same-case.
        norms = [normalize_label(m) for m in members]
        title_cased = [n for n in norms
                       if n and not n.isupper() and not n.islower()]
        if title_cased:
            display[k] = Counter(title_cased).most_common(1)[0][0]
        else:
            display[k] = max(norms, key=len) if norms else members[0]
    return dict(groups), dict(pages), display


# ---------------------------------------------------------------------------
# Pass 2: LLM clustering
# ---------------------------------------------------------------------------

_DESCRIBE_SYSTEM = """You write a scope description plus a small example
list for each U.S. Treasury Bulletin canonical chapter.

You will receive a list of canonical chapters, each with the raw
sub-chapter / variant names that were absorbed into it during merge.
For each chapter, produce:

  1. `description` — a concrete prose statement of what the chapter
     covers. Length is your call; stop when adding another phrase
     wouldn't help a retriever distinguish this chapter from the
     others. Some chapters need a sentence; chapters with many distinct
     sub-areas may need 2–3.

  2. `examples` — a list of 4–10 concrete raw sub-chapter / topic names
     drawn from the input that exemplify what lives in this chapter.
     Pick names that:
       - cover the chapter's distinct sub-areas (don't list 5 variants
         of the same topic),
       - include era-specific or special-program entries that a prose
         description would naturally smooth over (e.g.,
         "PUBLIC WORKS ADMINISTRATION", "WAR ACTIVITIES BY GOVERNMENT
         AGENCIES" for fiscal chapters),
       - are short — single phrases, not full table captions.
     These names are the retriever's string-match anchors for questions
     that use specific era / program / topic vocabulary.

Both fields will be shown to a downstream retriever LLM. Aim for the
description and examples to be complementary, not redundant.

Output a SINGLE JSON object (no prose, no fences):
  {"chapters": {
    "<canonical chapter>": {
      "description": "<scope description>",
      "examples": ["<sub-area>", ...]
    },
    ...
  }}

Every input chapter MUST appear as a key in the output.
"""


_CONSOLIDATE_SYSTEM = """You consolidate U.S. Treasury Bulletin canonical
chapters by rolling up obvious sub-chapters into their broader parents.

You will receive a list of canonical chapters, each annotated with its
page count and the set of raw L1 names it absorbed in the previous
clustering pass.

Some of these canonical chapters are SUB-chapters of broader recurring
chapters in the Treasury Bulletin's structure. The downstream
retriever picks ONE chapter per question; finer sub-chapter splits make
that decision harder without adding information. Examples of clear
sub-chapter → parent rollups:

  - "Ownership of Federal Securities" → Federal Debt
  - "Market Quotations on Treasury Securities" → Federal Debt
  - "Average Yields of Long-Term Bonds" → Federal Debt
  - "U.S. Savings Bonds and Notes" → Federal Debt
  - "Public Debt Operations" → Federal Debt (same parent chapter)
  - "Monetary Statistics" → International Financial Statistics
  - "Account of the U.S. Treasury" → Federal Fiscal Operations
  - "Internal Revenue Statistics / Collections" → Federal Fiscal Operations
  - "Federal Obligations" → Federal Fiscal Operations
  - "Federal Agencies Financial Reports" → Government Corporations and
    Business-Type Activities
  - "Bureau of the Fiscal Service Operations" → Federal Fiscal Operations
  - "Federal Credit Programs" → Federal Fiscal Operations

Target structure: the 8 recurring chapters of the modern Treasury
Bulletin (Federal Fiscal Operations, Federal Debt, Capital Movements,
Foreign Currency Positions, International Financial Statistics, Trust
Funds, Government Corporations and Business-Type Activities, Profile of
the Economy), plus 1-3 era-specific chapters that genuinely don't fit
(e.g. "War Activities Program" if the data supports it). Do NOT split a
parent chapter into multiple clusters. Do NOT force every input into a
modern chapter if it's truly distinct — but the bar is high.

CRITICAL — no holding-pen buckets:
Do NOT create catch-all chapters named "Special Reports", "Special
Articles", "Miscellaneous", "Other", "Reports and Studies", or similar
non-topical names. These never get picked by the downstream retriever
because real questions name a topic, not a publication form. Instead,
route the would-be members by content shape:

  - Customs / vessel-clearance / import tariff / shipping-tonnage
    tables (pre-1960 fiscal-statistical pages) → Federal Fiscal Operations
  - Treasury Financing Operations narrative + auction announcements
    + debt-issuance writeups → Federal Debt
  - Speeches by Treasury officials, congressional testimony, special
    articles, narrative analytical reports → Profile of the Economy
  - Social Security / OASI / trust-fund narrative reports → Trust Funds
  - Internal revenue / tax-policy narrative → Federal Fiscal Operations
  - Bulletin masthead / front-matter / cumulative table-of-contents
    pages → Profile of the Economy (they're navigation/meta; lump with
    the closest narrative chapter rather than create a junk bucket)
  - War-era program appropriations → "War Activities Program" if the
    data is dense enough; otherwise Federal Fiscal Operations

When in doubt for a sub-area that COULD fit a topical chapter, fit it
there. Only keep a separate chapter when its members are so era-specific
that no modern chapter applies AND the bucket has enough pages to be
worth a separate retrieve target.

Output a SINGLE JSON object (no prose, no markdown fences):

  {"consolidations": [
    {"parent": "Federal Debt",
     "members": ["Federal Debt", "Public Debt Operations",
                 "Ownership of Federal Securities",
                 "Market Quotations on Treasury Securities",
                 "Average Yields of Long-Term Bonds",
                 "U.S. Savings Bonds and Notes"]},
    {"parent": "Federal Fiscal Operations",
     "members": ["Federal Fiscal Operations", "Account of the U.S. Treasury",
                 "Internal Revenue Statistics", "Federal Obligations"]},
    ...
  ]}

Every input canonical MUST appear in exactly one consolidation's
`members` list. If a canonical is its own parent (no rollup), include
it in a one-member consolidation.
"""


_CLUSTER_SYSTEM = """You cluster U.S. Treasury Bulletin chapter headings into
canonical chapters.

You will receive a list of distinct top-level chapter names observed
across many bulletins (1939-2025). Each represents a recurring or
era-specific chapter from one or more bulletins. Some are clear
synonyms ("FEDERAL DEBT" / "Federal debt" / "Public debt and guaranteed
obligations of the United States Government" all refer to the Federal
Debt chapter). Some are era-specific and may or may not have a modern
analogue.

Cluster them into canonical chapters. Each cluster represents one
recurring chapter concept across decades.

Rules:
  - Pick a CANONICAL name per cluster — short, Title Case, no trailing
    punctuation. Match modern Treasury Bulletin usage when possible
    (e.g. "Federal Fiscal Operations", "Federal Debt", "Capital
    Movements", "Foreign Currency Positions", "International Financial
    Statistics", "Trust Funds", "Government Corporations and
    Business-Type Activities", "Profile of the Economy").
  - Every input chapter name MUST appear in exactly one cluster's
    `members` list.
  - Do NOT force a target count. Let the data decide. If a 1940s-only
    chapter is meaningfully distinct from every modern chapter, leave it
    as its own one-member cluster.
  - Cluster ESF / "Exchange Stabilization Fund" entries into whatever
    parent the data supports: if they appear distinct enough across
    eras, keep them split between Capital Movements and Foreign
    Currency Positions members; if they look unified, merge them.
  - Do NOT split a single canonical chapter into multiple clusters.

Output a SINGLE JSON object (no prose, no markdown fences):

  {"clusters": [
    {"canonical": "Federal Debt",
     "members": ["FEDERAL DEBT", "Federal debt", "Public debt and
      guaranteed obligations of the United States Government", ...]},
    {"canonical": "Federal Fiscal Operations",
     "members": ["FEDERAL FISCAL OPERATIONS", "Federal fiscal
      operations", "Receipts and expenditures", ...]},
    ...
  ]}
"""


def _llm_describe_chapters(
    chapters_with_members: dict[str, list[str]], llm: LLMClient,
    *, verbose: bool = False,
) -> dict[str, dict[str, Any]]:
    """One LLM call → {chapter → {description, examples}}.

    `chapters_with_members` maps each final canonical chapter name to
    the sorted list of raw L1 names it absorbed. Missing chapters fall
    back to empty description + empty examples.
    """
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
    resp = llm.call(system=_DESCRIBE_SYSTEM, user=user,
                    temperature=0.0, thinking_budget=0)
    if verbose:
        print(f"  [merge] Pass-4 description LLM in "
              f"{time.monotonic() - t0:.1f}s", flush=True)
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


def _llm_consolidate(
    canonical_chapters: list[dict[str, Any]], llm: LLMClient,
    *, verbose: bool = False,
) -> dict[str, str]:
    """Pass-3 consolidation: roll up Pass-2 canonicals into broader parents.

    `canonical_chapters` is a list of dicts shaped like:
        {"canonical": "Ownership of Federal Securities",
         "n_pages": 4312,
         "members": ["OWNERSHIP OF FEDERAL SECURITIES", ...]}

    Returns a mapping {pass2_canonical → final_parent}. Any name omitted
    by the LLM falls back to its own parent (no rollup).
    """
    if not canonical_chapters:
        return {}
    user = (
        f"Pass-2 canonical chapters ({len(canonical_chapters)}):\n"
        f"{json.dumps(canonical_chapters, ensure_ascii=False, indent=1)}\n"
    )
    t0 = time.monotonic()
    resp = llm.call(system=_CONSOLIDATE_SYSTEM, user=user,
                    temperature=0.0, thinking_budget=0)
    if verbose:
        print(f"  [merge] Pass-3 consolidation LLM in "
              f"{time.monotonic() - t0:.1f}s", flush=True)
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

    # Backfill: any chapter the LLM omitted stays as its own parent.
    for c in canonical_chapters:
        name = c["canonical"]
        mapping.setdefault(name, name)
    return mapping


def _llm_cluster(
    display_names: list[str], llm: LLMClient, *, verbose: bool = False,
) -> dict[str, str]:
    """One LLM call to cluster `display_names` into canonical chapters.

    Returns a mapping `{display_name → canonical}`. Names omitted from
    the LLM response fall back to their own canonical (one-member).
    """
    if not display_names:
        return {}

    user = (
        f"Distinct chapter names to cluster ({len(display_names)}):\n"
        f"{json.dumps(display_names, ensure_ascii=False, indent=1)}\n"
    )
    t0 = time.monotonic()
    resp = llm.call(system=_CLUSTER_SYSTEM, user=user,
                    temperature=0.0, thinking_budget=0)
    if verbose:
        print(f"  [merge] LLM clustering in {time.monotonic() - t0:.1f}s",
              flush=True)
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

    # Backfill: any display name the LLM omitted becomes its own canonical.
    for name in display_names:
        mapping.setdefault(name, name)
    return mapping


# ---------------------------------------------------------------------------
# Tree assembly
# ---------------------------------------------------------------------------

def _posting_for_row(r: PageCatalogRow) -> dict[str, Any]:
    return {
        "bulletin": r.bulletin,
        "page": r.page,
        "key_phrases": list(r.keywords),
    }


def build_tree(
    catalog: list[PageCatalogRow],
    llm: LLMClient,
    *,
    drop_unfiled: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """End-to-end Phase 3.

    Walks every content row in `catalog`; runs deterministic
    normalization → LLM clustering → flat tree assembly.
    """
    # Single pass: capture every retrievable row and tally its l1_local.
    # Reused at tree-assembly time, so we don't re-filter the catalog.
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
        print(f"  [merge] {len(counts)} distinct l1_local names "
              f"covering {sum(counts.values())} pages", flush=True)

    # ── Pass 1: deterministic normalization ────────────────────────────
    groups, group_pages, group_display = _normalize_groups(dict(counts))
    if verbose:
        print(f"  [merge] Pass 1 collapse: {len(counts)} → "
              f"{len(groups)} normalized groups", flush=True)

    # raw_l1_name → display form (post-pass-1 canonical)
    raw_to_display: dict[str, str] = {}
    for k, raws in groups.items():
        for raw in raws:
            raw_to_display[raw] = group_display[k]

    # ── Pass 2: LLM clustering on display forms ───────────────────────
    display_names = sorted(set(group_display.values()),
                           key=lambda x: -group_pages[norm_key(normalize_label(x))])
    display_to_canonical = _llm_cluster(display_names, llm, verbose=verbose)

    if verbose:
        canonicals = set(display_to_canonical.values())
        print(f"  [merge] Pass 2 clustering: {len(display_names)} → "
              f"{len(canonicals)} canonical chapters", flush=True)

    # raw_l1_name → Pass-2 canonical
    raw_to_pass2: dict[str, str] = {
        raw: display_to_canonical.get(display, display)
        for raw, display in raw_to_display.items()
    }

    # ── Pass 3: consolidate Pass-2 canonicals into broader parents ────
    # Build the Pass-3 input: each Pass-2 canonical with its page count
    # and the L1 names it absorbed.
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
        print(f"  [merge] Pass 3 consolidation: {len(pass2_canonicals)} → "
              f"{len(finals)} final canonical chapters", flush=True)

    raw_to_canonical: dict[str, str] = {
        raw: pass2_to_final.get(p2, p2) for raw, p2 in raw_to_pass2.items()
    }

    # ── Pass 4: one-sentence scope description per chapter ───────────
    chapter_to_members: dict[str, list[str]] = {}
    for raw, canonical in raw_to_canonical.items():
        chapter_to_members.setdefault(canonical, []).append(raw)
    for canonical in chapter_to_members:
        chapter_to_members[canonical] = sorted(chapter_to_members[canonical])
    descriptions = _llm_describe_chapters(chapter_to_members, llm, verbose=verbose)

    # ── Assemble flat tree ─────────────────────────────────────────────
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
        print(f"  [merge] final tree: {len(chapters)} chapters, "
              f"{sum(c['n_pages'] for c in chapters.values())} pages",
              flush=True)
        for chapter, data in sorted(chapters.items(),
                                    key=lambda kv: -kv[1]["n_pages"]):
            print(f"    {data['n_pages']:>6}  {chapter}", flush=True)
            print(f"           desc: {data['description']}", flush=True)
            print(f"           examples: {data['examples']}", flush=True)
    return tree
