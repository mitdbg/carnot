"""v0.4 concept tree — LLM-batch clustering, no embeddings.

Pipeline:
  A. propose_labels: shuffle the corpus-wide dateless keyword vocabulary, batch
     into chunks of ~300, ask an LLM per batch to produce 10-25 short cluster
     labels covering the batch. Union the labels across batches.
  B. dedup_labels: one LLM call merges near-paraphrase labels into a canonical
     set; emits a {raw_label → canonical_label} rewrite map.
  C. assign_keywords: in batches of ~1000, ask the LLM to assign each keyword
     a (cluster_label, section_label) pair, picking from the canonical label set.
     section_label is free-text — it falls through as the keyword's home only
     when no page banner is available.
  D. build_tree: deterministic walk over the catalog. For each (keyword, page),
     look up the assignment, then key the posting under the page's actual banner
     section (the LLM-picked section is only the fallback for banner-less pages).

Output JSON shape:
  {"sections": {<section>: {"clusters": {<label>: {"keywords":
      {<kw>: [{"bulletin": "YYYY-MM", "page": int, "dates": [...]}, ...]}}}}},
   "meta": {...}}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from skunk.common import LLMClient
from skunk.config import SkunkConfig

from .schema import PageCatalogRow


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def load_catalog(catalog_dir: Path) -> list[PageCatalogRow]:
    """Load every row from every JSONL in `catalog_dir`."""
    out: list[PageCatalogRow] = []
    for p in sorted(catalog_dir.glob("*.jsonl")):
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            out.append(PageCatalogRow.from_json(line))
    return out


# ---------------------------------------------------------------------------
# Step A — propose cluster labels in batches.
# ---------------------------------------------------------------------------

_PROPOSE_SYSTEM = """You are organizing U.S. Treasury Bulletin keywords into a
small set of CONCEPT CLUSTER LABELS.

You will receive a randomized batch of dateless keywords (each a short Treasury
noun phrase). Produce a small set of cluster labels that would cover this batch.

Rules:
  - Each label is a short noun phrase (2-5 words) describing a coherent concept.
  - Aim for 10-25 labels per batch. Fewer is better when keywords cluster cleanly.
  - Labels MUST NOT contain dates, years, months, fiscal/calendar year tokens.
  - Prefer labels that match Treasury's own vocabulary
    (e.g. "Marketable securities", "Statutory debt limitation", "Capital movements").
  - Output a SINGLE JSON object (no prose, no fences):
      {"labels": ["...", "...", ...]}
"""


def _parse_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        if text.endswith("```"):
            text = text[: -3]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def propose_labels(
    vocab: list[str],
    llm: LLMClient,
    *,
    batch_size: int = 300,
    seed: int = 42,
    workers: int = 1,
    verbose: bool = False,
) -> list[str]:
    """Run ~ceil(len(vocab)/batch_size) LLM calls (optionally in parallel);
    return the union of labels in deterministic order."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    shuffled = list(vocab)
    random.Random(seed).shuffle(shuffled)

    n_batches = (len(shuffled) + batch_size - 1) // batch_size

    def _run(i: int) -> tuple[int, list[str]]:
        batch = shuffled[i * batch_size:(i + 1) * batch_size]
        user = (
            f"Batch {i+1} of {n_batches}. "
            f"{len(batch)} Treasury Bulletin keywords:\n\n"
            f"{json.dumps(batch, ensure_ascii=False, indent=1)}\n"
        )
        resp = llm.call(system=_PROPOSE_SYSTEM, user=user, temperature=0.2)
        obj = _parse_json(resp.text)
        labels = (obj or {}).get("labels", []) if isinstance(obj, dict) else []
        return i, [str(lab).strip() for lab in labels if isinstance(lab, str)]

    per_batch: dict[int, list[str]] = {}
    t0 = time.monotonic()
    if workers <= 1:
        for i in range(n_batches):
            _, labs = _run(i)
            per_batch[i] = labs
            if verbose:
                print(f"  [propose] batch {i+1}/{n_batches}  "
                      f"out={len(labs)}  ({time.monotonic()-t0:.1f}s)", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run, i) for i in range(n_batches)]
            done = 0
            for f in as_completed(futs):
                i, labs = f.result()
                per_batch[i] = labs
                done += 1
                if verbose:
                    print(f"  [propose] batch {i+1}/{n_batches}  "
                          f"out={len(labs)}  done={done}/{n_batches}  "
                          f"({time.monotonic()-t0:.1f}s)", flush=True)

    # Deterministic union: walk batches in index order.
    all_labels: list[str] = []
    seen: set[str] = set()
    for i in range(n_batches):
        for lab in per_batch.get(i, []):
            if not lab or lab.lower() in seen:
                continue
            seen.add(lab.lower())
            all_labels.append(lab)
    return all_labels


# ---------------------------------------------------------------------------
# Step B — dedup labels into a canonical set.
# ---------------------------------------------------------------------------

_DEDUP_SYSTEM = """You are deduplicating concept-cluster labels for a Treasury
Bulletin retrieval index.

You will receive a list of candidate labels. Some are paraphrases or near-duplicates
of each other (e.g. "Marketable Securities" vs. "Marketable securities" vs.
"Public marketable securities"). Merge them.

Rules:
  - Pick a small CANONICAL label per merge group (Title Case, no trailing dot).
  - Every input label MUST be assigned to exactly one canonical label (it may be
    its own canonical if nothing merges into it).
  - Output a SINGLE JSON object (no prose, no fences):
      {"merges": [
         {"canonical": "Marketable securities",
          "members": ["Marketable Securities", "Marketable securities", "Public marketable securities"]},
         {"canonical": "Statutory debt limitation",
          "members": ["Statutory Debt Limitation", "Status under Limitation"]},
         ...
      ]}
"""


def dedup_labels(
    raw_labels: list[str],
    llm: LLMClient,
    *,
    verbose: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """Returns (canonical_labels, rewrite_map: raw_lower → canonical)."""
    if not raw_labels:
        return [], {}

    user = (
        f"Candidate labels ({len(raw_labels)}):\n"
        f"{json.dumps(raw_labels, ensure_ascii=False, indent=1)}\n"
    )
    t0 = time.monotonic()
    resp = llm.call(system=_DEDUP_SYSTEM, user=user, temperature=0.0)
    if verbose:
        print(f"  [dedup] LLM in {time.monotonic()-t0:.1f}s", flush=True)
    obj = _parse_json(resp.text)
    merges = (obj or {}).get("merges", []) if isinstance(obj, dict) else []

    rewrite: dict[str, str] = {}
    canonical_set: list[str] = []
    canonical_seen: set[str] = set()
    for m in merges:
        if not isinstance(m, dict):
            continue
        canonical = str(m.get("canonical") or "").strip()
        members = m.get("members") or []
        if not canonical or not isinstance(members, list):
            continue
        if canonical.lower() not in canonical_seen:
            canonical_seen.add(canonical.lower())
            canonical_set.append(canonical)
        for mb in members:
            key = str(mb).strip().lower()
            if key:
                rewrite[key] = canonical

    # Backfill: any raw label the LLM forgot becomes its own canonical.
    for lab in raw_labels:
        key = lab.strip().lower()
        if key not in rewrite:
            rewrite[key] = lab
            if lab.lower() not in canonical_seen:
                canonical_seen.add(lab.lower())
                canonical_set.append(lab)

    return canonical_set, rewrite


# ---------------------------------------------------------------------------
# Step B.5 — canonicalize raw section banners (Stage A deterministic +
# Stage B LLM-against-seed-list).
# ---------------------------------------------------------------------------

# Stage A: small explicit OCR substitution table — empirically observed in
# this corpus (parsed-JSON had OCR errors on older bulletins). Applied
# case-insensitively as whole-word substring replacements where the
# replacement makes sense as a whole word.
_OCR_SUBS: list[tuple[re.Pattern, str]] = [
    # Letter-substitution OCR errors that turn "DEBT" → "OEBT", etc.
    (re.compile(r"\bOEBT\b", re.IGNORECASE), "DEBT"),
    (re.compile(r"\bOATA\b", re.IGNORECASE), "DATA"),
    (re.compile(r"\bYIELOS\b", re.IGNORECASE), "YIELDS"),
    (re.compile(r"\bBONOS\b", re.IGNORECASE), "BONDS"),
    (re.compile(r"\bPISCAL\b", re.IGNORECASE), "FISCAL"),
    (re.compile(r"\bFEERAL\b", re.IGNORECASE), "FEDERAL"),
    (re.compile(r"\bACOUNT\b", re.IGNORECASE), "ACCOUNT"),
    (re.compile(r"\bFUNOS\b", re.IGNORECASE), "FUNDS"),
    (re.compile(r"\bFUNO\b", re.IGNORECASE), "FUND"),
    (re.compile(r"\bUNITEO\b", re.IGNORECASE), "UNITED"),
    (re.compile(r"\bSAV INGS\b", re.IGNORECASE), "SAVINGS"),
    (re.compile(r"\bRE PORT\b", re.IGNORECASE), "REPORT"),
    (re.compile(r"\bFIS CAL\b", re.IGNORECASE), "FISCAL"),
    # Missing-whitespace OCR concatenations.
    (re.compile(r"\bTRUSTFUNDS\b", re.IGNORECASE), "TRUST FUNDS"),
    (re.compile(r"\bFOREIGNCURRENCYPOSITIONS\b", re.IGNORECASE),
     "FOREIGN CURRENCY POSITIONS"),
    (re.compile(r"\bPUBLICDEBTOPERATIONS\b", re.IGNORECASE),
     "PUBLIC DEBT OPERATIONS"),
    (re.compile(r"\bEXCHANGESTABILIZATION\b", re.IGNORECASE),
     "EXCHANGE STABILIZATION"),
    (re.compile(r"U\.S\.TREASURY", re.IGNORECASE), "U.S. TREASURY"),
    (re.compile(r"\bD\.S\.", re.IGNORECASE), "U.S."),
    # Truncation artifacts.
    (re.compile(r"\bCAPAL\b", re.IGNORECASE), "CAPITAL"),
    (re.compile(r"\bCAPITAL MOVEM\b", re.IGNORECASE), "CAPITAL MOVEMENTS"),
    (re.compile(r"\bMOVEM\b", re.IGNORECASE), "MOVEMENTS"),
]

# Year-marker suffixes ("FISCAL YEAR 1988 (PROTOTYPE)", "FISCAL 1996 (EXCERPT)",
# "AS OF SEPT. 30, 1984.", etc.) — strip the year + parenthetical kind.
_YEAR_SUFFIX_RE = re.compile(
    r",?\s*FISCAL(?:\s+YEAR)?\s+\d{4}\s*\(?(?:PROTOTYPE|EXCERPT|EXTRACT|EXCERPTED)?\)?\.?$",
    re.IGNORECASE,
)
_AS_OF_DATE_RE = re.compile(
    r"\s+AS\s+OF\s+\w+\.?\s+\d{1,2},?\s+\d{4}\.?$",
    re.IGNORECASE,
)
# Roman-numeral / leading-decimal prefixes: "V. ", "VI. ", "II. ", "1. ", etc.
_PREFIX_NUMERAL_RE = re.compile(r"^\s*(?:[IVX]+|\d+)\.\s+")
# Continuation suffix (reuse the same pattern as build.py).
_CONT_SUFFIX_RE = re.compile(
    r"[,\s\-–—]+(?:con|cont|continued)\s*\.?\s*\)?\s*$", re.IGNORECASE,
)
# Trailing punctuation / whitespace.
_TRAILING_PUNCT_RE = re.compile(r"[\s\.,;:\-–—]+$")


def normalize_label(raw: str) -> str:
    """Stage A deterministic normalization.

    Strips OCR errors, year suffixes, roman-numeral prefixes, continuation
    suffixes, and trailing punctuation. Collapses whitespace. The result
    is suitable as a grouping key for variants of the same section.

    Returns the cleaned string in its original case (so downstream Stage B
    sees a clean form). The case-merge groups happen later via lowercased
    keying.
    """
    s = raw.strip()
    # Iterate OCR + suffix stripping until fixed point — some labels have
    # multiple issues stacked (e.g. roman-numeral + trailing year).
    for _ in range(4):
        prev = s
        for pattern, repl in _OCR_SUBS:
            s = pattern.sub(repl, s)
        s = _YEAR_SUFFIX_RE.sub("", s).strip()
        s = _AS_OF_DATE_RE.sub("", s).strip()
        s = _PREFIX_NUMERAL_RE.sub("", s).strip()
        s = _CONT_SUFFIX_RE.sub("", s).strip()
        s = _TRAILING_PUNCT_RE.sub("", s).strip()
        if s == prev:
            break
    # Collapse internal whitespace runs.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_key(s: str) -> str:
    """Lowercased grouping key for case-insensitive matching."""
    return s.lower().strip()


# Stage B: LLM canonicalization against a seed list.
_CANON_SYSTEM = """You are mapping U.S. Treasury Bulletin section labels to
a canonical list. The seed canonical list represents the recurring chapter
headings in the bulletin's Table of Contents.

For each input label, output the closest canonical from the seed list.

Older bulletins (1940s-50s) use longer or roman-numeral-prefixed phrasing
for what later became standard chapter names. Examples:
  - "Public debt and guaranteed obligations of the United States Government"
    → "Public Debt Operations"
  - "Ownership of Government securities" → "Ownership of Federal Securities"
  - "Receipts and expenditures and appropriations" → "Federal Fiscal Operations"

If an input is clearly a one-off historical article or report title (e.g.
"The Role of Saving in a Dynamic U.S. Economy", "Findings of the Joint
Report on the Government Securities Market") and does NOT match any seed
chapter, map it to "Special Article".

Output a SINGLE JSON object (no prose, no fences):
  {"map": {"<input verbatim>": "<canonical from seed list or 'Special Article'>",
           ...}}

Every input label MUST appear as a key in the output map. Use the EXACT
seed-list string for the value (preserve casing as shown).
"""


def canonicalize_banners(
    raw_banners: list[str],
    banner_page_counts: dict[str, int],
    llm: LLMClient,
    *,
    seed_n: int = 30,
    verbose: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """Two-stage canonicalization.

    Stage A: apply `normalize_label` deterministically to every raw banner;
    group by lowercased normalized key.
    Stage B: take the top-`seed_n` normalized groups by total page count as
    seed canonicals (these are the obvious recurring ToC chapters). Send
    the remaining normalized labels to one LLM call asking for a mapping
    into the seed list (or "Special Article" for genuine one-offs).

    Returns (canonical_set, rewrite_map) where `rewrite_map` is keyed by
    `raw.strip().lower()` and maps to the canonical label. Compatible with
    the on-disk shape used by prior `dedup_banners` so callers (e.g.
    `build_tree`) don't change.
    """
    if not raw_banners:
        return [], {}

    # Stage A: normalize every raw banner.
    raw_to_norm: dict[str, str] = {r: normalize_label(r) for r in raw_banners}

    # Aggregate page counts to a normalized-key view. Group by lowercased
    # normalized form so case-only and trailing-punct variants merge.
    norm_groups: dict[str, list[str]] = {}     # _norm_key → [raw, ...]
    norm_pages: dict[str, int] = {}             # _norm_key → total pages
    for raw, norm in raw_to_norm.items():
        if not norm:
            continue
        k = _norm_key(norm)
        norm_groups.setdefault(k, []).append(raw)
        norm_pages[k] = norm_pages.get(k, 0) + banner_page_counts.get(raw, 0)

    # Pick the display form per normalized group: prefer the title-cased
    # variant (mixed case) among the raw banners that landed in this group,
    # else the longest one.
    def _display(rawvars: list[str]) -> str:
        # Apply Stage A to each, then pick the most "title-case" looking one.
        norms = [normalize_label(r) for r in rawvars]
        title_cased = [n for n in norms if n and not n.isupper() and not n.islower()]
        if title_cased:
            # Pick most common title-cased form.
            from collections import Counter
            return Counter(title_cased).most_common(1)[0][0]
        return max(norms, key=len) if norms else rawvars[0]

    norm_display: dict[str, str] = {k: _display(rawvars)
                                    for k, rawvars in norm_groups.items()}

    # Stage B seeds: top `seed_n` normalized groups by page count.
    seed_keys = sorted(norm_pages.keys(), key=lambda k: -norm_pages[k])[:seed_n]
    seed_canonicals = [norm_display[k] for k in seed_keys]
    seed_set_lower = {_norm_key(s) for s in seed_canonicals}

    if verbose:
        print(f"  [canonicalize] {len(raw_to_norm)} raw → "
              f"{len(norm_groups)} normalized groups", flush=True)
        print(f"  [canonicalize] seed (top-{seed_n} by pages):", flush=True)
        for k in seed_keys:
            print(f"    {norm_pages[k]:6}  {norm_display[k]!r}", flush=True)

    # Labels to send to the LLM: every normalized display that isn't already
    # a seed canonical. Send normalized form so the LLM sees clean text.
    to_route = [norm_display[k] for k in norm_groups if k not in seed_set_lower]

    routing: dict[str, str] = {}      # lowercased norm_display → canonical
    if to_route:
        user = (
            f"Seed canonical list ({len(seed_canonicals)}):\n"
            f"{json.dumps(seed_canonicals, ensure_ascii=False, indent=1)}\n\n"
            f"Input labels to map ({len(to_route)}):\n"
            f"{json.dumps(to_route, ensure_ascii=False, indent=1)}\n"
        )
        t0 = time.monotonic()
        resp = llm.call(system=_CANON_SYSTEM, user=user, temperature=0.0)
        if verbose:
            print(f"  [canonicalize] LLM in {time.monotonic()-t0:.1f}s",
                  flush=True)
        obj = _parse_json(resp.text)
        mp = (obj or {}).get("map", {}) if isinstance(obj, dict) else {}
        valid = {_norm_key(s): s for s in seed_canonicals}
        valid["special article"] = "Special Article"
        for label, target in (mp or {}).items():
            t_norm = _norm_key(str(target).strip())
            canonical = valid.get(t_norm)
            if canonical is None:
                # LLM emitted something off-list — keep input as its own canonical.
                canonical = str(label).strip()
            routing[_norm_key(str(label))] = canonical

    # Assemble the rewrite map (keyed on raw.lower).
    rewrite: dict[str, str] = {}
    canonical_set: list[str] = list(seed_canonicals)
    canonical_set.append("Special Article")
    canonical_seen = {_norm_key(c) for c in canonical_set}

    for k, rawvars in norm_groups.items():
        display = norm_display[k]
        if k in seed_set_lower:
            canonical = display
        else:
            canonical = routing.get(_norm_key(display), display)
            if _norm_key(canonical) not in canonical_seen:
                canonical_seen.add(_norm_key(canonical))
                canonical_set.append(canonical)
        for raw in rawvars:
            rewrite[raw.strip().lower()] = canonical

    return canonical_set, rewrite


def _collect_banner_page_counts(catalog: list[PageCatalogRow]) -> dict[str, int]:
    """Per-bulletin section banner → page count across the corpus."""
    out: dict[str, int] = {}
    for r in catalog:
        if r.page_kind not in ("table", "chart", "prose"):
            continue
        sec = (r.section or "").strip()
        if sec:
            out[sec] = out.get(sec, 0) + 1
    return out


# Back-compat alias so existing callsites keep working. The arity matches
# the prior signature (`raw_banners, llm, *, batch_size=..., workers=...,
# verbose=...`) but the body delegates to `canonicalize_banners`. The
# `banner_page_counts` argument is required; new callers should use
# `canonicalize_banners` directly.
def dedup_banners(
    raw_banners: list[str],
    llm: LLMClient,
    *,
    banner_page_counts: dict[str, int] | None = None,
    verbose: bool = False,
    **_legacy_kw,
) -> tuple[list[str], dict[str, str]]:
    counts = banner_page_counts or {r: 1 for r in raw_banners}
    return canonicalize_banners(raw_banners, counts, llm, verbose=verbose)


# ---------------------------------------------------------------------------
# Step C — assign each keyword to a (cluster_label, section_label).
# ---------------------------------------------------------------------------

_ASSIGN_SYSTEM = """You are routing Treasury Bulletin keywords to a concept-cluster
index for retrieval.

You will receive:
  - The full set of canonical cluster labels.
  - A batch of dateless keywords.

For each keyword, pick the closest cluster_label from the canonical list.
The page's actual section is determined elsewhere from its banner — your job
is just the concept assignment.

Rules:
  - cluster_label MUST be an exact string from the canonical list (case
    matters; copy verbatim).
  - Assign EVERY keyword in the batch. Do not skip any. If a keyword genuinely
    has no good match, pick the closest available — anything beats "missing".
  - Output a SINGLE JSON object (no prose, no fences):
      {"assignments": [
          {"keyword": "Marketable", "cluster_label": "Marketable securities"},
          ...
      ]}
"""


def _assign_pass(
    vocab: list[str],
    canonical_labels: list[str],
    llm: LLMClient,
    *,
    batch_size: int,
    workers: int,
    label_tag: str,
    verbose: bool,
) -> dict[str, str]:
    """One pass through `vocab` in batches; returns keyword → cluster_label."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out: dict[str, str] = {}
    label_map = {lab.lower(): lab for lab in canonical_labels}

    if not vocab:
        return out

    n_batches = (len(vocab) + batch_size - 1) // batch_size
    t0 = time.monotonic()

    def _run_batch(i: int) -> tuple[int, list[dict], int]:
        batch = vocab[i * batch_size:(i + 1) * batch_size]
        user = (
            f"Canonical cluster labels ({len(canonical_labels)}):\n"
            f"{json.dumps(canonical_labels, ensure_ascii=False, indent=1)}\n\n"
            f"Keyword batch {i+1}/{n_batches}  ({len(batch)} keywords). "
            f"Return exactly {len(batch)} entries, one per keyword.\n"
            f"{json.dumps(batch, ensure_ascii=False, indent=1)}\n"
        )
        resp = llm.call(system=_ASSIGN_SYSTEM, user=user, temperature=0.0)
        obj = _parse_json(resp.text)
        entries = (obj or {}).get("assignments", []) if isinstance(obj, dict) else []
        return i, entries if isinstance(entries, list) else [], len(batch)

    def _merge(i: int, entries: list[dict], batch_size_actual: int) -> int:
        n_ok = 0
        for e in entries:
            if not isinstance(e, dict):
                continue
            kw = str(e.get("keyword") or "").strip()
            cl = str(e.get("cluster_label") or "").strip()
            if not kw or not cl:
                continue
            canonical = label_map.get(cl.lower())
            if canonical is None:
                continue
            out[kw] = canonical
            n_ok += 1
        return n_ok

    if workers <= 1:
        for i in range(n_batches):
            _, entries, bsz = _run_batch(i)
            n_ok = _merge(i, entries, bsz)
            if verbose:
                print(f"  [{label_tag}] batch {i+1}/{n_batches}  "
                      f"in={bsz}  ok={n_ok}  total={len(out)}  "
                      f"({time.monotonic()-t0:.1f}s)", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_batch, i) for i in range(n_batches)]
            done = 0
            for f in as_completed(futs):
                i, entries, bsz = f.result()
                n_ok = _merge(i, entries, bsz)
                done += 1
                if verbose:
                    print(f"  [{label_tag}] batch {i+1}/{n_batches}  "
                          f"in={bsz}  ok={n_ok}  done={done}/{n_batches}  "
                          f"total={len(out)}  ({time.monotonic()-t0:.1f}s)",
                          flush=True)
    return out


def assign_keywords(
    vocab: list[str],
    canonical_labels: list[str],
    llm: LLMClient,
    *,
    batch_size: int = 1000,
    workers: int = 1,
    max_retries: int = 4,
    retry_shrink: float = 0.5,
    verbose: bool = False,
) -> dict[str, str]:
    """For each keyword, return cluster_label.

    Runs an initial pass, then verifies coverage and retries the missing
    keywords with progressively smaller batch sizes (default halving each
    retry — long prompts truncate the LLM's output, so smaller batches
    naturally complete). Stops when every keyword is assigned, max_retries
    is hit, or two consecutive retries make no progress.
    """
    assignments: dict[str, str] = {}

    def _missing() -> list[str]:
        return [kw for kw in vocab if kw not in assignments]

    current_batch = batch_size
    pass_idx = 0
    while True:
        todo = _missing()
        if not todo:
            break
        if verbose:
            print(f"\n[assign pass {pass_idx}] {len(todo)} keywords pending, "
                  f"batch_size={current_batch}, workers={workers}", flush=True)
        before = len(assignments)
        partial = _assign_pass(
            todo, canonical_labels, llm,
            batch_size=current_batch, workers=workers,
            label_tag=f"assign p{pass_idx}", verbose=verbose,
        )
        assignments.update(partial)
        gained = len(assignments) - before
        if verbose:
            print(f"[assign pass {pass_idx}] +{gained} new "
                  f"({len(assignments)}/{len(vocab)} total)", flush=True)

        pass_idx += 1
        if pass_idx > max_retries:
            if verbose:
                print(f"[assign] max_retries={max_retries} hit; "
                      f"{len(_missing())} keywords remain unassigned", flush=True)
            break
        if gained == 0:
            if verbose:
                print(f"[assign] no progress this pass; stopping with "
                      f"{len(_missing())} unassigned", flush=True)
            break
        # Shrink the batch size for the retry — smaller prompts give the LLM
        # room to return every entry without hitting output limits.
        current_batch = max(50, int(current_batch * retry_shrink))

    return assignments


# ---------------------------------------------------------------------------
# Step D — assemble the tree.
# ---------------------------------------------------------------------------

def _collect_keyword_vocab(catalog: list[PageCatalogRow]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in catalog:
        if r.page_kind not in ("table", "chart", "prose"):
            continue
        for kw in r.keywords:
            k = kw.strip()
            if not k:
                continue
            kl = k.lower()
            if kl in seen:
                continue
            seen.add(kl)
            out.append(k)
    return out


def _collect_section_labels(catalog: list[PageCatalogRow]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in catalog:
        sec = (r.section or "").strip()
        if not sec or sec.lower() in seen:
            continue
        seen.add(sec.lower())
        out.append(sec)
    return out


def _collect_raw_banners(catalog: list[PageCatalogRow]) -> list[str]:
    """Sorted unique raw section banners across the catalog."""
    seen: set[str] = set()
    for r in catalog:
        if r.page_kind not in ("table", "chart", "prose"):
            continue
        sec = (r.section or "").strip()
        if sec:
            seen.add(sec)
    return sorted(seen)


def build_tree(
    catalog: list[PageCatalogRow],
    assignments: dict[str, str],
    banner_rewrite: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Walk the catalog and bucket (keyword, bulletin, page, dates, table_title) postings.

    `assignments` is keyword → cluster_label. The section comes from the page's
    actual banner (`row.section`) after `banner_rewrite` collapses raw variants
    into canonical ToC sections. Rows without a banner go under "Unsectioned"
    so they remain reachable in the tree.
    """
    rewrite = banner_rewrite or {}
    # Lowercased lookup view so case-variant keywords still resolve. The vocab
    # generator preserves the first-seen case form (e.g. "TREASURY BILLS" from
    # a 1942 row), so a later catalog row with "Treasury Bills" would otherwise
    # fail both the case-sensitive and the `.get(kw.lower())` lookup.
    assignments_lower = {k.lower(): v for k, v in assignments.items()}
    tree: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    # tree[section][cluster_label][keyword] = [{bulletin, page, dates, table_title}, ...]

    for r in catalog:
        if r.page_kind not in ("table", "chart", "prose"):
            continue
        raw_section = (r.section or "").strip()
        if raw_section:
            section = rewrite.get(raw_section.lower(), raw_section)
        else:
            section = "Unsectioned"
        for kw in r.keywords:
            k = kw.strip()
            if not k:
                continue
            cluster_label = assignments.get(k) or assignments_lower.get(k.lower())
            if cluster_label is None:
                continue
            # Bucket under the lowercased keyword so case variants merge into
            # the same posting list inside the tree.
            tree.setdefault(section, {}) \
                .setdefault(cluster_label, {}) \
                .setdefault(k.lower(), []) \
                .append({
                    "bulletin": r.bulletin,
                    "page": r.page,
                    "dates": list(r.dates),
                    "table_title": r.table_title or "",
                })

    # Wrap into the on-disk shape, computing summary counts.
    sections_out: dict[str, Any] = {}
    for section, clusters in tree.items():
        clusters_out: dict[str, Any] = {}
        for cluster, kws in clusters.items():
            n_pages = sum(len(v) for v in kws.values())
            clusters_out[cluster] = {
                "n_keywords": len(kws),
                "n_pages": n_pages,
                "keywords": kws,
            }
        sections_out[section] = {
            "n_clusters": len(clusters_out),
            "n_pages_in_section": sum(c["n_pages"] for c in clusters_out.values()),
            "clusters": clusters_out,
        }

    return {
        "sections": sections_out,
        "meta": {
            "n_sections": len(sections_out),
            "n_clusters": sum(len(s["clusters"]) for s in sections_out.values()),
            "n_keywords_assigned": len(assignments),
            "banner_rewrite": dict(rewrite),
        },
    }


# ---------------------------------------------------------------------------
# Alternative L2: 5-year time buckets (no concept clustering).
# ---------------------------------------------------------------------------

def _bucket_label(bulletin: str, bucket_years: int) -> str | None:
    """Map "YYYY-MM" → "<YYYY>-<YYYY+span-1>" floored to bucket boundary.
    Returns None if the bulletin month doesn't parse.
    """
    try:
        year = int(bulletin.split("-", 1)[0])
    except (ValueError, IndexError):
        return None
    start = (year // bucket_years) * bucket_years
    return f"{start}-{start + bucket_years - 1}"


def build_time_tree(
    catalog: list[PageCatalogRow],
    banner_rewrite: dict[str, str] | None = None,
    *,
    bucket_years: int = 5,
) -> dict[str, Any]:
    """L1 = canonical section (as in build_tree). L2 = N-year time bucket
    keyed on the source bulletin's publication year. Leaf grain stays
    `keyword → [postings]` so the existing walker (`retrieve_hierarchical`)
    works unchanged.
    """
    rewrite = banner_rewrite or {}
    tree: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}

    for r in catalog:
        if r.page_kind not in ("table", "chart", "prose"):
            continue
        raw_section = (r.section or "").strip()
        if raw_section:
            section = rewrite.get(raw_section.lower(), raw_section)
        else:
            section = "Unsectioned"
        bucket = _bucket_label(r.bulletin, bucket_years)
        if bucket is None:
            continue
        # Pages with no keywords still need to be reachable — use a
        # synthetic "_page" key keyed on (bulletin, page) so the leaf
        # flatten downstream still sees them.
        keys = [k.strip().lower() for k in r.keywords if k.strip()] or [
            f"_p{r.bulletin}_{r.page}"
        ]
        for k in keys:
            tree.setdefault(section, {}) \
                .setdefault(bucket, {}) \
                .setdefault(k, []) \
                .append({
                    "bulletin": r.bulletin,
                    "page": r.page,
                    "dates": list(r.dates),
                    "table_title": r.table_title or "",
                })

    sections_out: dict[str, Any] = {}
    for section, buckets in tree.items():
        # Sort buckets chronologically so the walker sees a tidy list.
        buckets_out: dict[str, Any] = {}
        for bucket in sorted(buckets.keys()):
            kws = buckets[bucket]
            n_pages = sum(len(v) for v in kws.values())
            buckets_out[bucket] = {
                "n_keywords": len(kws),
                "n_pages": n_pages,
                "keywords": kws,
            }
        sections_out[section] = {
            "n_clusters": len(buckets_out),
            "n_pages_in_section": sum(b["n_pages"] for b in buckets_out.values()),
            "clusters": buckets_out,
        }

    return {
        "sections": sections_out,
        "meta": {
            "mode": "time",
            "bucket_years": bucket_years,
            "n_sections": len(sections_out),
            "n_clusters": sum(len(s["clusters"]) for s in sections_out.values()),
            "banner_rewrite": dict(rewrite),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    _load_env(Path(__file__).resolve().parents[3] / ".env")

    ap = argparse.ArgumentParser(description="Build LLM-clustered concept tree from page-index catalog.")
    ap.add_argument("--catalog-dir", type=Path, default=Path("cache/page_index"))
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON path (default: <catalog-dir>/concept_tree.json)")
    ap.add_argument("--propose-batch-size", type=int, default=300)
    ap.add_argument("--assign-batch-size", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=32,
                    help="Parallelism for Step A (propose) and Step C (assign). "
                         "LLMClient handles rate limiting + retries.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-assignments", type=Path, default=None,
                    help="Path to an existing concept_tree.json. Skip the LLM "
                         "clustering steps; just re-emit the tree using the "
                         "assignments embedded in that file. Used to refresh "
                         "the leaf payload without re-paying for clustering.")
    ap.add_argument("--refresh-banners", action="store_true",
                    help="When used with --reuse-assignments, force a fresh "
                         "dedup_banners pass against the current catalog's "
                         "raw banners (instead of reusing the prior tree's "
                         "banner_rewrite). Use after a catalog rebuild that "
                         "changed the banner set, e.g. adding harvest_toc.")
    ap.add_argument("--mode", choices=("concept", "time"), default="concept",
                    help="L2 grouping. 'concept' = LLM-clustered keywords "
                         "(default). 'time' = N-year buckets of bulletin "
                         "publication year (no LLM calls, needs only the "
                         "banner_rewrite via --reuse-assignments).")
    ap.add_argument("--bucket-years", type=int, default=5,
                    help="With --mode=time, the bucket span in years.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    catalog = load_catalog(args.catalog_dir)
    if not catalog:
        print(f"No catalog rows found under {args.catalog_dir}", file=sys.stderr)
        return 2
    print(f"Loaded {len(catalog)} catalog rows from {args.catalog_dir}", flush=True)

    vocab = _collect_keyword_vocab(catalog)
    raw_banners = _collect_raw_banners(catalog)
    banner_page_counts = _collect_banner_page_counts(catalog)
    print(f"  {len(vocab)} distinct keywords, {len(raw_banners)} raw section banners",
          flush=True)

    def _load_prior_assignments(
        path: Path,
    ) -> tuple[dict[str, str], list[str], dict[str, str]]:
        """Read a prior concept_tree.json. Return
        (assignments, canonical_labels, banner_rewrite)."""
        prior = json.loads(path.read_text())
        prior_assigns: dict[str, str] = {}
        canonical: list[str] = []
        seen_canonical: set[str] = set()
        for section, sdata in prior.get("sections", {}).items():
            for cluster_label, cdata in sdata.get("clusters", {}).items():
                if cluster_label.lower() not in seen_canonical:
                    seen_canonical.add(cluster_label.lower())
                    canonical.append(cluster_label)
                for kw in cdata.get("keywords", {}):
                    prior_assigns.setdefault(kw, cluster_label)
        prior_rewrite = prior.get("meta", {}).get("banner_rewrite") or {}
        if not isinstance(prior_rewrite, dict):
            prior_rewrite = {}
        return prior_assigns, canonical, prior_rewrite

    # Time-bucket mode: skip all the LLM clustering steps. Just need a
    # banner_rewrite (loaded from --reuse-assignments or built fresh).
    if args.mode == "time":
        if args.reuse_assignments is not None:
            print(f"Reusing banner_rewrite from {args.reuse_assignments}",
                  flush=True)
            _, _, banner_rewrite = _load_prior_assignments(args.reuse_assignments)
            if not banner_rewrite or args.refresh_banners:
                cfg = SkunkConfig.from_env()
                llm = LLMClient(cfg)
                print(f"  running canonicalize_banners on {len(raw_banners)} "
                      f"raw banners (model: {cfg.llm_model})", flush=True)
                _, banner_rewrite = canonicalize_banners(
                    raw_banners, banner_page_counts, llm,
                    verbose=args.verbose,
                )
            else:
                print(f"  recovered banner_rewrite with {len(banner_rewrite)} "
                      f"entries", flush=True)
        else:
            cfg = SkunkConfig.from_env()
            llm = LLMClient(cfg)
            print(f"No --reuse-assignments; running canonicalize_banners on "
                  f"{len(raw_banners)} raw banners", flush=True)
            _, banner_rewrite = canonicalize_banners(
                raw_banners, banner_page_counts, llm, verbose=args.verbose,
            )
        tree = build_time_tree(catalog, banner_rewrite,
                               bucket_years=args.bucket_years)
        out_path = args.out or (args.catalog_dir / "time_tree.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(tree, ensure_ascii=False, indent=2))
        print(f"\nWrote time tree → {out_path}")
        print(f"  sections: {tree['meta']['n_sections']}  "
              f"buckets: {tree['meta']['n_clusters']}  "
              f"bucket_years: {tree['meta']['bucket_years']}")
        return 0

    if args.reuse_assignments is not None:
        print(f"Reusing assignments from {args.reuse_assignments}", flush=True)
        assignments, _, banner_rewrite = _load_prior_assignments(args.reuse_assignments)
        print(f"  recovered {len(assignments)} keyword → cluster_label assignments",
              flush=True)
        # If prior tree predates the v0.6 banner-dedup step, run dedup_banners
        # once so this rebuild still gets canonical sections. Otherwise reuse,
        # unless --refresh-banners forces a fresh pass against the current
        # raw banner set (used after a catalog rebuild that changed banners).
        if not banner_rewrite or args.refresh_banners:
            cfg = SkunkConfig.from_env()
            llm = LLMClient(cfg)
            reason = ("--refresh-banners" if args.refresh_banners
                      else "no banner_rewrite in prior tree")
            print(f"  {reason} — running canonicalize_banners on "
                  f"{len(raw_banners)} raw banners (model: {cfg.llm_model})",
                  flush=True)
            _, banner_rewrite = canonicalize_banners(
                raw_banners, banner_page_counts, llm, verbose=args.verbose,
            )
        else:
            print(f"  recovered banner_rewrite with {len(banner_rewrite)} entries",
                  flush=True)
        print("", flush=True)
        tree = build_tree(catalog, assignments, banner_rewrite)
        out_path = args.out or (args.catalog_dir / "concept_tree.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(tree, ensure_ascii=False, indent=2))
        print(f"\nWrote concept tree → {out_path}")
        print(f"  sections: {tree['meta']['n_sections']}  "
              f"clusters: {tree['meta']['n_clusters']}  "
              f"keywords assigned: {tree['meta']['n_keywords_assigned']}")
        return 0

    cfg = SkunkConfig.from_env()
    llm = LLMClient(cfg)
    print(f"  model: {cfg.llm_model}\n", flush=True)

    print(f"Step A — proposing cluster labels (workers={args.workers})...", flush=True)
    raw_labels = propose_labels(
        vocab, llm, batch_size=args.propose_batch_size, seed=args.seed,
        workers=args.workers, verbose=args.verbose,
    )
    print(f"  {len(raw_labels)} raw labels proposed\n", flush=True)

    print("Step B — deduping labels...", flush=True)
    canonical_labels, rewrite = dedup_labels(raw_labels, llm, verbose=args.verbose)
    print(f"  {len(canonical_labels)} canonical labels (from {len(raw_labels)} raw)\n",
          flush=True)

    print(f"Step B.5 — canonicalizing section banners ({len(raw_banners)} raw)...",
          flush=True)
    canonical_banners, banner_rewrite = canonicalize_banners(
        raw_banners, banner_page_counts, llm, verbose=args.verbose,
    )
    print(f"  {len(canonical_banners)} canonical banners "
          f"(from {len(raw_banners)} raw)\n", flush=True)

    print(f"Step C — assigning keywords to cluster_label (workers={args.workers})...",
          flush=True)
    assignments = assign_keywords(
        vocab, canonical_labels, llm,
        batch_size=args.assign_batch_size, workers=args.workers,
        verbose=args.verbose,
    )
    coverage = len(assignments) / max(1, len(vocab))
    print(f"  {len(assignments)}/{len(vocab)} keywords assigned ({coverage:.1%} coverage)\n",
          flush=True)

    print("Step D — assembling tree...", flush=True)
    tree = build_tree(catalog, assignments, banner_rewrite)

    out_path = args.out or (args.catalog_dir / "concept_tree.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tree, ensure_ascii=False, indent=2))

    print(f"\nWrote concept tree → {out_path}")
    print(f"  sections: {tree['meta']['n_sections']}  "
          f"clusters: {tree['meta']['n_clusters']}  "
          f"keywords assigned: {tree['meta']['n_keywords_assigned']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
