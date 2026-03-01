from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from _internal.ontology import Ontology
from _internal.type_utils import is_str_type

logger = logging.getLogger(__name__)


def _collect_unique_values(
    results: Dict[str, Dict[str, Any]],
    active_keys: Set[str],
) -> Dict[str, Set[str]]:
    """Single pass over all documents to collect unique string values per key."""
    unique: Dict[str, Set[str]] = {k: set() for k in active_keys}
    for doc_vals in results.values():
        for key, val in doc_vals.items():
            if key not in active_keys:
                continue
            if isinstance(val, list):
                for v in val:
                    if v is not None:
                        unique[key].add(str(v))
            elif val is not None:
                unique[key].add(str(val))
    return unique


def canonicalize_results(
    results: Dict[str, Dict[str, Any]],
    concept_schema_cols: List[Dict[str, Any]],
    ontologies: Dict[str, Ontology],
) -> Dict[str, Dict[str, Any]]:
    """Resolve normalized strings to canonical identifiers (e.g. Wikidata QIDs).

    Uses a two-phase approach:
      Phase 1 – collect unique values per ontology key (one pass over docs).
      Phase 2 – resolve each unique value exactly once (cache + API).
      Phase 3 – apply the lookup table to every document (pure dict lookups).

    This avoids O(docs * values_per_doc) redundant cache reads when many
    documents share the same values.
    """
    str_keys: Set[str] = set()
    for col in concept_schema_cols:
        if is_str_type(col["type"]):
            str_keys.add(col["name"])

    active_keys = str_keys & set(ontologies.keys())
    if not active_keys:
        return results

    # --- Phase 1: collect unique values per key ---
    unique_per_key = _collect_unique_values(results, active_keys)

    # --- Phase 2: resolve each unique value once -> build lookup ---
    # Uses resolve_batch when available to collapse N per-value SPARQL calls
    # into 1 batched call per key.  This is the critical performance fix —
    # without it, canonicalization makes hundreds of individual SPARQL queries.
    resolved_lookup: Dict[str, Dict[str, Optional[str]]] = {}
    resolved_count = 0
    unresolved_count = 0
    for key in sorted(active_keys):
        ont = ontologies[key]
        values = sorted(unique_per_key[key])
        if hasattr(ont, "resolve_batch"):
            lookup = ont.resolve_batch(values)
        else:
            lookup = {sv: ont.resolve(sv) for sv in values}
        resolved_count += sum(1 for v in lookup.values() if v is not None)
        unresolved_count += sum(1 for v in lookup.values() if v is None)
        resolved_lookup[key] = lookup

    # --- Phase 3: apply lookup to every document ---
    out: Dict[str, Dict[str, Any]] = {}
    for doc_id, doc_vals in results.items():
        new_doc: Dict[str, Any] = {}
        for key, val in doc_vals.items():
            if key not in active_keys:
                new_doc[key] = val
                continue
            lookup = resolved_lookup[key]
            if isinstance(val, list):
                canon: List[str] = []
                seen: Set[str] = set()
                for v in val:
                    if v is None:
                        continue
                    sv = str(v)
                    c = lookup.get(sv)
                    c = c if c is not None else sv   # keep original when unresolved
                    if c not in seen:
                        seen.add(c)
                        canon.append(c)
                if canon:
                    new_doc[key] = canon
            elif val is not None:
                sv = str(val)
                c = lookup.get(sv)
                new_doc[key] = c if c is not None else sv
        out[doc_id] = new_doc

    logger.info(
        "Canonicalization: %d resolved, %d unresolved across %d keys (%d unique values)",
        resolved_count, unresolved_count, len(active_keys),
        sum(len(v) for v in unique_per_key.values()),
    )
    return out
