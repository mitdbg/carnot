from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any, Dict, List, Set

from _internal.ontology import Ontology
from _internal.type_utils import is_str_type

logger = logging.getLogger(__name__)

DEFAULT_MAX_DEPTH = 5
_QID_RE = re.compile(r"^Q[1-9]\d*$")
_COMPOUND_SPLIT_RE = re.compile(r"\s*(?:,|/|;|\||&|\+|\band\b|\bor\b)\s*", flags=re.IGNORECASE)
_PARENT_PROPERTY_CANDIDATES = ("P131", "P279", "P171", "P361")
_PROPERTY_SUPPORT_MIN_RATIO = 0.20
_PROPERTY_SUPPORT_MIN_HITS = 2
_PROPERTY_SUPPORT_SAMPLE = 24


def _collect_global_value_set(
    results: Dict[str, Dict[str, Any]],
    key: str,
) -> Set[str]:
    """Collect every unique string value for *key* across all documents."""
    vals: Set[str] = set()
    for doc_vals in results.values():
        v = doc_vals.get(key)
        if v is None:
            continue
        if isinstance(v, list):
            for item in v:
                if item is not None:
                    vals.add(str(item))
        else:
            vals.add(str(v))
    return vals


def _expand_value(
    value: str,
    ontology: Ontology,
    global_set: Set[str],
    max_depth: int,
) -> Set[str]:
    """Walk up the hierarchy from *value*, collecting ancestors in *global_set*."""
    added: Set[str] = set()
    visited: Set[str] = {value}
    frontier: deque[tuple[str, int]] = deque([(value, 0)])

    while frontier:
        current, depth = frontier.popleft()
        if depth >= max_depth:
            continue
        parents = ontology.get_parents(current)
        if not parents:
            continue
        for parent in parents:
            if parent in visited:
                continue
            visited.add(parent)
            if parent in global_set:
                added.add(parent)
            # Continue traversal even when parent is not in global_set;
            # a higher ancestor might still be in global_set.
            frontier.append((parent, depth + 1))

    return added


def _expand_compound_literal(
    value: str,
    global_set: Set[str],
) -> Set[str]:
    """Expand composite free-text values into known global values.

    Example: if value is "crime and action" and global_set contains both
    "crime" and "action", return {"crime", "action"}.
    """
    if not value or _QID_RE.match(value):
        return set()
    parts = [p.strip() for p in _COMPOUND_SPLIT_RE.split(value) if p.strip()]
    if len(parts) < 2:
        return set()
    return {p for p in parts if p in global_set}


def _discover_runtime_parent_properties(
    ontology: Ontology,
    global_set: Set[str],
) -> List[str]:
    """Discover useful parent properties from observed values for this key.

    Uses only QIDs present in the key's global set, so behavior adapts to
    column semantics without hardcoding concept names.
    """
    if not hasattr(ontology, "get_parents_for_property"):
        return []
    qids = [v for v in sorted(global_set) if _QID_RE.match(v)]
    if not qids:
        return []
    sample_qids = qids[:_PROPERTY_SUPPORT_SAMPLE]
    selected: List[str] = []
    for prop in _PARENT_PROPERTY_CANDIDATES:
        non_empty = 0
        for qid in sample_qids:
            parents = ontology.get_parents_for_property(qid, prop)
            if parents:
                non_empty += 1
        ratio = non_empty / max(1, len(sample_qids))
        if non_empty >= _PROPERTY_SUPPORT_MIN_HITS and ratio >= _PROPERTY_SUPPORT_MIN_RATIO:
            selected.append(prop)
    return selected


def materialize_closure(
    results: Dict[str, Dict[str, Any]],
    concept_schema_cols: List[Dict[str, Any]],
    ontologies: Dict[str, Ontology],
    max_depth: int = DEFAULT_MAX_DEPTH,
    *,
    warmup: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Expand each value with ancestor identifiers from the ontology hierarchy.

    When *warmup* is True (default), batch-fetches parent hierarchies for all
    QIDs before the per-document expansion.  This replaces the separate
    ``_warmup_parent_caches`` call and avoids a redundant pass over all docs.
    """
    str_keys: Set[str] = set()
    for col in concept_schema_cols:
        if is_str_type(col["type"]):
            str_keys.add(col["name"])

    active_keys = str_keys & set(ontologies.keys())
    if not active_keys:
        return results

    # Collect every unique value per key (single pass over docs).
    global_sets: Dict[str, Set[str]] = {}
    for key in active_keys:
        global_sets[key] = _collect_global_value_set(results, key)
        ont = ontologies[key]
        runtime_props = _discover_runtime_parent_properties(ont, global_sets[key])
        if runtime_props and hasattr(ont, "set_runtime_parent_properties"):
            ont.set_runtime_parent_properties(runtime_props)
            logger.info(
                "Materialization runtime parent properties for %s: %s",
                key, runtime_props,
            )

    # Warmup: batch-prefetch parent hierarchies so _expand_value hits cache.
    # This used to be a separate _warmup_parent_caches() pass over all docs.
    if warmup:
        for key in sorted(active_keys):
            ont = ontologies[key]
            if hasattr(ont, "warmup_parents"):
                qids = sorted(global_sets[key])
                if qids:
                    ont.warmup_parents(qids)

    # Per-document expansion using (now-cached) parent lookups.
    out: Dict[str, Dict[str, Any]] = {}
    total_added = 0

    for doc_id, doc_vals in results.items():
        new_doc: Dict[str, Any] = {}
        for key, val in doc_vals.items():
            if key not in active_keys:
                new_doc[key] = val
                continue
            ont = ontologies[key]
            gset = global_sets[key]

            if isinstance(val, list):
                existing: Set[str] = set()
                for v in val:
                    if v is not None:
                        existing.add(str(v))
                added: Set[str] = set()
                for v in existing:
                    added |= _expand_value(v, ont, gset, max_depth)
                    added |= _expand_compound_literal(v, gset)
                invalid_added = added - gset
                if invalid_added:
                    raise RuntimeError(
                        f"Materialization invariant violation for key {key!r}: "
                        f"added values outside global set: {sorted(invalid_added)[:10]}"
                    )
                added -= existing
                if added:
                    total_added += len(added)
                new_doc[key] = sorted(existing | added)
            elif val is not None:
                sv = str(val)
                added = _expand_value(sv, ont, gset, max_depth) | _expand_compound_literal(sv, gset)
                invalid_added = added - gset
                if invalid_added:
                    raise RuntimeError(
                        f"Materialization invariant violation for key {key!r}: "
                        f"added values outside global set: {sorted(invalid_added)[:10]}"
                    )
                added.discard(sv)
                if added:
                    total_added += len(added)
                    new_doc[key] = sorted({sv} | added)
                else:
                    new_doc[key] = val
        out[doc_id] = new_doc

    logger.info(
        "Materialization: %d ancestor values added across %d keys",
        total_added, len(active_keys),
    )
    return out
