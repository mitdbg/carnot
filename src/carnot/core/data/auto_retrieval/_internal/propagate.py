from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, Iterable, List, Set, Tuple

from _internal.type_utils import is_str_type

logger = logging.getLogger(__name__)


def _parse_key(key: str) -> Tuple[str, str] | None:
    if ":" not in key:
        return None
    parts = [p.strip() for p in key.split(":")]
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    subject = parts[0].casefold()
    facet = ":".join(parts[1:]).casefold()
    return subject, facet


def _normalize_hierarchy(
    concept_hierarchy: Dict[str, List[str]],
) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for child, parents in concept_hierarchy.items():
        c = str(child).strip().casefold()
        if not c:
            continue
        out.setdefault(c, set())
        for p in parents or []:
            parent = str(p).strip().casefold()
            if parent and parent != c:
                out[c].add(parent)
                out.setdefault(parent, set())
    return out


def _ancestor_subjects(subject: str, edges: Dict[str, Set[str]]) -> Set[str]:
    if subject not in edges:
        return set()
    seen: Set[str] = set()
    queue: deque[str] = deque([subject])
    while queue:
        cur = queue.popleft()
        for parent in edges.get(cur, set()):
            if parent in seen:
                continue
            seen.add(parent)
            queue.append(parent)
    return seen


def _iter_values(v: Any) -> Iterable[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x is not None]
    return [str(v)]


def _merge_values(existing: Any, incoming: Any) -> Any:
    existing_vals = [x for x in _iter_values(existing) if x]
    incoming_vals = [x for x in _iter_values(incoming) if x]
    if not existing_vals and not incoming_vals:
        return None
    if not existing_vals:
        return incoming_vals if isinstance(incoming, list) else incoming_vals[0]
    if not incoming_vals:
        return existing

    merged: List[str] = []
    seen: Set[str] = set()
    for item in existing_vals + incoming_vals:
        norm = item.casefold()
        if norm in seen:
            continue
        seen.add(norm)
        merged.append(item)

    if isinstance(existing, list) or isinstance(incoming, list):
        return merged
    if len(merged) == 1:
        return merged[0]
    return merged


def propagate_across_concepts(
    results: Dict[str, Dict[str, Any]],
    concept_schema_cols: List[Dict[str, Any]],
    concept_hierarchy: Dict[str, List[str]],
    *,
    drop_source_keys: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Propagate values child->parent across concepts sharing the same facet.

    Example:
      bird:location -> animal:location
    when concept_hierarchy has bird -> animal and both keys exist in schema.
    """
    if not results or not concept_hierarchy:
        return results

    string_keys: Set[str] = {
        col["name"] for col in concept_schema_cols if is_str_type(col["type"])
    }
    if not string_keys:
        return results

    key_parts: Dict[str, Tuple[str, str]] = {}
    for key in string_keys:
        parsed = _parse_key(key)
        if parsed is not None:
            key_parts[key] = parsed
    if not key_parts:
        return results

    key_by_subject_and_facet: Dict[Tuple[str, str], str] = {
        v: k for k, v in key_parts.items()
    }
    hierarchy_edges = _normalize_hierarchy(concept_hierarchy)
    if not hierarchy_edges:
        return results

    propagation_pairs_set: Set[Tuple[str, str]] = set()
    for src_key, (src_subject, src_facet) in key_parts.items():
        for parent_subject in _ancestor_subjects(src_subject, hierarchy_edges):
            tgt_key = key_by_subject_and_facet.get((parent_subject, src_facet))
            if tgt_key and tgt_key != src_key:
                propagation_pairs_set.add((src_key, tgt_key))
    propagation_pairs = sorted(propagation_pairs_set)

    if not propagation_pairs:
        logger.info("Cross-concept propagation: no valid propagation pairs found")
        return results

    source_keys_to_drop: Set[str] = {src for src, _ in propagation_pairs} if drop_source_keys else set()

    out: Dict[str, Dict[str, Any]] = {}
    total_propagated = 0
    for doc_id, doc_vals in results.items():
        new_doc = dict(doc_vals)
        for src_key, tgt_key in propagation_pairs:
            if src_key not in doc_vals:
                continue
            before = new_doc.get(tgt_key)
            after = _merge_values(before, doc_vals[src_key])
            if after is None:
                continue
            if before != after:
                total_propagated += 1
            new_doc[tgt_key] = after
        if source_keys_to_drop:
            for src_key in source_keys_to_drop:
                new_doc.pop(src_key, None)
        out[doc_id] = new_doc

    logger.info(
        "Cross-concept propagation: %d source->target pairs, %d updates, %d source keys dropped",
        len(propagation_pairs), total_propagated, len(source_keys_to_drop),
    )
    return out


def infer_source_keys_to_drop(
    concept_schema_cols: List[Dict[str, Any]],
    concept_hierarchy: Dict[str, List[str]],
) -> Set[str]:
    """Infer child keys that should be dropped after child->parent propagation."""
    if not concept_hierarchy:
        return set()
    string_keys: Set[str] = {
        col["name"] for col in concept_schema_cols if is_str_type(col["type"])
    }
    if not string_keys:
        return set()

    key_parts: Dict[str, Tuple[str, str]] = {}
    for key in string_keys:
        parsed = _parse_key(key)
        if parsed is not None:
            key_parts[key] = parsed
    if not key_parts:
        return set()

    key_by_subject_and_facet: Dict[Tuple[str, str], str] = {
        v: k for k, v in key_parts.items()
    }
    hierarchy_edges = _normalize_hierarchy(concept_hierarchy)
    if not hierarchy_edges:
        return set()

    source_keys: Set[str] = set()
    for src_key, (src_subject, src_facet) in key_parts.items():
        for parent_subject in _ancestor_subjects(src_subject, hierarchy_edges):
            tgt_key = key_by_subject_and_facet.get((parent_subject, src_facet))
            if tgt_key and tgt_key != src_key:
                source_keys.add(src_key)
                break
    return source_keys
