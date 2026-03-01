from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

from _internal.normalizer import normalize_value
from _internal.type_utils import is_str_type
from _internal.wikidata_api import WikidataAPI
from _internal.wikidata_cache import WikidataCache

logger = logging.getLogger(__name__)

MAX_PARENTS_PER_ENTITY = 15
_TIMEZONE_LABEL_RE = re.compile(r"^(?:UTC|GMT)[+-]\d{1,2}(?::\d{2})?$", re.IGNORECASE)


class WikidataOntology:
    def __init__(
        self,
        concept_key: str,
        *,
        cache: WikidataCache,
        api: WikidataAPI,
        type_constraints: Optional[List[str]] = None,
        parents_properties: Optional[List[str]] = None,
        max_depth: int = 0,
        resolve_enabled: bool = True,
        materialize_enabled: bool = False,
    ) -> None:
        self._concept_key = concept_key
        self._cache = cache
        self._api = api
        self._type_constraints = list(type_constraints or [])
        self._parents_properties = list(parents_properties or [])
        self._runtime_parent_properties: Optional[List[str]] = None
        self._max_depth = max_depth
        self._resolve_enabled = resolve_enabled
        self._materialize_enabled = materialize_enabled and bool(self._parents_properties) and self._max_depth > 0

    @property
    def resolve_enabled(self) -> bool:
        return self._resolve_enabled

    @property
    def materialize_enabled(self) -> bool:
        return self._materialize_enabled

    def set_runtime_parent_properties(self, properties: List[str]) -> None:
        props = [str(p).strip() for p in properties if str(p).strip()]
        self._runtime_parent_properties = props if props else None

    def _effective_parent_properties(self) -> List[str]:
        if self._runtime_parent_properties:
            return list(self._runtime_parent_properties)
        return list(self._parents_properties)

    def get_parents_for_property(self, canonical: str, property_id: str) -> list[str]:
        if not self._materialize_enabled:
            return []
        qid = canonical
        cached = self._cache.get_parents(
            self._concept_key, qid, property_id, self._max_depth,
        )
        if cached is not None:
            return [p[0] for p in cached]
        try:
            batch = self._api.sparql_parents_batch(
                [qid], property_id, max_depth=self._max_depth,
            )
        except Exception:
            logger.exception("sparql_parents_batch failed for %s/%s", qid, property_id)
            self._cache.put_parents(
                self._concept_key, qid, property_id, self._max_depth, [],
            )
            return []
        parents = batch.get(qid, [])[:MAX_PARENTS_PER_ENTITY]
        self._cache.put_parents(
            self._concept_key, qid, property_id, self._max_depth, parents,
        )
        return [p[0] for p in parents]

    def resolve(self, value: str) -> str | None:
        """Resolve a single value. Prefer :meth:`resolve_batch` for many values."""
        if not self._resolve_enabled:
            return None
        normalized = normalize_value(value)
        if not normalized:
            return None
        cached = self._cache.get_resolved(self._concept_key, normalized)
        if cached is not None:
            qid, label, _desc = cached
            if not self._should_ignore_cached_resolution(normalized, qid, label):
                return qid if qid else None
        try:
            candidates = self._api.search_entities(value, limit=5)
        except Exception:
            logger.exception("search_entities failed for %r", value)
            return None
        if not candidates:
            self._cache.put_resolved(self._concept_key, normalized, "", "", "")
            return None
        if self._type_constraints:
            candidate_qids = [c["qid"] for c in candidates]
            try:
                valid = self._api.validate_types_batch(
                    candidate_qids, self._type_constraints,
                )
            except Exception:
                logger.exception("validate_types_batch failed")
                valid = set(candidate_qids)
            candidates = [c for c in candidates if c["qid"] in valid]
        candidates = self._filter_ambiguous_candidates(normalized, candidates)
        if not candidates:
            self._cache.put_resolved(self._concept_key, normalized, "", "", "")
            return None
        chosen = self._rank(normalized, candidates)
        if chosen is None:
            self._cache.put_resolved(self._concept_key, normalized, "", "", "")
            return None
        self._cache.put_resolved(
            self._concept_key, normalized,
            chosen["qid"], chosen["label"], chosen.get("description", ""),
        )
        return chosen["qid"]

    def resolve_batch(self, values: List[str]) -> Dict[str, Optional[str]]:
        """Resolve many values with ONE batched SPARQL type-validation call.

        Flow:
          1. Normalize + check cache → split into cached / uncached.
          2. Search each uncached value → collect ALL candidate QIDs.
          3. ONE ``validate_types_batch`` call for all candidates at once.
          4. Rank, pick best, cache each result.

        The old per-value ``resolve()`` did step 3 per-value, causing N SPARQL
        calls.  This method does it once, cutting WDQS load by ~100x.
        """
        results: Dict[str, Optional[str]] = {}
        if not self._resolve_enabled:
            return {v: None for v in values}

        # -- Step 1: normalize + cache check --
        uncached: List[tuple[str, str]] = []   # (original_value, normalized)
        for v in values:
            norm = normalize_value(v)
            if not norm:
                results[v] = None
                continue
            cached = self._cache.get_resolved(self._concept_key, norm)
            if cached is not None:
                qid, label, _desc = cached
                if self._should_ignore_cached_resolution(norm, qid, label):
                    uncached.append((v, norm))
                else:
                    results[v] = qid if qid else None
            else:
                uncached.append((v, norm))

        if not uncached:
            return results

        n_cached = len(values) - len(uncached)
        logger.info("[%s] resolve_batch: %d cached, %d to search",
                    self._concept_key, n_cached, len(uncached))

        # -- Step 2: search each uncached value --
        search_results: Dict[tuple[str, str], List[Dict[str, str]]] = {}
        all_candidate_qids: Set[str] = set()
        for i, (v, norm) in enumerate(uncached):
            if i > 0 and i % 50 == 0:
                logger.info("[%s]   searched %d/%d ...",
                            self._concept_key, i, len(uncached))
            try:
                candidates = self._api.search_entities(v, limit=5)
            except Exception:
                logger.debug("search_entities failed for %r", v)
                candidates = []
            search_results[(v, norm)] = candidates
            for c in candidates:
                all_candidate_qids.add(c["qid"])

        # -- Step 3: ONE batched SPARQL for type validation --
        valid_qids: Set[str] = all_candidate_qids  # default: accept all
        if self._type_constraints and all_candidate_qids:
            logger.info("[%s]   validating %d candidate QIDs against %d type constraints",
                        self._concept_key, len(all_candidate_qids), len(self._type_constraints))
            try:
                valid_qids = self._api.validate_types_batch(
                    sorted(all_candidate_qids), self._type_constraints,
                )
            except Exception:
                logger.exception("[%s] validate_types_batch failed — accepting all candidates",
                                 self._concept_key)

        # -- Step 4: rank, pick best, cache --
        for (v, norm), candidates in search_results.items():
            if not candidates:
                self._cache.put_resolved(self._concept_key, norm, "", "", "")
                results[v] = None
                continue

            if self._type_constraints:
                candidates = [c for c in candidates if c["qid"] in valid_qids]
            candidates = self._filter_ambiguous_candidates(norm, candidates)

            if not candidates:
                self._cache.put_resolved(self._concept_key, norm, "", "", "")
                results[v] = None
                continue

            chosen = self._rank(norm, candidates)
            if chosen is None:
                self._cache.put_resolved(self._concept_key, norm, "", "", "")
                results[v] = None
            else:
                self._cache.put_resolved(
                    self._concept_key, norm,
                    chosen["qid"], chosen["label"], chosen.get("description", ""),
                )
                results[v] = chosen["qid"]

        n_resolved = sum(1 for v in results.values() if v is not None)
        logger.info("[%s]   done: %d resolved, %d unresolved",
                    self._concept_key, n_resolved, len(values) - n_resolved)
        return results

    @staticmethod
    def _looks_like_timezone_text(text: str) -> bool:
        return bool(_TIMEZONE_LABEL_RE.match(str(text).strip()))

    @classmethod
    def _filter_ambiguous_candidates(
        cls,
        normalized: str,
        candidates: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Drop timezone-like labels when the query itself is not timezone-like.

        This prevents cases like "indochina" being resolved to "UTC+07:00".
        """
        if cls._looks_like_timezone_text(normalized):
            return candidates
        filtered = [
            c for c in candidates
            if not cls._looks_like_timezone_text(c.get("label", ""))
        ]
        return filtered

    @classmethod
    def _should_ignore_cached_resolution(
        cls,
        normalized: str,
        qid: str,
        label: str,
    ) -> bool:
        if not qid:
            return False
        return (not cls._looks_like_timezone_text(normalized)) and cls._looks_like_timezone_text(label)

    @staticmethod
    def _rank(
        normalized: str, candidates: List[Dict[str, str]],
    ) -> Optional[Dict[str, str]]:
        for c in candidates:
            if c["label"].lower() == normalized:
                return c
        return candidates[0] if candidates else None

    def get_parents(self, canonical: str) -> list[str]:
        if not self._materialize_enabled:
            return []
        all_parents: list[str] = []
        for prop in self._effective_parent_properties():
            all_parents.extend(self.get_parents_for_property(canonical, prop))
        return all_parents

    def label_for(self, canonical: str) -> str | None:
        return self._cache.get_label(canonical)

    def warmup_parents(self, qids: List[str]) -> None:
        if not self._materialize_enabled:
            return
        for prop in self._effective_parent_properties():
            uncached = [
                q for q in qids
                if self._cache.get_parents(
                    self._concept_key, q, prop, self._max_depth,
                ) is None
            ]
            if not uncached:
                continue
            logger.info("[%s] parents warmup: fetching %d/%d for %s",
                        self._concept_key, len(uncached), len(qids), prop)
            try:
                batch = self._api.sparql_parents_batch(
                    uncached, prop, max_depth=self._max_depth,
                )
            except Exception:
                logger.exception("[%s] warmup SPARQL failed for %s",
                                 self._concept_key, prop)
                for q in uncached:
                    self._cache.put_parents(
                        self._concept_key, q, prop, self._max_depth, [],
                    )
                continue
            for q in uncached:
                parents = batch.get(q, [])[:MAX_PARENTS_PER_ENTITY]
                self._cache.put_parents(
                    self._concept_key, q, prop, self._max_depth, parents,
                )


def _collect_qids(
    results: Dict[str, Dict[str, Any]],
    concept_schema_cols: List[Dict[str, Any]],
    active_keys: Set[str],
) -> Set[str]:
    qids: Set[str] = set()
    for doc_vals in results.values():
        for key, val in doc_vals.items():
            if key not in active_keys:
                continue
            if isinstance(val, list):
                for v in val:
                    if v is not None:
                        qids.add(str(v))
            elif val is not None:
                qids.add(str(val))
    return qids


def populate_labels(
    ontologies: Dict[str, "WikidataOntology"],
    qids: Set[str],
) -> None:
    if not ontologies or not qids:
        return
    ont = next(iter(ontologies.values()))
    cache = ont._cache
    api = ont._api

    existing = cache.get_labels_batch(list(qids))
    missing = qids - set(existing.keys())
    if not missing:
        return

    cached_labels = cache.collect_all_cached_labels()
    found: List[tuple] = []
    for qid in list(missing):
        lbl = cached_labels.get(qid)
        if lbl:
            found.append((qid, lbl))
            missing.discard(qid)
    if found:
        cache.put_labels_batch(found)
    if not missing:
        return

    logger.info("Fetching %d missing labels from Wikidata API", len(missing))
    try:
        fetched = api.fetch_labels_batch(list(missing))
    except Exception:
        logger.exception("fetch_labels_batch failed")
        return
    if fetched:
        cache.put_labels_batch(list(fetched.items()))


def resolve_to_labels(
    results: Dict[str, Dict[str, Any]],
    concept_schema_cols: List[Dict[str, Any]],
    ontologies: Dict[str, WikidataOntology],
) -> Dict[str, Dict[str, Any]]:
    """Replace QIDs with human-readable labels.

    Uses a single batch SQL read to build a {qid -> label} dict, then
    applies it via pure dict lookups (no per-value SQL queries).
    """
    active_keys: Set[str] = set()
    for col in concept_schema_cols:
        if is_str_type(col["type"]) and col["name"] in ontologies:
            active_keys.add(col["name"])
    if not active_keys:
        return results

    # Collect every QID, populate labels table, then batch-read into a dict.
    all_qids = _collect_qids(results, concept_schema_cols, active_keys)
    populate_labels(ontologies, all_qids)

    ont_sample = next(iter(ontologies.values()))
    cache = ont_sample._cache
    # One batch SQL query instead of O(total_values) individual get_label calls.
    label_map: Dict[str, str] = cache.get_labels_batch(list(all_qids))

    out: Dict[str, Dict[str, Any]] = {}
    resolved = 0
    passthrough = 0
    for doc_id, doc_vals in results.items():
        new_doc: Dict[str, Any] = {}
        for key, val in doc_vals.items():
            if key not in active_keys:
                new_doc[key] = val
                continue
            if isinstance(val, list):
                labels: List[str] = []
                seen: Set[str] = set()
                for v in val:
                    if v is None:
                        continue
                    sv = str(v)
                    lbl = label_map.get(sv)           # dict lookup, not SQL
                    display = lbl if lbl else sv
                    low = display.lower()
                    if low not in seen:
                        seen.add(low)
                        labels.append(display)
                        resolved += 1 if lbl else 0
                        passthrough += 0 if lbl else 1
                if labels:
                    new_doc[key] = labels
            elif val is not None:
                sv = str(val)
                lbl = label_map.get(sv)               # dict lookup, not SQL
                if lbl:
                    resolved += 1
                    new_doc[key] = lbl
                else:
                    passthrough += 1
                    new_doc[key] = sv
        out[doc_id] = new_doc

    logger.info("resolve_to_labels: %d resolved, %d passthrough across %d keys",
                resolved, passthrough, len(active_keys))
    return out
