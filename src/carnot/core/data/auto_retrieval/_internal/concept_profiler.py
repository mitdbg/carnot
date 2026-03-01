from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from _internal.normalizer import normalize_value
from _internal.type_utils import is_str_type
from _internal.wikidata_api import WikidataAPI
from _internal.wikidata_cache import WikidataCache

logger = logging.getLogger(__name__)

PROPERTY_CANDIDATES = ("P131", "P279", "P171", "P361")
PROP_DEFAULT_DEPTH = {"P131": 4, "P171": 4, "P279": 3, "P361": 2}
MAX_SAMPLES_PER_KEY = 40
MAX_TYPE_CONSTRAINTS = 4
P31_SUPPORT_MIN_RATIO = 0.25
P31_SUPPORT_MIN_COUNT = 2
SEARCH_LIMIT = 15
MAX_SCOPES = 3
AMBIGUITY_MARGIN = 0.03
PROFILE_PIPELINE_VERSION = "concept-key-class-first-v3"

YEAR_RE = re.compile(r"^(?:\d{4}|(?:early|late)\s+\d{4}s|\d{3,4}s)$", flags=re.IGNORECASE)
_QUERY_SPLIT_RE = re.compile(r"[\s:_\-/]+")
_NON_WORD_RE = re.compile(r"[^a-z0-9 ]+")
_STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "with", "by",
}


@dataclass
class InferredConceptProfile:
    concept_key: str
    resolve_enabled: bool
    type_constraints: List[str] = field(default_factory=list)
    parents_properties: List[str] = field(default_factory=list)
    max_depth: int = 0
    confidence: float = 0.0
    sample_count: int = 0
    resolved_count: int = 0
    evidence: Dict[str, Any] = field(default_factory=dict)


def profile_to_json(profile: InferredConceptProfile) -> str:
    return json.dumps(asdict(profile), sort_keys=True)


def profile_from_json(payload: str) -> InferredConceptProfile:
    raw = json.loads(payload)
    return InferredConceptProfile(
        concept_key=raw["concept_key"],
        resolve_enabled=bool(raw.get("resolve_enabled", False)),
        type_constraints=list(raw.get("type_constraints") or []),
        parents_properties=list(raw.get("parents_properties") or []),
        max_depth=int(raw.get("max_depth", 0)),
        confidence=float(raw.get("confidence", 0.0)),
        sample_count=int(raw.get("sample_count", 0)),
        resolved_count=int(raw.get("resolved_count", 0)),
        evidence=dict(raw.get("evidence") or {}),
    )


def _profile_cache_key_from_schema(
    concept_schema_cols: List[Dict[str, Any]],
) -> str:
    """Cache key for concept-key-driven profiling (schema-only, no data samples)."""
    h = hashlib.sha1()
    for col in sorted(concept_schema_cols, key=lambda c: c["name"]):
        h.update(col["name"].encode("utf-8"))
        h.update(str(col.get("desc", "")).encode("utf-8"))
    h.update(PROFILE_PIPELINE_VERSION.encode("utf-8"))
    return h.hexdigest()


def _concept_query_variants(concept_key: str) -> List[str]:
    """Generate minimal robust query variants without schema-shape assumptions."""
    raw = " ".join(str(concept_key).strip().split())
    if not raw:
        return []
    variants: List[str] = []
    seen: Set[str] = set()

    def _add(text: str) -> None:
        t = " ".join(text.strip().split())
        if not t:
            return
        n = normalize_value(t)
        if not n or n in seen:
            return
        seen.add(n)
        variants.append(t)

    _add(raw)
    normalized_punct = re.sub(r"[:/_-]+", " ", raw)
    _add(normalized_punct)
    ascii_clean = _NON_WORD_RE.sub(" ", normalize_value(raw))
    _add(ascii_clean)

    tokens = [t for t in _QUERY_SPLIT_RE.split(normalize_value(raw)) if t and t not in _STOPWORDS]
    if tokens:
        _add(" ".join(tokens))
        if len(tokens) >= 2:
            _add(" ".join(tokens[-2:]))
            _add(" ".join(tokens[:2]))
        if len(tokens) >= 3:
            _add(" ".join(tokens[-3:]))
            _add(" ".join(tokens[:3]))
        for tok in tokens:
            _add(tok)
    return variants[:12]


def _tokenize(text: str) -> Set[str]:
    toks = []
    for t in _QUERY_SPLIT_RE.split(normalize_value(text)):
        t = _NON_WORD_RE.sub("", t).strip()
        if not t or t in _STOPWORDS:
            continue
        toks.append(t)
    return set(toks)


def _is_year_like_ratio(samples: List[str]) -> float:
    if not samples:
        return 0.0
    count = sum(1 for s in samples if YEAR_RE.match(s.strip()))
    return count / len(samples)


def _collect_scope_candidates(
    concept_key: str,
    api: WikidataAPI,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    variants = _concept_query_variants(concept_key)
    candidates_by_qid: Dict[str, Dict[str, Any]] = {}
    for query in variants:
        try:
            candidates = api.search_entities(query, limit=SEARCH_LIMIT)
        except Exception:
            logger.debug("search_entities failed during concept profiling for %r", query)
            candidates = []
        for rank, c in enumerate(candidates):
            qid = str(c.get("qid", "")).strip()
            if not qid:
                continue
            entry = candidates_by_qid.setdefault(
                qid,
                {
                    "qid": qid,
                    "label": c.get("label", ""),
                    "description": c.get("description", ""),
                    "best_rank": rank,
                    "hits": 0,
                },
            )
            entry["hits"] += 1
            entry["best_rank"] = min(entry["best_rank"], rank)
            if not entry["label"] and c.get("label"):
                entry["label"] = c["label"]
            if not entry["description"] and c.get("description"):
                entry["description"] = c["description"]
    return variants, candidates_by_qid


def _score_candidate(
    concept_tokens: Set[str],
    cand: Dict[str, Any],
    signals: Dict[str, int],
    p279_parents: List[Tuple[str, str]],
) -> Dict[str, Any]:
    label = str(cand.get("label", "")).strip()
    description = str(cand.get("description", "")).strip()
    best_rank = int(cand.get("best_rank", SEARCH_LIMIT))
    hits = int(cand.get("hits", 1))

    label_tokens = _tokenize(label)
    desc_tokens = _tokenize(description)
    parent_label_tokens: Set[str] = set()
    for _qid, parent_label in p279_parents:
        parent_label_tokens |= _tokenize(parent_label)

    p279_count = int(signals.get("p279_count", 0))
    p31_count = int(signals.get("p31_count", 0))

    rank_score = max(0.0, 1.0 - (best_rank / max(1.0, SEARCH_LIMIT - 1)))
    hit_score = min(1.0, hits / 3.0)
    token_overlap = 0.0
    if concept_tokens:
        overlap_tokens = concept_tokens & (label_tokens | desc_tokens | parent_label_tokens)
        token_overlap = len(overlap_tokens) / len(concept_tokens)
    class_likeness = p279_count / max(1.0, (p279_count + p31_count))
    instance_penalty = 1.0 if (p279_count == 0 and p31_count > 0) else 0.0

    final_score = (
        0.30 * rank_score +
        0.15 * hit_score +
        0.35 * class_likeness +
        0.25 * token_overlap -
        0.20 * instance_penalty
    )
    return {
        "qid": cand["qid"],
        "label": label,
        "description": description,
        "score": round(final_score, 4),
        "components": {
            "rank_score": round(rank_score, 4),
            "hit_score": round(hit_score, 4),
            "class_likeness": round(class_likeness, 4),
            "token_overlap": round(token_overlap, 4),
            "instance_penalty": round(instance_penalty, 4),
            "p279_count": p279_count,
            "p31_count": p31_count,
        },
    }


def _derive_constraints_from_scopes(
    ranked_scopes: List[Dict[str, Any]],
    class_signals: Dict[str, Dict[str, int]],
    api: WikidataAPI,
) -> Tuple[List[str], str, Dict[str, int], Dict[str, str]]:
    if not ranked_scopes:
        return [], "none", {}, {}
    class_like_qids = [
        s["qid"]
        for s in ranked_scopes
        if class_signals.get(s["qid"], {}).get("p279_count", 0) > 0
    ]
    if class_like_qids:
        selected = class_like_qids[:MAX_TYPE_CONSTRAINTS]
        labels = {s["qid"]: s.get("label", "") for s in ranked_scopes if s["qid"] in selected}
        return selected, "class_anchor_p279", {q: 1 for q in selected}, labels

    instance_qids = [s["qid"] for s in ranked_scopes[:MAX_SCOPES]]
    p31_map = api.fetch_p31_types_batch(instance_qids)
    type_counts: Counter[str] = Counter()
    type_labels: Dict[str, str] = {}
    for qid in instance_qids:
        for t_qid, t_label in p31_map.get(qid, []):
            type_counts[t_qid] += 1
            if t_label:
                type_labels[t_qid] = t_label
    if not type_counts:
        return [], "instance_backoff_p31", {}, {}
    selected: List[str] = []
    total = max(1, len(instance_qids))
    min_count = 1 if total <= 3 else P31_SUPPORT_MIN_COUNT
    for t_qid, count in type_counts.most_common():
        if count >= min_count and (count / total) >= P31_SUPPORT_MIN_RATIO:
            selected.append(t_qid)
        if len(selected) >= MAX_TYPE_CONSTRAINTS:
            break
    return selected, "instance_backoff_p31", dict(type_counts), type_labels


def _infer_parent_properties_from_scopes(
    scope_qids: List[str],
    api: WikidataAPI,
) -> List[str]:
    if not scope_qids:
        return []
    supported: List[str] = []
    probe_qids = scope_qids[:MAX_SCOPES]
    for prop in PROPERTY_CANDIDATES:
        try:
            batch = api.sparql_parents_batch(probe_qids, prop, max_depth=2)
        except Exception:
            logger.debug("sparql_parents_batch failed for prop %s during scope profiling", prop)
            continue
        non_empty = sum(1 for q in probe_qids if batch.get(q))
        ratio = non_empty / max(1, len(probe_qids))
        if ratio >= 0.34:
            supported.append(prop)
    return supported


def _compute_profile_confidence(
    ranked_scopes: List[Dict[str, Any]],
    class_signals: Dict[str, Dict[str, int]],
) -> Tuple[float, bool]:
    if not ranked_scopes:
        return 0.0, True
    top = ranked_scopes[0]
    top_score = float(top.get("score", 0.0))
    second_score = float(ranked_scopes[1].get("score", 0.0)) if len(ranked_scopes) > 1 else 0.0
    margin = top_score - second_score
    top_sig = class_signals.get(top["qid"], {})
    class_strength = top_sig.get("p279_count", 0) / max(
        1.0, top_sig.get("p279_count", 0) + top_sig.get("p31_count", 0)
    )
    confidence = (
        0.65 * min(1.0, max(0.0, top_score)) +
        0.25 * min(1.0, max(0.0, margin / 0.30)) +
        0.10 * min(1.0, max(0.0, class_strength))
    )
    ambiguous = len(ranked_scopes) > 1 and margin < AMBIGUITY_MARGIN
    return round(confidence, 4), ambiguous


def infer_concept_profiles_from_keys(
    concept_schema_cols: List[Dict[str, Any]],
    cache: WikidataCache,
    api: WikidataAPI,
    *,
    cache_mode: str = "reuse",
    confidence_threshold: float = 0.55,
) -> Tuple[Dict[str, InferredConceptProfile], Dict[str, Any]]:
    """Infer concept profiles from schema concept keys (not sampled values)."""
    cache_key = _profile_cache_key_from_schema(concept_schema_cols)
    if cache_mode == "reuse":
        cached_blob = cache.get_inferred_profile_set(cache_key)
        if cached_blob:
            profiles = {
                k: profile_from_json(v)
                for k, v in cached_blob.items()
            }
            report = {
                "profile_cache_key": cache_key,
                "profile_cache_hit": True,
                "mode": "concept_key",
                "profiles": {
                    k: asdict(v) for k, v in profiles.items()
                },
            }
            return profiles, report

    profiles: Dict[str, InferredConceptProfile] = {}
    report_profiles: Dict[str, Any] = {}

    for col in concept_schema_cols:
        key = col["name"]
        if not is_str_type(col["type"]):
            continue

        variants, candidates_by_qid = _collect_scope_candidates(key, api)
        year_ratio = _is_year_like_ratio(variants)
        candidate_qids = sorted(candidates_by_qid.keys())
        class_signals = api.fetch_class_instance_signals_batch(candidate_qids) if candidate_qids else {}
        p279_parents = api.sparql_parents_batch(candidate_qids, "P279", max_depth=2) if candidate_qids else {}
        concept_tokens = _tokenize(key)

        scored: List[Dict[str, Any]] = []
        for qid in candidate_qids:
            scored.append(
                _score_candidate(
                    concept_tokens,
                    candidates_by_qid[qid],
                    class_signals.get(qid, {"p279_count": 0, "p31_count": 0}),
                    p279_parents.get(qid, []),
                )
            )
        ranked_scopes = sorted(scored, key=lambda x: x["score"], reverse=True)
        selected_scopes = ranked_scopes[:MAX_SCOPES]
        selected_scope_qids = [s["qid"] for s in selected_scopes]

        type_constraints, scope_strategy, type_counts, type_labels = _derive_constraints_from_scopes(
            selected_scopes,
            class_signals,
            api,
        )
        parent_props = _infer_parent_properties_from_scopes(selected_scope_qids, api)
        max_depth = max((PROP_DEFAULT_DEPTH.get(p, 2) for p in parent_props), default=0)
        confidence, ambiguous = _compute_profile_confidence(selected_scopes, class_signals)
        if year_ratio >= 0.5:
            confidence = max(confidence, 0.60)

        resolve_enabled = bool(type_constraints or parent_props or year_ratio >= 0.5)
        ambiguity_override = confidence >= (confidence_threshold + 0.12)
        resolve_enabled = resolve_enabled and (confidence >= confidence_threshold)
        if year_ratio >= 0.5:
            parent_props = [p for p in parent_props if p == "P279"]
            max_depth = min(max_depth or 2, 2)

        profile = InferredConceptProfile(
            concept_key=key,
            resolve_enabled=resolve_enabled,
            type_constraints=type_constraints,
            parents_properties=parent_props,
            max_depth=max_depth,
            confidence=round(confidence, 4),
            sample_count=len(variants),
            resolved_count=len(selected_scope_qids),
            evidence={
                "mode": "concept_key",
                "scope_strategy": scope_strategy,
                "ambiguous_scope": ambiguous,
                "ambiguity_override": ambiguity_override,
                "year_like_ratio": round(year_ratio, 4),
                "query_variants": variants,
                "chosen_scope_qids": selected_scope_qids,
                "chosen_scope_labels": {
                    s["qid"]: s.get("label", "")
                    for s in selected_scopes
                },
                "candidate_scores": selected_scopes,
                "class_instance_signals": {
                    q: class_signals.get(q, {"p279_count": 0, "p31_count": 0})
                    for q in selected_scope_qids
                },
                "type_counts": type_counts,
                "type_labels": type_labels,
                "parent_properties_signal": parent_props,
            },
        )
        profiles[key] = profile
        report_profiles[key] = asdict(profile)

    cache.put_inferred_profile_set(
        cache_key,
        {k: profile_to_json(v) for k, v in profiles.items()},
    )
    report = {
        "profile_cache_key": cache_key,
        "profile_cache_hit": False,
        "mode": "concept_key",
        "profiles": report_profiles,
    }
    return profiles, report


def infer_subject_hierarchy_from_profiles(
    concept_schema_cols: List[Dict[str, Any]],
    profiles: Dict[str, InferredConceptProfile],
    api: WikidataAPI,
    *,
    confidence_threshold: float,
) -> Dict[str, List[str]]:
    """Infer child->parent subject edges from concept profiles via KG types."""
    subject_types: Dict[str, Set[str]] = defaultdict(set)
    subject_conf: Dict[str, float] = defaultdict(float)
    subject_counts: Dict[str, int] = defaultdict(int)

    for col in concept_schema_cols:
        key = col["name"]
        if ":" not in key:
            continue
        subj = key.split(":", 1)[0].strip().casefold()
        prof = profiles.get(key)
        if not prof or prof.confidence < confidence_threshold:
            continue
        for t in prof.type_constraints:
            subject_types[subj].add(t)
        subject_conf[subj] += prof.confidence
        subject_counts[subj] += 1

    subjects = sorted(subject_types.keys())
    out: Dict[str, List[str]] = {}
    for child in subjects:
        child_types = sorted(subject_types[child])
        if not child_types:
            continue
        child_avg_conf = subject_conf[child] / max(1, subject_counts[child])
        if child_avg_conf < confidence_threshold:
            continue

        parents: List[str] = []
        for parent in subjects:
            if parent == child:
                continue
            parent_types = sorted(subject_types[parent])
            if not parent_types:
                continue
            try:
                matches = api.match_types_batch(child_types, parent_types)
            except Exception:
                logger.debug("match_types_batch failed for %s->%s", child, parent)
                continue
            matched_child_types = sum(1 for t in child_types if matches.get(t))
            ratio = matched_child_types / max(1, len(child_types))
            if ratio >= 0.50:
                parents.append(parent)
        if parents:
            out[child] = sorted(set(parents))
    return out

