from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _internal.concept_profiler import (
    InferredConceptProfile,
    infer_concept_profiles_from_keys,
    infer_subject_hierarchy_from_profiles,
)
from _internal.type_utils import is_str_type
from _internal.wikidata_api import WikidataAPI
from _internal.wikidata_cache import WikidataCache
from _internal.wikidata_ontology import WikidataOntology

logger = logging.getLogger(__name__)


@dataclass
class ConceptOverride:
    disable_key: bool = False
    force_type_constraints: List[str] = field(default_factory=list)
    force_parent_properties: List[str] = field(default_factory=list)
    force_max_depth: Optional[int] = None


@dataclass
class OverrideConfig:
    concept_overrides: Dict[str, ConceptOverride] = field(default_factory=dict)
    cross_concept_edges: Dict[str, List[str]] = field(default_factory=dict)


def _load_override_config(path: Optional[Path]) -> OverrideConfig:
    if path is None or not path.exists():
        return OverrideConfig()
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise RuntimeError("PyYAML required for .yaml config. pip install pyyaml")
        raw = yaml.safe_load(text) or {}
    elif suffix == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {suffix}")

    concept_overrides: Dict[str, ConceptOverride] = {}
    for key, cfg in (raw.get("overrides", {}).get("concepts") or {}).items():
        if not isinstance(cfg, dict):
            continue
        concept_overrides[str(key)] = ConceptOverride(
            disable_key=bool(cfg.get("disable_key", False)),
            force_type_constraints=list(cfg.get("force_type_constraints") or []),
            force_parent_properties=list(cfg.get("force_parent_properties") or []),
            force_max_depth=(
                int(cfg["force_max_depth"]) if cfg.get("force_max_depth") is not None else None
            ),
        )
    cross_edges: Dict[str, List[str]] = {}
    for child, parents in (raw.get("overrides", {}).get("cross_concept_edges") or {}).items():
        c = str(child).strip()
        if not c:
            continue
        pp = [str(p).strip() for p in (parents or []) if str(p).strip()]
        if pp:
            cross_edges[c] = pp
    return OverrideConfig(
        concept_overrides=concept_overrides,
        cross_concept_edges=cross_edges,
    )


def _apply_override(
    profile: InferredConceptProfile,
    override: Optional[ConceptOverride],
) -> InferredConceptProfile:
    if override is None:
        return profile
    merged = InferredConceptProfile(
        concept_key=profile.concept_key,
        resolve_enabled=False if override.disable_key else profile.resolve_enabled,
        type_constraints=list(profile.type_constraints),
        parents_properties=list(profile.parents_properties),
        max_depth=profile.max_depth,
        confidence=profile.confidence,
        sample_count=profile.sample_count,
        resolved_count=profile.resolved_count,
        evidence=dict(profile.evidence),
    )
    if override.force_type_constraints:
        merged.type_constraints = list(override.force_type_constraints)
        merged.resolve_enabled = True
    if override.force_parent_properties:
        merged.parents_properties = list(override.force_parent_properties)
    if override.force_max_depth is not None:
        merged.max_depth = int(override.force_max_depth)
    if override.disable_key:
        merged.parents_properties = []
        merged.max_depth = 0
    return merged


def build_ontologies_from_config(
    concept_schema_cols: List[Dict[str, Any]],
    config_path: str | Path | None,
    cache_db_path: str,
    *,
    api: Optional[WikidataAPI] = None,
    profile_cache_mode: str = "reuse",
    kg_confidence_threshold: float = 0.45,
) -> Tuple[Dict[str, WikidataOntology], Dict[str, Any], Dict[str, InferredConceptProfile], Dict[str, List[str]]]:
    cache = WikidataCache(cache_db_path)
    if api is None:
        api = WikidataAPI()

    overrides = _load_override_config(Path(config_path) if config_path else None)
    profiles, profiling_report = infer_concept_profiles_from_keys(
        concept_schema_cols,
        cache,
        api,
        cache_mode=profile_cache_mode,
        confidence_threshold=kg_confidence_threshold,
    )
    merged_profiles: Dict[str, InferredConceptProfile] = {}
    for key, profile in profiles.items():
        merged_profiles[key] = _apply_override(
            profile,
            overrides.concept_overrides.get(key),
        )

    ontologies: Dict[str, WikidataOntology] = {}
    skipped: List[str] = []
    entries: List[Dict[str, Any]] = []
    for col in concept_schema_cols:
        key_name = col["name"]
        if not is_str_type(col["type"]):
            continue
        prof = merged_profiles.get(key_name)
        if prof is None:
            skipped.append(key_name)
            entries.append({"concept_key": key_name, "reason": "no_profile"})
            continue
        if not prof.resolve_enabled:
            skipped.append(key_name)
            entries.append({
                "concept_key": key_name,
                "reason": "resolve_disabled",
                "confidence": prof.confidence,
            })
            continue
        materialize = bool(prof.parents_properties) and prof.max_depth > 0
        ontologies[key_name] = WikidataOntology(
            concept_key=key_name,
            cache=cache,
            api=api,
            type_constraints=prof.type_constraints,
            parents_properties=prof.parents_properties,
            max_depth=prof.max_depth,
            resolve_enabled=True,
            materialize_enabled=materialize,
        )
        entries.append({
            "concept_key": key_name,
            "confidence": prof.confidence,
            "type_constraints": prof.type_constraints,
            "parents_properties": prof.parents_properties,
            "max_depth": prof.max_depth,
        })

    inferred_hierarchy = infer_subject_hierarchy_from_profiles(
        concept_schema_cols,
        merged_profiles,
        api,
        confidence_threshold=kg_confidence_threshold,
    )
    for child, parents in overrides.cross_concept_edges.items():
        inferred_hierarchy.setdefault(child, [])
        inferred_hierarchy[child] = sorted(set(inferred_hierarchy[child]) | set(parents))

    coverage_report = {
        "entries": entries,
        "skipped": sorted(set(skipped)),
        "profiling_report": profiling_report,
        "overrides_applied": {
            k: asdict(v) for k, v in overrides.concept_overrides.items()
        },
        "inferred_cross_concept_hierarchy": inferred_hierarchy,
    }
    logger.info(
        "Ontology coverage (concept-key): %d ontologies, %d skipped",
        len(ontologies), len(skipped),
    )
    return ontologies, coverage_report, merged_profiles, inferred_hierarchy


def load_concept_hierarchy(
    config_path: str | Path | None,
    *,
    inferred_hierarchy: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    overrides = _load_override_config(Path(config_path) if config_path else None)
    merged: Dict[str, List[str]] = {}
    if inferred_hierarchy:
        for child, parents in inferred_hierarchy.items():
            c = str(child).strip()
            if not c:
                continue
            merged[c] = [str(p).strip() for p in parents if str(p).strip()]
    for child, parents in overrides.cross_concept_edges.items():
        merged.setdefault(child, [])
        merged[child] = sorted(set(merged[child]) | set(parents))
    return merged
