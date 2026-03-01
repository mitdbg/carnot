from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from pprint import pprint
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple, get_args, get_origin

DEFAULT_TOP_K_CONCEPTS = 30
CONCEPTS_JSON_REL_PATH = Path("results/quest_concepts/two_stage/per_query_concepts.json")
INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
FLOAT_PATTERN = re.compile(r"^[+-]?(?:\d+\.\d+|\d+\.\d*|\.\d+)$")


def _concepts_output_path() -> Path:
    return Path(__file__).resolve().parent / CONCEPTS_JSON_REL_PATH


@lru_cache(maxsize=1)
def _load_concepts_output(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _select_top_domain_facets(concepts_output: Mapping[str, Any], top_k: int) -> List[Tuple[str, str]]:
    facet_frequencies = concepts_output.get("facet_frequencies", {})
    if not isinstance(facet_frequencies, Mapping):
        raise ValueError("Expected 'facet_frequencies' to be a mapping in concepts output JSON.")

    def _as_int(val: Any) -> int:
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    ranked_items = sorted(
        facet_frequencies.items(),
        key=lambda item: (-_as_int(item[1]), str(item[0])),
    )[:top_k]

    pairs: List[Tuple[str, str]] = []
    for key, _ in ranked_items:
        name = str(key)
        if "|" not in name:
            continue
        domain, facet = name.split("|", 1)
        domain = domain.strip()
        facet = facet.strip()
        if domain and facet:
            pairs.append((domain, facet))
    return pairs


def _collect_query_examples(
    concepts_output: Mapping[str, Any],
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    queries = concepts_output.get("queries", {})
    if not isinstance(queries, Mapping):
        raise ValueError("Expected 'queries' to be a mapping in concepts output JSON.")

    examples: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for query_text, triples in queries.items():
        if not isinstance(triples, list):
            continue
        normalized_triples: List[List[str]] = []
        for t in triples:
            if not isinstance(t, list) or len(t) < 3:
                continue
            normalized_triples.append([str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()])

        if not normalized_triples:
            continue

        by_pair: Dict[Tuple[str, str], List[List[str]]] = {}
        for domain, facet, value in normalized_triples:
            if not domain or not facet or not value:
                continue
            by_pair.setdefault((domain, facet), []).append([domain, facet, value])

        query = str(query_text).strip()
        if not query:
            continue
        for pair, pair_triplets in by_pair.items():
            examples.setdefault(pair, []).append(
                {
                    "query": query,
                    "triplets": pair_triplets,
                }
            )
    return examples


def _sample_examples(
    values: Sequence[Mapping[str, Any]], max_examples: int, rng: random.Random
) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in values:
        query = str(raw.get("query", "")).strip()
        triplets = raw.get("triplets", [])
        if not isinstance(triplets, list) or not triplets:
            continue
        clean_triplets: List[List[str]] = []
        for triplet in triplets:
            if not isinstance(triplet, list) or len(triplet) < 3:
                continue
            clean_triplets.append(
                [str(triplet[0]).strip(), str(triplet[1]).strip(), str(triplet[2]).strip()]
            )
        if not query or not clean_triplets:
            continue
        key = f"{query.casefold()}|{json.dumps(clean_triplets, ensure_ascii=False)}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "query": query,
                "triplets": clean_triplets,
            }
        )

    if len(deduped) <= max_examples:
        return deduped
    return rng.sample(deduped, max_examples)


def _is_int_like(value: str) -> bool:
    return bool(INTEGER_PATTERN.fullmatch(value.strip()))


def _is_float_like(value: str) -> bool:
    s = value.strip()
    return _is_int_like(s) or bool(FLOAT_PATTERN.fullmatch(s))


def _list_of_str_type() -> Any:
    """Prefer built-in generic list[str], with py<3.9 fallback."""
    try:
        return list[str]
    except TypeError:
        return List[str]


def _infer_schema_type(example_values: Sequence[str]) -> Any:
    if example_values and all(_is_int_like(v) for v in example_values):
        return int
    if example_values and all(_is_float_like(v) for v in example_values):
        return float
    return _list_of_str_type()


def _build_desc(
    domain: str,
    facet: str,
    sampled_examples: Sequence[Mapping[str, Any]],
) -> str:
    if sampled_examples:
        example_parts: List[str] = []
        for idx, ex in enumerate(sampled_examples, start=1):
            query = str(ex.get("query", "")).strip()
            triplets = ex.get("triplets", [])
            values = [
                str(t[2]).strip()
                for t in triplets
                if isinstance(t, list) and len(t) >= 3 and str(t[2]).strip()
            ]
            example_parts.append(
                (
                    f"Example {idx} - text chunk: {query!r}; "
                    f"values: {json.dumps(values, ensure_ascii=False)}"
                )
            )
        examples_str = " ".join(example_parts)
    else:
        examples_str = "None available."
    return (
    f"Extract only explicit values for (domain: {domain}, facet: {facet}) from the text chunk. Return null if the text does not explicitly state a {facet} for a {domain}. "
    f"Return standardized, canonical values when the text clearly refers to a known value, using a consistent form across documents. Do not change the meaning. If standardization is uncertain, use the exact text-supported value (or return null if the value itself is uncertain). "
    f"Return only concise entity-like values (i.e., node-style values suitable for a knowledge graph), not descriptive phrases, clauses, or sentences."
    "If the expected type is list[type] and only one value is present, return it as a single-element list (e.g., [value]). "
    "If the expected type is int, return a single numeric value (not a list). "
    "An int type means exactly one number or null. "
    "A list[str] type means it may return one string (as a single-element list), multiple strings, or null.\n\n"
    "The following examples show you how to extract values for this domain and facet. "
    f"Example: {examples_str} "
)


def _build_concept_schema_cols(
    top_k: int = DEFAULT_TOP_K_CONCEPTS,
    concepts_output_path: str | None = None,
) -> List[Dict[str, Any]]:
    output_path = concepts_output_path or str(_concepts_output_path())
    concepts_output = _load_concepts_output(output_path)

    top_pairs = _select_top_domain_facets(concepts_output, top_k=top_k)
    query_examples = _collect_query_examples(concepts_output)
    rng = random.Random(0)

    cols: List[Dict[str, Any]] = []
    for domain, facet in top_pairs:
        sampled = _sample_examples(
            query_examples.get((domain, facet), []), max_examples=2, rng=rng
        )
        sampled_values: List[str] = []
        for ex in sampled:
            triplets = ex.get("triplets", [])
            if not isinstance(triplets, list):
                continue
            for triplet in triplets:
                if not isinstance(triplet, list) or len(triplet) < 3:
                    continue
                sampled_values.append(str(triplet[2]).strip())
        cols.append(
            {
                "name": f"{domain}:{facet}",
                "type": _infer_schema_type(sampled_values),
                "desc": _build_desc(domain=domain, facet=facet, sampled_examples=sampled),
            }
        )
    return cols


def _dedupe_list(vals: Any) -> Any:
    if not isinstance(vals, list):
        return vals
    out: List[Any] = []
    seen: set[str] = set()
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str):
            s = " ".join(v.strip().split())
            if not s:
                continue
            k = s.casefold()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
        else:
            k = str(v)
            if k in seen:
                continue
            seen.add(k)
            out.append(v)
    return out


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if INTEGER_PATTERN.fullmatch(s):
            return int(s)
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if _is_float_like(s):
            try:
                return float(s)
            except ValueError:
                return None
    return None


def _normalize_by_schema_type(value: Any, schema_type: Any) -> Any:
    value = _dedupe_list(value)
    if value is None or value == "" or value == []:
        return None

    values = value if isinstance(value, list) else [value]

    if schema_type is int:
        for item in values:
            coerced = _coerce_int(item)
            if coerced is not None:
                return coerced
        return None

    if schema_type is float:
        for item in values:
            coerced = _coerce_float(item)
            if coerced is not None:
                return coerced
        return None

    if _is_list_type(schema_type):
        normalized_items: List[str] = []
        for item in values:
            if item is None:
                continue
            s = " ".join(str(item).strip().split())
            if s:
                normalized_items.append(s)
        normalized_items = _dedupe_list(normalized_items)
        return normalized_items if normalized_items else None

    return value

def sem_map(
    *,
    data: Sequence[Mapping[str, str]],
    concepts_output_path: str | None = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    import palimpzest as pz

    concept_schema_cols = _build_concept_schema_cols(top_k=DEFAULT_TOP_K_CONCEPTS, concepts_output_path=concepts_output_path)
    
    if len(concept_schema_cols) != DEFAULT_TOP_K_CONCEPTS:
        raise RuntimeError(
            f"Expected {DEFAULT_TOP_K_CONCEPTS} schema columns, got {len(concept_schema_cols)}."
        )

    names = [c["name"] for c in concept_schema_cols]
    if len(names) != len(set(names)):
        raise RuntimeError("Schema contains duplicate column names.")
    
    # payload = {
    #     "top_k": DEFAULT_TOP_K_CONCEPTS,
    #     "num_cols": len(concept_schema_cols),
    #     "schema_preview": [
    #         {"name": c["name"], "type": c["type"], "desc": c["desc"]}
    #         for c in concept_schema_cols[:10]
    #     ],
    # }
    # pprint(payload, sort_dicts=False)
    # import pdb; pdb.set_trace()
    
    rows: List[Dict[str, str]] = []
    for d in data:
        doc_id = str(d.get("id", "")).strip()
        text = str(d.get("text", "")).strip()
        if doc_id and text:
            rows.append({"id": doc_id, "text": text})

    if not rows:
        return {}, concept_schema_cols

    dataset = pz.MemoryDataset(id="sem-map", vals=rows)
    cols_for_pz = [dict(c) for c in concept_schema_cols]
    dataset = dataset.sem_map(cols=cols_for_pz)
    output = dataset.run(max_quality=True)
    
    col_names = [c["name"] for c in concept_schema_cols]
    type_by_name = {c["name"]: c["type"] for c in concept_schema_cols}
    results: Dict[str, Dict[str, Any]] = {}

    for rec in getattr(output, "data_records", []):
        doc_id = str(getattr(rec, "id", "")).strip()
        if not doc_id:
            continue

        out: Dict[str, Any] = {}
        for c in col_names:
            v = getattr(rec, c, None)
            v = _normalize_by_schema_type(v, type_by_name[c])
            if v is None:
                continue
            out[c] = v

        results[doc_id] = out    

    return results, concept_schema_cols


def _is_list_type(tp: Any) -> bool:
    origin = get_origin(tp)
    return origin is list

def _is_taggable_type(tp: Any) -> bool:
    return tp is str or _is_list_type(tp)

def _canon_tag_suffix(v: Any) -> str:
    """Canonical string suffix used in the tag key (keeps spaces, trims/normalizes)."""
    if v is None:
        return ""
    if isinstance(v, str):
        s = " ".join(v.strip().split())
        return s
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else str(v)
    return str(v).strip()


def expand_sem_map_results_to_tags(
    results: Dict[str, Dict[str, Any]],
    schema: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """
    Returns:
      - expanded_results: doc_id -> { tag_col: True, scalar_col: value, ... }
      - expanded_schema: schema where taggable cols are replaced by bool tag columns
      - expanded_stats: present/total/selectivity for each expanded column
    """
    type_by_name: Dict[str, Any] = {c["name"]: c["type"] for c in schema}
    taggable_cols = {name for name, tp in type_by_name.items() if _is_taggable_type(tp)}

    tag_values: Dict[str, set[str]] = {c: set() for c in taggable_cols}
    for _, cols in results.items():
        for col, v in cols.items():
            if col not in taggable_cols:
                continue
            vs = v if isinstance(v, list) else [v]
            for x in vs:
                sfx = _canon_tag_suffix(x)
                if sfx:
                    tag_values[col].add(sfx)

    expanded_schema: List[Dict[str, Any]] = []
    for col_def in schema:
        name = col_def["name"]
        if name not in taggable_cols:
            expanded_schema.append(col_def)
            continue

        for sfx in sorted(tag_values[name], key=lambda s: s.casefold()):
            tag_name = f"{name}:{sfx}"
            expanded_schema.append(
                {
                    "name": tag_name,
                    "type": bool,
                    "desc": f"True if {sfx!r} appears in {name}.",
                }
            )

    expanded_results: Dict[str, Dict[str, Any]] = {}
    
    all_bool_cols = [col["name"] for col in expanded_schema if col["type"] is bool]
    
    for doc_id, cols in results.items():
        out: Dict[str, Any] = {k: False for k in all_bool_cols}
        
        for col, v in cols.items():
            if col not in taggable_cols:
                out[col] = v
                continue
                
            vs = v if isinstance(v, list) else [v]
            for x in vs:
                sfx = _canon_tag_suffix(x)
                if sfx:
                    out[f"{col}:{sfx}"] = True
        expanded_results[doc_id] = out

    total = float(len(results))
    present: Dict[str, float] = {c["name"]: 0.0 for c in expanded_schema}
    
    for _, cols in expanded_results.items():
        for k, v in cols.items():
            if k not in present:
                continue
            if isinstance(v, bool):
                if v:
                    present[k] += 1.0
            else:
                if v is not None:
                    present[k] += 1.0

    expanded_stats: Dict[str, Dict[str, float]] = {}
    for k, p in present.items():
        sel = (p / total) if total else 0.0
        expanded_stats[k] = {"present": p, "total": total, "selectivity": sel}

    return expanded_results, expanded_schema, expanded_stats
