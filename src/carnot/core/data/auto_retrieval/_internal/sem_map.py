from __future__ import annotations

import json
from dataclasses import dataclass
import os
from enum import Enum
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union, get_args, get_origin

from pathlib import Path
import dspy
import palimpzest as pz

class SemMapStrategy(str, Enum):
    HIERARCHY_FIRST = "hierarchy_first"
    FLAT = "flat"

class FlatConceptSchemaSignature(dspy.Signature):
    concepts = dspy.InputField(
        desc="List of concept phrases. Normalize them into a FLAT, canonical schema."
    )
    concept_schema = dspy.OutputField(
        desc=(
            "Return a JSON array of objects with keys: name, type, desc.\n"
            "Format rules:\n"
            "- name: Create a colon-delimited hierarchy (e.g., 'film genre' -> 'film:genre').\n"
            "  * CONSTRAINT: Use EXACT words from the input. Do not introduce new words.\n"
            "- type: Native Python type string: 'str', 'int', 'float', or 'List[...]'.\n"
            "  * Determine the concept's cardinality. If the concept tends to have multiple mentions per document, use 'List[...]'.\n"
            "    If it implies a single value, use 'str', 'int', or 'float'.\n"
            "    Example: locations often have multiple mentions -> 'List[str]'; year is usually a single value -> 'int'.\n"
            "- desc: Natural-language description of the leaf node.\n"
            "Output must be valid JSON only. No markdown."
        )
    )


class HierarchyFirstConceptSchemaSignature(dspy.Signature):
    concepts = dspy.InputField(
        desc="List of concept phrases. Normalize them into a HIERARCHY-FIRST schema."
    )
    concept_schema = dspy.OutputField(
        desc=(
            "Return a JSON array of objects with keys: name, type, desc.\n"
            "Format rules:\n"
            "- name: Create a hierarchy. Base hierarchy must preserve the same word order as the input and use EXACT words from the input. Allow expansion (':subtype') ONLY when essential and necessary for granularity.\n"
            "  * Example: 'film location' -> 'film:location:city', 'film:location:state', 'film:location:province', 'film:location:country', 'film:location:continent'\n"
            "- type: Native Python type string: 'str', 'int', 'float', or 'List[...]'.\n"
            "  * Determine the concept's cardinality. If the concept tends to have multiple mentions per document, use 'List[...]'.\n"
            "    If it implies a single value, use 'str', 'int', or 'float'.\n"
            "    Example: locations often have multiple mentions -> 'List[str]'; year is usually a single value -> 'int'.\n"
            "- desc: Natural-language description of the leaf node.\n"
            "Output must be valid JSON only. No markdown."
        )
    )

class HierarchyFirstConceptSchemaModel(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(HierarchyFirstConceptSchemaSignature)
        self._few_shot_examples = [
            dspy.Example(
                concepts=["film location"],
                concept_schema=json.dumps([
                    {"name": "film:location:city", "type": "List[str]", "desc": "Distinct film cities mentioned. If the document does not mention any cities, set this field to None."},
                    {"name": "film:location:state", "type": "List[str]", "desc": "Distinct film states mentioned. If the document does not mention any states, set this field to None."},
                    {"name": "film:location:province", "type": "List[str]", "desc": "Distinct film provinces mentioned. If the document does not mention any provinces, set this field to None."},
                    {"name": "film:location:country", "type": "List[str]", "desc": "Distinct film countries mentioned. If the document does not mention any countries, set this field to None."},
                    {"name": "film:location:continent", "type": "List[str]", "desc": "Distinct film continents mentioned. If the document does not mention any continents, set this field to None."},
                ])
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["book release-year"],
                concept_schema=json.dumps([
                    {"name": "book:release-year", "type": "int", "desc": "Year the book was released. If the document does not mention a book release year, set this field to None."},
                ])
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["film decade"],
                concept_schema=json.dumps([
                    {"name": "film:decade", "type": "int", "desc": "Decade the film was released. If the document does not mention a film decade, set this field to None."},
                ])
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["person name"],
                concept_schema=json.dumps([
                    {"name": "person:name:first", "type": "str", "desc": "First name of the person. If the document does not mention a person's first name, set this field to None."},
                    {"name": "person:name:last", "type": "str", "desc": "Last name of the person. If the document does not mention a person's last name, set this field to None."},
                ])
            ).with_inputs("concepts")
        ]

    def forward(self, concepts: List[str]) -> str:
        result = self._predict(concepts=concepts, demos=self._few_shot_examples)
        return result.concept_schema

class FlatConceptSchemaModel(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(FlatConceptSchemaSignature)
        self._few_shot_examples = [
            dspy.Example(
                concepts=["film location"],
                concept_schema=json.dumps([
                    {"name": "film:location", "type": "List[str]", "desc": "Distinct film locations mentioned. If the document does not mention any film locations, set this field to None."},
                ])
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["book release-year"],
                concept_schema=json.dumps([
                    {"name": "book:release-year", "type": "int", "desc": "Year the book was released. If the document does not mention a book release year, set this field to None."},
                ])
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["bird color"],
                concept_schema=json.dumps([
                    {"name": "bird:color", "type": "List[str]", "desc": "Distinct bird colors mentioned. If the document does not mention any bird colors, set this field to None."},
                ])
            ).with_inputs("concepts")
        ]

    def forward(self, concepts: List[str]) -> str:
        result = self._predict(concepts=concepts, demos=self._few_shot_examples)
        return result.concept_schema

@dataclass(frozen=True)
class PruneSemMapResults:
    # TODO: start simple, evolve this into a utility score combining selectivity and query frequency.
    pass


def _type_from_str(t: str) -> Any:
    # Simplified parser to map clean strings to types
    t_clean = str(t).strip()
    if t_clean == "int": return int
    if t_clean == "float": return float
    if t_clean == "str": return str
    if t_clean == "List[str]": return List[str]
    if t_clean == "List[int]": return List[int]
    if t_clean == "List[float]": return List[float]
    return List[str] # Default fallback


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

def sem_map(
    *,
    concepts: Sequence[str],
    data: Sequence[Mapping[str, str]],
    strategy: Union[SemMapStrategy, str] = SemMapStrategy.FLAT,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    strat = SemMapStrategy(strategy)
    model: dspy.Module = FlatConceptSchemaModel() if strat is SemMapStrategy.FLAT else HierarchyFirstConceptSchemaModel()
    
    concept_schema_cols: List[Dict[str, Any]] = []

    for concept in concepts:
        raw_output = model([concept])
        
        if not isinstance(raw_output, str):
            raise TypeError(f"DSPy concept_schema output expected string, got {type(raw_output)}")
        
        try:
            schema_list = json.loads(raw_output)
            # Ensure it's a list
            if not isinstance(schema_list, list):
                 raise ValueError("Output is not a JSON list")
        except Exception as e:
            print(f"DEBUG: Raw='{raw_output}'")
            raise ValueError(f"Failed to parse DSPy concept_schema output as JSON for concept '{concept}'.") from e

        for col in schema_list:
            concept_schema_cols.append(
                {"name": col["name"], "type": _type_from_str(col["type"]), "desc": col.get("desc", "...")}
            )

    rows: List[Dict[str, str]] = []
    for d in data:
        doc_id = str(d.get("id", "")).strip()
        text = str(d.get("text", "")).strip()
        if doc_id and text:
            rows.append({"id": doc_id, "text": text})

    if not rows:
        return {}, concept_schema_cols

    dataset = pz.MemoryDataset(id="sem-map", vals=rows)
    dataset = dataset.sem_map(cols=concept_schema_cols)
    output = dataset.run(max_quality=True)
    df = output.to_df()

    col_names = [c["name"] for c in concept_schema_cols]
    results: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        doc_id = str(row.get("id", "")).strip()
        if not doc_id:
            continue
        out: Dict[str, Any] = {}
        for c in col_names:
            v = row.get(c, None)
            v = _dedupe_list(v)
            if v is None or v == "" or v == []:
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


# Example usage
if __name__ == "__main__":
    import random
    
    api_key = os.environ.get("OPENAI_API_KEY", "")
    lm = dspy.LM("openai/gpt-5.1", temperature=1.0, max_tokens=16000, api_key=api_key)
    dspy.configure(lm=lm)

    concepts=['amphibian location', 'film classification', 'fish classification', 'book region', 'crime film theme']
    
    print(f"concepts={concepts}")

    for strat in (SemMapStrategy.FLAT, SemMapStrategy.HIERARCHY_FIRST):
        model: dspy.Module = FlatConceptSchemaModel() if strat is SemMapStrategy.FLAT else HierarchyFirstConceptSchemaModel()
        
        concept_schema_cols: List[Dict[str, Any]] = []

        for concept in concepts:
            raw_output = model([concept])
            
            try:
                schema_list = json.loads(raw_output)
                if not isinstance(schema_list, list):
                     print(f"Warning: Output for '{concept}' is not a list. Skipping.")
                     continue
            except Exception as e:
                print(f"Error parsing '{concept}': {e}")
                continue

            for col in schema_list:
                concept_schema_cols.append(
                    {"name": col["name"], "type": _type_from_str(col["type"]), "desc": col.get("desc", "...")}
                )

        print(f"\nstrategy={strat.value}")
        print("concept_schema_cols=")
        for c in concept_schema_cols:
            print(c)

    raise SystemExit(0)
