"""Postprocess raw sem_map output: value normalization + LLM-based hierarchy augmentation."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict, deque
from pathlib import Path as _Path
from typing import Any, Dict, List, Optional, Set, Tuple

import dspy

logger = logging.getLogger(__name__)

class InferHierarchySignature(dspy.Signature):
    """Infer a minimal graph of direct broader-parent relations among metadata values.

    You are given a list of unique string values from one structured metadata column
    (identified by column_name). Return DIRECT parent relations where one value is a
    narrower geographic, administrative, regional, temporal, or categorical member of another.

    Rules:
    - Return a MINIMAL graph of DIRECT parent edges only.
    - Do NOT include transitive ancestors.
      Example: if A -> B and B -> C, do NOT also output A -> C.
    - A value may have MULTIPLE direct parents when multiple broader groupings are valid
      and all of them appear in the input list.
      Example: "California" may have direct parents ["United States", "West Coast"].
    - Every child and parent MUST appear verbatim in the input values list.
    - Do NOT introduce any new value.
    - Do NOT include reflexive edges.
    - Prefer precision over recall: only include relations you are highly confident about.
    - If the column has no clear hierarchy, return {}.

    Notes:
    - Valid parent relations include administrative containment, geographic containment,
      regional grouping membership, temporal period membership, and other clear broader-category relations.
    - Do NOT include weak topical association or semantic relatedness.
    - Python will compute the transitive closure later, so only output the direct edges needed.
    """

    column_name: str = dspy.InputField(
        desc="The domain:facet column name (e.g. 'film:location', 'animal:location')."
    )
    values_json: str = dspy.InputField(
        desc="JSON array of the unique string values present in this column."
    )
    hierarchy_json: str = dspy.OutputField(
    desc=(
        "Return ONLY a JSON object mapping each child value to a list of its DIRECT parent values. "
        "Use minimal non-redundant edges only. "
        "A value may have multiple direct parents. "
        "Omit values with no parent. "
    )
)

# Few-shot examples.
# Example 1: DAG with multiple parents + regional subdivision + empty non-geographic values.
# Example 2: Sparse semantic column that correctly returns {}.
_FEW_SHOT_EXAMPLES = [
    dspy.Example(
        column_name="film:location",
        values_json=json.dumps([
            "Southern California", "California", "United States", "West Coast", "London", "England", "United Kingdom",
        ]),
        hierarchy_json=json.dumps({
            "Southern California": ["California"],
            "California": ["United States", "West Coast"],
            "London": ["England"],
            "England": ["United Kingdom"],
        }),
    ).with_inputs("column_name", "values_json"),
    dspy.Example(
        column_name="film:subject",
        values_json=json.dumps(["war", "romance", "death", "family", "identity"]),
        hierarchy_json="{}",
    ).with_inputs("column_name", "values_json"),
]


class HierarchyInferenceModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(InferHierarchySignature)

    def forward(self, column_name: str, values_json: str) -> str:
        result = self._predict(
            column_name=column_name,
            values_json=values_json,
            demos=_FEW_SHOT_EXAMPLES,
        )
        return getattr(result, "hierarchy_json", "{}").strip()

def _compute_transitive_closure(
    direct_parents: Dict[str, List[str]],
    valid_values: Set[str],
) -> Dict[str, Set[str]]:
    # Filter edges to valid values only.
    adj: Dict[str, Set[str]] = defaultdict(set)
    for child, parents in direct_parents.items():
        if child not in valid_values:
            continue
        for parent in parents:
            if parent in valid_values and parent != child:
                adj[child].add(parent)

    ancestors: Dict[str, Set[str]] = {}

    def _ancestors_of(node: str) -> Set[str]:
        if node in ancestors:
            return ancestors[node]
        # Guard against cycles: seed with empty set before recursing.
        ancestors[node] = set()
        result: Set[str] = set()
        queue = deque(adj.get(node, set()))
        while queue:
            current = queue.popleft()
            if current in result:
                continue
            result.add(current)
            for grandparent in adj.get(current, set()):
                if grandparent not in result:
                    queue.append(grandparent)
        ancestors[node] = result
        return result

    for value in valid_values:
        _ancestors_of(value)

    return ancestors

def _normalize_sem_map(
    sem_results: Dict[str, Dict[str, Any]],
    list_col_names: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """Strip trailing whitespace from string values in list-typed columns only."""
    normalized: Dict[str, Dict[str, Any]] = {}
    for doc_id, doc in sem_results.items():
        new_doc: Dict[str, Any] = {}
        for col, val in doc.items():
            if col in list_col_names and isinstance(val, list):
                new_doc[col] = [v.rstrip() if isinstance(v, str) else v for v in val]
            else:
                new_doc[col] = val
        normalized[doc_id] = new_doc
    return normalized

def _parse_hierarchy_json(
    raw: str,
    valid_values: Set[str],
) -> Dict[str, List[str]]:
    """Parse LLM output into a validated direct-parents mapping."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Distinguish truncation (no closing brace) from a malformed response.
        if text.startswith("{") and not text.endswith("}"):
            logger.error(
                "Hierarchy JSON was TRUNCATED (max_tokens too low). "
                "All hierarchy relations for this column are lost. Raw tail: ...%s",
                raw[-100:],
            )
        else:
            logger.warning("Failed to parse hierarchy JSON: %s", raw[:200])
        return {}

    if not isinstance(data, dict):
        logger.warning("Expected a JSON object from hierarchy inference, got: %s", type(data))
        return {}

    validated: Dict[str, List[str]] = {}
    for child, parents in data.items():
        if not isinstance(child, str) or child not in valid_values:
            continue
        if not isinstance(parents, list):
            continue
        valid_parents = [
            p for p in parents
            if isinstance(p, str) and p in valid_values and p != child
        ]
        if valid_parents:
            validated[child] = valid_parents
    return validated


def _infer_column_hierarchy(
    module: HierarchyInferenceModule,
    column_name: str,
    global_values: Set[str],
) -> Dict[str, Set[str]]:
    """Run LLM hierarchy inference for one column; return ancestor map."""
    values_json = json.dumps(sorted(global_values), ensure_ascii=False)

    try:
        raw = module(column_name=column_name, values_json=values_json)
    except Exception as exc:
        logger.error("LLM call failed for column '%s': %s", column_name, exc)
        return {}

    direct_parents = _parse_hierarchy_json(raw, global_values)

    if direct_parents:
        total_edges = sum(len(v) for v in direct_parents.values())
        logger.info(
            "Column '%s': %d direct edges inferred (from %d values)",
            column_name, total_edges, len(global_values),
        )
    else:
        logger.info(
            "Column '%s': no containment relations inferred (from %d values)",
            column_name, len(global_values),
        )

    return _compute_transitive_closure(direct_parents, global_values)

def _augment_sem_map(
    sem_results: Dict[str, Dict[str, Any]],
    ancestors_by_col: Dict[str, Dict[str, Set[str]]],
) -> Dict[str, Dict[str, Any]]:
    """Expand each document's list columns by appending transitive ancestor values."""
    augmented: Dict[str, Dict[str, Any]] = {}
    for doc_id, doc in sem_results.items():
        new_doc: Dict[str, Any] = {}
        for col, val in doc.items():
            if col in ancestors_by_col and isinstance(val, list):
                col_ancestors = ancestors_by_col[col]
                extended: List[Any] = list(val)
                seen: Set[Any] = set(val)
                for item in val:
                    if isinstance(item, str):
                        for ancestor in col_ancestors.get(item, set()):
                            if ancestor not in seen:
                                extended.append(ancestor)
                                seen.add(ancestor)
                new_doc[col] = extended
            else:
                new_doc[col] = val
        augmented[doc_id] = new_doc
    return augmented

def postprocess_sem_map(
    sem_results: Dict[str, Dict[str, Any]],
    concept_schema_cols: List[Dict[str, Any]],
    out_dir: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    _out: Optional[_Path] = _Path(out_dir) if out_dir else None
    if _out:
        _out.mkdir(parents=True, exist_ok=True)

    def _save(name: str, payload: Any) -> None:
        if _out is None:
            return
        (_out / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")

    lm = dspy.LM("openai/gpt-5.1", temperature=1.0, max_tokens=16000, api_key=api_key)
    dspy.configure(lm=lm)
    module = HierarchyInferenceModule()

    # Identify list-typed columns from schema.
    list_col_names: Set[str] = {
        col["name"]
        for col in concept_schema_cols
        if str(col.get("type", "")).startswith("list[")
    }

    logger.info(
        "postprocess_sem_map: %d list-typed columns, %d documents",
        len(list_col_names), len(sem_results),
    )

    # Step 1: Normalize (trailing whitespace only).
    sem_results = _normalize_sem_map(sem_results, list_col_names)
    _save("postprocess_step1_normalized.json", sem_results)
    logger.info("Step 1 done: value normalization.")

    # Step 2: Collect global unique values per list column, then infer hierarchy.
    global_values_by_col: Dict[str, Set[str]] = {col: set() for col in list_col_names}
    for doc in sem_results.values():
        for col in list_col_names:
            val = doc.get(col)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        global_values_by_col[col].add(item)

    ancestors_by_col: Dict[str, Dict[str, Set[str]]] = {}
    for col in sorted(list_col_names):
        global_values = global_values_by_col[col]
        if not global_values:
            continue
        ancestors_by_col[col] = _infer_column_hierarchy(module, col, global_values)

    # Serialize hierarchy: { column -> { value -> [ancestor, ...] } }
    hierarchy_payload: Dict[str, Any] = {
        col: {
            value: sorted(ancestor_set)
            for value, ancestor_set in col_ancestors.items()
            if ancestor_set
        }
        for col, col_ancestors in ancestors_by_col.items()
    }
    _save("postprocess_step2_hierarchy.json", hierarchy_payload)
    logger.info("Step 2 done: hierarchy inference.")

    # Step 3: Augment each document with ancestor values.
    sem_results = _augment_sem_map(sem_results, ancestors_by_col)
    _save("postprocess_step3_augmented.json", sem_results)
    logger.info("Step 3 done: row-level augmentation.")

    return sem_results
