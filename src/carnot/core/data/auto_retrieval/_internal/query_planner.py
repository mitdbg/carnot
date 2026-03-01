from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os

import dspy

logger = logging.getLogger(__name__)

class FilterSelectionSignature(dspy.Signature):
    """Select metadata filters to apply to a search query.

    You are given a search query and a list of available metadata filters.
    Each filter is shown with its domain:facet name and a few sample values
    so you can judge relevance.

    Goal: identify filters that will CONFIDENTLY narrow the search space.
    Pure vector search is the safe fallback — use it whenever uncertain.

    Rules:
    - Return [] if you are not highly confident any filter applies.
    - Select a filter ONLY if ALL of the following hold:
        1. The filter's domain (film, book, animal, plant, ...) matches the entity
           type the query is clearly about.
        2. The filter's facet (location, genre, year, subject, ...) is directly
           relevant to what the query is asking about.
        3. The sample values suggest that matching values plausibly exist.
    - Do NOT select a filter on vague or partial topic overlap.
    - If the query asks about a property no available facet covers
      (e.g. species taxonomy, film cinematography style), return [].
    - If the query spans multiple domains ("films or books about X"), select
      filters from all applicable domains.
    - When in doubt, return []. A missed filter is always safer than a wrong one.
    """

    query: str = dspy.InputField(desc="The user's search query.")
    available_filters: str = dspy.InputField(
        desc=(
            "Available metadata filters, one per line: "
            "'name (type) — e.g.: val1, val2, ...'"
        )
    )
    selected_filters: str = dspy.OutputField(
        desc=(
            "Return ONLY a JSON array of filter names to apply. "
            "Return [] if uncertain or no filter clearly applies. "
            'Example: ["film:location", "film:genre"]'
        )
    )


class ValueSelectionSignature(dspy.Signature):
    """Select specific values for chosen metadata filters.

    You are given the search query and the full allowed-values list for
    each already-selected filter. Select the values that match the query.

    Goal: high recall within each selected filter — include every value that
    could plausibly satisfy the query to avoid false negatives.

    Rules:
    - Copy values EXACTLY as they appear in the allowed list.
      Do not invent, paraphrase, or alter any value.
    - For each filter, include ALL values that plausibly satisfy the query.
      When the query uses alternatives ("A or B"), include values for both A and B.
    - Filter type matters for how values are encoded:
        * bool filters: values are strings (tag names). Return them as strings.
          Example: {"film:genre": {"include": ["comedy", "drama"], "exclude": []}}
        * int filters: values are numbers. Return them as JSON numbers, not strings.
          Example: {"film:release_year": {"include": [1891], "exclude": []}}
    - Use exclude[] ONLY when the query explicitly negates something
      ("not", "without", "excluding", "except"). For int filters, leave exclude empty.
    - If nothing in the allowed list confidently applies, return empty
      include/exclude for that filter.
    """

    query: str = dspy.InputField(desc="The user's search query.")
    filter_schema: str = dspy.InputField(
        desc="JSON object: {filter_name: {type, allowed_values: [...]}}."
    )
    selection: str = dspy.OutputField(
        desc=(
            "Return a JSON object mapping each filter name to "
            "{include: [...], exclude: [...]}. "
            "For bool filters, values are strings. "
            "For int filters, values are JSON numbers. "
            'Example: {"film:genre": {"include": ["comedy"], "exclude": []}, '
            '"film:release_year": {"include": [1988, 2000], "exclude": []}}'
        )
    )

class FilterSelectionModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(FilterSelectionSignature)

    def forward(self, query: str, available_filters: str) -> str:
        result = self._predict(query=query, available_filters=available_filters)
        return getattr(result, "selected_filters", "[]").strip()


class ValueSelectionModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self._predict = dspy.Predict(ValueSelectionSignature)

    def forward(self, query: str, filter_schema: str) -> str:
        result = self._predict(query=query, filter_schema=filter_schema)
        return getattr(result, "selection", "{}").strip()

def _parse_json(raw: str, default: Any) -> Any:
    if not raw:
        return default
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse JSON: %s...", raw[:120])
        return default


@dataclass(frozen=True)
class FilterInfo:
    name: str
    ftype: str
    allowed: List[str]


def _format_available_filters(
    filter_catalog: Dict[str, Any],
    n_samples: int = 5,
) -> str:
    """Format the filter listing for Step 1.

    Each line: 'name (type) — e.g.: val1, val2, ...'
    Sample values are the n_samples most frequent ones.
    """
    lines = []
    for name, info in sorted(filter_catalog.items()):
        ftype = info.get("type", "unknown")
        allowed_raw = info.get("allowed_values", []) or []
        # Values are either dicts {"value": x, "frequency": n} or plain scalars.
        # They are already sorted by frequency descending in build_filter_catalog.
        sample_vals = []
        for v in allowed_raw[:n_samples]:
            sample_vals.append(str(v["value"]) if isinstance(v, dict) else str(v))
        sample_str = ", ".join(sample_vals) if sample_vals else "(no values)"
        lines.append(f"{name} ({ftype}) — e.g.: {sample_str}")
    return "\n".join(lines)


def _extract_filter_infos(
    filter_catalog: Dict[str, Any],
    selected: List[str],
) -> Dict[str, FilterInfo]:
    out: Dict[str, FilterInfo] = {}
    for name in selected:
        info = filter_catalog.get(name) or {}
        ftype = info.get("type", "unknown")
        allowed_raw = info.get("allowed_values", []) or []
        allowed: List[str] = []
        for v in allowed_raw:
            allowed.append(str(v["value"]) if isinstance(v, dict) else str(v))
        out[name] = FilterInfo(name=name, ftype=ftype, allowed=allowed)
    return out


def _format_filter_schema_json(filter_infos: Dict[str, FilterInfo]) -> str:
    obj = {
        fname: {"type": finfo.ftype, "allowed_values": finfo.allowed}
        for fname, finfo in filter_infos.items()
    }
    return json.dumps(obj, ensure_ascii=False)

def _normalize_where(clause: dict) -> Optional[dict]:
    if not clause:
        return None
    for op in ("$and", "$or"):
        if op in clause:
            items = clause[op]
            if isinstance(items, list) and len(items) >= 2:
                return clause
            if isinstance(items, list) and len(items) == 1:
                return items[0]
            return None
    items = [{k: v} for k, v in clause.items()]
    if len(items) == 0:
        return None
    if len(items) == 1:
        return items[0]
    return {"$and": items}


def _compile_bool_tag_filter(
    fname: str,
    include: List[str],
    exclude: List[str],
) -> Optional[dict]:
    """
    Bool tag encoding: {f"{fname}:{value}": True/False}.
    include → OR over tags set True.
    exclude → AND over tags set False (hard constraints).
    """
    parts: List[dict] = []

    if include:
        inc_terms = [{f"{fname}:{v}": True} for v in include]
        parts.append({"$or": inc_terms} if len(inc_terms) >= 2 else inc_terms[0])

    if exclude:
        exc_terms = [{f"{fname}:{v}": False} for v in exclude]
        parts.append({"$and": exc_terms} if len(exc_terms) >= 2 else exc_terms[0])

    if not parts:
        return None
    return {"$and": parts} if len(parts) >= 2 else parts[0]


def _compile_int_filter(fname: str, include: List[Any]) -> Optional[dict]:
    """
    Int scalar encoding: {fname: {"$eq": value}}.
    Multiple values → OR over equality checks.
    """
    terms: List[dict] = []
    for raw in include:
        try:
            val = int(raw)
        except (TypeError, ValueError):
            continue
        terms.append({fname: {"$eq": val}})
    if not terms:
        return None
    return {"$or": terms} if len(terms) >= 2 else terms[0]


def _validate_selection(
    selection: Dict[str, Any],
    filter_infos: Dict[str, FilterInfo],
) -> Dict[str, Dict[str, List[str]]]:
    cleaned: Dict[str, Dict[str, List[str]]] = {}
    if not isinstance(selection, dict):
        return cleaned

    for fname, payload in selection.items():
        if fname not in filter_infos:
            continue
        if not isinstance(payload, dict):
            continue
        inc = payload.get("include", [])
        exc = payload.get("exclude", [])
        if not isinstance(inc, list):
            inc = []
        if not isinstance(exc, list):
            exc = []

        allowed_set = set(filter_infos[fname].allowed)
        inc2 = [str(v) for v in inc if str(v) in allowed_set]
        exc2 = [str(v) for v in exc if str(v) in allowed_set]

        # Remove values that appear in both include and exclude.
        inc_set = set(inc2)
        exc2 = [v for v in exc2 if v not in inc_set]

        if inc2 or exc2:
            cleaned[fname] = {"include": inc2, "exclude": exc2}

    return cleaned


def _infer_top_join_op(query: str) -> str:
    """$or if query offers alternatives; $and if it stacks constraints."""
    q = f" {query.lower()} "
    if " or " in q:
        return "$or"
    if any(tok in q for tok in (" and ", " with ", " from ", " in ", " that are ", " which are ")):
        return "$and"
    return "$or"


def _compile_where(
    query: str,
    cleaned_selection: Dict[str, Dict[str, List[str]]],
    filter_infos: Dict[str, FilterInfo],
) -> Optional[dict]:
    parts: List[dict] = []
    for fname, sel in cleaned_selection.items():
        ftype = filter_infos[fname].ftype
        if ftype == "int":
            clause = _compile_int_filter(fname, sel.get("include", []))
        else:
            clause = _compile_bool_tag_filter(
                fname, sel.get("include", []), sel.get("exclude", [])
            )
        if clause:
            parts.append(clause)

    if not parts:
        return None

    join_op = _infer_top_join_op(query)
    out = {join_op: parts} if len(parts) >= 2 else parts[0]
    return _normalize_where(out)

class LLMQueryPlanner:
    def __init__(self, filter_catalog_path: Union[str, Path]) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment.")

        lm = dspy.LM("openai/gpt-4o", temperature=0.0, max_tokens=4096, api_key=api_key)
        dspy.configure(lm=lm)

        self.filter_catalog = self._load_catalog(filter_catalog_path)
        self._select_filters = FilterSelectionModule()
        self._select_values = ValueSelectionModule()

        # Pre-format the filter listing (stable across all queries).
        self._available_filters_str = _format_available_filters(self.filter_catalog)

    def _load_catalog(self, path: Union[str, Path]) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            logger.warning("Filter catalog not found: %s", p)
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Failed to load filter catalog: %s", e)
            return {}

    def plan(self, query: str) -> Optional[Dict[str, Any]]:
        """Return a ChromaDB where-clause, or None for pure vector search."""
        if not self.filter_catalog:
            return None

        # Step 1: Select which filters apply (conservative / confidence-first).
        raw_selected = self._select_filters(
            query=query,
            available_filters=self._available_filters_str,
        )
        selected = _parse_json(raw_selected, default=[])
        if not isinstance(selected, list):
            selected = []
        # Keep only valid catalog keys.
        selected = [str(x) for x in selected if str(x) in self.filter_catalog]

        if not selected:
            logger.info("No filters selected for: %s", query)
            return None

        logger.info("Selected filters: %s", selected)

        # Step 2: Select values for each chosen filter (recall-oriented).
        filter_infos = _extract_filter_infos(self.filter_catalog, selected)
        schema_str = _format_filter_schema_json(filter_infos)
        raw_values = self._select_values(query=query, filter_schema=schema_str)
        raw_sel = _parse_json(raw_values, default={})
        cleaned_sel = _validate_selection(raw_sel, filter_infos)

        # Step 3: Compile ChromaDB where-clause.
        clause = _compile_where(query, cleaned_sel, filter_infos)

        if clause:
            logger.info("Where clause: %s", json.dumps(clause))
            return clause

        logger.info("No where clause generated (empty after validation).")
        return None
