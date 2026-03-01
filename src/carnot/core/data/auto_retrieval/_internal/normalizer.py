from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Set

from _internal.type_utils import is_str_type


def normalize_value(raw: str) -> str:
    """Casefold, NFC-normalize, collapse whitespace, strip articles/trailing dots."""
    s = raw.strip()
    s = unicodedata.normalize("NFC", s)
    s = s.casefold()
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\s+", " ", s).strip()
    if s.startswith("the "):
        s = s[4:]
    if s.endswith("."):
        s = s[:-1].rstrip()
    return s


def normalize_results(
    results: Dict[str, Dict[str, Any]],
    concept_schema_cols: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Normalize all string-typed values: casefold, dedup lists, drop empties."""
    str_keys: Set[str] = set()
    for col in concept_schema_cols:
        if is_str_type(col["type"]):
            str_keys.add(col["name"])

    out: Dict[str, Dict[str, Any]] = {}
    for doc_id, doc_vals in results.items():
        new_doc: Dict[str, Any] = {}
        for key, val in doc_vals.items():
            if key not in str_keys:
                new_doc[key] = val
                continue
            if isinstance(val, list):
                normed = []
                seen: Set[str] = set()
                for v in val:
                    if v is None:
                        continue
                    n = normalize_value(str(v))
                    if n and n not in seen:
                        seen.add(n)
                        normed.append(n)
                if normed:
                    new_doc[key] = normed
            elif val is not None:
                n = normalize_value(str(val))
                if n:
                    new_doc[key] = n
        out[doc_id] = new_doc
    return out
