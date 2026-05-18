"""Shared helpers used by multiple page-index modules.

Kept deliberately small — only utilities that would otherwise be
duplicated across stages or corpora.
"""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)


def strip_code_fence(s: str) -> str:
    """Strip a leading ```lang and trailing ``` markdown fence from an
    LLM response. Idempotent on un-fenced text."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def safe_json_loads(text: str, *, context: str = "") -> dict | list | None:
    """Parse a possibly-fenced JSON blob from an LLM. Returns None on
    parse failure with a logged warning — callers treat None as
    "skip this batch" rather than raising.
    """
    try:
        return json.loads(strip_code_fence(text))
    except (json.JSONDecodeError, TypeError) as e:
        ctx = f" [{context}]" if context else ""
        log.warning("safe_json_loads%s failed: %s; head=%r", ctx, e, text[:120])
        return None
