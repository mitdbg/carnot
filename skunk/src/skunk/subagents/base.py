"""Shared utilities for subagents: exception, LLM call, exec, context."""

from __future__ import annotations

import ast
import base64
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from google import genai
from google.genai import errors as genai_errors
from google.genai import types


class StepFailed(Exception):
    def __init__(self, op: str, reason: str):
        super().__init__(f"[{op}] {reason}")
        self.op = op
        self.reason = reason


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------

_GEMINI_MODEL = "gemini-2.5-flash"

# Transient-failure retry tunables. Gemini 2.5 Flash returns 503 under load and 429
# under per-minute quota; both are recoverable by waiting. Retries are bounded so a
# fully-down API still fails the step cleanly (orchestrator records it in the trace).
_RETRY_DELAY_S = float(os.environ.get("SKUNK_GEMINI_RETRY_DELAY", "30"))
_MAX_RETRIES = int(os.environ.get("SKUNK_GEMINI_MAX_RETRIES", "5"))


def _is_transient(exc: genai_errors.APIError) -> bool:
    code = getattr(exc, "code", None)
    return code in (429, 500, 503, 504)


def call_gemini(
    system: str,
    user: str,
    images: list[tuple[str, str]] | None = None,
    temperature: float = 0.0,
) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)

    parts: list[Any] = []
    if images:
        for mime_type, b64_data in images:
            parts.append(types.Part.from_bytes(data=base64.b64decode(b64_data), mime_type=mime_type))
    parts.append(types.Part.from_text(text=user))

    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=8192,
        temperature=temperature,
    )

    for attempt in range(1, _MAX_RETRIES + 2):
        try:
            resp = client.models.generate_content(model=_GEMINI_MODEL, contents=parts, config=config)
            return (resp.text or "").strip()
        except genai_errors.APIError as e:
            if not _is_transient(e) or attempt > _MAX_RETRIES:
                raise
            print(
                f"[call_gemini] {e.code} from Gemini ({e.status or 'transient'}), "
                f"sleeping {_RETRY_DELAY_S}s then retrying (attempt {attempt}/{_MAX_RETRIES})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(_RETRY_DELAY_S)

    raise RuntimeError("unreachable: retry loop fell through")


def load_image_b64(path: str | Path) -> tuple[str, str]:
    path = Path(path)
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
        path.suffix.lower(), "image/png"
    )
    return mime, base64.standard_b64encode(path.read_bytes()).decode()


# ---------------------------------------------------------------------------
# Python execution
# ---------------------------------------------------------------------------

# Strip only a leading ```[lang] fence and a trailing ``` fence. Never strips
# inline-string lines, even if they happen to start with ```.
_LEADING_FENCE_RE = re.compile(r"\A```[a-zA-Z]*[ \t]*\n")
_TRAILING_FENCE_RE = re.compile(r"\n```[ \t]*\Z")


def strip_code_fences(code: str) -> str:
    s = code.strip()
    s = _LEADING_FENCE_RE.sub("", s)
    s = _TRAILING_FENCE_RE.sub("", s)
    return s.strip()


def exec_python(code: str, local_vars: dict[str, Any] | None = None) -> Any:
    code = strip_code_fences(code)

    env: dict[str, Any] = {
        "math": math,
        "np": np, "numpy": np,
        "pd": pd, "pandas": pd,
        "sm": sm, "statsmodels": sm,
    }
    env.update(local_vars or {})
    exec(compile(code, "<sandbox>", "exec"), env)  # noqa: S102
    if "result" not in env:
        raise StepFailed("sandbox", f"Code did not set `result`:\n{code}")
    return env["result"]


# ---------------------------------------------------------------------------
# LLM value parsing
# ---------------------------------------------------------------------------

def parse_llm_value(raw: str) -> tuple[Any, str, str]:
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip() and not ln.strip().startswith("```")]
    if not lines:
        raise ValueError("Empty LLM response")
    value = ast.literal_eval(lines[0])
    unit = lines[1] if len(lines) > 1 else ""
    if isinstance(value, list):
        dtype = "list[scalar]"
    elif isinstance(value, str):
        dtype = "text"
    else:
        dtype = "scalar"
    return value, dtype, unit
