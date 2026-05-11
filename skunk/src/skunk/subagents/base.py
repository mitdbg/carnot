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

_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Transient-failure retry tunables. Gemini 2.5 Flash returns 503 under load and 429
# under per-minute quota; both are recoverable by waiting. Retries are bounded so a
# fully-down API still fails the step cleanly (orchestrator records it in the trace).
_RETRY_DELAY_S = float(os.environ.get("SKUNK_GEMINI_RETRY_DELAY", "30"))
_MAX_RETRIES = int(os.environ.get("SKUNK_GEMINI_MAX_RETRIES", "5"))

_gemini_client: "genai.Client | None" = None
_vertex_credentials_json: str | None = None


def _use_vertex() -> bool:
    return os.environ.get("SKUNK_USE_VERTEX", "").lower() in ("1", "true", "yes") \
        or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")


def _get_gemini_client() -> "genai.Client":
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (and SKUNK_USE_VERTEX not enabled)")
    _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _get_vertex_credentials_json() -> str | None:
    """Load the ADC / service-account JSON file from GOOGLE_APPLICATION_CREDENTIALS
    once, and return its contents as a JSON string for litellm. Returns None to
    let litellm fall back to ambient ADC if no file path is set."""
    global _vertex_credentials_json
    if _vertex_credentials_json is not None:
        return _vertex_credentials_json
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path:
        return None
    import json as _json  # noqa: PLC0415
    with open(cred_path) as f:
        _vertex_credentials_json = _json.dumps(_json.load(f))
    return _vertex_credentials_json


def _is_transient_litellm(exc: Exception) -> bool:
    code = getattr(exc, "status_code", None)
    return code in (429, 500, 503, 504)


def _is_transient(exc: genai_errors.APIError) -> bool:
    code = getattr(exc, "code", None)
    return code in (429, 500, 503, 504)


def _call_via_litellm(system: str, user: str, images: list[tuple[str, str]] | None,
                      temperature: float, model: str) -> str:
    """Vertex backend via litellm. Model id should be prefixed with `vertex_ai/`
    if not already (e.g. `vertex_ai/gemini-2.5-flash-preview-09-2025`)."""
    from litellm import completion  # noqa: PLC0415
    import litellm.exceptions as _lexc  # noqa: PLC0415

    if not model.startswith("vertex_ai/"):
        model = f"vertex_ai/{model}"

    user_content: list[dict[str, Any]] | str
    if images:
        user_content = []
        for mime_type, b64_data in images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
            })
        user_content.append({"type": "text", "text": user})
    else:
        user_content = user

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 8192,
    }
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    if project:
        kwargs["vertex_project"] = project
    if location:
        kwargs["vertex_location"] = location
    creds_json = _get_vertex_credentials_json()
    if creds_json is not None:
        kwargs["vertex_credentials"] = creds_json

    for attempt in range(1, _MAX_RETRIES + 2):
        try:
            resp = completion(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except _lexc.APIError as e:
            if not _is_transient_litellm(e) or attempt > _MAX_RETRIES:
                raise
            print(
                f"[call_gemini/vertex] {getattr(e, 'status_code', '?')} from Vertex, "
                f"sleeping {_RETRY_DELAY_S}s then retrying (attempt {attempt}/{_MAX_RETRIES})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(_RETRY_DELAY_S)

    raise RuntimeError("unreachable: retry loop fell through")


def call_gemini(
    system: str,
    user: str,
    images: list[tuple[str, str]] | None = None,
    temperature: float = 0.0,
) -> str:
    model = os.environ.get("SKUNK_GEMINI_MODEL", _DEFAULT_GEMINI_MODEL)

    if _use_vertex():
        return _call_via_litellm(system, user, images, temperature, model)

    client = _get_gemini_client()

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
            resp = client.models.generate_content(model=model, contents=parts, config=config)
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
