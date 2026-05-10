"""Shared utilities for subagents: exception, LLM call, exec, context."""

from __future__ import annotations

import ast
import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class StepFailed(Exception):
    def __init__(self, op: str, reason: str):
        super().__init__(f"[{op}] {reason}")
        self.op = op
        self.reason = reason


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------

_GEMINI_MODEL = "gemini-2.5-flash"


def call_gemini(
    system: str,
    user: str,
    images: list[tuple[str, str]] | None = None,
) -> str:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)

    parts: list[Any] = []
    if images:
        for mime_type, b64_data in images:
            parts.append(types.Part.from_bytes(data=base64.b64decode(b64_data), mime_type=mime_type))
    parts.append(types.Part.from_text(text=user))

    resp = client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=8192,
            temperature=0.0,
        ),
    )
    return (resp.text or "").strip()


def load_image_b64(path: str | Path) -> tuple[str, str]:
    path = Path(path)
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
        path.suffix.lower(), "image/png"
    )
    return mime, base64.standard_b64encode(path.read_bytes()).decode()


# ---------------------------------------------------------------------------
# Python execution
# ---------------------------------------------------------------------------

def exec_python(code: str, local_vars: dict[str, Any] | None = None) -> Any:
    import math
    import numpy as np
    import pandas as pd

    code = code.strip()
    if code.startswith("```"):
        code = "\n".join(l for l in code.splitlines() if not l.strip().startswith("```")).strip()

    env: dict[str, Any] = {"math": math, "np": np, "numpy": np, "pd": pd, "pandas": pd}
    if local_vars:
        env.update(local_vars)
    exec(compile(code, "<sandbox>", "exec"), env)  # noqa: S102
    if "result" not in env:
        raise StepFailed("sandbox", f"Code did not set `result`:\n{code}")
    return env["result"]


# ---------------------------------------------------------------------------
# LLM value parsing
# ---------------------------------------------------------------------------

def parse_llm_value(raw: str) -> tuple[Any, str, str]:
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip() and not l.strip().startswith("```")]
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


# ---------------------------------------------------------------------------
# Harness context
# ---------------------------------------------------------------------------

@dataclass
class HarnessContext:
    question: str
    manifest_path: str | None = None
    cache_dir: str = "cache"
    golden_handle: "DocHandle | None" = None  # noqa: F821
    cache_only: bool = False

    def __post_init__(self) -> None:
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
