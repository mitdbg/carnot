"""Base subagent contract, shared utilities, and StepFailed exception."""

from __future__ import annotations

import base64
import os
import re
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skunk.dsl import DocHandle, FormattedString, OpNode, TypedValue

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StepFailed(Exception):
    """Raised when a subagent exhausts all local strategies."""
    def __init__(self, op: str, reason: str):
        super().__init__(f"[{op}] {reason}")
        self.op = op
        self.reason = reason


# ---------------------------------------------------------------------------
# Subagent protocol
# ---------------------------------------------------------------------------

class Subagent:
    """Base class for per-op subagents."""

    op_name: str = ""

    def prewarm(self, op: OpNode, input_desc: str) -> None:
        """Called eagerly with the *description* of the forthcoming input.
        Do prep work here (load pages, draft prompts) before the value arrives.
        Default: no-op."""

    def run(
        self, op: OpNode, prev: DocHandle | TypedValue | None, ctx: "HarnessContext"
    ) -> DocHandle | TypedValue | FormattedString:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# LLM backend (pluggable)
# ---------------------------------------------------------------------------

_GEMINI_MODEL = "gemini-2.5-flash"


def call_gemini(
    system: str,
    user: str,
    images: list[tuple[str, str]] | None = None,  # list of (mime_type, base64_data)
) -> str:
    """Call Gemini and return the text response."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    client = genai.Client(api_key=api_key)

    parts: list[Any] = []
    if images:
        for mime_type, b64_data in images:
            image_bytes = base64.b64decode(b64_data)
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
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
    """Return (mime_type, base64_data) for a PNG/JPEG file."""
    path = Path(path)
    suffix = path.suffix.lower()
    mime = {"png": "image/png", ".png": "image/png",
            "jpg": "image/jpeg", ".jpg": "image/jpeg",
            "jpeg": "image/jpeg", ".jpeg": "image/jpeg"}.get(suffix, "image/png")
    data = path.read_bytes()
    return mime, base64.standard_b64encode(data).decode()


# ---------------------------------------------------------------------------
# Sandboxed Python execution
# ---------------------------------------------------------------------------

_SANDBOX_GLOBALS: dict[str, Any] = {}


def _build_sandbox() -> dict[str, Any]:
    """Build a restricted globals dict for sandboxed exec."""
    import builtins
    allowed_builtins = {
        k: getattr(builtins, k)
        for k in ("abs", "all", "any", "bool", "dict", "enumerate",
                  "filter", "float", "frozenset", "int", "isinstance",
                  "issubclass", "len", "list", "map", "max", "min",
                  "print", "range", "reversed", "round", "set", "slice",
                  "sorted", "str", "sum", "tuple", "type", "zip",
                  "NotImplementedError", "ValueError", "TypeError",
                  "KeyError", "IndexError", "StopIteration", "True", "False", "None")
        if hasattr(builtins, k)
    }
    g: dict[str, Any] = {"__builtins__": allowed_builtins}
    # Add safe scientific packages
    try:
        import math
        g["math"] = math
    except ImportError:
        pass
    try:
        import numpy as np
        g["np"] = np
        g["numpy"] = np
    except ImportError:
        pass
    try:
        import pandas as pd
        g["pd"] = pd
        g["pandas"] = pd
    except ImportError:
        pass
    try:
        import statsmodels
        g["statsmodels"] = statsmodels
    except ImportError:
        pass
    return g


def exec_python(code: str, local_vars: dict[str, Any] | None = None) -> Any:
    """Execute `code` in a sandbox; return the value of `result` local var.

    The code should set a variable named `result`.
    """
    sandbox = _build_sandbox()
    if local_vars:
        sandbox.update(local_vars)
    try:
        exec(compile(code, "<sandbox>", "exec"), sandbox)  # noqa: S102
    except Exception as e:
        raise StepFailed("sandbox", f"Python execution error: {e}\nCode:\n{textwrap.indent(code, '  ')}") from e
    if "result" not in sandbox:
        raise StepFailed("sandbox", f"Code did not set `result`. Code:\n{textwrap.indent(code, '  ')}")
    return sandbox["result"]


# ---------------------------------------------------------------------------
# Typed value parsing from LLM output strings
# ---------------------------------------------------------------------------

def parse_llm_value(raw: str) -> tuple[Any, str, str]:
    """Parse a two-line LLM response into (value, dtype, unit).

    Expected format — no JSON, no labels:
      Line 1: Python literal  — int, float, list of numbers, or "quoted string"
      Line 2: unit string     — e.g. 'fx_rate', 'year', 'usd_millions', 'count'
              (optional; defaults to '')

    dtype is inferred from the Python type of the parsed value.
    """
    import ast as _ast

    lines = [
        l.strip() for l in raw.strip().splitlines()
        if l.strip() and not l.strip().startswith("```")
    ]
    if not lines:
        raise ValueError(f"Empty LLM response")

    value_str = lines[0]
    try:
        value = _ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        # Fallback 1: bare float (possibly with commas)
        try:
            value = float(value_str.replace(",", ""))
        except ValueError:
            # Fallback 2: comma-separated numbers → list
            try:
                parts = [float(p.strip()) for p in value_str.split(",") if p.strip()]
                value = parts[0] if len(parts) == 1 else parts
            except ValueError:
                value = value_str  # treat as text

    unit = lines[1] if len(lines) > 1 else ""

    if isinstance(value, list):
        dtype = "list[scalar]"
    elif isinstance(value, str):
        dtype = "text"
    else:
        dtype = "scalar"

    return value, dtype, unit


# ---------------------------------------------------------------------------
# LLM config and generic call_llm wrapper
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    model: str = "gemini-2.5-flash"
    api_key: str | None = None


def call_llm(
    system: str,
    user: str,
    llm_config: "LLMConfig | None" = None,
    images: list[tuple[str, str]] | None = None,
) -> str:
    """Route an LLM call through the configured backend (currently Gemini)."""
    return call_gemini(system, user, images=images)


# ---------------------------------------------------------------------------
# Harness context (passed through the executor)
# ---------------------------------------------------------------------------

@dataclass
class HarnessContext:
    question: str
    manifest_path: str | None = None   # path to manifest.csv
    cache_dir: str = "cache"           # page render cache (PNG/TXT files)
    concept_dict_path: str | None = None
    llm_config: LLMConfig = None
    golden_handle: "DocHandle | None" = None  # injected by --golden; bypasses retrieve
    cache_only: bool = False           # if True, never make LLM calls

    def __post_init__(self) -> None:
        if self.llm_config is None:
            self.llm_config = LLMConfig()
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
