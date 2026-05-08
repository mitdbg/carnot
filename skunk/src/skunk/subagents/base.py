"""Base subagent contract, shared utilities, and StepFailed exception."""

from __future__ import annotations

import base64
import json
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

@dataclass
class LLMConfig:
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    temperature: float = 0.0


def _get_client() -> Any:
    """Return an Anthropic client (lazy import)."""
    try:
        import anthropic
        return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    except ImportError as e:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic") from e


def call_llm(
    system: str,
    user: str,
    config: LLMConfig | None = None,
    images: list[tuple[str, str]] | None = None,  # list of (mime_type, base64_data)
) -> str:
    """Call Claude and return the text response."""
    cfg = config or LLMConfig()
    client = _get_client()

    content: list[Any] = []
    if images:
        for mime_type, b64_data in images:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime_type, "data": b64_data},
            })
    content.append({"type": "text", "text": user})

    resp = client.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text


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
# JSON extraction from LLM responses
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Any:
    """Extract the first JSON block from an LLM response."""
    # Try ```json ... ``` block first
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Try bare JSON object/array
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    raise ValueError(f"No JSON found in: {text[:200]!r}")


# ---------------------------------------------------------------------------
# Harness context (passed through the executor)
# ---------------------------------------------------------------------------

@dataclass
class HarnessContext:
    question: str
    manifest_path: str | None = None   # path to manifest.csv
    cache_dir: str = "cache"
    concept_dict_path: str | None = None
    llm_config: LLMConfig = None
    cache_only: bool = False           # True → never hit live sources
    golden_handle: "DocHandle | None" = None  # injected by --golden; bypasses retrieve

    def __post_init__(self) -> None:
        if self.llm_config is None:
            self.llm_config = LLMConfig()
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
