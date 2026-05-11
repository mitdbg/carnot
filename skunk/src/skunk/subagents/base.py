"""Shared utilities for subagents: exception, code execution, value parsing."""

from __future__ import annotations

import ast
import base64
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd
import statsmodels.api as sm

if TYPE_CHECKING:
    from skunk.common.context import HarnessContext
    from skunk.dsl import OpNode


class SubagentFn(Protocol):
    """Contract for every registered subagent callable: fn(op, prev, ctx) -> result."""
    def __call__(self, op: OpNode, prev: Any, ctx: HarnessContext) -> Any: ...


class StepFailed(Exception):
    def __init__(self, op: str, reason: str):
        super().__init__(f"[{op}] {reason}")
        self.op = op
        self.reason = reason


class MissingData(Exception):
    """Compute determined its input is insufficient. The orchestrator catches this
    and may run one recovery round (fire an additional branch) before failing."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

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

def parse_llm_value(raw: str) -> tuple[Any, str]:
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip() and not ln.strip().startswith("```")]
    if not lines:
        raise ValueError("Empty LLM response")
    value = ast.literal_eval(lines[0])
    unit = lines[1] if len(lines) > 1 else ""
    return value, unit
