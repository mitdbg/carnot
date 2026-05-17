"""Shared operator surface: OpNode dispatch envelope, exceptions, code execution, value parsing.

The 4 operators (`retrieve`, `extract`, `lookup_external`, `compute`) are the
DSL's query-plan nodes; the orchestrator translates `Branch` AST nodes into
`OpNode(op, args)` calls and dispatches to the matching operator. Operators
read `op.args[...]` by key and return their output. `OperatorFn` is the
protocol every registered operator must satisfy. Failure modes are
`StepFailed` (terminal) and `MissingData` (triggers an orchestrator replan).
"""

from __future__ import annotations

import ast
import base64
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd
import statsmodels.api as sm

if TYPE_CHECKING:
    from skunk.common import HarnessContext


# ---------------------------------------------------------------------------
# Operator dispatch envelope
# ---------------------------------------------------------------------------

VALID_OPS = frozenset({"retrieve", "extract", "lookup_external", "compute"})


@dataclass
class OpNode:
    """The orchestrator-to-operator call wrapper. `op` selects the operator;
    `args` carries the call's keyword arguments. Operators read `op.args[...]`
    by key."""
    op: str
    args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.op not in VALID_OPS:
            raise ValueError(f"Unknown op: {self.op!r}. Must be one of {sorted(VALID_OPS)}")


class OperatorFn(Protocol):
    """Contract for every registered operator callable: fn(op, prev, ctx) -> result."""
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
    """Exec `code` in a sandboxed env. Returns env["result"] or raises."""
    _, result = exec_python_with_env(code, local_vars)
    return result

def _hp_filter(y: "np.ndarray | list", lamb: float = 1600) -> "tuple[np.ndarray, np.ndarray]":
    """Hodrick-Prescott filter implemented in pure numpy.

    Returns (cycle, trend). Use lamb=100 for annual data, 1600 for quarterly,
    14400 for monthly.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    A = np.eye(n) + lamb * D.T @ D
    trend = np.linalg.solve(A, y)
    return y - trend, trend

def exec_python_with_env(
    code: str, local_vars: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Exec `code` and return both the post-exec env and env["result"].

    Useful when the caller needs auxiliary variables the code may have set
    (e.g. `result_unit`, `result_kind`) alongside the primary `result`.
    """
    code = strip_code_fences(code)
    env: dict[str, Any] = {
        "math": math,
        "np": np, "numpy": np,
        "pd": pd, "pandas": pd,
        "sm": sm, "statsmodels": sm,
        "hp_filter": _hp_filter,
    }
    env.update(local_vars or {})
    exec(compile(code, "<sandbox>", "exec"), env)  # noqa: S102
    if "result" not in env:
        raise StepFailed("sandbox", f"Code did not set `result`:\n{code}")
    return env, env["result"]

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
