"""Sandbox for operator-generated Python code: a thin wrapper around the
smolagents-derived `LocalPythonExecutor` so `compute` and `lookup_external` get the
same AST-walked evaluation (import/module/function blocklists, no dunder access) the
search agent uses.

Not a hardened sandbox: the wrapped modules (numpy, pandas, statsmodels) run as
trusted Python, so a determined caller can still reach the host. Real isolation
(subprocess + seccomp, WASM, container) is future work."""

from __future__ import annotations

import datetime
import math
import re
import statistics
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from skunk.errors import StepFailed
from skunk.local_python_executor import InterpreterError, LocalPythonExecutor

# Modules code may `import`; also pre-injected as globals (`_PRELOADED_GLOBALS`) so
# code that omits the import still works — matches what the system prompts advertise.
_AUTHORIZED_IMPORTS = ["json", "numpy", "numpy.*", "pandas", "statsmodels", "statsmodels.*"]

_PRELOADED_GLOBALS: dict[str, Any] = {
    "math": math,
    "statistics": statistics,
    "datetime": datetime,
    "np": np, "numpy": np,
    "pd": pd, "pandas": pd,
    "sm": sm, "statsmodels": sm,
}

# Strip only a leading ```[lang] fence and a trailing ``` fence (not inline lines).
_LEADING_FENCE_RE = re.compile(r"\A```[a-zA-Z]*[ \t]*\n")
_TRAILING_FENCE_RE = re.compile(r"\n```[ \t]*\Z")


def strip_code_fences(code: str) -> str:
    s = code.strip()
    s = _LEADING_FENCE_RE.sub("", s)
    s = _TRAILING_FENCE_RE.sub("", s)
    return s.strip()


def _new_executor(
    extra_vars: dict[str, Any],
    extra_tools: dict[str, Any] | None = None,
) -> LocalPythonExecutor:
    """Build a fresh sandboxed executor with the operator preload set (fresh state per
    call). `send_tools` is called even when empty — `__call__` requires `static_tools`
    populated, which only happens on that first call."""
    ex = LocalPythonExecutor(additional_authorized_imports=_AUTHORIZED_IMPORTS)
    ex.send_tools(extra_tools or {})
    ex.send_variables({**_PRELOADED_GLOBALS, **extra_vars})
    return ex


def exec_python_with_env(
    code: str, local_vars: dict[str, Any] | None = None, *, require_result: bool = True,
) -> tuple[dict[str, Any], Any]:
    """Exec `code` and return `(post-exec state, state.get("result"))`. The state lets the
    caller read auxiliary variables the code set alongside `result`. `require_result=False`
    tolerates code that sets no `result` (compute's partial-progress form, where the code
    sets `missing`/`committed`/`keep` instead)."""
    code = strip_code_fences(code)
    ex = _new_executor(local_vars or {})
    try:
        ex(code)
    except InterpreterError as e:
        raise StepFailed("pyexec", str(e)) from e
    if require_result and "result" not in ex.state:
        raise StepFailed("pyexec", f"Code did not set `result`:\n{code}")
    return ex.state, ex.state.get("result")


