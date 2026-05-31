"""Sandbox for operator-generated Python code.

A thin wrapper around the smolagents-derived `LocalPythonExecutor`
(`src/skunk/search_agent/utils/local_python_executor.py`) so `compute` and
`lookup_external` get the same actual sandboxing the search agent loop uses:

- AST-walked evaluation (no raw `exec()` of attacker-controlled strings).
- Allowed-import list — `numpy`, `pandas`, `statsmodels`, plus the
  smolagents `BASE_BUILTIN_MODULES` (math, statistics, datetime, etc.).
- Dangerous-module blocklist — `os`, `subprocess`, `socket`, `sys`,
  `pathlib`, `io`, `multiprocessing`, `pty`, `shutil`, `builtins`.
- Dangerous-function blocklist — `exec`, `eval`, `compile`, `globals`,
  `locals`, `__import__`, `os.system`, `os.popen`.
- Dunder access is rejected (no `obj.__class__`, no `obj.__dict__`).

Limitation: this is not a hardened sandbox. The wrapped modules
(numpy, pandas, statsmodels) themselves run as trusted Python — if the
attacker can route execution into them with the right arguments, they
can still touch the host process. Real isolation (subprocess + seccomp,
WASM, container) is future work; this raises the bar without claiming
to close it.

Public API (unchanged from the previous in-process `exec()`-based
implementation, so call-sites in `compute.py` and `lookup_external.py`
need no edits):

- `exec_python_with_env(code, local_vars=None)` — runs `code`, expects
  it to assign `result`, returns `(state, state["result"])`.
- `exec_python_capture_stdout(code, local_vars=None)` — runs `code`
  with stdout captured; returns `(state, stdout)`. Callable entries in
  `local_vars` route to `send_tools()` (sandbox-immutable); data
  entries route to `send_variables()`.
- `strip_code_fences(code)` — removes a leading ```[lang] fence and a
  trailing ``` fence; idempotent on already-stripped input.
"""

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
from skunk.search_agent.utils import InterpreterError, LocalPythonExecutor

# Modules user-generated code may `import` (in addition to smolagents'
# BASE_BUILTIN_MODULES). The same names are also pre-injected as global
# bindings (see `_PRELOADED_GLOBALS`) so code that omits the import
# statement also works — matches the contract of the prior pyexec
# `_default_env()` and what the compute / lookup_external system prompts
# advertise as "already in scope".
_AUTHORIZED_IMPORTS = ["json", "numpy", "numpy.*", "pandas", "statsmodels", "statsmodels.*"]

_PRELOADED_GLOBALS: dict[str, Any] = {
    "math": math,
    "statistics": statistics,
    "datetime": datetime,
    "np": np, "numpy": np,
    "pd": pd, "pandas": pd,
    "sm": sm, "statsmodels": sm,
}

# Strip only a leading ```[lang] fence and a trailing ``` fence. Never strips
# inline-string lines, even if they happen to start with ```.
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
    """Build a fresh sandboxed executor with the operator preload set.

    Fresh state per call (no leakage between attempts). `send_tools` is
    called even when `extra_tools` is empty because `LocalPythonExecutor`
    leaves `static_tools=None` until that first call and `__call__`
    requires it populated.
    """
    ex = LocalPythonExecutor(additional_authorized_imports=_AUTHORIZED_IMPORTS)
    ex.send_tools(extra_tools or {})
    ex.send_variables({**_PRELOADED_GLOBALS, **extra_vars})
    return ex


def exec_python_with_env(
    code: str, local_vars: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Exec `code` and return both the post-exec state and state["result"].

    Useful when the caller needs auxiliary variables the code may have set
    (e.g. `result_unit`, `result_kind`) alongside the primary `result`.
    """
    code = strip_code_fences(code)
    ex = _new_executor(local_vars or {})
    try:
        ex(code)
    except InterpreterError as e:
        raise StepFailed("pyexec", str(e)) from e
    if "result" not in ex.state:
        raise StepFailed("pyexec", f"Code did not set `result`:\n{code}")
    return ex.state, ex.state["result"]


def exec_python_capture_stdout(
    code: str, local_vars: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Exec `code` with print() output captured; returns (state, stdout).

    Used by operators whose contract is "call helpers and print() the
    answer" (e.g. `lookup_external`) rather than "assign to `result`"
    (`compute`). Callable entries in `local_vars` are routed through
    `send_tools()` so the sandboxed code cannot rebind them; data
    entries flow through `send_variables()`.
    """
    code = strip_code_fences(code)
    tools: dict[str, Any] = {}
    vars_: dict[str, Any] = {}
    for k, v in (local_vars or {}).items():
        (tools if callable(v) else vars_)[k] = v
    ex = _new_executor(vars_, extra_tools=tools)
    try:
        out = ex(code)
    except InterpreterError as e:
        raise StepFailed("pyexec", str(e)) from e
    return ex.state, str(out.logs)
