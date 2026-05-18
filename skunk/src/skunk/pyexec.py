"""In-process Python exec for operator-generated code.

`exec_python_with_env` runs a snippet in an env preloaded with
numpy/pandas/statsmodels; callers pass in their own local variables.
`strip_code_fences` removes a leading/trailing markdown fence.

NOTE: this is not a security boundary. The snippet runs in the host
process with full builtins, imports, filesystem, and network access.
Real isolation (subprocess + seccomp, WASM, container) is future work.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from skunk.errors import StepFailed

# TODO: Should probably add a minimal sandbox / merge with search agent code execution sandbox

# Strip only a leading ```[lang] fence and a trailing ``` fence. Never strips
# inline-string lines, even if they happen to start with ```.
_LEADING_FENCE_RE = re.compile(r"\A```[a-zA-Z]*[ \t]*\n")
_TRAILING_FENCE_RE = re.compile(r"\n```[ \t]*\Z")


def strip_code_fences(code: str) -> str:
    s = code.strip()
    s = _LEADING_FENCE_RE.sub("", s)
    s = _TRAILING_FENCE_RE.sub("", s)
    return s.strip()


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
    }
    env.update(local_vars or {})
    exec(compile(code, "<pyexec>", "exec"), env)  # noqa: S102
    if "result" not in env:
        raise StepFailed("pyexec", f"Code did not set `result`:\n{code}")
    return env, env["result"]
