"""Load skunk's `.env` BEFORE importing skunk, so `LLMClient` (and the embedding
clients) see the API keys / config. Mirrors `eval/eval_e2e.py:_load_env`.

Entry points must call `load_env()` as their first action, before any `import skunk`.
"""

from __future__ import annotations

import os
from pathlib import Path

from qatfd.paths import SKUNK_DIR


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            # setdefault: a value already in the real environment wins over .env.
            os.environ.setdefault(k.strip(), v.strip())


def load_env() -> None:
    """Load skunk/.env (the one the skunk stack is configured against)."""
    _load_env_file(SKUNK_DIR / ".env")
