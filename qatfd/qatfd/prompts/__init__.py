"""QATFD prompt templates, stored as YAML alongside this module.

Each `.yaml` maps a prompt key -> a template string. Templates use Jinja `{{ var }}`
placeholders and are rendered by the module / class that owns them (see e.g. `SearchAgent`).
Files are read via `importlib.resources`, so this works from any CWD and from an installed
wheel (the YAMLs ship as package data — see pyproject `[tool.setuptools.package-data]`).
Results are cached: templates are static.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

import yaml


@lru_cache(maxsize=None)
def load_qatfd_prompts(name: str) -> dict[str, str]:
    """Return the `{key: template}` map from `skunk/prompts/<name>.yaml` (cached)."""
    text = resources.files(__name__).joinpath(f"{name}.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"prompts/{name}.yaml must be a mapping of prompt-key -> template")
    return data
