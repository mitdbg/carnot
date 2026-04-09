"""Auto-detection of available model IDs from API keys in ``llm_config``.

Each trusted provider maps to a tiered list of models spanning a range
of cost/quality trade-offs — from expensive, slow, and powerful models
to cheap, fast, and lower-quality ones.  When the caller does not
supply explicit ``available_model_ids``, the optimizer enumerates all
providers whose API key is present in ``llm_config`` and collects their
tiered models.
"""

from __future__ import annotations

import enum
import logging

logger = logging.getLogger(__name__)


class ModelSize(enum.Enum):
    """Coarse capability tier for a model.

    Representation invariant:
        Values are ``"small"``, ``"medium"``, or ``"large"``.

    Abstraction function:
        Maps to the three tiers used in per-provider model lists and
        quality estimation for agentic operators.
    """

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

# ── Per-provider tiered model lists ──────────────────────────────────
#
# Models are ordered from *most capable / expensive* to
# *least capable / cheapest* within each provider.  The optimizer's
# cost model will assign different per-token costs to each, enabling
# the Pareto search to explore the quality–cost trade-off.

_OPENAI_MODELS: list[str] = [
    # large
    "gpt-5-2025-08-07",
    # medium
    "gpt-5-mini-2025-08-07",
    # small
    "gpt-5-nano-2025-08-07"
]

_ANTHROPIC_MODELS: list[str] = [
    # large
    "anthropic/claude-sonnet-4-5-20250929",
    # medium
    "anthropic/claude-sonnet-4-20250514",
    # small
    "anthropic/claude-haiku-4-5-20250929",
]

_GEMINI_MODELS: list[str] = [
    # large
    "gemini/gemini-3-pro",
    # medium
    "gemini/gemini-3-flash",
    # small
    "gemini/gemini-2.5-flash",
    # small
    "gemini-2.5-flash-lite",
]

_TOGETHER_AI_MODELS: list[str] = [
    # large
    "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    # small
    "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
]

# Maps an ``llm_config`` API-key name to its provider's tiered model
# list.  Iteration order determines priority when selecting a default
# planner model (first match wins).
_API_KEY_TO_MODELS: dict[str, list[str]] = {
    "OPENAI_API_KEY": _OPENAI_MODELS,
    "GEMINI_API_KEY": _GEMINI_MODELS,
    "GOOGLE_API_KEY": _GEMINI_MODELS,
    "ANTHROPIC_API_KEY": _ANTHROPIC_MODELS,
    "TOGETHER_API_KEY": _TOGETHER_AI_MODELS,
}

# ── Model-ID → size mapping ─────────────────────────────────────────
#
# Derived from the inline ``# large / # medium / # small`` annotations
# above.  If a model is absent, ``get_model_size`` falls back to
# ``ModelSize.MEDIUM``.

_MODEL_SIZE: dict[str, ModelSize] = {
    # OpenAI
    "gpt-5-2025-08-07": ModelSize.LARGE,
    "gpt-5-mini-2025-08-07": ModelSize.MEDIUM,
    "gpt-5-nano-2025-08-07": ModelSize.SMALL,
    # Anthropic
    "anthropic/claude-sonnet-4-5-20250929": ModelSize.LARGE,
    "anthropic/claude-sonnet-4-20250514": ModelSize.MEDIUM,
    "anthropic/claude-haiku-4-5-20250929": ModelSize.SMALL,
    # Gemini
    "gemini/gemini-3-pro": ModelSize.LARGE,
    "gemini/gemini-3-flash": ModelSize.MEDIUM,
    "gemini/gemini-2.5-flash": ModelSize.SMALL,
    "gemini-2.5-flash-lite": ModelSize.SMALL,
    # Together AI
    "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": ModelSize.LARGE,
    "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo": ModelSize.SMALL,
}


def get_model_size(model_id: str) -> ModelSize:
    """Return the capability tier for *model_id*.

    Requires:
        - *model_id* is a non-empty string.

    Returns:
        The ``ModelSize`` associated with *model_id* if it appears in the
        known mapping, otherwise ``ModelSize.MEDIUM`` as a safe default.

    Raises:
        None.
    """
    return _MODEL_SIZE.get(model_id, ModelSize.MEDIUM)


def get_api_key_for_model(model_id: str, llm_config: dict) -> str | None:
    """Return the API key from *llm_config* for the provider of *model_id*.

    Requires:
        - *model_id* is a non-empty string.
        - *llm_config* is a dict mapping API-key names to their values.

    Returns:
        The API-key string for the provider that owns *model_id*, or
        ``None`` if *model_id* is not found in any provider's model
        list or the corresponding key is absent/empty in *llm_config*.

    Raises:
        None.
    """
    for key_name, models in _API_KEY_TO_MODELS.items():
        if model_id in models:
            value = llm_config.get(key_name)
            if value:
                return value
    return None


def get_best_available_model_id(llm_config: dict) -> str | None:
    """Return the highest-quality model ID available in *llm_config*.

    Models are ordered first by size (large > medium > small) and then
    by provider priority (OpenAI > Gemini > Anthropic > Together AI).

    Requires:
        - *llm_config* is a dict mapping API-key names to their values.

    Returns:
        The model-ID string of the most capable model whose provider key
        is present and non-empty, or ``None`` if no recognised keys are
        found.

    Raises:
        None.
    """
    available = get_available_model_ids(llm_config)
    if not available:
        return None

    # available is already ordered provider-by-provider from most to
    # least capable, so among all available models the first ``LARGE``
    # (then ``MEDIUM``, then ``SMALL``) is the best.
    size_priority = {ModelSize.LARGE: 0, ModelSize.MEDIUM: 1, ModelSize.SMALL: 2}
    return min(available, key=lambda mid: size_priority.get(get_model_size(mid), 1))


def get_available_model_ids(llm_config: dict) -> list[str]:
    """Return all tiered model IDs whose provider API key is set in *llm_config*.

    Requires:
        - *llm_config* is a dict mapping API-key names to their values
          (e.g. ``{"OPENAI_API_KEY": "sk-...", "GEMINI_API_KEY": "AIza..."}``).

    Returns:
        A de-duplicated list of model-ID strings aggregated from every
        provider whose key is present and non-empty in *llm_config*.
        Models are ordered provider-by-provider (OpenAI first, then
        Anthropic, Gemini, Together AI) and within each provider from
        most capable to cheapest.  Returns an empty list if no
        recognised API keys are found.

    Raises:
        None.
    """
    model_ids: list[str] = []
    seen: set[str] = set()

    for key_name, models in _API_KEY_TO_MODELS.items():
        api_key = llm_config.get(key_name)
        if api_key:
            for mid in models:
                if mid not in seen:
                    model_ids.append(mid)
                    seen.add(mid)

    if not model_ids:
        logger.warning(
            "No recognised API keys found in llm_config; "
            "available_model_ids will be empty."
        )

    return model_ids
