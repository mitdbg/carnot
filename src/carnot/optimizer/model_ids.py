"""Auto-detection of available model IDs from API keys in ``llm_config``.

Each trusted provider maps to a tiered list of models spanning a range
of cost/quality trade-offs — from expensive, slow, and powerful models
to cheap, fast, and lower-quality ones.  When the caller does not
supply explicit ``available_model_ids``, the optimizer enumerates all
providers whose API key is present in ``llm_config`` and collects their
tiered models.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Per-provider tiered model lists ──────────────────────────────────
#
# Models are ordered from *most capable / expensive* to
# *least capable / cheapest* within each provider.  The optimizer's
# cost model will assign different per-token costs to each, enabling
# the Pareto search to explore the quality–cost trade-off.

_OPENAI_MODELS: list[str] = [
    "gpt-5-2025-08-07",
    # "gpt-5-mini-2025-08-07",
    # "gpt-5-nano-2025-08-07"
]

_ANTHROPIC_MODELS: list[str] = [
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-sonnet-4-5-20250929",
    "anthropic/claude-haiku-4-5-20250929",
]

_GEMINI_MODELS: list[str] = [
    "gemini/gemini-3-pro",
    "gemini/gemini-3-flash",
    "gemini/gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

_TOGETHER_AI_MODELS: list[str] = [
    "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
]

# Maps an ``llm_config`` API-key name to its provider's tiered model
# list.  Iteration order determines priority when selecting a default
# planner model (first match wins).
_API_KEY_TO_MODELS: dict[str, list[str]] = {
    "OPENAI_API_KEY": _OPENAI_MODELS,
    "ANTHROPIC_API_KEY": _ANTHROPIC_MODELS,
    "GEMINI_API_KEY": _GEMINI_MODELS,
    "GOOGLE_API_KEY": _GEMINI_MODELS,
    "TOGETHER_API_KEY": _TOGETHER_AI_MODELS,
}


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
