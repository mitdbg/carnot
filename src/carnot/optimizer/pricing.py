"""Model pricing lookup for the cost-based optimizer.

Resolves LLM model identifiers to their per-token dollar costs using
``litellm.model_cost`` as the authoritative data source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import litellm

logger = logging.getLogger(__name__)

# Providers whose pricing we trust.  Entries from other providers
# (e.g. ``vertex_ai``, ``databricks``) are ignored to avoid
# accidentally using reseller pricing for first-party models.
_TRUSTED_PROVIDERS = frozenset({"openai", "anthropic", "gemini", "together_ai"})


@dataclass(frozen=True)
class ModelPricing:
    """Per-token costs for a single model.

    Representation invariant:
        - All float fields are non-negative.
        - ``max_input_tokens >= 0`` and ``max_output_tokens >= 0``.

    Abstraction function:
        The dollar cost of processing one token of each modality
        through this model, together with context-window limits.
    """

    input_cost_per_token: float
    output_cost_per_token: float
    max_input_tokens: int
    max_output_tokens: int


class ModelPricingLookup:
    """Resolves a model identifier to its per-token pricing.

    Uses ``litellm.model_cost`` as the authoritative source.  Falls
    back to a configurable default when a model is not found.

    Representation invariant:
        - ``_cache`` keys are model identifier strings.
        - Every value in ``_cache`` is a ``ModelPricing`` instance.

    Abstraction function:
        A memoized mapping from model identifiers (as used by
        operators' ``model_id`` fields) to ``ModelPricing`` objects.
    """

    def __init__(self, default_pricing: ModelPricing | None = None):
        """Construct a ``ModelPricingLookup``.

        Requires:
            None.

        Returns:
            A new ``ModelPricingLookup`` instance.

        Raises:
            None.
        """
        self._default = default_pricing
        self._cache: dict[str, ModelPricing] = {}

    def get(self, model_id: str) -> ModelPricing:
        """Return the pricing for *model_id*.

        Requires:
            - *model_id* is a non-empty string in the same format
              used by physical operators (e.g. ``"gpt-4o"``,
              ``"gemini/gemini-2.0-flash"``).

        Returns:
            A ``ModelPricing`` for the model.  Uses a cached value
            if available, otherwise looks up ``litellm.model_cost``
            and caches the result.  Falls back to ``default_pricing``
            if the model is not found.

        Raises:
            ValueError: If the model is not found and no default
            pricing was provided.
        """
        if model_id in self._cache:
            return self._cache[model_id]

        pricing = self._lookup(model_id)
        self._cache[model_id] = pricing
        return pricing

    def _lookup(self, model_id: str) -> ModelPricing:
        """Look up *model_id* in ``litellm.model_cost``.

        Requires:
            - *model_id* is a non-empty string.

        Returns:
            A ``ModelPricing`` for the model, or the default pricing
            if the model is not found.

        Raises:
            ValueError: If the model is not found and no default
            pricing was provided.
        """
        entry = litellm.model_cost.get(model_id)
        if entry is not None:
            provider = entry.get("litellm_provider", "")
            if provider in _TRUSTED_PROVIDERS:
                return self._entry_to_pricing(entry)
            else:
                logger.warning(
                    "Model '%s' found in litellm.model_cost but provider '%s' "
                    "is not in trusted set %s; ignoring.",
                    model_id,
                    provider,
                    _TRUSTED_PROVIDERS,
                )

        if self._default is not None:
            logger.debug(
                "Model '%s' not found in litellm.model_cost; using default pricing.",
                model_id,
            )
            return self._default

        raise ValueError(
            f"Model '{model_id}' not found in litellm.model_cost and no "
            f"default pricing was provided."
        )

    @staticmethod
    def _entry_to_pricing(entry: dict) -> ModelPricing:
        """Convert a ``litellm.model_cost`` entry dict to a ``ModelPricing``.

        Requires:
            - *entry* is a dict from ``litellm.model_cost``.

        Returns:
            A ``ModelPricing`` populated from the entry.

        Raises:
            None.
        """
        return ModelPricing(
            input_cost_per_token=entry.get("input_cost_per_token", 0.0),
            output_cost_per_token=entry.get("output_cost_per_token", 0.0),
            max_input_tokens=entry.get("max_input_tokens", 0),
            max_output_tokens=entry.get("max_output_tokens", 0),
        )
