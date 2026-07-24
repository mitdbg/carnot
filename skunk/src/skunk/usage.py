"""Token + cost accounting for LLM calls.

`LLMResponse` carries per-call token counts but no aggregation. `UsageTracker`
accumulates them across one `LLMClient`'s lifetime: every `LLMClient` owns a
`.usage` tracker that `_build_response` feeds on each successful generation, so
the agent loop and every system call are counted with no wrapper and no
positional-arg fishing. Read the tracker after a run (the eval harness builds one
client per question, so its tracker is naturally per-question scoped).

Each call is bucketed by a caller-supplied `key` (`key_to_usage` / `key_to_embed_usage`),
so several agents sharing one client keep separate per-model token/cost tallies — an
agent passes its `agent_id`, unkeyed calls land in `"default"`. `cost(key)` / `embed_cost(key)`
report one caller's spend; `cost()` (key=None) and the `total_*` properties aggregate across
all keys.

Cost comes from a price table on the `InferenceConfig` (`llm_prices`), a map of
`model-substring -> {"in"/"out"/"cached": $/Mtok}`; unmatched models cost 0, and
models served by a local vLLM server (`free_models`, from `vllm_base_urls`) cost 0
even when the table prices them. Chat responses don't report a dollar cost, so this
table is the sole cost source. Cached input tokens (a subset of input tokens) are
billed at the model's `cached` rate when given, else at its `in` rate.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skunk.llm_client import LLMResponse

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    n_calls: int = 0
    model_to_input_tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    model_to_output_tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    model_to_cached_tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    model_to_think_tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))

@dataclass
class EmbedUsage:
    embed_tokens: int = 0
    n_embed_calls: int = 0
    model_to_embed_tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))

class UsageTracker:
    """Accumulates token usage (and derives cost) across one client's LLM calls.

    `prices` is the `InferenceConfig.llm_prices` table (see module docstring); an empty
    table means every call costs 0. `free_models` are models whose calls cost $0
    REGARDLESS of the table — the ones routed to a local vLLM server
    (`InferenceConfig.vllm_base_urls` keys, passed in by `LLMClient`) — so a model priced
    for OpenRouter runs is still free in a run that serves it locally."""

    def __init__(
        self, default_model: str, prices: dict | None = None, free_models: set[str] | None = None
    ) -> None:
        # Guards the mutating accumulators below: a single client is hit concurrently when
        # tools fan LLM calls across a thread pool. Reads (`cost()`, snapshots) run after
        # the writers have joined, so only the writers need the lock.
        self._lock = threading.Lock()
        self.default_model = default_model
        self.prices = prices or {}
        self.free_models = set(free_models or ())
        self.key_to_usage: dict[str, Usage] = {"default": Usage()}
        self.key_to_embed_usage: dict[str, EmbedUsage] = {"default": EmbedUsage()}

    @property
    def total_input_tokens(self) -> int:
        """Total number of input tokens across all keys."""
        total = 0
        for _, usage in self.key_to_usage.items():
            total += usage.input_tokens
        return total

    @property
    def total_output_tokens(self) -> int:
        """Total number of output tokens across all keys."""
        total = 0
        for _, usage in self.key_to_usage.items():
            total += usage.output_tokens
        return total

    @property
    def total_cached_tokens(self) -> int:
        """Total number of cached tokens across all keys."""
        total = 0
        for _, usage in self.key_to_usage.items():
            total += usage.cached_tokens
        return total

    @property
    def total_embed_tokens(self) -> int:
        """Total number of embed tokens across all keys."""
        total = 0
        for _, usage in self.key_to_embed_usage.items():
            total += usage.embed_tokens
        return total

    @property
    def total_embed_calls(self) -> int:
        """Total number of embed calls across all keys."""
        total = 0
        for _, usage in self.key_to_embed_usage.items():
            total += usage.n_embed_calls
        return total

    def add(self, resp: LLMResponse, model: str | None, key: str) -> None:
        """Fold one `LLMResponse` into the running totals. Any token field may be
        None (streaming can omit usage) — treat those as 0."""
        in_tok = resp.input_tokens or 0
        out_tok = resp.output_tokens or 0
        cache_tok = resp.cache_input_tokens or 0
        think_tok = resp.thinking_tokens or 0
        m = model or self.default_model
        with self._lock:
            self.key_to_usage.setdefault(key, Usage())
            self.key_to_usage[key].input_tokens += in_tok
            self.key_to_usage[key].output_tokens += out_tok
            self.key_to_usage[key].cached_tokens += cache_tok
            self.key_to_usage[key].model_to_input_tokens[m] += in_tok
            self.key_to_usage[key].model_to_output_tokens[m] += out_tok
            self.key_to_usage[key].model_to_cached_tokens[m] += cache_tok
            self.key_to_usage[key].model_to_think_tokens[m] += think_tok
            self.key_to_usage[key].n_calls += 1

    def add_embed(self, model: str | None, input_tokens: int, key: str) -> None:
        """Fold one embedding call into the running totals. `input_tokens` is the
        provider-reported prompt-token count (0 when the backend doesn't report one)."""
        m = model or self.default_model
        with self._lock:
            self.key_to_embed_usage.setdefault(key, EmbedUsage())
            self.key_to_embed_usage[key].embed_tokens += input_tokens
            self.key_to_embed_usage[key].model_to_embed_tokens[m] += input_tokens
            self.key_to_embed_usage[key].n_embed_calls += 1

    def cost(self, key: str | None = None) -> float:
        """Total USD cost from the price table — generation plus embeddings - for the caller / agent
        specified by `key`. If `key` is `None`, reports the aggregate cost across all callers. Cached
        input tokens are billed at the model's `cached` rate (falling back to `in` when unset) and
        the remaining (uncached) input tokens at `in`; output AND thinking tokens at `out` (providers
        bill thoughts at the output rate). Every call adds to `model_to_input_tokens`, so iterating
        it covers all models seen."""
        if not self.prices:
            return 0.0

        # if key is specified but missing from map; return None
        if key is not None and key not in self.key_to_usage:
            return 0.0

        total = 0.0
        keys = self.key_to_usage.keys() if key is None else [key]
        for k in keys:
            usage = self.key_to_usage[k]
            for model, in_tok in usage.model_to_input_tokens.items():
                p = self._price_for(model)
                if not p:
                    continue
                in_rate = p.get("in", 0.0)
                cached = usage.model_to_cached_tokens.get(model, 0)
                uncached_in = max(0, in_tok - cached)
                total += uncached_in / 1_000_000 * in_rate
                total += cached / 1_000_000 * p.get("cached", in_rate)
                out_and_think = usage.model_to_output_tokens.get(model, 0) + usage.model_to_think_tokens.get(model, 0)
                total += out_and_think / 1_000_000 * p.get("out", 0.0)

        return total + self.embed_cost(key)

    def price_call(
        self, model: str | None, in_tok: int, cached_tok: int, out_tok: int, think_tok: int = 0
    ) -> float | None:
        """USD cost of a single generation call, priced exactly as `cost()` aggregates (so
        per-call costs sum to the question total): uncached input at `in`, cached input at
        `cached` (→ `in` when unset), output + thinking at `out`. Returns None when there is
        no price table, no matching entry, or a `free_models` (locally served) model, so
        callers can distinguish "unpriced" from "$0.00"."""
        if not self.prices:
            return None
        p = self._price_for(model or self.default_model)
        if not p:
            return None
        # Streaming can omit the usage chunk → token counts arrive as None; treat as 0
        # (mirrors add()). Without this, a completion that streamed fine but lacked usage
        # raised `TypeError: NoneType - int` here, which the retry loop mistook for a failed
        # call and discarded — wasting good work and adding load to a rate-limited endpoint.
        in_tok = in_tok or 0
        cached_tok = cached_tok or 0
        out_tok = out_tok or 0
        in_rate = p.get("in", 0.0)
        uncached = max(0, in_tok - cached_tok)
        return (
            uncached / 1_000_000 * in_rate
            + cached_tok / 1_000_000 * p.get("cached", in_rate)
            + (out_tok + think_tok) / 1_000_000 * p.get("out", 0.0)
        )

    def price_embed(self, model: str | None, in_tok: int) -> float | None:
        """USD cost of a single embedding call (input-only, billed at the model's `in` rate);
        None when unpriced. Mirrors `embed_cost`'s per-model lookup for one call."""
        if not self.prices:
            return None
        p = self._price_for(model or self.default_model)
        return None if not p else in_tok / 1_000_000 * p.get("in", 0.0)

    def embed_cost(self, key: str | None = None) -> float:
        """USD cost of embedding calls alone for the caller / agent specified by `key`. If `key` is `None`,
        reports the aggregate cost across all callers. Priced from the same table (embeddings are input-only,
        billed at the model's `in` rate). Looked up by the embedding model id, so add an entry for it to
        `llm_prices` (e.g. `{"qwen3-embedding": {"in": ...}}`); an unpriced embedding model costs 0. Folded
        into `cost()`; exposed separately so a report can show embedding spend as its own column."""
        if not self.prices:
            return 0.0

        # if key is specified but missing from map; return None
        if key is not None and key not in self.key_to_embed_usage:
            return 0.0

        total = 0.0
        keys = self.key_to_embed_usage.keys() if key is None else [key]
        for k in keys:
            usage = self.key_to_embed_usage[k]
            for model, in_tok in usage.model_to_embed_tokens.items():
                p = self._price_for(model)
                if not p:
                    continue
                total += in_tok / 1_000_000 * p.get("in", 0.0)

        return total

    def _price_for(self, model: str) -> dict | None:
        """The model's price entry, or None when unpriced — including every `free_models`
        entry: a locally served model costs $0 even when the table prices it (the same id
        can be a paid OpenRouter model in one run and a vLLM-served one in another)."""
        if model in self.free_models:
            return None
        return _match_price(model, self.prices)


def match_model_entry(model: str, mapping: dict[str, Any]) -> Any | None:
    """Look up a per-model value in a `model id/substring -> value` map: exact
    (case-insensitive) match first, then substring match (e.g. "qwen/qwen3-..." matches a
    "qwen3" key). Returns None when nothing matches. Shared by the price table and any other
    per-model config map (e.g. context-window limits) so they resolve model ids identically."""
    m = model.lower()
    by_lower = {k.lower(): v for k, v in mapping.items()}
    if m in by_lower:
        return by_lower[m]
    for key, val in mapping.items():
        if key.lower() in m:
            return val
    return None


def _match_price(model: str, prices: dict[str, dict]) -> dict | None:
    return match_model_entry(model, prices)
