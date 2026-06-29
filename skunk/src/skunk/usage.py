"""Token + cost accounting for LLM calls.

`LLMResponse` carries per-call token counts but no aggregation. `UsageTracker`
accumulates them across one `LLMClient`'s lifetime: every `LLMClient` owns a
`.usage` tracker that `_build_response` feeds on each successful generation, so
the agent loop and every system call are counted with no wrapper and no
positional-arg fishing. Read the tracker after a run (the eval harness builds one
client per question, so its tracker is naturally per-question scoped), and call
`.reset()` to zero it for the next accounting window (e.g. to separately bill a
build/prep phase that reuses a long-lived client).

Cost comes from a price table on the `SystemConfig` (`llm_prices`), a map of
`model-substring -> {"in"/"out"/"cached": $/Mtok}`; unmatched models cost 0.
genai/OpenRouter chat responses don't report a dollar cost, so this table is the
sole cost source. Cached input tokens (a subset of input tokens) are billed at the
model's `cached` rate when given, else at its `in` rate.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skunk.llm_client import LLMResponse


class UsageTracker:
    """Accumulates token usage (and derives cost) across one client's LLM calls.

    `prices` is the `SystemConfig.llm_prices` table (see module docstring); an empty
    table means every call costs 0."""

    def __init__(self, default_model: str, prices: dict | None = None) -> None:
        # Guards the mutating accumulators below: a single client is hit concurrently when
        # tools fan LLM calls across a thread pool (e.g. semfilter's per-doc judges), and the
        # `+=` increments are read-modify-write across bytecodes, so concurrent adds can drop
        # updates (under-count) without it. Reads (`cost()`, snapshots) run after the writers
        # have joined, so only the writers need the lock.
        self._lock = threading.Lock()
        self.default_model = default_model
        self.prices = prices or {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_input_tokens = 0
        self.n_calls = 0
        # per-model token sums, for cost when a run mixes models. cached is a subset
        # of input (its discounted portion), tracked separately so cost can price it.
        self.by_model_in: dict[str, int] = defaultdict(int)
        self.by_model_out: dict[str, int] = defaultdict(int)
        self.by_model_cached: dict[str, int] = defaultdict(int)
        # Embeddings are accounted SEPARATELY from generation: they have only input
        # tokens (no output / no cache), a distinct price, and a distinct call count.
        # Kept out of the generation counters above so a report's `total_input_tokens`
        # stays "tokens the LLM read" and embedding spend is its own line item.
        self.embed_tokens = 0
        self.n_embed_calls = 0
        self.by_emb_model_in: dict[str, int] = defaultdict(int)

    def add(self, resp: LLMResponse, model: str | None) -> None:
        """Fold one `LLMResponse` into the running totals. Any token field may be
        None (streaming can omit usage) — treat those as 0."""
        in_tok = resp.input_tokens or 0
        out_tok = resp.output_tokens or 0
        cache_tok = resp.cache_input_tokens or 0
        m = model or self.default_model
        with self._lock:
            self.input_tokens += in_tok
            self.output_tokens += out_tok
            self.cache_input_tokens += cache_tok
            self.by_model_in[m] += in_tok
            self.by_model_out[m] += out_tok
            self.by_model_cached[m] += cache_tok
            self.n_calls += 1

    def add_embed(self, model: str | None, input_tokens: int) -> None:
        """Fold one embedding call into the running totals. `input_tokens` is the exact
        prompt-token count when the provider reports it (OpenRouter) or a char/4 estimate
        for backends that don't (local SentenceTransformers)."""
        m = model or self.default_model
        with self._lock:
            self.embed_tokens += input_tokens
            self.by_emb_model_in[m] += input_tokens
            self.n_embed_calls += 1

    def reset(self) -> None:
        """Zero all counters. Use to start a fresh accounting window on a reused
        client (e.g. separating a build/prep phase from the query phase)."""
        with self._lock:
            self.input_tokens = 0
            self.output_tokens = 0
            self.cache_input_tokens = 0
            self.n_calls = 0
            self.by_model_in.clear()
            self.by_model_out.clear()
            self.by_model_cached.clear()
            self.embed_tokens = 0
            self.n_embed_calls = 0
            self.by_emb_model_in.clear()

    def cost(self) -> float:
        """Total USD cost from the price table — generation plus embeddings. Cached input
        tokens are billed at the model's `cached` rate (falling back to `in` when unset)
        and the remaining (uncached) input tokens at `in`; output tokens at `out`. Every
        call adds to `by_model_in`, so iterating it covers all models seen."""
        if not self.prices:
            return 0.0
        total = 0.0
        for model, in_tok in self.by_model_in.items():
            p = _match_price(model, self.prices)
            if not p:
                continue
            in_rate = p.get("in", 0.0)
            cached = self.by_model_cached.get(model, 0)
            uncached_in = max(0, in_tok - cached)
            total += uncached_in / 1_000_000 * in_rate
            total += cached / 1_000_000 * p.get("cached", in_rate)
            total += self.by_model_out.get(model, 0) / 1_000_000 * p.get("out", 0.0)
        return total + self.embed_cost()

    def price_call(self, model: str | None, in_tok: int, cached_tok: int, out_tok: int) -> float | None:
        """USD cost of a single generation call, priced exactly as `cost()` aggregates (so
        per-call costs sum to the question total): uncached input at `in`, cached input at
        `cached` (→ `in` when unset), output at `out`. Returns None when there is no price
        table or no matching entry, so callers can distinguish "unpriced" from "$0.00"."""
        if not self.prices:
            return None
        p = _match_price(model or self.default_model, self.prices)
        if not p:
            return None
        in_rate = p.get("in", 0.0)
        uncached = max(0, in_tok - cached_tok)
        return (
            uncached / 1_000_000 * in_rate
            + cached_tok / 1_000_000 * p.get("cached", in_rate)
            + out_tok / 1_000_000 * p.get("out", 0.0)
        )

    def price_embed(self, model: str | None, in_tok: int) -> float | None:
        """USD cost of a single embedding call (input-only, billed at the model's `in` rate);
        None when unpriced. Mirrors `embed_cost`'s per-model lookup for one call."""
        if not self.prices:
            return None
        p = _match_price(model or self.default_model, self.prices)
        return None if not p else in_tok / 1_000_000 * p.get("in", 0.0)

    def embed_cost(self) -> float:
        """USD cost of embedding calls alone, priced from the same table (embeddings are
        input-only, billed at the model's `in` rate). Looked up by the embedding model id,
        so add an entry for it to `llm_prices` (e.g. `{"qwen3-embedding": {"in": ...}}`);
        an unpriced embedding model costs 0. Folded into `cost()`; exposed separately so a
        report can show embedding spend as its own column."""
        if not self.prices:
            return 0.0
        total = 0.0
        for model, in_tok in self.by_emb_model_in.items():
            p = _match_price(model, self.prices)
            if not p:
                continue
            total += in_tok / 1_000_000 * p.get("in", 0.0)
        return total


def _match_price(model: str, prices: dict[str, dict]) -> dict | None:
    # exact (case-insensitive) match first, then substring match (e.g. "qwen/qwen3-..." matches a "qwen3" key).
    m = model.lower()
    by_lower = {k.lower(): v for k, v in prices.items()}
    if m in by_lower:
        return by_lower[m]
    for key, val in prices.items():
        if key.lower() in m:
            return val
    return None
