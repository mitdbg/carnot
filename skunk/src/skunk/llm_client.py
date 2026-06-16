"""The LLM client and its provider-specific scaffolding.

All LLM traffic goes through either the direct Gemini API (AI Studio) via
`google-genai` (authenticated by `GEMINI_API_KEY`, the default) or OpenRouter
(`provider=openrouter`, authenticated by `OPENROUTER_API_KEY`). LLM calls are paced
by the process-wide token-bucket limiters from `skunk.common` (the per-model
`llm:<model>` bucket); only transient SDK exceptions (HTTP 429 + 5xx, network
timeouts / connection resets — see `_is_retryable`) are retried with exponential
backoff up to `llm_max_retries` times. The limiter paces traffic; the retry rides
out throttling and blips the limiter can't prevent (provider-side 429, TPM quotas,
multi-process fan-out)."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
import requests
from google.genai import errors as genai_errors
from google.genai import types

from skunk.common import (
    Effort,
    make_genai_client,
    get_rate_limiter,
)

if TYPE_CHECKING:
    from google import genai
    from openrouter import OpenRouter

    from skunk.common import B64Image, ExecutionContext
    from skunk.config import SkunkConfig

# Process-scoped logger: retries happen with no per-question ctx in scope (see
# `_retry_call`), so they go through stdlib logging rather than `ctx.emit`.
log = logging.getLogger(__name__)


def _is_retryable(e: BaseException) -> bool:
    """True for transient failures worth retrying: HTTP 429 (throttling) and 5xx
    (server-side), plus network-layer timeouts / connection resets from the
    underlying transport (`httpx` async, `requests` sync). Non-429 4xx — bad
    request, auth, context-length overflow — is a permanent error that will never
    succeed, so it raises immediately instead of burning the retry budget."""
    if isinstance(e, genai_errors.APIError):
        code = getattr(e, "code", None)
        return code == 429 or (code is not None and 500 <= code < 600)
    # OpenRouter SDK errors all derive from `OpenRouterError`, which exposes the
    # HTTP `status_code` (lazy import — the SDK is only loaded on the openrouter path).
    try:
        from openrouter.errors import OpenRouterError
    except ImportError:
        OpenRouterError = ()  # type: ignore[assignment]
    if isinstance(e, OpenRouterError):
        code = getattr(e, "status_code", None)
        return code == 429 or (code is not None and 500 <= code < 600)
    # Transport faults: connection resets, read timeouts, DNS failures, etc.
    # `TimeoutError` covers both asyncio timeouts on the async SDK path and our own
    # per-request wall-clock cap (asyncio.wait_for in the streaming path raises builtin
    # TimeoutError) — treat a tripped timeout as a transient fault worth retrying, same as a
    # transport-level read timeout. The genai async transport is aiohttp, whose connection
    # faults (`ClientOSError`: broken pipe / reset) derive from `ClientConnectionError` — none
    # of which the httpx/requests types below catch.
    retryable: tuple[type[BaseException], ...] = (
        TimeoutError,
        httpx.TimeoutException,
        httpx.TransportError,
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
    )
    try:
        import aiohttp

        retryable += (aiohttp.ClientConnectionError,)
    except ImportError:
        pass
    return isinstance(e, retryable)


def _is_429(e: BaseException) -> bool:
    """True iff `e` is an HTTP 429 (rate-limit / quota) from either provider SDK — the
    one signal the failover monitor counts (`_ProviderFailover.record_429`). Mirrors
    the 429 arm of `_is_retryable`; lazy-imports the OpenRouter error type."""
    if isinstance(e, genai_errors.APIError):
        return getattr(e, "code", None) == 429
    try:
        from openrouter.errors import OpenRouterError
    except ImportError:
        OpenRouterError = ()  # type: ignore[assignment]
    if isinstance(e, OpenRouterError):
        return getattr(e, "status_code", None) == 429
    return False


_MODEL_RPM: dict[str, float] | None = None

# Built-in per-model request caps (provider account limits). Overridable per model via
# SKUNK_MODEL_RPM ("model=rpm,..."); a model in neither falls back to SKUNK_LLM_RPM.
_DEFAULT_RPM: dict[str, float] = {
    "gemini-3.5-flash": 4000.0,
    "gemini-3-flash-preview": 4000.0,  # fine-grained filter scan model — separate quota bucket, same RPM as 3.5-flash
    "gemini-3.1-flash-lite": 4000.0,
    "gemini-3.1-pro-preview": 2000.0,
}


def _llm_model_rpm(model: str) -> float:
    """Per-minute request cap for an LLM `model`. Parsed once from `SKUNK_MODEL_RPM`
    ("model=rpm,..."); a model not listed falls back to its `_DEFAULT_RPM`, then to
    `SKUNK_LLM_RPM` (default 1000). Each model gets its own limiter bucket
    (`llm:<model>`), so mixed-model runs pace independently."""
    global _MODEL_RPM
    if _MODEL_RPM is None:
        out: dict[str, float] = {}
        for entry in os.environ.get("SKUNK_MODEL_RPM", "").split(","):
            entry = entry.strip()
            if not entry:
                continue
            name, _, rpm = entry.partition("=")
            out[name.strip()] = float(rpm.strip())
        _MODEL_RPM = out
    if model in _MODEL_RPM:
        return _MODEL_RPM[model]
    if model in _DEFAULT_RPM:
        return _DEFAULT_RPM[model]
    return float(os.environ.get("SKUNK_LLM_RPM", "1000"))


# ---------------------------------------------------------------------------
# Tokens-per-minute (TPM) throttle — async, opt-in via SKUNK_MODEL_TPM.
#
# The RPM limiter alone can't bound token throughput: one request can carry tens
# of thousands of tokens, so a request-paced run still blows a TPM quota (the
# full-text page-index filter pushed ~36M tok/min and 429-stormed). This bucket
# meters estimated *input* tokens per call. Mirrors `_RateLimiter.acquire_async`'s
# cross-loop safety (threading.Lock around refill+deduct, sleep outside the lock).
# Off unless `SKUNK_MODEL_TPM` names the model, so it's scoped to experiments.
# ---------------------------------------------------------------------------

class _AsyncTokenBudget:
    """Like `_RateLimiter.acquire_async` but `acquire(amount)` deducts a variable token
    count (the call's estimated input tokens). `capacity` allows a short burst
    and must exceed the largest single request, or `acquire` would cap-clamp it."""

    def __init__(self, rate_per_sec: float, capacity: float) -> None:
        if rate_per_sec <= 0:
            raise ValueError(f"rate_per_sec must be > 0 (got {rate_per_sec})")
        self._rate = rate_per_sec
        self._capacity = max(capacity, rate_per_sec)
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_refill = now

    async def acquire(self, amount: float) -> None:
        amount = max(0.0, min(float(amount), self._capacity))
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                wait_s = (amount - self._tokens) / self._rate
            await asyncio.sleep(wait_s)

    def settle(self, delta: float) -> None:
        """Post-call correction: charge `delta` (actual minus estimated tokens)
        without awaiting. May drive the balance negative so the next `acquire`
        waits longer — this is what makes the throttle track ACTUAL token usage
        even when the pre-call estimate is off. No-op for delta 0."""
        if not delta:
            return
        with self._lock:
            self._refill_locked()
            self._tokens -= delta


_ASYNC_TPM_LOCK = threading.Lock()
_ASYNC_TPM_LIMITERS: dict[str, _AsyncTokenBudget] = {}


def get_async_tpm_limiter(model: str, tpm: float) -> _AsyncTokenBudget:
    """Process-wide TPM bucket for `model`, paced at `tpm` tokens/min. Capacity is
    ~4s of budget so a few large concurrent requests can burst, then throttle."""
    with _ASYNC_TPM_LOCK:
        lim = _ASYNC_TPM_LIMITERS.get(model)
        if lim is None:
            rate_per_sec = tpm / 60.0
            lim = _AsyncTokenBudget(rate_per_sec, capacity=max(rate_per_sec * 4.0, 256_000.0))
            _ASYNC_TPM_LIMITERS[model] = lim
        return lim


_MODEL_TPM: dict[str, float] | None = None

# Built-in per-model token-per-minute caps (provider account limits). Overridable per model
# via SKUNK_MODEL_TPM ("model=tpm,..."); a model in neither is unthrottled (None) and paced
# by RPM alone.
_DEFAULT_TPM: dict[str, float] = {
    "gemini-3.5-flash": 10_000_000.0,
    "gemini-3-flash-preview": 10_000_000.0,  # fine-grained filter scan model — separate quota bucket, same TPM as 3.5-flash
    "gemini-3.1-flash-lite": 25_000_000.0,
    "gemini-3.1-pro-preview": 8_000_000.0,
}


def _llm_model_tpm(model: str) -> float | None:
    """Per-minute *token* cap for `model`, parsed once from `SKUNK_MODEL_TPM`
    ("model=tpm,..."), else its `_DEFAULT_TPM`. Returns None (no throttle) when neither
    sets it — so the TPM bucket is inert for unlisted models (paced by RPM alone)."""
    global _MODEL_TPM
    if _MODEL_TPM is None:
        out: dict[str, float] = {}
        for entry in os.environ.get("SKUNK_MODEL_TPM", "").split(","):
            entry = entry.strip()
            if not entry:
                continue
            name, _, tpm = entry.partition("=")
            out[name.strip()] = float(tpm.strip())
        _MODEL_TPM = out
    if model in _MODEL_TPM:
        return _MODEL_TPM[model]
    return _DEFAULT_TPM.get(model)


def _estimate_prompt_tokens(system: str, user: str) -> float:
    """Cheap pre-call input-token estimate (~4 chars/token) for the TPM bucket.
    Approximate by design — it only paces throughput, it doesn't bill."""
    return (len(system) + len(user)) / 4.0


def _requires_thinking(model: str) -> bool:
    """True for models that mandate thinking mode (reject `thinking_budget=0`).
    Gemini 3.x Pro tiers are thinking-only; Flash/Flash-Lite accept budget=0."""
    m = model.lower()
    return "gemini-3" in m and "pro" in m


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    # Cached (prompt-cache hit) input tokens, when the provider reports them.
    # Subset of input_tokens; None when unknown.
    cache_input_tokens: int | None = None
    # Thinking/reasoning tokens (billed at the output rate); None when unreported.
    thinking_tokens: int | None = None


def _make_openrouter_client() -> "OpenRouter":
    """Build an OpenRouter client from `OPENROUTER_API_KEY` (for `provider=openrouter`).
    Lazy import: the SDK is only needed when this provider is selected."""
    from openrouter import OpenRouter

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set (required for provider=openrouter)")
    return OpenRouter(api_key=api_key)


# ---------------------------------------------------------------------------
# Automatic Gemini→OpenRouter failover (self-healing).
#
# Gemini is the front path. A process-global monitor counts HTTP 429s in a rolling
# per-second window (`_Rolling429Window`); once `threshold` of them land inside the window
# it routes ALL generation to OpenRouter. Every call reads the window when picking a
# provider, so failover is instant for new traffic and needs no broadcast / registry /
# cancellation of in-flight calls. The provider is re-checked per RETRY ATTEMPT (see
# `_retry_call`), so a call already on Gemini that 429s after the trip finishes on
# OpenRouter at its next attempt — no call is ever cancelled.
#
# Recovery is built into the window: it DRAINS by the second. Once on OpenRouter no traffic
# hits Gemini, so no fresh 429s arrive and the per-second buckets age out; when the windowed
# total falls back under `threshold` the next call flips back to Gemini. No canary probes —
# if Gemini is still storming, the resumed load simply re-trips. See `SkunkConfig.llm_failover_*`.
# ---------------------------------------------------------------------------


class _Rolling429Window:
    """A self-draining rolling window of HTTP-429 counts over the last `window_s` seconds,
    bucketed one count per second. The data structure is maintained COLLABORATIVELY — there
    is no background thread; each caller advances the ring by the wall-clock seconds elapsed
    since the last touch, so old seconds age out on their own. All state is private.

    Simple API: `log_429()` records one failure; `check_status()` returns True when the
    windowed total has reached `threshold` (i.e. the backup provider should be used).

    Thread-safe: mutation/advance happens under a lock; `check_status` short-circuits the
    lock when it can tell no second has rolled over yet (an atomic int read of the cached
    epoch), so the per-call hot path is uncontended within a given second."""

    def __init__(self, *, window_s: int, threshold: int) -> None:
        self._n = max(1, int(window_s))
        self._threshold = threshold
        self._buckets = [0] * self._n        # ring of per-second 429 counts
        self._head = 0                        # index of the current second's bucket
        self._epoch = int(time.monotonic())   # wall-clock second that `_head` represents
        self._total = 0                       # cached sum(_buckets) — the windowed count
        self._lock = threading.Lock()

    def _advance_locked(self, now_s: int) -> None:
        """Drain the buckets for each whole second elapsed since `_epoch` (caller holds the
        lock). This is the collaborative maintenance step — whoever touches the window next
        pays for the seconds that have passed."""
        steps = now_s - self._epoch
        if steps <= 0:
            return
        if steps >= self._n:  # the whole window has aged out — nothing survives
            self._buckets = [0] * self._n
            self._head = 0
            self._total = 0
        else:
            for _ in range(steps):
                self._head = (self._head + 1) % self._n
                self._total -= self._buckets[self._head]
                self._buckets[self._head] = 0
        self._epoch = now_s

    def log_429(self) -> None:
        now_s = int(time.monotonic())
        with self._lock:
            self._advance_locked(now_s)
            self._buckets[self._head] += 1
            self._total += 1

    def check_status(self) -> bool:
        """True ⇒ the windowed 429 count has reached `threshold` (use the backup)."""
        now_s = int(time.monotonic())
        if now_s == self._epoch:  # same second — no drain owed; atomic reads, lock-free
            return self._total >= self._threshold
        with self._lock:
            self._advance_locked(now_s)
            return self._total >= self._threshold


class _ProviderFailover:
    """Process-global failover policy: owns provider routing + the model map + once-per-
    transition logging, and delegates all 429 windowing/draining to `_Rolling429Window`.
    Inert (always routes to the configured provider) when `enabled` is False."""

    def __init__(
        self,
        *,
        enabled: bool,
        threshold: int,
        window_s: float,
        fallback_model: str | None,
        model_map: dict[str, str],
    ) -> None:
        self._enabled = enabled
        self._fallback_model = fallback_model
        self._model_map = dict(model_map)
        self._window = _Rolling429Window(window_s=int(window_s), threshold=threshold)
        self._on_backup = False  # last-logged status, so TRIP/RECOVER log once per flip
        self._log_lock = threading.Lock()

    def record_429(self) -> None:
        """Record a 429 into the rolling window. No-op when disabled."""
        if self._enabled:
            self._window.log_429()

    def route(self, provider: str, model: str) -> tuple[str, str]:
        """(provider, model) for the next attempt. Disabled — or already targeting
        openrouter — passes through unchanged. Otherwise, while the window is over threshold
        Gemini calls become OpenRouter (mapped/fallback model); when it drains back under
        threshold, traffic returns to Gemini."""
        if not self._enabled or provider == "openrouter":
            return provider, model
        on_backup = self._window.check_status()
        self._note_transition(on_backup)
        if on_backup:
            return "openrouter", self._model_map.get(model, self._fallback_model or model)
        return provider, model

    def _note_transition(self, on_backup: bool) -> None:
        """Log TRIPPED / RECOVERED once per actual flip (double-checked under the lock)."""
        if on_backup == self._on_backup:
            return
        with self._log_lock:
            if on_backup == self._on_backup:
                return
            self._on_backup = on_backup
            if on_backup:
                log.warning(
                    "LLM failover TRIPPED: ≥ threshold HTTP 429s in the rolling window — "
                    "routing ALL generation to OpenRouter (default model=%s); will retry "
                    "Gemini once the window drains", self._fallback_model,
                )
            else:
                log.warning(
                    "LLM failover RECOVERED: 429 window drained — routing generation back "
                    "to Gemini",
                )

    @property
    def tripped(self) -> bool:
        return self._enabled and self._window.check_status()


_FAILOVER: _ProviderFailover | None = None
_FAILOVER_LOCK = threading.Lock()


def _warn_free_failover_models(fallback: str | None, model_map: dict[str, str]) -> None:
    """One-time warning if any failover target is a `:free` OpenRouter id — those cap at
    ~20 RPM, BELOW the Gemini quota, so a 429 storm would only worsen on them."""
    free = sorted({m for m in [fallback, *model_map.values()] if m and m.endswith(":free")})
    if free:
        log.warning(
            "LLM failover target(s) %s are `:free` tier (~20 RPM) — too small to absorb "
            "a 429 storm; configure paid OpenRouter model ids instead.", free,
        )


def get_provider_failover(config: SkunkConfig) -> _ProviderFailover:
    """Process-wide failover monitor, built from the FIRST config seen and frozen for the
    run (same first-use-wins contract as `get_rate_limiter`). Armed only when failover is
    enabled AND `OPENROUTER_API_KEY` AND a failover model are all present; otherwise it's
    INERT (logs once, never trips) so a 429 storm degrades to retries, never to the hard
    failure an unconfigured OpenRouter client would raise."""
    global _FAILOVER
    with _FAILOVER_LOCK:
        if _FAILOVER is None:
            has_key = bool(os.environ.get("OPENROUTER_API_KEY"))
            armed = bool(config.llm_failover_enabled and has_key and config.llm_failover_model)
            if config.llm_failover_enabled and not armed:
                log.warning(
                    "LLM failover enabled but INERT (%s) — staying on the configured "
                    "provider with no automatic OpenRouter failover.",
                    "OPENROUTER_API_KEY not set" if not has_key
                    else "SKUNK_LLM_FAILOVER_MODEL not set",
                )
            if armed:
                _warn_free_failover_models(config.llm_failover_model, config.llm_failover_model_map)
            _FAILOVER = _ProviderFailover(
                enabled=armed,
                threshold=config.llm_failover_429_threshold,
                window_s=config.llm_failover_window_s,
                fallback_model=config.llm_failover_model,
                model_map=config.llm_failover_model_map,
            )
        return _FAILOVER


class LLMClient:
    """LLM client. Generation calls route to either the AI Studio Gemini API
    (`provider=genai`, default) or OpenRouter (`provider=openrouter`); both share the
    same rate-limit / retry / logging scaffolding. Under sustained Gemini 429s an
    automatic latch fails the whole process over to OpenRouter (`_ProviderFailover`),
    re-decided per retry attempt. Embeddings stay on Gemini."""

    def __init__(self, config: SkunkConfig) -> None:
        self._config = config
        self._gemini_client: genai.Client | None = None
        self._openrouter_client: OpenRouter | None = None
        # Process-global, shared across every LLMClient (one per question) so a 429
        # storm spanning all in-flight questions is seen as one signal.
        self._failover = get_provider_failover(config)

    def _get_gemini_client(self) -> genai.Client:
        if self._gemini_client is None:
            self._gemini_client = make_genai_client()
        return self._gemini_client

    def _get_openrouter_client(self) -> "OpenRouter":
        if self._openrouter_client is None:
            self._openrouter_client = _make_openrouter_client()
        return self._openrouter_client

    def call(
        self,
        system: str,
        user: str,
        images: list[B64Image] | None = None,
        temperature: float = 0.0,
        effort: Effort = "off",
        ctx: ExecutionContext | None = None,
        call_site: str = "llm",
        model: str | None = None,
    ) -> LLMResponse:
        def attempt(provider: str, m: str) -> LLMResponse:
            args = (system, user, images, temperature, effort, ctx, call_site, m)
            if provider == "openrouter":
                return self._call_openrouter_once(*args)
            return self._call_gemini_once(*args)

        return self._retry_call(attempt, self._config.llm_provider, model or self._config.llm_model)

    async def acall(
        self,
        system: str,
        user: str,
        images: list[B64Image] | None = None,
        temperature: float = 0.0,
        effort: "Effort" = "off",
        ctx: "ExecutionContext | None" = None,
        call_site: str = "llm",
        model: str | None = None,
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Async twin of `call` for the request path. `max_output_tokens` /
        `timeout_s` mirror `astream`'s caps (None → provider default / no cap)."""
        async def attempt(provider: str, m: str) -> LLMResponse:
            args = (system, user, images, temperature, effort, ctx, call_site, m,
                    max_output_tokens, timeout_s)
            if provider == "openrouter":
                return await self._acall_openrouter_once(*args)
            return await self._acall_gemini_once(*args)

        return await self._aretry_call(
            attempt, self._config.llm_provider, model or self._config.llm_model
        )

    def _retry_call(
        self,
        attempt: Callable[[str, str], LLMResponse],
        configured_provider: str,
        model: str,
    ) -> LLMResponse:
        """Drive `attempt(provider, model)` with exponential-backoff retry. The
        provider+model are re-routed through the failover latch ON EACH ATTEMPT (not
        once up front), so a call that 429s after the latch trips runs its next attempt
        on OpenRouter — no in-flight call is cancelled. 429s feed `record_429`; the
        rate-limiter bucket follows the *effective* model. `attempt` owns the API call,
        timing, parsing, and success emit."""
        max_retries = self._config.llm_max_retries
        delay = self._config.llm_retry_initial_delay_s

        for i in range(max_retries + 1):
            provider, eff_model = self._failover.route(configured_provider, model)
            limiter = get_rate_limiter(f"llm:{eff_model}", rate_per_min=_llm_model_rpm(eff_model))
            limiter.acquire()
            try:
                return attempt(provider, eff_model)
            except Exception as e:
                if _is_429(e):
                    self._failover.record_429()
                # Log EVERY failure with its message — including the final one
                # before we re-raise — so a fatal error is never silent.
                stop = i == max_retries or not _is_retryable(e)
                # Skip the backoff when the next attempt re-routes to a different
                # (healthy) provider — no reason to wait out a quota we're abandoning.
                rerouting = not stop and self._failover.route(configured_provider, model)[0] != provider
                log.warning(
                    "llm call failed (attempt %d/%d, provider=%s model=%s): %s: %s%s",
                    i + 1, max_retries + 1, provider, eff_model, type(e).__name__, e,
                    "" if stop else (
                        "; re-routing to backup provider" if rerouting else f"; retrying in {delay:.1f}s"
                    ),
                )
                if stop:
                    raise
                if not rerouting:
                    time.sleep(delay)
                    delay *= 2
        raise RuntimeError("unreachable: retry loop fell through")

    async def _aretry_call(
        self,
        attempt: Callable[[str, str], Awaitable[LLMResponse]],
        configured_provider: str,
        model: str,
    ) -> LLMResponse:
        """Async twin of `_retry_call`: awaits the effective model's async rate limiter
        and the coroutine `attempt(provider, model)`, backing off via `asyncio.sleep`
        (never blocking the event loop). Re-routes per attempt and records 429s exactly
        like the sync path."""
        max_retries = self._config.llm_max_retries
        delay = self._config.llm_retry_initial_delay_s

        for i in range(max_retries + 1):
            provider, eff_model = self._failover.route(configured_provider, model)
            limiter = get_rate_limiter(f"llm:{eff_model}", rate_per_min=_llm_model_rpm(eff_model))
            await limiter.acquire_async()
            try:
                return await attempt(provider, eff_model)
            except Exception as e:
                if _is_429(e):
                    self._failover.record_429()
                stop = i == max_retries or not _is_retryable(e)
                rerouting = not stop and self._failover.route(configured_provider, model)[0] != provider
                log.warning(
                    "llm call failed (attempt %d/%d, provider=%s model=%s): %s: %s%s",
                    i + 1, max_retries + 1, provider, eff_model, type(e).__name__, e,
                    "" if stop else (
                        "; re-routing to backup provider" if rerouting else f"; retrying in {delay:.1f}s"
                    ),
                )
                if stop:
                    raise
                if not rerouting:
                    await asyncio.sleep(delay)
                    delay *= 2
        raise RuntimeError("unreachable: retry loop fell through")

    def embed(
        self,
        texts: list[str],
        *,
        task_type: str = "CLUSTERING",
        dim: int = 768,
        model: str = "gemini-embedding-001",
        batch_size: int = 100,
    ) -> list[list[float]]:
        """Batched embedding — build-time / offline corpus-prep only (the query
        path is embedding-free). Output is L2-unnormalized; callers normalize
        before cosine. Chunked at `batch_size` (endpoint caps at 100/request).
        No retries / no rate limiter — one-shot build call."""
        if not texts:
            return []
        client = self._get_gemini_client()
        cfg = types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=dim,
        )
        out: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            resp = client.models.embed_content(
                model=model, contents=chunk, config=cfg,  # type: ignore[arg-type]
            )
            if not resp.embeddings:
                raise RuntimeError(f"embed model {model!r} returned no embeddings for {len(chunk)} inputs")
            vecs = [list(e.values or ()) for e in resp.embeddings]
            if len(vecs) != len(chunk):
                raise RuntimeError(
                    f"embed model {model!r} returned {len(vecs)} vectors "
                    f"for {len(chunk)} inputs; model likely requires batch_size=1"
                )
            out.extend(vecs)
        return out

    @staticmethod
    def _gemini_parts(user: str, images: list[B64Image] | None) -> list[Any]:
        parts: list[Any] = []
        if images:
            for img in images:
                parts.append(
                    types.Part.from_bytes(data=base64.b64decode(img.data), mime_type=img.mime)
                )
        parts.append(types.Part.from_text(text=user))
        return parts

    @staticmethod
    def _effort_to_thinking_config(effort: Effort, model: str) -> "types.ThinkingConfig":
        # ThinkingLevel is the only knob that hard-caps thinking spend for Gemini 3
        # (the legacy thinking_budget int is soft-bucketed).
        if _requires_thinking(model):
            # Gemini 3.x Pro: thinking is mandatory AND MINIMAL is unsupported, so
            # "off"/"minimal" floor at LOW (the cheapest tier this model accepts).
            level_map = {
                "off": types.ThinkingLevel.LOW,
                "minimal": types.ThinkingLevel.LOW,
                "low": types.ThinkingLevel.LOW,
                "medium": types.ThinkingLevel.MEDIUM,
                "high": types.ThinkingLevel.HIGH,
            }
            return types.ThinkingConfig(thinking_level=level_map[effort])
        if effort == "off":
            return types.ThinkingConfig(thinking_budget=0)
        level_map = {
            "minimal": types.ThinkingLevel.MINIMAL,
            "low": types.ThinkingLevel.LOW,
            "medium": types.ThinkingLevel.MEDIUM,
            "high": types.ThinkingLevel.HIGH,
        }
        return types.ThinkingConfig(thinking_level=level_map[effort])

    @staticmethod
    def _gemini_config(
        system: str,
        temperature: float,
        effort: Effort,
        model: str,
        max_output_tokens: int = 65535,
        timeout_s: float | None = None,
    ) -> types.GenerateContentConfig:
        # NOTE: on Gemini 3.x `max_output_tokens` is a COMBINED budget for thinking
        # + visible tokens, so it must stay comfortably above the effort tier's
        # thinking spend (medium ≈ 2.5K, high ≈ 16K thinking tokens) or the visible
        # answer is starved to empty (finish_reason=MAX_TOKENS). `timeout_s` sets the
        # SDK's per-read HTTP timeout; the streaming path additionally enforces a
        # hard wall-clock cap via asyncio.wait_for.
        http_options = (
            types.HttpOptions(timeout=int(timeout_s * 1000)) if timeout_s is not None else None
        )
        return types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            thinking_config=LLMClient._effort_to_thinking_config(effort, model),
            http_options=http_options,
        )

    @staticmethod
    def _usage_tokens(usage: Any) -> dict:
        """Token counts from a Gemini `usage_metadata` (best-effort — any may be
        None, e.g. when streaming omits usage)."""
        return {
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
            "thinking_tokens": getattr(usage, "thoughts_token_count", None),
            "cache_input_tokens": getattr(usage, "cached_content_token_count", None),
        }

    def _call_gemini_once(
        self,
        system: str,
        user: str,
        images: list[B64Image] | None,
        temperature: float,
        effort: Effort,
        ctx: ExecutionContext | None,
        call_site: str = "llm",
        model: str | None = None,
    ) -> LLMResponse:
        """One Gemini call (no retry — the retry loop owns that). `call_site`
        attributes the envelope log to the caller."""
        client = self._get_gemini_client()
        parts = self._gemini_parts(user, images)
        model = model or self._config.llm_model
        gen_config = self._gemini_config(system, temperature, effort, model)

        def do() -> LLMResponse:
            t0 = time.monotonic()
            api_resp = client.models.generate_content(
                model=model, contents=parts, config=gen_config,
            )
            latency_s = time.monotonic() - t0
            usage = api_resp.usage_metadata
            output_text = (api_resp.text or "").strip()
            toks = self._usage_tokens(usage)
            if ctx is not None:
                ctx.emit(
                    f"call call_site={call_site} model={model} temp={temperature} "
                    f"effort={effort} latency_s={round(latency_s, 3)} "
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                    f"think_tok={toks['thinking_tokens']}"
                )
            return LLMResponse(
                text=output_text,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
                cache_input_tokens=toks.get("cache_input_tokens"),
                thinking_tokens=toks.get("thinking_tokens"),
            )

        return do()

    async def _acall_gemini_once(
        self,
        system: str,
        user: str,
        images: list[B64Image] | None,
        temperature: float,
        effort: Effort,
        ctx: ExecutionContext | None,
        call_site: str = "llm",
        model: str | None = None,
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Async twin of `_call_gemini_once` — uses `client.aio.models.generate_content`."""
        client = self._get_gemini_client()
        parts = self._gemini_parts(user, images)
        model = model or self._config.llm_model
        gen_config = self._gemini_config(
            system, temperature, effort, model,
            max_output_tokens if max_output_tokens is not None else 65535,
            timeout_s,
        )

        async def do() -> LLMResponse:
            # TPM throttle (opt-in via SKUNK_MODEL_TPM): meter input tokens so
            # throughput stays under quota. Charge an estimate up front for pacing,
            # then `settle` the actual-vs-estimate delta after the call so the
            # bucket tracks REAL token usage (the char/4 estimate runs ~2x low on
            # dense tabular text). Inside `do` so each retry re-charges. Separate
            # from the RPM limiter.
            tpm = _llm_model_tpm(model)
            tpm_lim, est = None, 0.0
            if tpm:
                est = _estimate_prompt_tokens(system, user)
                tpm_lim = get_async_tpm_limiter(model, tpm)
                await tpm_lim.acquire(est)
            t0 = time.monotonic()
            # Hard wall-clock cap mirroring the streaming path: the http_options
            # timeout in `gen_config` bounds reads, wait_for bounds the whole call.
            coro = client.aio.models.generate_content(
                model=model, contents=parts, config=gen_config,
            )
            if timeout_s is not None:
                api_resp = await asyncio.wait_for(coro, timeout_s)
            else:
                api_resp = await coro
            latency_s = time.monotonic() - t0
            usage = api_resp.usage_metadata
            output_text = (api_resp.text or "").strip()
            toks = self._usage_tokens(usage)
            if tpm_lim is not None:
                tpm_lim.settle((toks["input_tokens"] or 0) - est)
            if ctx is not None:
                ctx.emit(
                    f"call call_site={call_site} model={model} temp={temperature} "
                    f"effort={effort} latency_s={round(latency_s, 3)} "
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                    f"think_tok={toks['thinking_tokens']}"
                )
            return LLMResponse(
                text=output_text,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
                cache_input_tokens=toks.get("cache_input_tokens"),
                thinking_tokens=toks.get("thinking_tokens"),
            )

        return await do()

    def stream(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str | None = None,
        should_stop: Callable[[str], bool] | None = None,
        temperature: float = 0.0,
        effort: Effort = "off",
        ctx: ExecutionContext | None = None,
        call_site: str = "llm",
    ) -> LLMResponse:
        """Multi-turn streaming call, accumulating chunks until `should_stop(acc)`
        or the stream ends. `messages` are the {role, content} turns after the
        system message ('assistant' → model role, else user). Lets multi-turn
        agents share this client's rate-limit + retry + logging. `temperature`
        defaults to 0.0 — agent loops are deterministic like every other call site.
        `effort` maps onto Gemini's thinking config exactly like the single-shot
        path (`_gemini_config`), so agent loops are tunable like every other call site."""
        def attempt(provider: str, m: str) -> LLMResponse:
            kw = dict(system=system, messages=messages, model=m, should_stop=should_stop,
                      temperature=temperature, effort=effort, ctx=ctx, call_site=call_site)
            if provider == "openrouter":
                return self._stream_openrouter_once(**kw)  # type: ignore
            return self._stream_gemini_once(**kw)  # type: ignore

        return self._retry_call(attempt, self._config.llm_provider, model or self._config.llm_model)

    def _stream_gemini_once(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str | None = None,
        should_stop: "Callable[[str], bool] | None" = None,
        temperature: float = 0.0,
        effort: "Effort" = "off",
        ctx: "ExecutionContext | None" = None,
        call_site: str = "llm",
    ) -> LLMResponse:
        client = self._get_gemini_client()
        model_id = (model or self._config.llm_model).removeprefix("google/")
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=self._gemini_parts(m["content"], m.get("images")),
            )
            for m in messages
        ]
        gen_config = self._gemini_config(system, temperature, effort, model_id)

        def do() -> LLMResponse:
            t0 = time.monotonic()
            resp_stream = client.models.generate_content_stream(
                model=model_id, contents=contents, config=gen_config,  # type: ignore[arg-type]
            )
            accumulated = ""
            usage = None
            for chunk in resp_stream:
                accumulated += chunk.text or ""
                usage = getattr(chunk, "usage_metadata", None) or usage
                if should_stop is not None and should_stop(accumulated):
                    break
            close = getattr(resp_stream, "close", None)
            if close is not None:
                try:  # noqa: SIM105
                    close()
                except Exception:
                    pass
            latency_s = time.monotonic() - t0
            toks = self._usage_tokens(usage)
            if ctx is not None:
                ctx.emit(
                    f"call call_site={call_site} model={model_id} temp={temperature} "
                    f"effort={effort} latency_s={round(latency_s, 3)} "
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                    f"think_tok={toks['thinking_tokens']}"
                )
            return LLMResponse(
                text=accumulated,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
                cache_input_tokens=toks.get("cache_input_tokens"),
                thinking_tokens=toks.get("thinking_tokens"),
            )

        return do()

    async def astream(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str | None = None,
        should_stop: Callable[[str], bool] | None = None,
        temperature: float = 0.0,
        effort: Effort = "off",
        ctx: ExecutionContext | None = None,
        call_site: str = "llm",
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Async twin of `stream` — uses `client.aio.models.generate_content_stream`
        and `async for`, so the event loop runs other tasks between chunks.

        `max_output_tokens` overrides the per-call output cap (None → provider
        default); `timeout_s` enforces a hard per-request wall-clock cap. Both are
        opt-in and currently used only by the search agent (see `MultiTurnAgent`)."""
        async def attempt(provider: str, m: str) -> LLMResponse:
            kw = dict(system=system, messages=messages, model=m, should_stop=should_stop,
                      temperature=temperature, effort=effort, ctx=ctx, call_site=call_site,
                      max_output_tokens=max_output_tokens, timeout_s=timeout_s)
            if provider == "openrouter":
                return await self._astream_openrouter_once(**kw)  # type: ignore
            return await self._astream_gemini_once(**kw)  # type: ignore

        return await self._aretry_call(
            attempt, self._config.llm_provider, model or self._config.llm_model
        )

    async def _astream_gemini_once(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str | None = None,
        should_stop: Callable[[str], bool] | None = None,
        temperature: float = 0.0,
        effort: Effort = "off",
        ctx: ExecutionContext | None = None,
        call_site: str = "llm",
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        client = self._get_gemini_client()
        model_id = (model or self._config.llm_model).removeprefix("google/")
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=self._gemini_parts(m["content"], m.get("images")),
            )
            for m in messages
        ]
        gen_config = self._gemini_config(
            system, temperature, effort, model_id,
            max_output_tokens if max_output_tokens is not None else 65535,
            timeout_s,
        )

        async def _consume() -> tuple[str, Any]:
            resp_stream = await client.aio.models.generate_content_stream(
                model=model_id, contents=contents, config=gen_config,  # type: ignore[arg-type]
            )
            accumulated = ""
            usage = None
            try:
                async for chunk in resp_stream:
                    accumulated += chunk.text or ""
                    usage = getattr(chunk, "usage_metadata", None) or usage
                    if should_stop is not None and should_stop(accumulated):
                        break
            finally:
                # Runs on normal completion AND on wait_for cancellation, so a
                # timed-out stream still releases its connection.
                aclose = getattr(resp_stream, "aclose", None)
                if aclose is not None:
                    try:  # noqa: SIM105
                        await aclose()
                    except Exception:
                        pass
            return accumulated, usage

        async def do() -> LLMResponse:
            t0 = time.monotonic()
            if timeout_s is not None:
                # Hard wall-clock cap: http_options.timeout is only per-read, so a
                # slow-but-steady runaway would never trip it. On timeout this raises
                # TimeoutError, which `_is_retryable` treats as a transient fault.
                accumulated, usage = await asyncio.wait_for(_consume(), timeout_s)
            else:
                accumulated, usage = await _consume()
            latency_s = time.monotonic() - t0
            toks = self._usage_tokens(usage)
            if ctx is not None:
                ctx.emit(
                    f"call call_site={call_site} model={model_id} temp={temperature} "
                    f"effort={effort} latency_s={round(latency_s, 3)} "
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                    f"think_tok={toks['thinking_tokens']}"
                )
            return LLMResponse(
                text=accumulated,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
                cache_input_tokens=toks.get("cache_input_tokens"),
                thinking_tokens=toks.get("thinking_tokens"),
            )

        return await do()

    # --- OpenRouter generation path (provider="openrouter") ---
    # Mirrors the Gemini helpers above: same do()/_retry_call shape, same ctx.emit
    # envelope and LLMResponse, so the rate-limit/retry/logging scaffolding is shared.

    @staticmethod
    def _effort_to_reasoning(effort: Effort) -> dict:
        """Map our Effort tier onto OpenRouter's `reasoning` arg. "off" maps to the
        lowest tier ("minimal") rather than disabling reasoning outright, since some
        endpoints (e.g. Gemini 3) mandate reasoning and reject `effort=none` with a
        400. The rest pass straight through (OpenRouter's `effort` accepts
        minimal/low/medium/high). Models without reasoning ignore it."""
        return {"effort": "minimal" if effort == "off" else effort}

    @staticmethod
    def _openrouter_content(user: str, images: list[B64Image] | None) -> Any:
        """User-turn content: a plain string, or OpenAI-style content parts when
        images are attached (base64 `data:` URLs), mirroring `_gemini_parts`."""
        if not images:
            return user
        parts: list[dict] = [
            {"type": "image_url", "image_url": {"url": f"data:{img.mime};base64,{img.data}"}}
            for img in images
        ]
        parts.append({"type": "text", "text": user})
        return parts

    @staticmethod
    def _openrouter_text(resp: Any) -> str:
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ""
        msg = getattr(choices[0], "message", None)
        return getattr(msg, "content", None) or ""

    @staticmethod
    def _usage_tokens_openrouter(usage: Any) -> dict:
        """Token counts from an OpenRouter `ChatUsage` (best-effort — any may be None)."""
        details = getattr(usage, "prompt_tokens_details", None)
        return {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "cache_input_tokens": getattr(details, "cached_tokens", None) if details else None,
        }

    def _call_openrouter_once(
        self,
        system: str,
        user: str,
        images: list[B64Image] | None,
        temperature: float,
        effort: Effort,
        ctx: ExecutionContext | None,
        call_site: str = "llm",
        model: str | None = None,
    ) -> LLMResponse:
        """One OpenRouter chat call (no retry — the retry loop owns that). `call_site`
        attributes the envelope log."""
        client = self._get_openrouter_client()
        model = model or self._config.llm_model
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": self._openrouter_content(user, images)})
        reasoning = self._effort_to_reasoning(effort)

        def do() -> LLMResponse:
            t0 = time.monotonic()
            resp = client.chat.send(
                model=model, messages=messages, stream=False, # type: ignore
                temperature=temperature, reasoning=reasoning, # type: ignore
            )
            latency_s = time.monotonic() - t0
            output_text = self._openrouter_text(resp).strip()
            toks = self._usage_tokens_openrouter(getattr(resp, "usage", None))
            if ctx is not None:
                ctx.emit(
                    f"call call_site={call_site} model={model} temp={temperature} "
                    f"effort={effort} latency_s={round(latency_s, 3)} "
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                    f"think_tok={toks['thinking_tokens']}"
                )
            return LLMResponse(
                text=output_text,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
                cache_input_tokens=toks.get("cache_input_tokens"),
                thinking_tokens=toks.get("thinking_tokens"),
            )

        return do()

    async def _acall_openrouter_once(
        self,
        system: str,
        user: str,
        images: list[B64Image] | None,
        temperature: float,
        effort: Effort,
        ctx: ExecutionContext | None,
        call_site: str = "llm",
        model: str | None = None,
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Async twin of `_call_openrouter_once` — uses `client.chat.send_async`."""
        client = self._get_openrouter_client()
        model = model or self._config.llm_model
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": self._openrouter_content(user, images)})
        reasoning = self._effort_to_reasoning(effort)

        async def do() -> LLMResponse:
            # TPM throttle (opt-in via SKUNK_MODEL_TPM) — identical to the Gemini path.
            tpm = _llm_model_tpm(model)
            tpm_lim, est = None, 0.0
            if tpm:
                est = _estimate_prompt_tokens(system, user)
                tpm_lim = get_async_tpm_limiter(model, tpm)
                await tpm_lim.acquire(est)
            extra = {"max_tokens": max_output_tokens} if max_output_tokens is not None else {}
            t0 = time.monotonic()
            coro = client.chat.send_async(
                model=model, messages=messages, stream=False, # type: ignore
                temperature=temperature, reasoning=reasoning, **extra, # type: ignore
            )
            resp = await (asyncio.wait_for(coro, timeout_s) if timeout_s is not None else coro)
            latency_s = time.monotonic() - t0
            output_text = self._openrouter_text(resp).strip()
            toks = self._usage_tokens_openrouter(getattr(resp, "usage", None))
            if tpm_lim is not None:
                tpm_lim.settle((toks["input_tokens"] or 0) - est)
            if ctx is not None:
                ctx.emit(
                    f"call call_site={call_site} model={model} temp={temperature} "
                    f"effort={effort} latency_s={round(latency_s, 3)} "
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                    f"think_tok={toks['thinking_tokens']}"
                )
            return LLMResponse(
                text=output_text,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
                cache_input_tokens=toks.get("cache_input_tokens"),
                thinking_tokens=toks.get("thinking_tokens"),
            )

        return await do()

    @staticmethod
    def _openrouter_chat_messages(system: str, messages: list[dict]) -> list[dict]:
        """Multi-turn messages in OpenRouter format: a leading system message (if any)
        then the {role, content} turns ('assistant' → assistant, else user)."""
        out: list[dict] = []
        if system:
            out.append({"role": "system", "content": system})
        out.extend(
            {
                "role": "assistant" if m["role"] == "assistant" else "user",
                "content": LLMClient._openrouter_content(m["content"], m.get("images")),
            }
            for m in messages
        )
        return out

    @staticmethod
    def _openrouter_chunk_text(chunk: Any) -> str:
        text = ""
        for choice in getattr(chunk, "choices", None) or []:
            delta = getattr(choice, "delta", None)
            text += getattr(delta, "content", None) or ""
        return text

    def _stream_openrouter_once(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str | None = None,
        should_stop: Callable[[str], bool] | None = None,
        temperature: float = 0.0,
        effort: Effort = "off",
        ctx: ExecutionContext | None = None,
        call_site: str = "llm",
    ) -> LLMResponse:
        client = self._get_openrouter_client()
        model_id = model or self._config.llm_model
        or_messages = self._openrouter_chat_messages(system, messages)
        reasoning = self._effort_to_reasoning(effort)

        def do() -> LLMResponse:
            t0 = time.monotonic()
            accumulated = ""
            usage = None
            with client.chat.send(
                model=model_id, messages=or_messages, stream=True, # type: ignore
                temperature=temperature, reasoning=reasoning, # type: ignore
            ) as resp_stream:
                for chunk in resp_stream:
                    accumulated += self._openrouter_chunk_text(chunk)
                    usage = getattr(chunk, "usage", None) or usage
                    if should_stop is not None and should_stop(accumulated):
                        break
            latency_s = time.monotonic() - t0
            toks = self._usage_tokens_openrouter(usage)
            if ctx is not None:
                ctx.emit(
                    f"call call_site={call_site} model={model_id} temp={temperature} "
                    f"effort={effort} latency_s={round(latency_s, 3)} "
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                    f"think_tok={toks['thinking_tokens']}"
                )
            return LLMResponse(
                text=accumulated,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
                cache_input_tokens=toks.get("cache_input_tokens"),
                thinking_tokens=toks.get("thinking_tokens"),
            )

        return do()

    async def _astream_openrouter_once(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str | None = None,
        should_stop: Callable[[str], bool] | None = None,
        temperature: float = 0.0,
        effort: Effort = "off",
        ctx: ExecutionContext | None = None,
        call_site: str = "llm",
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Async twin of `_stream_openrouter_once` — uses `client.chat.send_async` + `async for`."""
        client = self._get_openrouter_client()
        model_id = model or self._config.llm_model
        or_messages = self._openrouter_chat_messages(system, messages)
        reasoning = self._effort_to_reasoning(effort)
        extra = {"max_tokens": max_output_tokens} if max_output_tokens is not None else {}

        async def _consume() -> tuple[str, Any]:
            accumulated = ""
            usage = None
            async with await client.chat.send_async(
                model=model_id, messages=or_messages, stream=True, # type: ignore
                temperature=temperature, reasoning=reasoning, **extra, # type: ignore
            ) as resp_stream:
                async for chunk in resp_stream:
                    accumulated += self._openrouter_chunk_text(chunk)
                    usage = getattr(chunk, "usage", None) or usage
                    if should_stop is not None and should_stop(accumulated):
                        break
            return accumulated, usage

        async def do() -> LLMResponse:
            t0 = time.monotonic()
            if timeout_s is not None:
                accumulated, usage = await asyncio.wait_for(_consume(), timeout_s)
            else:
                accumulated, usage = await _consume()
            latency_s = time.monotonic() - t0
            toks = self._usage_tokens_openrouter(usage)
            if ctx is not None:
                ctx.emit(
                    f"call call_site={call_site} model={model_id} temp={temperature} "
                    f"effort={effort} latency_s={round(latency_s, 3)} "
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                    f"think_tok={toks['thinking_tokens']}"
                )
            return LLMResponse(
                text=accumulated,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
                cache_input_tokens=toks.get("cache_input_tokens"),
                thinking_tokens=toks.get("thinking_tokens"),
            )

        return await do()
