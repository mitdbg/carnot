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

import aiohttp
import asyncio
import base64
import os
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import httpx
import requests
from google.genai import errors as genai_errors
from google.genai import types

from skunk.common import (
    AsyncTokenBudget,
    Effort,
    get_async_tpm_limiter,
    get_rate_limiter,
    make_genai_client,
)
from skunk.usage import UsageTracker

if TYPE_CHECKING:
    from google import genai
    from openrouter import OpenRouter

    from skunk.common import B64Image, ExecutionContext
    from skunk.config import SystemConfig

def _warn(ctx: "ExecutionContext | None", message: str) -> None:
    """Route a client warning onto the owning question's event stream when a ctx
    is in scope (retry/usage warnings become attributable per question); ctx-less
    callers (offline corpus prep) fall back to plain stderr."""
    if ctx is not None:
        ctx.emit(message, level="warning")
    else:
        print(message, file=sys.stderr)

# Return type of an `attempt` driven by `_retry_call` — an `LLMResponse` for the
# generation paths, a `list[float]` for the embedding path. The retry loop never
# inspects the value, so it is generic over it.
R = TypeVar("R")


class EmptyCompletionError(RuntimeError):
    """A generation call returned HTTP 200 but with no message content. OpenRouter
    relays some upstream-provider failures this way — empty `choices` (or empty
    `content`) and zeroed usage, with no exception raised. For a text-generation call
    (final answer, judge, planner) an empty completion is never a usable result, so we
    treat it as a transient fault: the retry loop re-issues the call, and only if it
    stays empty past the retry budget does it surface as a failed row — far better than
    silently recording an empty answer that then scores 0."""


class EmptyEmbeddingError(EmptyCompletionError):
    """An embedding call returned no vector (empty `data`, or a zero-length embedding).
    Subclasses `EmptyCompletionError` so `_is_retryable` already treats it as transient:
    the same OpenRouter-relays-an-upstream-failure-as-200 mode applies to the embeddings
    endpoint, so re-issue rather than crash a question on a blip."""


def _is_retryable(e: BaseException) -> bool:
    """True for transient failures worth retrying: HTTP 429 (throttling) and 5xx
    (server-side), plus network-layer timeouts / connection resets from the
    underlying transport (`httpx` async, `requests` sync), and an empty-content 200
    (`EmptyCompletionError`). Non-429 4xx — bad request, auth, context-length overflow —
    is a permanent error that will never succeed, so it raises immediately instead of
    burning the retry budget."""
    if isinstance(e, EmptyCompletionError):
        return True
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
        aiohttp.ClientConnectionError,
        TimeoutError,
        httpx.TimeoutException,
        httpx.TransportError,
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
    )

    return isinstance(e, retryable)


def _error_detail(e: BaseException) -> str:
    """Diagnostic suffix for an LLM error. OpenRouter collapses upstream-provider
    failures into a terse `...ResponseError: Provider returned error`; the actionable
    cause — the HTTP status, the provider's raw error body, and OpenRouter's
    `metadata` (which carries the provider name + the raw upstream message) — lives on
    the exception, not in `str(e)`. Surface it so an intermittent 400 is debuggable
    straight from the logs. Best-effort: never raises, returns "" when nothing to add."""
    try:
        from openrouter.errors import OpenRouterError
    except ImportError:
        return ""
    if not isinstance(e, OpenRouterError):
        return ""
    parts: list[str] = []
    try:
        code = getattr(e, "status_code", None)
        if code is not None:
            parts.append(f"status={code}")
        err = getattr(getattr(e, "data", None), "error", None)
        meta = getattr(err, "metadata", None)
        if meta:
            parts.append(f"metadata={str(meta)[:1000]}")
        rr = getattr(e, "raw_response", None)
        body = getattr(rr, "text", None) if rr is not None else getattr(e, "body", None)
        if body:
            parts.append(f"body={str(body)[:1000]}")
    except Exception:  # noqa: BLE001 — diagnostics must never mask the original error
        return ""
    return (" | " + " ".join(parts)) if parts else ""


# The TPM throttle (AsyncTokenBudget + its process-wide registry) lives in
# `skunk.common`, next to the RPM registry — one home for all pacing state.


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


@dataclass(frozen=True)
class CallSpec:
    """One generation request, provider-agnostic — the value threaded from the
    public `call` / `acall` / `astream` methods into the per-provider bodies
    (which used to take it as 8–10 positional args). `model` is always resolved
    by the public method (never empty inside a provider body). Exactly one of
    `user` (single-shot) / `messages` (multi-turn streaming) is meaningful,
    matching the public method it came through."""

    system: str
    model: str
    user: str = ""
    messages: tuple[dict, ...] | None = None
    images: list[B64Image] | None = None
    temperature: float = 0.0
    effort: Effort = "off"
    ctx: ExecutionContext | None = None
    call_site: str = "llm"
    max_output_tokens: int | None = None
    timeout_s: float | None = None
    should_stop: Callable[[str], bool] | None = None
    # Per-call OpenRouter provider order (no fallback). None => use config.llm_provider_order.
    # Lets a single client route one model (e.g. a cheaper semantic-filter judge) to specific
    # providers while other models on the same client stay unpinned.
    provider_order: list[str] | None = None


# Raw HTTP body of the most recent OpenRouter response, captured by an httpx response
# hook (below) and read only when a completion comes back empty — so we can log the
# UNFILTERED upstream body (not the SDK's typed object, which could in principle drop a
# field) and settle whether an empty completion is genuinely empty or a parse artifact.
# A ContextVar isolates the value per async task / thread: the hook fires synchronously
# within the same send() call we're about to inspect, so no cross-call bleed.
_RAW_BODY_MAX = 2000
_last_openrouter_raw_body: ContextVar[str | None] = ContextVar(
    "_last_openrouter_raw_body", default=None
)


def _capture_response_body_sync(response: "httpx.Response") -> None:
    """httpx sync `response` event hook: buffer the body and stash it. `read()` caches
    `.content`, so the SDK's later `.json()` still works. Best-effort — never raise into
    the request path."""
    try:
        response.read()
        _last_openrouter_raw_body.set(response.text[:_RAW_BODY_MAX])
    except Exception:  # noqa: BLE001
        pass


async def _capture_response_body_async(response: "httpx.Response") -> None:
    """Async twin of `_capture_response_body_sync` (async clients require async hooks)."""
    try:
        await response.aread()
        _last_openrouter_raw_body.set(response.text[:_RAW_BODY_MAX])
    except Exception:  # noqa: BLE001
        pass


def _raw_body_suffix() -> str:
    """Diagnostic suffix carrying the last captured raw OpenRouter body, for empty-completion
    errors. The body is the ground truth the SDK's typed `ChatResult` is parsed from, so it
    reveals whether `content` was truly null on the wire vs. dropped in parsing."""
    body = _last_openrouter_raw_body.get()
    return f" raw_body={body}" if body else ""


def _make_openrouter_client(api_key: str | None = None) -> OpenRouter:
    """Build an OpenRouter client (for `provider=openrouter`), authenticated by the
    explicit `api_key` else `OPENROUTER_API_KEY`. Lazy import: the SDK is only needed
    when this provider is selected. Inject httpx clients carrying a response hook that
    captures each raw body (read on empty completions only) — with a generous timeout,
    since a bare httpx client defaults to 5s and would abort long generations (the app
    enforces its own per-request wall-clock cap)."""
    import httpx
    from openrouter import OpenRouter

    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set (required for provider=openrouter)")
    timeout = httpx.Timeout(600.0)
    sync_client = httpx.Client(timeout=timeout, event_hooks={"response": [_capture_response_body_sync]})
    async_client = httpx.AsyncClient(
        timeout=timeout, event_hooks={"response": [_capture_response_body_async]}
    )
    return OpenRouter(api_key=api_key, client=sync_client, async_client=async_client)


# Local SentenceTransformers embedders are cached PROCESS-WIDE (not per LLMClient): the
# eval builds one LLMClient per question and runs many concurrently, so a per-instance
# cache loaded N identical copies of the model — fine on CPU RAM, but on a single GPU that
# is N * ~1.2GB and OOMs the device (e.g. 32 copies > a 22GB L4). One shared instance is
# also correct: the model is read-only at inference, so concurrent .encode() calls from
# worker threads are safe. Keyed by model id; the lock only guards the one-time load.
_LOCAL_EMBEDDERS: dict[str, Any] = {}
_LOCAL_EMBEDDERS_LOCK = threading.Lock()


class LLMClient:
    """LLM client. Generation calls route to either the AI Studio Gemini API
    (`provider=genai`, default) or OpenRouter (`provider=openrouter`) per
    `config.llm_provider`; both share the same rate-limit / retry / logging
    scaffolding. Embeddings stay on Gemini."""

    def __init__(
        self,
        config: SystemConfig,
        *,
        gemini_api_key: str | None = None,
        openrouter_api_key: str | None = None,
    ) -> None:
        """`gemini_api_key` / `openrouter_api_key` override the environment
        (`GEMINI_API_KEY` / `OPENROUTER_API_KEY`) — the constructor is the seam for
        callers that manage credentials themselves; env remains the default."""
        self._config = config
        self._gemini_api_key = gemini_api_key
        self._openrouter_api_key = openrouter_api_key
        self._gemini_client: genai.Client | None = None
        self._openrouter_client: OpenRouter | None = None
        self.usage = UsageTracker(
            default_model=config.llm_model,
            prices=config.llm_prices,
        )

    def _get_gemini_client(self) -> genai.Client:
        if self._gemini_client is None:
            self._gemini_client = make_genai_client(self._gemini_api_key)
        return self._gemini_client

    def _get_openrouter_client(self) -> OpenRouter:
        if self._openrouter_client is None:
            self._openrouter_client = _make_openrouter_client(self._openrouter_api_key)
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
        provider_order: list[str] | None = None,
    ) -> LLMResponse:
        spec = CallSpec(
            system=system, user=user, images=images, temperature=temperature,
            effort=effort, ctx=ctx, call_site=call_site,
            model=model or self._config.llm_model,
            provider_order=provider_order,
        )
        impl = (
            self._openrouter_call
            if self._config.llm_provider == "openrouter"
            else self._gemini_call
        )
        return self._retry_call(lambda: impl(spec), self._config.llm_provider, spec.model, ctx=spec.ctx)

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
        spec = CallSpec(
            system=system, user=user, images=images, temperature=temperature,
            effort=effort, ctx=ctx, call_site=call_site,
            model=model or self._config.llm_model,
            max_output_tokens=max_output_tokens, timeout_s=timeout_s,
        )
        impl = (
            self._openrouter_acall
            if self._config.llm_provider == "openrouter"
            else self._gemini_acall
        )
        return await self._aretry_call(
            lambda: impl(spec), self._config.llm_provider, spec.model, ctx=spec.ctx
        )

    def _retry_call(
        self,
        attempt: Callable[[], R],
        provider: str,
        model: str,
        bucket: str | None = None,
        ctx: "ExecutionContext | None" = None,
    ) -> R:
        """Drive `attempt()` with exponential-backoff retry, paced by a
        rate-limiter bucket. Retries only transient faults (`_is_retryable`: 429 / 5xx /
        transport blips); a non-429 4xx (bad request, auth, context overflow) raises
        immediately. `attempt` owns the API call, timing, parsing, and the success emit;
        `provider`/`model` are for the failure warning (routed onto `ctx`'s event
        stream when given — see `_warn`). `bucket` overrides the limiter (a
        shared process-wide bucket like `"embed"`); when None, the per-model
        `llm:<model>` bucket is used at the configured RPM."""
        max_retries = self._config.llm_max_retries
        delay = self._config.llm_retry_initial_delay_s
        limiter = (
            get_rate_limiter(bucket)
            if bucket is not None
            else get_rate_limiter(
                f"llm:{model}",
                rate_per_min=self._config.llm_model_rpm.get(model, self._config.llm_default_rpm),
            )
        )

        for i in range(max_retries + 1):
            limiter.acquire()
            try:
                return attempt()
            except Exception as e:
                # Log EVERY failure with its message — including the final one
                # before we re-raise — so a fatal error is never silent.
                stop = i == max_retries or not _is_retryable(e)
                _warn(
                    ctx,
                    f"llm_call_failed attempt={i + 1}/{max_retries + 1} provider={provider} "
                    f"model={model} error={type(e).__name__}: {e}{_error_detail(e)}"
                    f"{'' if stop else f'; retrying in {delay:.1f}s'}",
                )
                if stop:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable: retry loop fell through")

    async def _aretry_call(
        self,
        attempt: Callable[[], Awaitable[LLMResponse]],
        provider: str,
        model: str,
        ctx: "ExecutionContext | None" = None,
    ) -> LLMResponse:
        """Async twin of `_retry_call`: awaits the model's async rate limiter and the
        coroutine `attempt()`, backing off via `asyncio.sleep` (never blocking the
        event loop). `provider`/`model` are for the failure warning."""
        max_retries = self._config.llm_max_retries
        delay = self._config.llm_retry_initial_delay_s
        limiter = get_rate_limiter(
            f"llm:{model}",
            rate_per_min=self._config.llm_model_rpm.get(model, self._config.llm_default_rpm),
        )

        for i in range(max_retries + 1):
            await limiter.acquire_async()
            try:
                return await attempt()
            except Exception as e:
                stop = i == max_retries or not _is_retryable(e)
                _warn(
                    ctx,
                    f"llm_call_failed attempt={i + 1}/{max_retries + 1} provider={provider} "
                    f"model={model} error={type(e).__name__}: {e}{_error_detail(e)}"
                    f"{'' if stop else f'; retrying in {delay:.1f}s'}",
                )
                if stop:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable: retry loop fell through")

    def _build_response(
        self,
        text: str,
        toks: dict,
        latency_s: float,
        *,
        model: str,
        temperature: float,
        effort: Effort,
        ctx: ExecutionContext | None,
        call_site: str,
    ) -> LLMResponse:
        """Shared tail for every generation path: emit the uniform `call ...` envelope
        (when `ctx` is set), accumulate the call into this client's `usage` tracker, and
        pack the provider-agnostic token dict (`_usage_tokens` / `_usage_tokens_openrouter`)
        into an `LLMResponse`. `toks` may omit any key — OpenRouter's dict has no
        `thinking_tokens`, hence the `.get`s. The single chokepoint every call path funnels
        through, so usage accounting lives here once rather than at each public method."""
        if ctx is not None:
            # Exact per-call USD (priced as `cost()` aggregates) → the trace viewer shows
            # per-step and cumulative spend without re-deriving it from a duplicate price table.
            cost = self.usage.price_call(
                model, toks["input_tokens"], toks.get("cache_input_tokens") or 0,
                toks["output_tokens"], toks.get("thinking_tokens") or 0,
            )
            ctx.emit(
                f"call call_site={call_site} model={model} temp={temperature} "
                f"effort={effort} latency_s={round(latency_s, 3)} "
                f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                f"think_tok={toks.get('thinking_tokens')}",
                data=None if cost is None else {"cost": cost},
            )
        resp = LLMResponse(
            text=text,
            latency_s=latency_s,
            input_tokens=toks["input_tokens"],
            output_tokens=toks["output_tokens"],
            cache_input_tokens=toks.get("cache_input_tokens"),
            thinking_tokens=toks.get("thinking_tokens"),
        )
        self.usage.add(resp, model)
        return resp

    async def _tpm_acquire(
        self, model: str, system: str, user: str
    ) -> tuple[AsyncTokenBudget | None, float]:
        """Acquire the per-model token budget (`config.llm_model_tpm`, falling back to
        `llm_default_tpm`) before a call, charging an up-front input-token estimate.
        Returns (limiter, estimate) so the caller can `_tpm_settle` the actual-vs-estimate
        delta afterward; (None, 0.0) when the model is unthrottled (paced by RPM alone)."""
        tpm = self._config.llm_model_tpm.get(model, self._config.llm_default_tpm)
        if not tpm:
            return None, 0.0
        est = _estimate_prompt_tokens(system, user)
        lim = get_async_tpm_limiter(model, tpm)
        await lim.acquire(est)
        return lim, est

    @staticmethod
    def _tpm_settle(lim: AsyncTokenBudget | None, est: float, input_tokens: int | None) -> None:
        """Post-call correction: charge the actual-minus-estimated input tokens so the
        bucket tracks REAL usage (the char/4 estimate runs ~2x low on dense tabular text)."""
        if lim is not None:
            lim.settle((input_tokens or 0) - est)

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

    # --- Query-time embedding (provider="openrouter" | "local"), usage-tracked --------

    def _get_local_embedder(self, model: str) -> Any:
        """Lazily build (and cache) a local SentenceTransformers model. `sentence-
        transformers` is an optional extra (`pip install skunk[embeddings]`); the import
        is deferred so a run that only uses API-backed embeddings never pays it."""
        st = _LOCAL_EMBEDDERS.get(model)
        if st is None:
            with _LOCAL_EMBEDDERS_LOCK:
                st = _LOCAL_EMBEDDERS.get(model)  # re-check: another thread may have loaded it
                if st is None:
                    try:
                        import torch
                        from sentence_transformers import SentenceTransformer
                    except ImportError as e:
                        raise ImportError(
                            "local embeddings need the `embeddings` extra: "
                            "pip install 'skunk[embeddings]'"
                        ) from e

                    st = SentenceTransformer(model)
                    # The Qwen3-Embedding checkpoints declare torch_dtype=bfloat16, so weights
                    # load as bf16. CPU matmul cannot mix bf16 weights with the fp32 activations
                    # the forward pass produces ("RuntimeError: expected m1 and m2 to have the
                    # same dtype, but got: c10::BFloat16 != float"), which silently zeroes out
                    # query embeddings and tanks retrieval. Force fp32 on CPU; on CUDA bf16 is
                    # supported and faster, so leave it.
                    if not torch.cuda.is_available():
                        st = st.float()
                    _LOCAL_EMBEDDERS[model] = st
        return st

    def embed_query(
        self,
        text: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        ctx: ExecutionContext | None = None,
    ) -> list[float]:
        """Embed a single query string for vector search, routing through the same
        rate-limit / retry / usage-accounting scaffolding as generation so embedding
        spend lands on this client's `usage` tracker (tokens, cost, calls).

        Backends (per `config.emb_provider`, overridable via `provider`):
          - "openrouter": the OpenRouter embeddings endpoint (e.g. Qwen3-Embedding-8B).
            Token count is the provider-reported `usage.prompt_tokens`.
          - "local": a local SentenceTransformers model (e.g. Qwen3-Embedding-0.6B),
            L2-normalized to match how the corpus vectors were built. No token count is
            reported, so it is estimated at ~4 chars/token (bills only; paces nothing).

        When `ctx` is given, emits the uniform `call ...` envelope (so the embed call's
        latency shows in the per-question trace like every LLM call); usage is recorded
        regardless of `ctx`."""
        model = model or self._config.emb_model_id
        provider = provider or self._config.emb_provider
        m = model

        def attempt() -> list[float]:
            t0 = time.monotonic()
            if provider == "local":
                st = self._get_local_embedder(m)
                vec = st.encode([text], normalize_embeddings=True)[0]
                vector = [float(x) for x in vec]
                in_tok = max(1, len(text) // 4)  # no provider token count; estimate.
            else:
                resp = self._get_openrouter_client().embeddings.generate(input=text, model=m)
                data = getattr(resp, "data", None) or []
                vector = list(data[0].embedding) if data and data[0].embedding else []
                if not vector:
                    raise EmptyEmbeddingError(
                        f"openrouter empty embedding (model={m} provider={provider})"
                    )
                usage = getattr(resp, "usage", None)
                in_tok = (getattr(usage, "prompt_tokens", None) or 0) if usage else 0
            latency_s = time.monotonic() - t0
            self.usage.add_embed(m, in_tok)
            if ctx is not None:
                cost = self.usage.price_embed(m, in_tok)
                ctx.emit(
                    f"call call_site=embed model={m} provider={provider} "
                    f"latency_s={round(latency_s, 3)} in_tok={in_tok} dim={len(vector)}",
                    kind="call",
                    data=None if cost is None else {"cost": cost},
                )
            return vector

        # Embeddings share one process-wide "embed" rate bucket across all workers
        # (separate from the per-model generation buckets).
        return self._retry_call(attempt, provider, model, bucket="embed", ctx=ctx)

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

    @staticmethod
    def _gemini_model_id(model: str) -> str:
        """Bare Gemini model id for the AI Studio SDK. Full OpenRouter-style ids
        (`google/gemini-...`) drop the prefix — consistently, on EVERY genai path
        (call/acall/astream), so a config written for one provider degrades sanely
        on the other."""
        return model.removeprefix("google/")

    def _gemini_call(self, spec: CallSpec) -> LLMResponse:
        """One Gemini call (no retry — the retry loop owns that)."""
        client = self._get_gemini_client()
        parts = self._gemini_parts(spec.user, spec.images)
        model_id = self._gemini_model_id(spec.model)
        gen_config = self._gemini_config(
            spec.system, spec.temperature, spec.effort, model_id
        )
        t0 = time.monotonic()
        api_resp = client.models.generate_content(
            model=model_id, contents=parts, config=gen_config,
        )
        latency_s = time.monotonic() - t0
        output_text = (api_resp.text or "").strip()
        toks = self._usage_tokens(api_resp.usage_metadata)
        return self._finish(spec, output_text, toks, latency_s, model_id)

    async def _gemini_acall(self, spec: CallSpec) -> LLMResponse:
        """Async twin of `_gemini_call` — uses `client.aio.models.generate_content`."""
        client = self._get_gemini_client()
        parts = self._gemini_parts(spec.user, spec.images)
        model_id = self._gemini_model_id(spec.model)
        gen_config = self._gemini_config(
            spec.system, spec.temperature, spec.effort, model_id,
            spec.max_output_tokens if spec.max_output_tokens is not None else 65535,
            spec.timeout_s,
        )
        # TPM throttle (opt-in via SKUNK_MODEL_TPM): meter input tokens so
        # throughput stays under quota, separate from the RPM limiter. Per attempt
        # (the retry loop re-invokes this body), so each retry re-charges.
        # `_tpm_settle` corrects the estimate post-call.
        tpm_lim, est = await self._tpm_acquire(model_id, spec.system, spec.user)
        t0 = time.monotonic()
        # Hard wall-clock cap mirroring the streaming path: the http_options
        # timeout in `gen_config` bounds reads, wait_for bounds the whole call.
        coro = client.aio.models.generate_content(
            model=model_id, contents=parts, config=gen_config,
        )
        if spec.timeout_s is not None:
            api_resp = await asyncio.wait_for(coro, spec.timeout_s)
        else:
            api_resp = await coro
        latency_s = time.monotonic() - t0
        output_text = (api_resp.text or "").strip()
        toks = self._usage_tokens(api_resp.usage_metadata)
        self._tpm_settle(tpm_lim, est, toks["input_tokens"])
        return self._finish(spec, output_text, toks, latency_s, model_id)

    def _finish(
        self, spec: CallSpec, text: str, toks: dict, latency_s: float, model_id: str
    ) -> LLMResponse:
        """`_build_response` with the spec's envelope fields unpacked — the one tail
        every provider body calls."""
        return self._build_response(
            text, toks, latency_s,
            model=model_id, temperature=spec.temperature, effort=spec.effort,
            ctx=spec.ctx, call_site=spec.call_site,
        )

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
        """Multi-turn streaming call, accumulating chunks until `should_stop(acc)`
        or the stream ends. `messages` are the {role, content} turns after the
        system message ('assistant' → model role, else user). Lets multi-turn
        agents share this client's rate-limit + retry + logging; the event loop
        runs other tasks between chunks.

        `max_output_tokens` overrides the per-call output cap (None → provider
        default); `timeout_s` enforces a hard per-request wall-clock cap. Both are
        opt-in and currently used only by the search agent (see `MultiTurnAgent`)."""
        spec = CallSpec(
            system=system, messages=tuple(messages), temperature=temperature,
            effort=effort, ctx=ctx, call_site=call_site, should_stop=should_stop,
            model=model or self._config.llm_model,
            max_output_tokens=max_output_tokens, timeout_s=timeout_s,
        )
        impl = (
            self._openrouter_astream
            if self._config.llm_provider == "openrouter"
            else self._gemini_astream
        )
        return await self._aretry_call(
            lambda: impl(spec), self._config.llm_provider, spec.model, ctx=spec.ctx
        )

    async def _gemini_astream(self, spec: CallSpec) -> LLMResponse:
        client = self._get_gemini_client()
        model_id = self._gemini_model_id(spec.model)
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=self._gemini_parts(m["content"], m.get("images")),
            )
            for m in spec.messages or ()
        ]
        gen_config = self._gemini_config(
            spec.system, spec.temperature, spec.effort, model_id,
            spec.max_output_tokens if spec.max_output_tokens is not None else 65535,
            spec.timeout_s,
        )
        should_stop = spec.should_stop

        async def _consume() -> tuple[str, Any]:
            resp_stream = await client.aio.models.generate_content_stream(
                model=model_id, contents=contents, config=gen_config,  # type: ignore[arg-type]
            )
            accumulated = ""
            stopped_text: str | None = None
            usage = None
            try:
                async for chunk in resp_stream:
                    accumulated += chunk.text or ""
                    usage = getattr(chunk, "usage_metadata", None) or usage
                    # Keep draining past `should_stop` so the final usage_metadata is
                    # captured (parity with the openrouter path); record the cut point
                    # for the returned text. Bounded by max_output_tokens + timeout.
                    if stopped_text is None and should_stop is not None and should_stop(accumulated):
                        stopped_text = accumulated
            finally:
                # Runs on normal completion AND on wait_for cancellation, so a
                # timed-out stream still releases its connection.
                aclose = getattr(resp_stream, "aclose", None)
                if aclose is not None:
                    try:  # noqa: SIM105
                        await aclose()
                    except Exception:
                        pass
            return (accumulated if stopped_text is None else stopped_text), usage

        t0 = time.monotonic()
        if spec.timeout_s is not None:
            # Hard wall-clock cap: http_options.timeout is only per-read, so a
            # slow-but-steady runaway would never trip it. On timeout this raises
            # TimeoutError, which `_is_retryable` treats as a transient fault.
            accumulated, usage = await asyncio.wait_for(_consume(), spec.timeout_s)
        else:
            accumulated, usage = await _consume()
        latency_s = time.monotonic() - t0
        if usage is None:
            _warn(
                spec.ctx,
                f"stream_no_usage cost undercounted (call_site={spec.call_site} model={model_id})",
            )
        toks = self._usage_tokens(usage)
        return self._finish(spec, accumulated, toks, latency_s, model_id)

    # --- OpenRouter generation path (provider="openrouter") ---
    # Mirrors the Gemini helpers above: same _retry_call shape and shared `_build_response`
    # tail (ctx.emit envelope + LLMResponse), so the rate-limit/retry/logging is shared.

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
    def _openrouter_finish_reason(resp: Any) -> str | None:
        """First choice's `finish_reason` (e.g. 'stop' / 'length' / 'content_filter' /
        'error'), or None when there are no choices — surfaced in EmptyCompletionError so
        an empty completion's cause is visible in the retry-warning log."""
        choices = getattr(resp, "choices", None) or []
        return getattr(choices[0], "finish_reason", None) if choices else None

    @staticmethod
    def _openrouter_error_detail(resp: Any) -> str:
        """Diagnostic suffix for an empty OpenRouter completion. Two cases:

        - If OpenRouter populated the response-level `error` (code + a message that usually
          carries the provider's raw text — rate-limit, capacity, content-filter, 5xx),
          surface it: that's the actionable cause.
        - On a `finish_reason=error` from an upstream Gemini fault, OpenRouter instead
          returns a NULL message (content/reasoning/refusal all null) and NO error body —
          there is no "why" in the response. The only handle is the generation `id` (look it
          up at openrouter.ai/activity or `GET /api/v1/generation?id=<id>`), so surface that
          plus the resolved model. The null message also rules out a refusal or a
          length/thinking truncation — this is purely provider-side.

        Best-effort; returns "" when nothing useful is present."""
        err = getattr(resp, "error", None)
        if err:
            code = getattr(err, "code", None)
            msg = getattr(err, "message", None)
            parts = ([f"code={code}"] if code is not None else []) + (
                [f"message={str(msg)[:800]}"] if msg else []
            )
            return (" upstream_error: " + " ".join(parts)) if parts else ""
        gen_id = getattr(resp, "id", None)
        resolved = getattr(resp, "model", None)
        parts: list[str] = []
        if gen_id:
            # Emit the generation-lookup URL inline so a failure is one click from its
            # native provider error. NOTE: this `GET /generation` endpoint 404s for most
            # `finish_reason=error` generations (OpenRouter doesn't persist a generation
            # that failed upstream) and has a ~20s propagation delay even for successes —
            # the web dashboard (openrouter.ai/activity) is the more reliable place to see
            # the native error. The URL is still worth surfacing: it resolves for some
            # failure modes and is the handle for the activity page.
            parts.append(f"gen_id={gen_id}")
            parts.append(f"lookup=https://openrouter.ai/api/v1/generation?id={gen_id}")
        if resolved:
            parts.append(f"resolved_model={resolved}")
        return (" no_error_body (" + " ".join(parts) + ")") if parts else ""

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

    def _openrouter_single_messages(self, spec: CallSpec) -> list[dict]:
        """Single-shot messages in OpenRouter format: leading system (if any) + the
        one user turn (with any images) — shared by the sync and async bodies."""
        messages: list[dict] = []
        if spec.system:
            messages.append({"role": "system", "content": spec.system})
        messages.append(
            {"role": "user", "content": self._openrouter_content(spec.user, spec.images)}
        )
        return messages

    def _openrouter_checked_text(self, resp: Any, spec: CallSpec) -> str:
        """The completion text, or `EmptyCompletionError` (with the full diagnostic
        suffix chain) when it came back empty — shared by the sync and async bodies."""
        output_text = self._openrouter_text(resp).strip()
        if not output_text:
            raise EmptyCompletionError(
                f"openrouter empty content (model={spec.model} call_site={spec.call_site} "
                f"finish_reason={self._openrouter_finish_reason(resp)}"
                f"{self._openrouter_error_detail(resp)}){_raw_body_suffix()}"
            )
        return output_text

    def _openrouter_provider_kwarg(self, spec: "CallSpec | None" = None) -> dict:
        """`{"provider": {...}}` pinning generation to a provider order (no fallback), or `{}` when
        unset. A per-call `spec.provider_order` wins over the client-wide `config.llm_provider_order`,
        so one model (e.g. a cheaper semantic-filter judge) can be routed to specific providers while
        the rest of the client stays unpinned. Lets a run route deterministically to one provider so
        its prompt-cache and pricing are the ones actually used."""
        order = (spec.provider_order if spec else None) or getattr(self._config, "llm_provider_order", None)
        if not order:
            return {}
        return {"provider": {"order": list(order), "allow_fallbacks": False}}

    def _openrouter_call(self, spec: CallSpec) -> LLMResponse:
        """One OpenRouter chat call (no retry — the retry loop owns that)."""
        client = self._get_openrouter_client()
        messages = self._openrouter_single_messages(spec)
        reasoning = self._effort_to_reasoning(spec.effort)
        t0 = time.monotonic()
        resp = client.chat.send(
            model=spec.model, messages=messages, stream=False, # type: ignore
            temperature=spec.temperature, reasoning=reasoning, # type: ignore
            **self._openrouter_provider_kwarg(spec),
        )
        latency_s = time.monotonic() - t0
        output_text = self._openrouter_checked_text(resp, spec)
        toks = self._usage_tokens_openrouter(getattr(resp, "usage", None))
        return self._finish(spec, output_text, toks, latency_s, spec.model)

    async def _openrouter_acall(self, spec: CallSpec) -> LLMResponse:
        """Async twin of `_openrouter_call` — uses `client.chat.send_async`."""
        client = self._get_openrouter_client()
        messages = self._openrouter_single_messages(spec)
        reasoning = self._effort_to_reasoning(spec.effort)
        # TPM throttle (opt-in via SKUNK_MODEL_TPM) — identical to the Gemini path.
        tpm_lim, est = await self._tpm_acquire(spec.model, spec.system, spec.user)
        extra = (
            {"max_tokens": spec.max_output_tokens}
            if spec.max_output_tokens is not None
            else {}
        )
        extra.update(self._openrouter_provider_kwarg(spec))
        t0 = time.monotonic()
        coro = client.chat.send_async(
            model=spec.model, messages=messages, stream=False, # type: ignore
            temperature=spec.temperature, reasoning=reasoning, **extra, # type: ignore
        )
        resp = await (
            asyncio.wait_for(coro, spec.timeout_s) if spec.timeout_s is not None else coro
        )
        latency_s = time.monotonic() - t0
        output_text = self._openrouter_checked_text(resp, spec)
        toks = self._usage_tokens_openrouter(getattr(resp, "usage", None))
        self._tpm_settle(tpm_lim, est, toks["input_tokens"])
        return self._finish(spec, output_text, toks, latency_s, spec.model)

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

    async def _openrouter_astream(self, spec: CallSpec) -> LLMResponse:
        """OpenRouter streaming body for `astream` — `client.chat.send_async` + `async for`."""
        client = self._get_openrouter_client()
        or_messages = self._openrouter_chat_messages(spec.system, list(spec.messages or ()))
        reasoning = self._effort_to_reasoning(spec.effort)
        extra = (
            {"max_tokens": spec.max_output_tokens}
            if spec.max_output_tokens is not None
            else {}
        )
        extra.update(self._openrouter_provider_kwarg(spec))
        should_stop = spec.should_stop

        async def _consume() -> tuple[str, Any]:
            accumulated = ""
            stopped_text: str | None = None
            usage = None
            async with await client.chat.send_async(
                model=spec.model, messages=or_messages, stream=True, # type: ignore
                temperature=spec.temperature, reasoning=reasoning, **extra, # type: ignore
            ) as resp_stream:
                async for chunk in resp_stream:
                    accumulated += self._openrouter_chunk_text(chunk)
                    usage = getattr(chunk, "usage", None) or usage
                    # OpenRouter sends the usage chunk LAST, so we must drain the stream
                    # to capture cost stats — breaking at `should_stop` drops them
                    # entirely. Record the cut point for the returned text but keep
                    # reading; the agent is prompted to emit one block then stop, so the
                    # stream usually ends right after (max_output_tokens + the request
                    # timeout bound the worst case).
                    if stopped_text is None and should_stop is not None and should_stop(accumulated):
                        stopped_text = accumulated
            return (accumulated if stopped_text is None else stopped_text), usage

        t0 = time.monotonic()
        if spec.timeout_s is not None:
            accumulated, usage = await asyncio.wait_for(_consume(), spec.timeout_s)
        else:
            accumulated, usage = await _consume()
        latency_s = time.monotonic() - t0
        if usage is None:
            _warn(
                spec.ctx,
                f"stream_no_usage cost undercounted (call_site={spec.call_site} model={spec.model})",
            )
        toks = self._usage_tokens_openrouter(usage)
        return self._finish(spec, accumulated, toks, latency_s, spec.model)
