"""Shared runtime: the process-wide rate limiters plus the `ExecutionContext`.

The LLM client itself lives in `skunk.llm_client`; this module owns the external
rate-limit infrastructure it (and the data-source tools) pace against — the
process-wide token-bucket limiters (see `_RATE_LIMITS` / `get_rate_limiter`)."""

from __future__ import annotations

import asyncio
import base64
import os
import re
import threading
import time
from chromadb import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

import tiktoken

from skunk.config import SkunkConfig
from skunk.constants import CHARS_PER_TOKEN_EST
from skunk.trace import Tracer

if TYPE_CHECKING:
    from skunk.llm_client import LLMClient
    from skunk.storage.document_map import DocumentMap

# Reasoning-effort knob, mapped onto Gemini's `thinking_level` enum. "off" means
# no thinking; "minimal" is the cheapest thinking tier.
Effort = Literal["off", "minimal", "low", "medium", "high"]
EFFORT_VALUES = get_args(Effort)

@dataclass
class B64Image:
    """A single rendered page image ready to pass to an LLM: MIME type + base64 data."""

    mime: str
    data: str


def pdf_path_for(doc: str, pdf_dir: Path | str) -> Path:
    """The corpus PDF for a doc id: `<pdf_dir>/<doc>.pdf`. `doc` is the filename stem
    (e.g. "combined_statement__modern__2024__c40") — the same key the page index uses."""
    return Path(pdf_dir) / f"{doc}.pdf"


def render_page_b64(
    stem: str | None,
    page: int | None,
    *,
    pdf_dir: Path | str,
    renders_dir: Path | str | None = None,
    dpi: int = 300,
    fmt: str = "png",
    jpg_quality: int | None = None,
) -> B64Image | None:
    """The single PDF-page rasterizer for the repo: render one page to in-memory image bytes via
    PyMuPDF. `stem` is the doc-id stem naming the PDF (`<pdf_dir>/<stem>.pdf`). Returns None when
    the PDF doesn't exist; PyMuPDF errors propagate. `fitz` is imported lazily so importing
    `common` doesn't pull in PyMuPDF.

    When `renders_dir` is given, a pre-rendered PNG at `<renders_dir>/<stem>_<page>.png` (the page
    render cache) is served directly — skipping the PDF open. `dpi`/`fmt`/`jpg_quality` are
    ignored for a cache hit (the cached PNG's own resolution applies); on a miss we fall back to
    rasterizing the PDF."""
    if stem is None or page is None or int(page) <= 0:
        return None
    if renders_dir is not None:
        cached = Path(renders_dir) / f"{stem}_{int(page)}.png"
        if cached.exists():
            return B64Image(
                mime="image/png",
                data=base64.standard_b64encode(cached.read_bytes()).decode(),
            )
    pdf_path = pdf_path_for(stem, pdf_dir)
    if not pdf_path.exists():
        return None
    import fitz

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    with fitz.open(pdf_path) as doc:
        pix = doc[int(page) - 1].get_pixmap(matrix=mat)
    if fmt == "jpg":
        data = pix.tobytes(
            "jpg", jpg_quality=jpg_quality if jpg_quality is not None else 95
        )
        mime = "image/jpeg"
    else:
        data = pix.tobytes("png")
        mime = "image/png"
    return B64Image(mime=mime, data=base64.standard_b64encode(data).decode())


class _RateLimiter:
    """Process-wide token-bucket rate limiter with BOTH interfaces over ONE bucket.

    `acquire()` blocks the calling thread; `acquire_async()` yields to the event loop. Both
    draw from the same tokens, so a name's rate cap holds whether it's hit from sync code
    (offline build, the `requests`-based lookup tools, embeddings prep, the sync LLM path) or
    from async code (the async request-path LLM calls).

    Thread-safe: a `threading.Condition` guards the refill+deduct. The async path holds that
    lock only across the (non-awaiting) token math and sleeps via `asyncio.sleep` OUTSIDE it,
    so it never blocks its event loop. Sync waiters block on `cond.wait(timeout)` and re-check
    on timeout, so no notify is needed when the async path deducts (and vice versa)."""

    def __init__(self, rate_per_sec: float) -> None:
        if rate_per_sec <= 0:
            raise ValueError(f"rate_per_sec must be > 0 (got {rate_per_sec})")
        self._rate = rate_per_sec
        # Burst capacity = ~1s of refill, floored at 1 token so sub-1/s rates still work.
        self._capacity = max(1.0, rate_per_sec)
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def acquire(self) -> None:
        """Sync: block the calling thread until 1 slot is free, then deduct it."""
        with self._cond:
            while True:
                self._refill_locked()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_s = (1.0 - self._tokens) / self._rate
                self._cond.wait(timeout=wait_s)

    async def acquire_async(self) -> None:
        """Async: yield to the loop until 1 slot is free, then deduct it. The lock is held
        only across the synchronous token math; the wait is `await asyncio.sleep` outside it."""
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_s = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait_s)


# Single source of truth for every external rate cap we pace against. Each row is
# `name: (env override, default rpm)`. The rpm is read ONCE, on first use of the
# name, and frozen for the rest of the run — buckets are created on demand and
# never rebuilt (rate limits are immutable for a run). To add a service: add a row
# and call `get_rate_limiter("<name>")` at the call site.
_RATE_LIMITS: dict[str, tuple[str, float]] = {
    # name          (env override,            default rpm)  # rationale
    "embed": (
        "SKUNK_EMBED_RPM",
        600.0,
    ),  # Gemini embeddings (search_agent.vector_search)
    "fred": ("SKUNK_FRED_RPM", 60.0),  # hard 120/min per API key; stay well under —
    # FRED escalates to an extended key-wide ban
    # (persistent 429s) once the cap is tripped.
    "bls": ("SKUNK_BLS_RPM", 50.0),  # 500/day (registered); smooth worker bursts
    "world_bank": (
        "SKUNK_WORLD_BANK_RPM",
        120.0,
    ),  # no published cap; stay a good citizen
    "tavily": ("SKUNK_TAVILY_RPM", 100.0),  # ~100/min on the dev tier
}

_LIMITERS_LOCK = threading.Lock()
_LIMITERS: dict[str, _RateLimiter] = {}


def _resolve_rate_per_sec(name: str, rate_per_min: float | None) -> float:
    """Rate (req/sec) for a limiter bucket. A known service in `_RATE_LIMITS`
    reads its env override (once); any other name (e.g. a per-model `llm:<model>`
    bucket) requires an explicit `rate_per_min`."""
    if name in _RATE_LIMITS:
        env_var, default = _RATE_LIMITS[name]
        return float(os.environ.get(env_var, default)) / 60.0
    if rate_per_min is None:
        raise KeyError(f"unknown rate-limit name {name!r} and no rate_per_min given")
    return rate_per_min / 60.0


def get_rate_limiter(name: str, rate_per_min: float | None = None) -> _RateLimiter:
    """Process-wide token-bucket limiter for service `name`, paced at its rpm — read ONCE at
    first use, then frozen for the run. ONE bucket per name, serving BOTH sync `acquire()` and
    async `acquire_async()` callers (so the cap holds across sync+async use). Known services
    key off `_RATE_LIMITS` (env override); dynamic buckets (per-model `llm:<model>`) pass an
    explicit `rate_per_min`. All threads in this process share the named bucket.

    Scope is per process, NOT per API key: separate processes get independent
    buckets and do not coordinate, so a multi-process deployment would not be
    bounded by the underlying per-key cap. The eval harness is single-process."""
    with _LIMITERS_LOCK:
        lim = _LIMITERS.get(name)
        if lim is None:
            lim = _RateLimiter(rate_per_sec=_resolve_rate_per_sec(name, rate_per_min))
            _LIMITERS[name] = lim
        return lim


# ---------------------------------------------------------------------------
# Tokens-per-minute (TPM) throttle — async; per-model caps come from the
# InferenceConfig (`llm_model_tpm`, falling back to `llm_default_tpm`).
#
# The RPM limiter alone can't bound token throughput: one request can carry tens
# of thousands of tokens, so a request-paced run still blows a TPM quota (the
# full-text page-index filter pushed ~36M tok/min and 429-stormed). This bucket
# meters estimated *input* tokens per call. Mirrors `_RateLimiter.acquire_async`'s
# cross-loop safety (threading.Lock around refill+deduct, sleep outside the lock).
# Inert for any model whose effective TPM is falsy (None/0) — paced by RPM alone.
# Housed here so BOTH process-wide pacing registries (RPM above, TPM below) live
# in one module with one idiom.
# ---------------------------------------------------------------------------


class AsyncTokenBudget:
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
_ASYNC_TPM_LIMITERS: dict[str, AsyncTokenBudget] = {}


def get_async_tpm_limiter(model: str, tpm: float) -> AsyncTokenBudget:
    """Process-wide TPM bucket for `model`, paced at `tpm` tokens/min. Capacity is
    ~4s of budget so a few large concurrent requests can burst, then throttle."""
    with _ASYNC_TPM_LOCK:
        lim = _ASYNC_TPM_LIMITERS.get(model)
        if lim is None:
            rate_per_sec = tpm / 60.0
            lim = AsyncTokenBudget(rate_per_sec, capacity=max(rate_per_sec * 4.0, 256_000.0))
            _ASYNC_TPM_LIMITERS[model] = lim
        return lim


# ---------------------------------------------------------------------------
# Token estimation — the one place text is converted to a token count.
# ---------------------------------------------------------------------------

# Above this many characters (~1M tokens at ~4 chars/token), `estimate_tokens` falls back
# to the char-count heuristic: exact BPE encoding is fast and fits in modest memory (~8GB)
# up to roughly this size, but beyond it the encode cost outweighs the estimate's error.
_EXACT_TOKENIZE_MAX_CHARS = 4_000_000

# Lazily built on first `estimate_tokens` call so importing `common` doesn't pay the
# BPE-table load. Benign race: concurrent first calls both build; one assignment wins.
_token_encoder: tiktoken.Encoding | None = None


def estimate_tokens(s: str) -> int:
    """Input-token estimate for `s`: an exact BPE count (o200k_base) for strings up to
    `_EXACT_TOKENIZE_MAX_CHARS` chars, falling back to the ~4 chars/token heuristic
    (`CHARS_PER_TOKEN_EST`) beyond that. Still an estimate, not a billing figure — the
    provider's tokenizer may differ from o200k_base — but far tighter than chars/4."""
    if len(s) > _EXACT_TOKENIZE_MAX_CHARS:
        return len(s) // CHARS_PER_TOKEN_EST
    global _token_encoder
    if _token_encoder is None:
        _token_encoder = tiktoken.get_encoding("o200k_base")
    return len(_token_encoder.encode(s))


_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\n?")


def strip_code_fence(s: str) -> str:
    """Strip a leading/trailing ```lang fence from an LLM response body."""
    s = s.strip()
    if s.startswith("```"):
        s = _CODE_FENCE_RE.sub("", s)
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


# --- Cross-cutting runtime types threaded between operators and the orchestrator ---


@dataclass(frozen=True)
class PageRef:
    """Canonical page coordinate. Frozen so it's hashable — usable as a dict key
    and set member (e.g. the page-index catalog is keyed by `PageRef`)."""

    stem: str | None = None  # doc-id stem, e.g. "combined_statement__historical__cs-1872" (search-agent / page-index key)
    page: int | None = None  # 1-based PDF page index (canonical)

    @property
    def year(self) -> int | None:
        """The calendar year for a `YYYY-MM` slot, or None. Returns None when the slot is not a
        bare month (the rekeyed page index stores a doc-id stem here — its year is parsed from
        the stem via `corpus.parse_doc_id`, not from this coordinate)."""
        if not self.stem or not self.stem[:4].isdigit():
            return None
        return int(self.stem[:4])

    def __post_init__(self) -> None:
        if self.page is not None and self.stem is None:
            raise ValueError(
                f"PageRef with page={self.page} requires stem for parsed-JSON lookup"
            )

    def __repr__(self) -> str:
        parts = []
        if self.year:
            parts.append(f"year={self.year}")
        if self.stem:
            parts.append(f"stem={self.stem}")
        if self.page is not None:
            parts.append(f"page={self.page}")
        return f"PageRef({', '.join(parts)})"


def page_key_to_pageref(key: str) -> PageRef:
    """Parse a search-agent page key `"<stem>_<page>"` into a `PageRef`. The doc-id stem (which
    itself contains underscores, e.g. `"combined_statement__historical__cs-1872"`) goes in the
    `stem` slot; the trailing integer is the 1-based page index."""
    try:
        stem, page_str = key.rsplit("_", 1)
        return PageRef(stem=stem, page=int(page_str))
    except ValueError as e:
        raise ValueError(f"page key {key!r} not in '<stem>_<page>' form") from e


# TODO: maybe make Tracer.emit() non-blocking?
# TODO: put chroma_collection & document map into StorageManager?
@dataclass
class ExecutionContext:
    """Execution state for a single instance of a Skunk agent."""

    llm_client: LLMClient
    config: SkunkConfig
    tracer: Tracer
    document_map: DocumentMap
    chroma_collection: Collection

    @classmethod
    def build(
        cls,
        config: SkunkConfig,
        *,
        document_map: DocumentMap,
        chroma_collection: Collection,
        tracer: Tracer,
    ) -> ExecutionContext:
        """
        This constructor builds the LLMClient from `config.inference`, so `llm_client`'s config
        and `config.inference` are the same object by construction. Tests inject stubs via the plain
        constructor instead.
        """
        # local import to avoid circular import at top-level (`skunk.llm_client` imports from this module)
        from skunk.llm_client import LLMClient

        return cls(
            config=config,
            llm_client=LLMClient(config.inference, tracer=tracer),
            tracer=tracer,
            document_map=document_map,
            chroma_collection=chroma_collection,
        )
