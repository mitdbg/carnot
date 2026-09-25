"""Shared runtime: the process-wide rate limiters plus the `ExecutionContext`.

The rate limiters (`_RATE_LIMITS` / `get_rate_limiter`) pace the lookup agent's external data
APIs (FRED, BLS, World Bank, Tavily), some of which ban a key outright when their cap is tripped.
LLM calls are NOT paced here: `skunk.llm_client` relies on retry with backoff instead."""

from __future__ import annotations

import asyncio
import base64
import os
import re
import threading
import time
from chromadb import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, get_args

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
    reads its env override (once); any other name requires an explicit `rate_per_min`."""
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
    key off `_RATE_LIMITS` (env override); other names pass an explicit `rate_per_min`. All
    threads in this process share the named bucket.

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


@dataclass
class B64Image:
    """A single rendered page image ready to pass to an LLM: MIME type + base64 data."""

    mime: str
    data: str


def render_page_b64(
    pdf_path: Path,
    page_num: int,
    *,
    renders_dir: Path | str | None = None,
    dpi: int = 300,
    fmt: str = "png",
    jpg_quality: int = 95,
) -> B64Image | None:
    """The single PDF-page rasterizer for the repo: render one page to in-memory image bytes via
    PyMuPDF.

    When `renders_dir` is given, a pre-rendered PNG at `<renders_dir>/<pdf_path.stem>_<page_num>.png`
    (the page render cache; e.g. "treasury_bulletin_1951_06_57.png" for page 57 of
    treasury_bulletin_1951_06.pdf) is served directly — skipping the PDF open. `dpi`/`fmt`/`jpg_quality` are
    ignored for a cache hit (the cached PNG's own resolution applies); on a miss we fall back to
    rasterizing the PDF."""
    if renders_dir is not None:
        cached = Path(renders_dir) / f"{pdf_path.stem}_{page_num}.png"
        if cached.exists():
            return B64Image(
                mime="image/png",
                data=base64.standard_b64encode(cached.read_bytes()).decode(),
            )

    if not pdf_path.exists():
        return None
    import fitz

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    with fitz.open(pdf_path) as doc:
        pix = doc[page_num - 1].get_pixmap(matrix=mat)
    if fmt == "jpg":
        data = pix.tobytes("jpg", jpg_quality=jpg_quality)
        mime = "image/jpeg"
    else:
        data = pix.tobytes("png")
        mime = "image/png"

    return B64Image(mime=mime, data=base64.standard_b64encode(data).decode())


@dataclass(frozen=True)
class PageSource:
    """Dataclass with metadata for locating the filepath and page number for a given doc_id."""
    # path to the PDF file containing this page
    pdf_path: Path
    # 1-indexed location of the page within the PDF
    page_num: int


class PageLocator(Protocol):
    """The doc_id --> PageSource lookup for a given benchmark. Returns None if the doc_id has no renderable page."""
    def lookup(self, doc_id: str) -> PageSource | None: ...


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
