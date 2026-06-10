"""Centralized configuration. All tuning constants live here, overridable via the
environment (env var noted next to each field). Construct via `SkunkConfig.from_env()`."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from skunk.common import Effort
    from skunk.common import PageRef


# Default corpus locations (overridable via env / explicit construction — see the
# `parsed_json_dir` / `pdf_dir` fields below). Homed here so `SkunkConfig` is the single
# source of truth for where the corpus lives; `corpus.py` resolves through it.
_DEFAULT_PARSED_JSON_DIR = Path.home() / "Desktop/officeqa/treasury_bulletins_parsed/jsons"
_DEFAULT_PDF_DIR = Path.home() / "Desktop/officeqa/treasury_bulletin_pdfs"


@dataclass
class SkunkConfig:
    # LLM model — all calls go through the AI Studio Gemini API (bare model names,
    # no `google/` prefix). Needs GEMINI_API_KEY. (env: SKUNK_LLM_MODEL)
    # LLM request pacing is a process-wide rate limit (env: SKUNK_LLM_RPM, plus
    # per-model overrides via SKUNK_MODEL_RPM), owned by `common._RATE_LIMITS` /
    # `llm_client._llm_model_rpm` alongside every other external service, not by config.
    llm_model: str = "gemini-3.5-flash"
    # LLM provider backend for all generation calls. "genai" (default) → Google AI
    # Studio Gemini SDK (needs GEMINI_API_KEY; bare model names). "openrouter" →
    # OpenRouter chat API (needs OPENROUTER_API_KEY); SKUNK_LLM_MODEL must then be a
    # full OpenRouter model id (e.g. "google/gemini-2.5-flash",
    # "qwen/qwen-2.5-72b-instruct"). Only generation is routed; embeddings
    # (LLMClient.embed / search-agent indexing) stay on their own model-id dispatch.
    # (env: SKUNK_LLM_PROVIDER)
    llm_provider: Literal["genai", "openrouter"] = "genai"
    # Per-call retry: only transient failures (HTTP 429 + 5xx, network timeouts /
    # connection resets) are retried — see `llm_client._is_retryable`; non-429 4xx
    # (bad request, auth, context overflow) raises immediately. Delay doubles each
    # attempt; the retry count bounds total wait on its own (1→2→4→8→16, ~31s
    # over 5 retries), so no separate delay cap is needed.
    llm_max_retries: int = 5
    llm_retry_initial_delay_s: float = 1.0

    # Per call-site effort override: `PromptedCall.name` → Effort tier. Missing key →
    # the call-site's `default_effort`; an explicit `effort=` arg still wins over both.
    # (env: SKUNK_EFFORT_OVERRIDES — comma-separated `name=tier` pairs)
    effort_overrides: dict[str, Effort] = field(default_factory=dict)

    # Per call-site model override: `PromptedCall.name` → model id. Missing key →
    # `llm_model`. Lets one run mix models (e.g. a strong default with cheap Flash
    # pinned on `question_explainer`). Agent loops use
    # `agent_model_id` instead. Each model is paced by its own RPM bucket — see
    # `llm_client._llm_model_rpm` (env SKUNK_MODEL_RPM).
    # (env: SKUNK_MODEL_OVERRIDES — comma-separated `name=model` pairs)
    model_overrides: dict[str, str] = field(default_factory=dict)

    # Compute operator
    compute_max_attempts: int = 3

    # Replan-on-MissingData loop. Total compute invocations ≤ recovery_max_rounds + 1.
    recovery_max_rounds: int = 2

    # Corpus directories. `parsed_json_dir` holds the per-bulletin parsed-JSON the text
    # accessors read (cleaner than PyMuPDF text on scanned pages); `pdf_dir` holds the
    # source bulletin PDFs used for page rendering. `corpus.py` resolves through these.
    # (env: OFFICEQA_PARSED_JSON_DIR, OFFICEQA_PDF_DIR)
    parsed_json_dir: Path = field(default_factory=lambda: _DEFAULT_PARSED_JSON_DIR)
    pdf_dir: Path = field(default_factory=lambda: _DEFAULT_PDF_DIR)

    # Prompt overrides YAML — corpus blurbs, few-shots, lessons. (env: SKUNK_PROMPT_OVERRIDES)
    prompt_overrides_path: str = "config/prompts/treasury_bulletin.yaml"

    # Ablation: golden page refs bypass the retrieve operator (eval runs only).
    golden_pages: list[PageRef] | None = field(default=None, repr=False)

    # Retrieve dispatch: "page_index" (ToC pick → year filter → coarse summary filter,
    # the default) or "search_agent" (iterative ChromaDB + LLM loop). (env: SKUNK_RETRIEVER)
    retriever: Literal["search_agent", "page_index"] = "page_index"

    # Search-agent corpus artifacts (built offline; agent fails fast if missing).
    # (env: SKUNK_CHROMADB_DIR, SKUNK_CHROMADB_COLLECTION, SKUNK_CLEAN_PAGE_MAP)
    chromadb_dir: str = "cache/chromadb"
    chromadb_collection: str = "treasury_pages"
    clean_page_map_path: str = "cache/clean_page_map.json"

    # Embedding model for vector_search (must match the stored embeddings).
    # (env: SKUNK_EMB_MODEL)
    emb_model_id: str = "gemini-embedding-001"

    # Per-question search-agent budget. (env: SKUNK_AGENT_MAX_STEPS, SKUNK_AGENT_MAX_PAGES_PER_TOOL_CALL)
    agent_max_steps: int = 20
    agent_max_pages_per_tool_call: int = 20

    # Search-agent per-LLM-call caps (RetrieveOp only). On Gemini 3.x
    # `max_output_tokens` is a COMBINED thinking+visible budget, so keep it above the
    # effort tier's thinking spend (medium ≈ 2.5K, high ≈ 16K thinking tokens) or the
    # visible answer is starved to empty (finish_reason=MAX_TOKENS). 4096 is safe at
    # the default medium effort; raise it if the search agent is moved to high.
    # `request_timeout_s` is a hard per-request wall-clock cap (retried on trip).
    # (env: SKUNK_SEARCH_MAX_OUTPUT_TOKENS, SKUNK_SEARCH_TIMEOUT_S)
    search_agent_max_output_tokens: int = 4096
    search_agent_request_timeout_s: float = 120.0

    # Step cap for the lookup_external agent (terminates earlier via its final-answer JSON block).
    # (env: SKUNK_LOOKUP_MAX_STEPS)
    lookup_max_steps: int = 8
    # Active lookup tools by name (see `lookup_tools._REGISTRY`); None → all tools.
    # (env: SKUNK_LOOKUP_TOOLS — comma-separated, e.g. "fetch_fred,tavily_search")
    lookup_tools: list[str] | None = None

    # Agent-loop chat model. None → `llm_model`. (env: SKUNK_AGENT_MODEL)
    agent_model_id: str | None = None

    # Page-index semantic filter: a single coarse pass over each year-filtered page's
    # metadata summary, judged against the full question on `semfilter_model`
    # (positional true/false per page). Set `semfilter_enabled=False` for a
    # ToC + year-filter-only ablation.
    # (env: SKUNK_SEMFILTER_ENABLED, SKUNK_SEMFILTER_MODEL, SKUNK_SEMFILTER_BATCH)
    semfilter_enabled: bool = True
    semfilter_model: str = "gemini-3.1-flash-lite"
    semfilter_batch_size: int = 20


    @classmethod
    def from_env(cls) -> SkunkConfig:
        return cls(
            llm_model=os.environ.get("SKUNK_LLM_MODEL", "gemini-3.5-flash"),
            llm_provider=os.environ.get("SKUNK_LLM_PROVIDER", "genai"),  # type: ignore[arg-type]
            effort_overrides=_parse_effort_overrides(os.environ.get("SKUNK_EFFORT_OVERRIDES", "")),
            model_overrides=_parse_model_overrides(os.environ.get("SKUNK_MODEL_OVERRIDES", "")),
            llm_max_retries=int(os.environ.get("SKUNK_LLM_MAX_RETRIES", "5")),
            llm_retry_initial_delay_s=float(os.environ.get("SKUNK_LLM_RETRY_INITIAL_DELAY", "1.0")),
            parsed_json_dir=Path(os.environ.get("OFFICEQA_PARSED_JSON_DIR") or _DEFAULT_PARSED_JSON_DIR),
            pdf_dir=Path(os.environ.get("OFFICEQA_PDF_DIR") or _DEFAULT_PDF_DIR),
            prompt_overrides_path=os.environ.get(
                "SKUNK_PROMPT_OVERRIDES", "config/prompts/treasury_bulletin.yaml"
            ),
            semfilter_enabled=os.environ.get("SKUNK_SEMFILTER_ENABLED", "true").lower() in ("1", "true", "yes"),
            semfilter_model=os.environ.get("SKUNK_SEMFILTER_MODEL", "gemini-3.1-flash-lite"),
            semfilter_batch_size=int(os.environ.get("SKUNK_SEMFILTER_BATCH", "20")),
            retriever=os.environ.get("SKUNK_RETRIEVER", "page_index"),  # type: ignore[arg-type]
            chromadb_dir=os.environ.get("SKUNK_CHROMADB_DIR", "cache/chromadb"),
            chromadb_collection=os.environ.get("SKUNK_CHROMADB_COLLECTION", "treasury_pages"),
            clean_page_map_path=os.environ.get("SKUNK_CLEAN_PAGE_MAP", "cache/clean_page_map.json"),
            emb_model_id=os.environ.get("SKUNK_EMB_MODEL", "gemini-embedding-001"),
            agent_max_steps=int(os.environ.get("SKUNK_AGENT_MAX_STEPS", "20")),
            agent_max_pages_per_tool_call=int(os.environ.get("SKUNK_AGENT_MAX_PAGES_PER_TOOL_CALL", "20")),
            search_agent_max_output_tokens=int(os.environ.get("SKUNK_SEARCH_MAX_OUTPUT_TOKENS", "4096")),
            search_agent_request_timeout_s=float(os.environ.get("SKUNK_SEARCH_TIMEOUT_S", "120")),
            lookup_max_steps=int(os.environ.get("SKUNK_LOOKUP_MAX_STEPS", "8")),
            lookup_tools=_parse_csv(os.environ.get("SKUNK_LOOKUP_TOOLS", "")),
            agent_model_id=os.environ.get("SKUNK_AGENT_MODEL") or None,
        )


def _parse_effort_overrides(raw: str) -> dict[str, "Effort"]:
    """Parse `SKUNK_EFFORT_OVERRIDES` ("name=tier,...") into a dict. Raises ValueError
    on malformed entries / unknown tiers so config typos fail loudly at startup."""
    from skunk.common import EFFORT_VALUES  # local to avoid import cycle

    out: dict[str, Effort] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"SKUNK_EFFORT_OVERRIDES entry {entry!r} missing '=' "
                f"(expected `name=tier`)"
            )
        name, _, tier = entry.partition("=")
        name = name.strip()
        tier = tier.strip()
        if tier not in EFFORT_VALUES:
            raise ValueError(
                f"SKUNK_EFFORT_OVERRIDES tier {tier!r} not in {EFFORT_VALUES}"
            )
        out[name] = tier  # type: ignore[assignment]
    return out


def _parse_model_overrides(raw: str) -> dict[str, str]:
    """Parse `SKUNK_MODEL_OVERRIDES` ("name=model,...") into a dict keyed by
    `PromptedCall.name`. Model ids are free-form (no enum to validate against);
    an unknown id surfaces as an API error at the call site. Raises ValueError on
    a missing '=' so config typos fail loudly at startup."""
    out: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"SKUNK_MODEL_OVERRIDES entry {entry!r} missing '=' "
                f"(expected `name=model`)"
            )
        name, _, model = entry.partition("=")
        name = name.strip()
        model = model.strip()
        if not model:
            raise ValueError(f"SKUNK_MODEL_OVERRIDES entry {entry!r} has empty model")
        out[name] = model
    return out


def _parse_csv(raw: str) -> list[str] | None:
    """Parse a comma-separated env var into a list of trimmed entries, or None when
    unset/empty (so the field falls back to its default)."""
    items = [s.strip() for s in raw.split(",") if s.strip()]
    return items or None
