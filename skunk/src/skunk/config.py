"""Centralized configuration. All tuning constants live here, overridable via the
environment (env var noted next to each field). Construct via `SkunkConfig.from_env()`."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from skunk.common import Effort
    from skunk.common import PageRef


@dataclass
class SkunkConfig:
    # Orchestrator + extract fan-out; sized so a typical question is bounded by the
    # LLM RPM limiter, not the thread pool.
    max_parallel_workers: int = 16

    # LLM model — all calls go through GCP Vertex AI (bare model names, no `google/`
    # prefix). Needs GOOGLE_CLOUD_PROJECT + ADC. (env: SKUNK_LLM_MODEL)
    # LLM request pacing is a process-wide rate limit (env: SKUNK_LLM_RPM), owned by
    # `common._RATE_LIMITS` alongside every other external service, not by config.
    llm_model: str = "gemini-3.5-flash"
    # Per-call retry: any SDK exception, delay doubling each attempt up to the cap.
    llm_max_retries: int = 10
    llm_retry_initial_delay_s: float = 0.05
    llm_retry_max_delay_s: float = 1.0

    # Per call-site effort override: `PromptedCall.name` → Effort tier. Missing key →
    # the call-site's `default_effort`; an explicit `effort=` arg still wins over both.
    # (env: SKUNK_EFFORT_OVERRIDES — comma-separated `name=tier` pairs)
    effort_overrides: dict[str, "Effort"] = field(default_factory=dict)

    # Extract operator (env: SKUNK_EXTRACT_N_SAMPLES, SKUNK_EXTRACT_SAMPLE_TEMPERATURE)
    extract_n_samples: int = 1       # single-sample; multi-sample + dedup disabled (no measured accuracy gain)
    extract_sample_temperature: float = 0.7
    extract_max_pages: int = 5       # page cap per tier

    # Compute operator
    compute_max_attempts: int = 3

    # Replan-on-MissingData loop. Total compute invocations ≤ recovery_max_rounds + 1.
    recovery_max_rounds: int = 2

    # Prompt overrides YAML — corpus blurbs, few-shots, lessons. (env: SKUNK_PROMPT_OVERRIDES)
    prompt_overrides_path: str = "config/prompts/treasury_bulletin.yaml"

    # Ablation: golden page refs bypass the retrieve operator (eval runs only).
    golden_pages: list[PageRef] | None = field(default=None, repr=False)

    # Retrieve dispatch: "search_agent" (iterative ChromaDB + LLM loop) or "page_index"
    # (ToC pick → year filter → semantic filter). (env: SKUNK_RETRIEVER)
    retriever: Literal["search_agent", "page_index"] = "search_agent"

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

    # Step cap for the lookup_external agent (terminates earlier via `final_answer`).
    # (env: SKUNK_LOOKUP_MAX_STEPS)
    lookup_max_steps: int = 8
    # Active lookup tools by name (see `lookup_tools._REGISTRY`); None → all tools.
    # (env: SKUNK_LOOKUP_TOOLS — comma-separated, e.g. "fetch_fred,tavily_search")
    lookup_tools: list[str] | None = None

    # Agent-loop chat model. None → `llm_model`. (env: SKUNK_AGENT_MODEL)
    agent_model_id: str | None = None

    # Page-index semantic filter: two-stage (coarse metadata → fine page text) cascade,
    # batched and parallel. Set False for a ToC + year-filter-only ablation.
    # (env: SKUNK_SEMFILTER_ENABLED, SKUNK_SEMFILTER_BATCH, SKUNK_SEMFILTER_WORKERS, SKUNK_SEMFILTER_MAX_PAGE_CHARS)
    semfilter_enabled: bool = True
    semfilter_batch_size: int = 20
    semfilter_workers: int = 16
    semfilter_max_page_chars: int = 12000

    @classmethod
    def from_env(cls) -> SkunkConfig:
        return cls(
            llm_model=os.environ.get("SKUNK_LLM_MODEL", "gemini-3.5-flash"),
            effort_overrides=_parse_effort_overrides(os.environ.get("SKUNK_EFFORT_OVERRIDES", "")),
            llm_max_retries=int(os.environ.get("SKUNK_LLM_MAX_RETRIES", "10")),
            llm_retry_initial_delay_s=float(os.environ.get("SKUNK_LLM_RETRY_INITIAL_DELAY", "0.05")),
            llm_retry_max_delay_s=float(os.environ.get("SKUNK_LLM_RETRY_MAX_DELAY", "1.0")),
            extract_n_samples=int(os.environ.get("SKUNK_EXTRACT_N_SAMPLES", "1")),
            extract_sample_temperature=float(os.environ.get("SKUNK_EXTRACT_SAMPLE_TEMPERATURE", "0.7")),
            prompt_overrides_path=os.environ.get(
                "SKUNK_PROMPT_OVERRIDES", "config/prompts/treasury_bulletin.yaml"
            ),
            semfilter_enabled=os.environ.get("SKUNK_SEMFILTER_ENABLED", "true").lower() in ("1", "true", "yes"),
            semfilter_batch_size=int(os.environ.get("SKUNK_SEMFILTER_BATCH", "20")),
            semfilter_workers=int(os.environ.get("SKUNK_SEMFILTER_WORKERS", "16")),
            semfilter_max_page_chars=int(os.environ.get("SKUNK_SEMFILTER_MAX_PAGE_CHARS", "12000")),
            retriever=os.environ.get("SKUNK_RETRIEVER", "search_agent"),  # type: ignore[arg-type]
            chromadb_dir=os.environ.get("SKUNK_CHROMADB_DIR", "cache/chromadb"),
            chromadb_collection=os.environ.get("SKUNK_CHROMADB_COLLECTION", "treasury_pages"),
            clean_page_map_path=os.environ.get("SKUNK_CLEAN_PAGE_MAP", "cache/clean_page_map.json"),
            emb_model_id=os.environ.get("SKUNK_EMB_MODEL", "gemini-embedding-001"),
            agent_max_steps=int(os.environ.get("SKUNK_AGENT_MAX_STEPS", "20")),
            agent_max_pages_per_tool_call=int(os.environ.get("SKUNK_AGENT_MAX_PAGES_PER_TOOL_CALL", "20")),
            lookup_max_steps=int(os.environ.get("SKUNK_LOOKUP_MAX_STEPS", "8")),
            lookup_tools=_parse_csv(os.environ.get("SKUNK_LOOKUP_TOOLS", "")),
            agent_model_id=os.environ.get("SKUNK_AGENT_MODEL") or None,
        )


def _parse_effort_overrides(raw: str) -> dict[str, "Effort"]:
    """Parse `SKUNK_EFFORT_OVERRIDES` ("name=tier,...") into a dict. Raises ValueError
    on malformed entries / unknown tiers so config typos fail loudly at startup."""
    from skunk.common import _EFFORT_VALUES  # local to avoid import cycle

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
        if tier not in _EFFORT_VALUES:
            raise ValueError(
                f"SKUNK_EFFORT_OVERRIDES tier {tier!r} not in {_EFFORT_VALUES}"
            )
        out[name] = tier  # type: ignore[assignment]
    return out


def _parse_csv(raw: str) -> list[str] | None:
    """Parse a comma-separated env var into a list of trimmed entries, or None when
    unset/empty (so the field falls back to its default)."""
    items = [s.strip() for s in raw.split(",") if s.strip()]
    return items or None
