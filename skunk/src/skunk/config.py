"""Centralized configuration for the skunk harness.

All tuning constants live here. Values can be overridden via environment variables
(documented next to each field). Construct via SkunkConfig.from_env() or pass
custom values directly for tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from skunk.common import Effort
    from skunk.models import PageRef


@dataclass
class SkunkConfig:
    # Orchestrator + extract fan-out. Sized so a typical question
    # (≤3 branches × ≤5 pages × ≤3 samples) is bounded by the LLM RPM
    # limiter rather than the thread pool.
    max_parallel_workers: int = 16

    # LLM model — all calls (planner, retrieve, extract, compute, lookup_external)
    # go through GCP Vertex AI. Use bare Vertex model names (no `google/` prefix).
    # Requires GOOGLE_CLOUD_PROJECT in the environment and ADC set up via
    # `gcloud auth application-default login`. GOOGLE_CLOUD_LOCATION defaults to
    # us-central1. (env: SKUNK_LLM_MODEL, SKUNK_LLM_RPM)
    llm_model: str = "gemini-3.5-flash"
    # Token-bucket rate limit applied to every LLM call (requests per minute).
    # Default sized for Vertex Gemini Flash paid-tier quotas; adjust for other models.
    llm_rpm: float = 1000.0
    # Per-call retry with exponential backoff. Re-tries any SDK exception;
    # delay doubles each attempt, capped at llm_retry_max_delay_s.
    llm_max_retries: int = 10
    llm_retry_initial_delay_s: float = 0.05
    llm_retry_max_delay_s: float = 1.0

    # Per call-site effort override. Maps `PromptedCall.name` (e.g.
    # "planner", "compute.codegen", "extract.text") → Effort tier
    # ("off"|"minimal"|"low"|"medium"|"high"). When a key is missing
    # the call-site's class-level `default_effort` applies. Operators
    # may still pass an explicit `effort=` argument at call time to
    # escalate on retry — that wins over both this dict and the default.
    # (env: SKUNK_EFFORT_OVERRIDES — comma-separated `name=tier` pairs)
    effort_overrides: dict[str, "Effort"] = field(default_factory=dict)

    # Extract operator (env: SKUNK_EXTRACT_N_SAMPLES, SKUNK_EXTRACT_SAMPLE_TEMPERATURE)
    extract_n_samples: int = 3       # LLM calls per tier 1; all surviving entries are merged + deduped
    extract_sample_temperature: float = 0.7
    extract_max_pages: int = 5       # page cap per tier

    # Compute operator
    compute_max_attempts: int = 3

    # Replan-on-MissingData loop: after compute raises MissingData, the
    # planner is re-invoked with the prior plan + current `prev` + the
    # missing-data signal. The orchestrator diffs the returned plan against
    # the prior plan and executes only the newly-added branches; their
    # outputs are appended to `prev` before compute is re-invoked. Total
    # compute invocations per question ≤ recovery_max_rounds + 1.
    recovery_max_rounds: int = 2

    # Prompt overrides YAML — corpus blurbs, few-shots, lessons. Loaded once
    # at the top level and threaded onto HarnessContext.prompt_overrides.
    # (env: SKUNK_PROMPT_OVERRIDES)
    prompt_overrides_path: str = "config/prompts/treasury_bulletin.yaml"

    # Ablation: supply golden page refs to bypass the retrieve operator.
    # Present for train/eval runs; None for production workloads.
    golden_pages: list[PageRef] | None = field(default=None, repr=False)

    # Retrieve operator dispatch. "search_agent" routes to the teammate's
    # iterative ChromaDB + LLM-loop retriever (default); "page_index" routes
    # to the legacy PageIndexRetrievePrototype kept for ablations.
    # (env: SKUNK_RETRIEVER)
    retriever: Literal["search_agent", "page_index"] = "search_agent"

    # Search-agent corpus artifacts (built offline; see
    # src/skunk/search_agent/prep/). The agent fails fast at first
    # non-golden call if either path is missing.
    # (env: SKUNK_CHROMADB_DIR, SKUNK_CHROMADB_COLLECTION, SKUNK_CLEAN_PAGE_MAP)
    chromadb_dir: str = "cache/chromadb"
    chromadb_collection: str = "treasury_pages"
    clean_page_map_path: str = "cache/clean_page_map.json"

    # Embedding model used by the agent's vector_search tool (must match
    # whatever produced the stored embeddings). Vertex bare name — no `google/` prefix.
    # (env: SKUNK_EMB_MODEL)
    emb_model_id: str = "gemini-embedding-001"

    # Per-question agent budget. The teammate's defaults are 20/20.
    # (env: SKUNK_AGENT_MAX_STEPS, SKUNK_AGENT_MAX_PAGES_PER_TOOL_CALL)
    agent_max_steps: int = 20
    agent_max_pages_per_tool_call: int = 20

    # Chat model used by the agent loop. None → fall back to `llm_model`.
    # (env: SKUNK_AGENT_MODEL)
    agent_model_id: str | None = None

    # BM25 rerank scaffold for the page-index retriever. Default OFF —
    # this is an opt-in experiment. When enabled, the retrieve operator
    # scores year-filtered survivors with an in-memory BM25 index over
    # title + headers + keywords; if the top score clearly dominates the
    # field (top1/median20 ≥ bm25_dominance_threshold) it truncates to
    # the top-K, otherwise it returns all survivors in BM25 order.
    # (env: SKUNK_BM25_ENABLED, SKUNK_BM25_TOP_K, SKUNK_BM25_DOMINANCE)
    bm25_enabled: bool = False
    bm25_top_k: int = 20
    bm25_dominance_threshold: float = 2.0

    @classmethod
    def from_env(cls) -> SkunkConfig:
        return cls(
            llm_model=os.environ.get("SKUNK_LLM_MODEL", "gemini-3.5-flash"),
            effort_overrides=_parse_effort_overrides(os.environ.get("SKUNK_EFFORT_OVERRIDES", "")),
            llm_rpm=float(os.environ.get("SKUNK_LLM_RPM", "1000")),
            llm_max_retries=int(os.environ.get("SKUNK_LLM_MAX_RETRIES", "10")),
            llm_retry_initial_delay_s=float(os.environ.get("SKUNK_LLM_RETRY_INITIAL_DELAY", "0.05")),
            llm_retry_max_delay_s=float(os.environ.get("SKUNK_LLM_RETRY_MAX_DELAY", "1.0")),
            extract_n_samples=int(os.environ.get("SKUNK_EXTRACT_N_SAMPLES", "3")),
            extract_sample_temperature=float(os.environ.get("SKUNK_EXTRACT_SAMPLE_TEMPERATURE", "0.7")),
            prompt_overrides_path=os.environ.get(
                "SKUNK_PROMPT_OVERRIDES", "config/prompts/treasury_bulletin.yaml"
            ),
            bm25_enabled=os.environ.get("SKUNK_BM25_ENABLED", "").lower() in ("1", "true", "yes"),
            bm25_top_k=int(os.environ.get("SKUNK_BM25_TOP_K", "20")),
            bm25_dominance_threshold=float(os.environ.get("SKUNK_BM25_DOMINANCE", "2.0")),
            retriever=os.environ.get("SKUNK_RETRIEVER", "search_agent"),  # type: ignore[arg-type]
            chromadb_dir=os.environ.get("SKUNK_CHROMADB_DIR", "cache/chromadb"),
            chromadb_collection=os.environ.get("SKUNK_CHROMADB_COLLECTION", "treasury_pages"),
            clean_page_map_path=os.environ.get("SKUNK_CLEAN_PAGE_MAP", "cache/clean_page_map.json"),
            emb_model_id=os.environ.get("SKUNK_EMB_MODEL", "gemini-embedding-001"),
            agent_max_steps=int(os.environ.get("SKUNK_AGENT_MAX_STEPS", "20")),
            agent_max_pages_per_tool_call=int(os.environ.get("SKUNK_AGENT_MAX_PAGES_PER_TOOL_CALL", "20")),
            agent_model_id=os.environ.get("SKUNK_AGENT_MODEL") or None,
        )


def _parse_effort_overrides(raw: str) -> dict[str, "Effort"]:
    """Parse `SKUNK_EFFORT_OVERRIDES` ("name=tier,name=tier,...") into a
    dict. Empty / whitespace-only input → empty dict. Raises ValueError
    on malformed entries or unknown tiers — we want config typos to
    fail loudly at startup rather than silently apply the default."""
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
