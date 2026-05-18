"""Centralized configuration for the skunk harness.

All tuning constants live here. Values can be overridden via environment variables
(documented next to each field). Construct via SkunkConfig.from_env() or pass
custom values directly for tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skunk.models import PageRef


@dataclass
class SkunkConfig:
    # Orchestrator + extract fan-out. Sized so a typical question
    # (≤3 branches × ≤5 pages × ≤3 samples) is bounded by the LLM RPM
    # limiter rather than the thread pool.
    max_parallel_workers: int = 16

    # OpenRouter LLM client — used for all non-search calls (env: SKUNK_LLM_MODEL, SKUNK_LLM_RPM)
    llm_model: str = "google/gemini-3-flash-preview"
    # Token-bucket rate limit applied to every LLM call (requests per minute).
    # Default matches Gemini 2.5 Flash paid-tier 1k RPM quota; adjust for other models.
    llm_rpm: float = 1000.0
    # Per-call retry with exponential backoff. Re-tries any SDK exception;
    # delay doubles each attempt, capped at gemini_retry_max_delay_s.
    gemini_max_retries: int = 10
    gemini_retry_initial_delay_s: float = 0.05
    gemini_retry_max_delay_s: float = 1.0
    # Direct Gemini model used only for lookup_external Google Search grounding
    # (env: SKUNK_GEMINI_MODEL). All other calls go through OpenRouter via llm_model.
    gemini_model: str = "gemini-3-flash-preview"
    # Vertex AI (env: SKUNK_USE_VERTEX). When True, uses Vertex AI instead of the direct Gemini API
    # for the Google Search path. Requires GOOGLE_CLOUD_PROJECT; optionally GOOGLE_CLOUD_LOCATION
    # (default: us-central1) and GOOGLE_APPLICATION_CREDENTIALS for service-account auth.
    use_vertex: bool = False
    # Route every non-search LLM call (planner, retrieve, extract, compute, …)
    # through the direct Gemini API instead of OpenRouter. Uses `gemini_model`
    # (default gemini-3-flash-preview) and GEMINI_API_KEY. Useful when
    # OpenRouter credits are unavailable. (env: SKUNK_USE_DIRECT_GEMINI)
    use_direct_gemini: bool = False

    # Extract operator (env: SKUNK_EXTRACT_N_SAMPLES, SKUNK_EXTRACT_SAMPLE_TEMPERATURE)
    extract_n_samples: int = 3       # LLM calls per tier 1; all surviving entries are merged + deduped
    extract_sample_temperature: float = 0.7
    extract_max_pages: int = 5       # page cap per tier

    # Compute operator
    compute_max_attempts: int = 3

    # Recovery loop: number of re-plan rounds allowed after the final compute
    # raises MissingData. Each round calls the planner (LLM) again with a one-shot
    # recovery lesson injected into ctx.prompt_overrides, then re-executes the
    # whole returned plan from scratch. Total execute() rounds ≤ recovery_max_rounds + 1.
    recovery_max_rounds: int = 1

    # Prompt overrides YAML — corpus blurbs, few-shots, lessons. Loaded once
    # at the top level and threaded onto HarnessContext.prompt_overrides.
    # (env: SKUNK_PROMPT_OVERRIDES)
    prompt_overrides_path: str = "config/prompts/treasury_bulletin.yaml"

    # Ablation: supply golden page refs to bypass the retrieve operator.
    # Present for train/eval runs; None for production workloads.
    golden_pages: list[PageRef] | None = field(default=None, repr=False)

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
            llm_model=os.environ.get("SKUNK_LLM_MODEL", "google/gemini-3-flash-preview"),
            llm_rpm=float(os.environ.get("SKUNK_LLM_RPM", "1000")),
            gemini_max_retries=int(os.environ.get("SKUNK_GEMINI_MAX_RETRIES", "10")),
            gemini_retry_initial_delay_s=float(os.environ.get("SKUNK_GEMINI_RETRY_INITIAL_DELAY", "0.05")),
            gemini_retry_max_delay_s=float(os.environ.get("SKUNK_GEMINI_RETRY_MAX_DELAY", "1.0")),
            gemini_model=os.environ.get("SKUNK_GEMINI_MODEL", "gemini-3-flash-preview"),
            use_vertex=os.environ.get("SKUNK_USE_VERTEX", "").lower() in ("1", "true", "yes"),
            use_direct_gemini=os.environ.get("SKUNK_USE_DIRECT_GEMINI", "").lower() in ("1", "true", "yes"),
            extract_n_samples=int(os.environ.get("SKUNK_EXTRACT_N_SAMPLES", "3")),
            extract_sample_temperature=float(os.environ.get("SKUNK_EXTRACT_SAMPLE_TEMPERATURE", "0.7")),
            prompt_overrides_path=os.environ.get(
                "SKUNK_PROMPT_OVERRIDES", "config/prompts/treasury_bulletin.yaml"
            ),
            bm25_enabled=os.environ.get("SKUNK_BM25_ENABLED", "").lower() in ("1", "true", "yes"),
            bm25_top_k=int(os.environ.get("SKUNK_BM25_TOP_K", "20")),
            bm25_dominance_threshold=float(os.environ.get("SKUNK_BM25_DOMINANCE", "2.0")),
        )
