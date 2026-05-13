"""Centralized configuration for the skunk harness.

All tuning constants live here. Values can be overridden via environment variables
(documented next to each field). Construct via SkunkConfig.from_env() or pass
custom values directly for tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from skunk.dsl import DocHandle


@dataclass
class SkunkConfig:
    # Orchestrator + extract fan-out. Sized so a typical question
    # (≤3 branches × ≤5 pages × ≤3 samples) is bounded by the Gemini RPM
    # limiter rather than the thread pool.
    max_parallel_workers: int = 16

    # Gemini LLM client (env: SKUNK_GEMINI_MODEL, SKUNK_GEMINI_RPM)
    gemini_model: str = "gemini-2.5-flash"
    # Token-bucket rate limit applied to every Gemini call (requests per minute).
    # Default matches Gemini 2.5 Flash paid-tier 1k RPM quota.
    gemini_rpm: float = 1000.0
    # Per-call retry with exponential backoff. Re-tries any Gemini exception;
    # delay doubles each attempt, capped at gemini_retry_max_delay_s.
    gemini_max_retries: int = 10
    gemini_retry_initial_delay_s: float = 0.05
    gemini_retry_max_delay_s: float = 1.0
    # Vertex AI (env: SKUNK_USE_VERTEX). When True, uses Vertex AI instead of the direct Gemini API.
    # Requires GOOGLE_CLOUD_PROJECT; optionally GOOGLE_CLOUD_LOCATION (default: us-central1)
    # and GOOGLE_APPLICATION_CREDENTIALS for service-account auth.
    use_vertex: bool = False

    # Extract subagent (env: SKUNK_EXTRACT_N_SAMPLES, SKUNK_EXTRACT_SAMPLE_TEMPERATURE)
    extract_n_samples: int = 3       # Gemini calls per tier 1; all surviving entries are merged + deduped
    extract_sample_temperature: float = 0.7
    extract_max_pages: int = 5       # page cap per tier

    # Compute subagent
    compute_max_attempts: int = 3
    # Max depth of the compute chain. 1 = flat (data → single compute, default).
    # 2 = one intermediate layer + one final aggregator.
    max_compute_depth: int = 1

    # Recovery loop: number of re-plan rounds allowed after the final compute
    # raises MissingData. Each round calls plan_recovery (LLM) for a list of
    # supplemental branches, runs them in parallel, appends to prev, and retries
    # compute. The total compute call count is bounded by recovery_max_rounds + 1.
    recovery_max_rounds: int = 1

    # DSL plan cache (env: SKUNK_PLAN_CACHE_CSV)
    plan_cache_csv: str = "data/dsl_planning_pass.csv"

    # Corpus paths (env: SKUNK_MANIFEST)
    manifest_path: str | None = None

    # Ablation: supply golden page refs to bypass the retrieve subagent.
    # Present for train/eval runs; None for production workloads.
    golden_handle: DocHandle | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> SkunkConfig:
        return cls(
            gemini_model=os.environ.get("SKUNK_GEMINI_MODEL", "gemini-2.5-flash"),
            gemini_rpm=float(os.environ.get("SKUNK_GEMINI_RPM", "1000")),
            gemini_max_retries=int(os.environ.get("SKUNK_GEMINI_MAX_RETRIES", "10")),
            gemini_retry_initial_delay_s=float(os.environ.get("SKUNK_GEMINI_RETRY_INITIAL_DELAY", "0.05")),
            gemini_retry_max_delay_s=float(os.environ.get("SKUNK_GEMINI_RETRY_MAX_DELAY", "1.0")),
            use_vertex=os.environ.get("SKUNK_USE_VERTEX", "").lower() in ("1", "true", "yes"),
            extract_n_samples=int(os.environ.get("SKUNK_EXTRACT_N_SAMPLES", "3")),
            extract_sample_temperature=float(os.environ.get("SKUNK_EXTRACT_SAMPLE_TEMPERATURE", "0.7")),
            max_compute_depth=int(os.environ.get("SKUNK_MAX_COMPUTE_DEPTH", "1")),
            plan_cache_csv=os.environ.get("SKUNK_PLAN_CACHE_CSV", "data/dsl_planning_pass.csv"),
            manifest_path=os.environ.get("SKUNK_MANIFEST"),
        )
