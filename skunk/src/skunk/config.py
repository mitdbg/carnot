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
    from skunk.dsl import DocHandle


@dataclass
class SkunkConfig:
    # Orchestrator
    max_parallel_workers: int = 4
    parallel_timeout_s: float = 120.0

    # Gemini LLM client (env: SKUNK_GEMINI_MAX_RETRIES, SKUNK_GEMINI_RETRY_DELAY, SKUNK_GEMINI_MODEL)
    gemini_model: str = "gemini-2.5-flash"
    gemini_max_retries: int = 5
    gemini_retry_delay_s: float = 30.0
    # Vertex AI (env: SKUNK_USE_VERTEX). When True, uses Vertex AI instead of the direct Gemini API.
    # Requires GOOGLE_CLOUD_PROJECT; optionally GOOGLE_CLOUD_LOCATION (default: us-central1)
    # and GOOGLE_APPLICATION_CREDENTIALS for service-account auth.
    use_vertex: bool = False

    # Extract subagent (env: SKUNK_EXTRACT_N_SAMPLES, SKUNK_EXTRACT_SAMPLE_TEMPERATURE)
    extract_n_samples: int = 3       # Gemini calls per tier for consensus sampling
    extract_quorum: int = 2          # minimum samples a bucket needs to pass consensus
    extract_sample_temperature: float = 0.7
    extract_max_pages: int = 5       # page cap per tier

    # Compute subagent
    compute_max_attempts: int = 3

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
            gemini_max_retries=int(os.environ.get("SKUNK_GEMINI_MAX_RETRIES", "5")),
            gemini_retry_delay_s=float(os.environ.get("SKUNK_GEMINI_RETRY_DELAY", "30")),
            use_vertex=os.environ.get("SKUNK_USE_VERTEX", "").lower() in ("1", "true", "yes"),
            extract_n_samples=int(os.environ.get("SKUNK_EXTRACT_N_SAMPLES", "3")),
            extract_sample_temperature=float(os.environ.get("SKUNK_EXTRACT_SAMPLE_TEMPERATURE", "0.7")),
            plan_cache_csv=os.environ.get("SKUNK_PLAN_CACHE_CSV", "data/dsl_planning_pass.csv"),
            manifest_path=os.environ.get("SKUNK_MANIFEST"),
        )
