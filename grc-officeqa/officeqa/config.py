"""OfficeQA app configuration. `SkunkConfig` extends the library's
`PipelineConfig` with the Treasury corpus paths, the OfficeQA prompt-override
file, per-stage model pinning, and the `SKUNK_*` env plumbing (`from_env`).
Moved out of the skunk library 2026-07-07 (REFACTOR_PLAN.md Phase 5)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from skunk.config import (
    PipelineConfig,
    parse_csv,
    parse_effort_overrides,
    parse_model_overrides,
)

# Default corpus locations (overridable via env / explicit construction — see the
# `parsed_json_dir` / `pdf_dir` fields below). Homed here so `SkunkConfig` is the single
# source of truth for where the corpus lives; `corpus.py` resolves through it.
_DEFAULT_PARSED_JSON_DIR = (
    Path.home() / "Desktop/officeqa/treasury_bulletins_parsed/jsons"
)
_DEFAULT_PDF_DIR = Path.home() / "Desktop/officeqa/treasury_bulletin_pdfs"


# USD per Mtok for the models the OfficeQA pipeline pins (see `llm_prices` below).
# Substring-matched, so full OpenRouter ids ("google/gemini-3.5-flash") price too.
# Thinking tokens are billed at the output rate by the UsageTracker.
_GEMINI_PRICES: dict[str, dict[str, float]] = {
    "gemini-3.5-flash": {"in": 1.50, "out": 9.00},  # list price
    "gemini-3-flash-preview": {"in": 0.50, "out": 3.00},  # verified ai.google.dev + OpenRouter, 2026-06
    "gemini-3.1-flash-lite": {"in": 0.10, "out": 0.40},
    "gemini-3.1-pro-preview": {"in": 2.00, "out": 12.00},
}


@dataclass
class SkunkConfig(PipelineConfig):
    # ---- OfficeQA defaults over the library's -------------------------------------
    name: str = "skunk"
    # Price table for cost accounting (`UsageTracker.cost()`; the eval report's
    # cost_usd column reads it). $/Mtok, substring-matched by model id.
    llm_prices: dict[str, dict[str, float]] = field(
        default_factory=lambda: dict(_GEMINI_PRICES)
    )
    # Embedding model for vector_search (must match the stored treasury embeddings).
    # (env: SKUNK_EMB_MODEL)
    emb_model_id: str = "Qwen/Qwen3-Embedding-8B"
    # Misfires: skunk runs give the model one extra non-progressing attempt (base: 5).
    # (env: SKUNK_AGENT_MAX_MISFIRES)
    agent_max_misfires: int = 6
    # Search-agent corpus artifacts (built offline; agent fails fast if missing).
    chromadb_collection: str = "treasury_pages"

    # ---- OfficeQA corpus paths -----------------------------------------------------
    # Corpus directories. `parsed_json_dir` holds the per-bulletin parsed-JSON the text
    # accessors read (cleaner than PyMuPDF text on scanned pages); `pdf_dir` holds the
    # source bulletin PDFs used for page rendering. `corpus.py` resolves through these.
    # (env: OFFICEQA_PARSED_JSON_DIR, OFFICEQA_PDF_DIR)
    parsed_json_dir: Path = field(default_factory=lambda: _DEFAULT_PARSED_JSON_DIR)
    pdf_dir: Path = field(default_factory=lambda: _DEFAULT_PDF_DIR)

    # Optional pre-rendered page-PNG cache (`<dir>/<stem>_<page>.png`). When set, page rendering
    # (`render_page_b64`, hence `view_figure`) serves from here instead of rasterizing the PDF —
    # the competition-latency win for corpora with a warmed render cache. None → always render
    # on the fly. (env: SKUNK_PAGE_RENDERS_DIR)
    page_renders_dir: Path | None = None

    # Prompt overrides YAML — corpus blurbs, few-shots, lessons. (env: SKUNK_PROMPT_OVERRIDES)
    prompt_overrides_path: str = "config/prompts/us_receipts_expenditures.yaml"

    # Build: the `vision_rescan` stage always re-reads `parse_broken` pages (mangled parses). Pages
    # flagged `has_unparsed_graphics` that are CHART-ONLY (a chart/figure with no table on the page,
    # so its data is otherwise lost) are re-read only when this is on. That set is the bulk of the
    # vision working set (~6k pages corpus-wide) and low-value for table-centric queries, so chart
    # re-reading is an opt-in phase, default off. (env: SKUNK_VISION_RESCAN_CHARTS=1)
    vision_rescan_charts: bool = False

    def __post_init__(self) -> None:
        # Route per-stage models through the override registry so `PromptedCall` resolves them
        # like every other call-site. Defaulted here unless a run pins them explicitly
        # (SKUNK_MODEL_OVERRIDES=stage=…). Efforts
        # come from each call-site's `default_effort` (compute=high, planner=high, replanner=high),
        # overridable via SKUNK_EFFORT_OVERRIDES.
        # - compute.codegen: 3.1 Pro at high thinking — codegen/reasoning over the retrieved pages.
        # - replanner: 3.1 Pro at high thinking — same as the planner; revising a failed plan
        #   needs the same decomposition quality as the initial plan.
        # - planner: 3.1 Pro at high thinking. The initial plan's decomposition quality
        #   (branch coverage, operator routing, period fidelity) is worth the per-question
        #   cost — the flash planner systematically dropped/misrouted branches that Pro/high
        #   gets right (plan-probe 2026-06-14: 19/20 known-bad dev plans fixed).
        # Everything else runs on the base `llm_model` (flash).
        self.model_overrides.setdefault("compute.codegen", "gemini-3.1-pro-preview")
        self.model_overrides.setdefault("planner", "gemini-3.1-pro-preview")
        self.model_overrides.setdefault("replanner", "gemini-3.1-pro-preview")

    @classmethod
    def from_env(cls) -> SkunkConfig:
        model_overrides = parse_model_overrides(
            os.environ.get("SKUNK_MODEL_OVERRIDES", "")
        )
        return cls(
            llm_model=os.environ.get("SKUNK_LLM_MODEL", "google/gemini-3.5-flash"),
            llm_provider=os.environ.get("SKUNK_LLM_PROVIDER", "openrouter"),  # type: ignore[arg-type]
            effort_overrides=parse_effort_overrides(
                os.environ.get("SKUNK_EFFORT_OVERRIDES", "")
            ),
            model_overrides=model_overrides,
            llm_max_retries=int(os.environ.get("SKUNK_LLM_MAX_RETRIES", "5")),
            llm_retry_initial_delay_s=float(
                os.environ.get("SKUNK_LLM_RETRY_INITIAL_DELAY", "1.0")
            ),
            compute_best_of_n=int(os.environ.get("SKUNK_COMPUTE_BEST_OF_N", "5")),
            parsed_json_dir=Path(
                os.environ.get("OFFICEQA_PARSED_JSON_DIR") or _DEFAULT_PARSED_JSON_DIR
            ),
            pdf_dir=Path(os.environ.get("OFFICEQA_PDF_DIR") or _DEFAULT_PDF_DIR),
            page_renders_dir=(
                Path(os.environ["SKUNK_PAGE_RENDERS_DIR"])
                if os.environ.get("SKUNK_PAGE_RENDERS_DIR")
                else None
            ),
            prompt_overrides_path=os.environ.get(
                "SKUNK_PROMPT_OVERRIDES", "config/prompts/us_receipts_expenditures.yaml"
            ),
            lookup_agent_max_output_tokens=int(
                os.environ.get("SKUNK_LOOKUP_AGENT_MAX_OUTPUT_TOKENS", "8192")
            ),
            lookup_agent_request_timeout_s=float(
                os.environ.get("SKUNK_LOOKUP_AGENT_TIMEOUT_S", "150")
            ),
            vision_rescan_charts=os.environ.get("SKUNK_VISION_RESCAN_CHARTS", "")
            not in ("", "0"),
            chromadb_dir=os.environ.get("SKUNK_CHROMADB_DIR", "cache/chromadb"),
            chromadb_collection=os.environ.get(
                "SKUNK_CHROMADB_COLLECTION", "treasury_pages"
            ),
            clean_page_map_path=os.environ.get(
                "SKUNK_CLEAN_PAGE_MAP", "cache/clean_page_map.json"
            ),
            chroma_server_host=os.environ.get("SKUNK_CHROMA_SERVER_HOST", "127.0.0.1"),
            chroma_server_port=int(os.environ.get("SKUNK_CHROMA_SERVER_PORT", "8001")),
            emb_model_id=os.environ.get("SKUNK_EMB_MODEL", "qwen/qwen3-embedding-8b"),
            agent_max_steps=int(os.environ.get("SKUNK_AGENT_MAX_STEPS", "20")),
            agent_max_misfires=int(os.environ.get("SKUNK_AGENT_MAX_MISFIRES", "6")),
            agent_max_pages_per_tool_call=int(
                os.environ.get("SKUNK_AGENT_MAX_PAGES_PER_TOOL_CALL", "20")
            ),
            grep_max_output_tokens=int(
                os.environ.get("SKUNK_GREP_MAX_OUTPUT_TOKENS", "200000")
            ),
            read_document_max_output_chars=int(
                os.environ.get("SKUNK_READ_DOCUMENT_MAX_OUTPUT_CHARS", "400000")
            ),
            search_agent_max_output_tokens=int(
                os.environ.get("SKUNK_SEARCH_MAX_OUTPUT_TOKENS", "4096")
            ),
            search_agent_request_timeout_s=float(
                os.environ.get("SKUNK_SEARCH_TIMEOUT_S", "120")
            ),
            lookup_max_steps=int(os.environ.get("SKUNK_LOOKUP_MAX_STEPS", "4")),
            lookup_tools=parse_csv(os.environ.get("SKUNK_LOOKUP_TOOLS", "")),
        )
