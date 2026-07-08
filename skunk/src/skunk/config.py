"""Centralized library configuration: the `SystemConfig` → `SearchAgentConfig` →
`PipelineConfig` hierarchy every operator and agent reads. Apps subclass
`PipelineConfig` to add their corpus paths / per-stage model pinning / env
plumbing (e.g. grc-officeqa's `SkunkConfig.from_env`, which reuses the
`_parse_*` helpers below)."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from skunk.common import Effort, PageRef

# --------------------------------------------------------------------------------
# System-specific configuration
# --------------------------------------------------------------------------------

@dataclass
class SystemConfig:
    # the name of the system
    name: str
    # the client to use for computing embeddings
    emb_provider: Literal["openrouter", "local"]
    # the model to use for computing embeddings
    emb_model_id: str
    # generation provider for all LLM calls
    llm_provider: Literal["genai", "openrouter"]
    # the model to use for the system
    agent_model_id: str
    # default model for calls that don't pass an explicit `model=` (agent loops pass `agent_model_id`).
    llm_model: str
    # per-call retry on transient faults only (429 / 5xx / transport blips); the delay doubles each attempt.
    llm_max_retries: int
    llm_retry_initial_delay_s: float
    # per-model request pacing (requests/min); each model gets its own `llm:<model>` token bucket; a model absent from the map uses `llm_default_rpm`.
    llm_model_rpm: dict[str, float]
    llm_default_rpm: float
    # per-model token pacing (tokens/min); a model absent from the map uses `llm_default_tpm`; a falsy effective value (null / 0) means unthrottled TPM.
    llm_model_tpm: dict[str, float]
    llm_default_tpm: float | None
    # USD price table for cost accounting; maps a model-substring -> {"in"/"out"/"cached": $/Mtok}.
    # Lookup is exact-first then substring, a model with no match costs 0.
    llm_prices: dict[str, dict[str, float]]
    # Per-call-site overrides keyed by `PromptedCall.name`, read by skunk's agent loop
    # (prompted_call._resolve_effort / _resolve_model). Empty = every call site uses its
    # own default effort and `llm_model`. The qatfd harness doesn't tune per-call-site,
    # so these default empty — but the agent code path requires the attributes to exist.
    effort_overrides: dict[str, "Effort"] = field(default_factory=dict)
    model_overrides: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> SystemConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


@dataclass
class SearchAgentConfig(SystemConfig):
    # the agent will answer the question directly if agent_mode == "answer", otherwise a separate
    # LLM computes an answer given the SearchAgent's retrieved documents
    agent_mode: str = "retrieve"

    # maximum number of tokens and seconds the agent can take togenerate a repsonse
    search_agent_max_output_tokens: int = 4096
    search_agent_request_timeout_s: float = 120.0

    # maximum number of tokens a single grep command can return; over-budget tool calls are dropped
    # with a note telling the agent to narrow its pattern / pass `limit`.
    grep_max_output_tokens: int = 200_000

    # maximum number of pages the agent can process in a single read_document tool call
    agent_max_pages_per_tool_call: int = 20

    # maximum number of characters the agent can output from a single read_document tool call
    read_document_max_output_chars: int = 400_000

    # maximum number of steps the agent can take in a single conversation
    agent_max_steps: int = 20

    # maximum number of failed agent steps before aborting
    agent_max_misfires: int = 5


# --------------------------------------------------------------------------------
# Pipeline (plan/orchestrate/retrieve/extract/lookup/compute) configuration.
# `PipelineConfig` extends `SearchAgentConfig` so the operator pipeline and the
# search agent share ONE config object with the library fields declared once.
# It is app-agnostic: corpus paths, prompt-override files, and per-stage model
# pinning live on the app's subclass (e.g. grc-officeqa's `SkunkConfig`).
# --------------------------------------------------------------------------------


@dataclass
class PipelineConfig(SearchAgentConfig):
    # ---- pipeline defaults for the base's required fields ------------------------
    name: str = "pipeline"
    # LLM model — bare Gemini names on the "genai" provider, full OpenRouter ids on
    # "openrouter". (env: SKUNK_LLM_MODEL). Request pacing is a per-model token
    # bucket (`llm:<model>`), paced from `llm_model_rpm` / `llm_default_rpm` — see
    # `LLMClient._retry_call`.
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
    # Per-model pacing: empty maps → every model at the defaults below.
    llm_model_rpm: dict[str, float] = field(default_factory=dict)
    llm_default_rpm: float = 1000.0
    llm_model_tpm: dict[str, float] = field(default_factory=dict)
    llm_default_tpm: float | None = None
    # USD price table for cost accounting; empty → every model costs 0.
    llm_prices: dict[str, dict[str, float]] = field(default_factory=dict)
    # Embedding backend for `LLMClient.embed_query` (vector_search): "openrouter" or
    # "local" (SentenceTransformers, needs the `embeddings` extra).
    emb_provider: Literal["openrouter", "local"] = "openrouter"
    # Embedding model for vector_search (must match the stored embeddings).
    # (env: SKUNK_EMB_MODEL)
    emb_model_id: str = "gemini-embedding-001"
    # Agent-loop chat model. None → `llm_model`. (env: SKUNK_AGENT_MODEL)
    agent_model_id: str | None = None

    # ---- operator knobs -----------------------------------------------------------
    # Compute operator
    compute_max_attempts: int = 3
    # Best-of-N: run this many independent codegen→exec trials per compute call (in
    # parallel) and commit the most frequent outcome. All `NeedsMore` trials pool into a
    # single missing-data candidate; a tie NEVER breaks in favor of missing-data (see
    # `ComputeOp._vote`). 1 = single-trial. (env: SKUNK_COMPUTE_BEST_OF_N)
    compute_best_of_n: int = 5
    # Bypass the QuestionExplainer's per-question selection and inject the ENTIRE
    # PRECOMPUTED_CONCEPTS catalog into every compute `## Concept references` block (skips
    # the selection LLM call). Default OFF (the explainer selects only the relevant
    # entries); flip per-run to A/B selection vs. full-dump on compute accuracy.
    # (env: SKUNK_PRECOMPUTED_CONCEPT_REFS=1)
    compute_precomputed_concept_refs: bool = False

    # Data-prep gate: codegen→exec attempts before failing safe (pool passes through
    # unchanged). Its own budget — no longer borrows `compute_max_attempts`.
    data_prep_max_attempts: int = 3

    # Replan-on-MissingData loop. Total compute invocations ≤ recovery_max_rounds + 1.
    recovery_max_rounds: int = 2

    # Ablation: golden page refs bypass the retrieve operator (eval runs only).
    golden_pages: list[PageRef] | None = field(default=None, repr=False)

    # Retrieve dispatch: "search_agent" (iterative ChromaDB + LLM loop) is the sole
    # backend; golden_pages is a separate eval bypass. (env: SKUNK_RETRIEVER)
    retriever: Literal["search_agent"] = "search_agent"

    # Search-agent corpus artifacts (built offline; agent fails fast if missing).
    # (env: SKUNK_CHROMADB_DIR, SKUNK_CHROMADB_COLLECTION, SKUNK_CLEAN_PAGE_MAP)
    chromadb_dir: str = "cache/chromadb"
    chromadb_collection: str = "corpus_pages"
    clean_page_map_path: str = "cache/clean_page_map.json"

    # ChromaDB server (HttpClient) the read paths connect to. The embedded PersistentClient
    # deadlocks under 15-way in-process concurrency; the server owns ChromaDB's concurrency.
    # Launch it over `chromadb_dir` with `scripts/run_chroma_server.sh`.
    # (env: SKUNK_CHROMA_SERVER_HOST, SKUNK_CHROMA_SERVER_PORT)
    chroma_server_host: str = "127.0.0.1"
    chroma_server_port: int = 8001

    # (The search-agent knobs — `agent_max_steps`, `agent_max_pages_per_tool_call`,
    # `grep_max_output_tokens`, `read_document_max_output_chars`,
    # `search_agent_max_output_tokens`, `search_agent_request_timeout_s` — are
    # inherited from `SearchAgentConfig`.)

    # Step cap for the lookup_external agent (terminates earlier via its final-answer JSON block).
    # (env: SKUNK_LOOKUP_MAX_STEPS)
    lookup_max_steps: int = 4
    # Active lookup tools by name (see `lookup_tools._REGISTRY`); None → all tools.
    # (env: SKUNK_LOOKUP_TOOLS — comma-separated, e.g. "fetch_fred,tavily_search")
    lookup_tools: list[str] | None = None

    # Per-LLM-call caps for the external-lookup agent's turns. Mirrors the search
    # agent: without a combined thinking+visible cap, Flash thrashed to ~63K thinking
    # tokens / ~285s per step and emitted no parseable tool call (parse-retry death
    # spiral). (env: SKUNK_LOOKUP_AGENT_MAX_OUTPUT_TOKENS, SKUNK_LOOKUP_AGENT_TIMEOUT_S)
    lookup_agent_max_output_tokens: int = 8192
    lookup_agent_request_timeout_s: float = 150.0

    # Per-call caps for the extract tiers (text/vision). Uncapped, individual Flash/Pro
    # extract calls hung for 260-480s and returned garbage that then burned a parse retry;
    # a hard timeout fails fast into the retry, which typically completes in seconds.
    # (env: SKUNK_EXTRACT_MAX_OUTPUT_TOKENS, SKUNK_EXTRACT_TIMEOUT_S)
    extract_max_output_tokens: int = 8192
    extract_request_timeout_s: float = 150.0

    # Extract: skip the parsed-text (OCR) tier entirely and read values straight off the rendered
    # page images (vision tier). Default OFF — parsed text first, vision as the fallback tier.
    # Vision-only is robust to OCR corruption on dense scanned tables (it recovered single-cell
    # OCR misses on the dev set) but costs more and can run away on thinking-only pro models;
    # enable per-run with SKUNK_EXTRACT_VISION_ONLY=1 when OCR quality is the binding issue.
    extract_vision_only: bool = False


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
