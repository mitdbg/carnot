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
    # the client to use for computing embeddings ("vllm" resolves the server URL from
    # `vllm_base_urls[emb_model_id]`)
    emb_provider: Literal["openrouter", "vllm"]
    # the model to use for computing embeddings
    emb_model_id: str
    # generation provider for every LLM call whose model has no `vllm_base_urls` entry
    # (routing is per call — see `vllm_base_urls` below)
    llm_provider: Literal["openrouter", "vllm"]
    # default model for every LLM call — one-shot prompted calls AND agent loops alike.
    # A call resolves its model as `model_overrides.get(call_site_name, llm_model)`; only a
    # per-call-site entry in `model_overrides` (below) overrides it.
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
    # Per-model context-window limits (tokens); maps a model-substring -> max context tokens, resolved
    # by the same exact-then-substring rule as `llm_prices`. Used by the semantic filter to head-truncate
    # a candidate document so the judge request fits the judge model's window; a model with no match is
    # treated as having no known limit (no truncation).
    llm_context_limits: dict[str, int]
    # Optional OpenRouter provider pin (ignored on the vllm path): an ordered list of provider
    # slugs (e.g. ["parasail"]). When set, generation is routed only to these providers with no
    # fallback, so a specific provider's prompt-cache / pricing is used deterministically. null =
    # let OpenRouter pick. Resolve slugs from the model's Providers tab (e.g. io.net => "io-net").
    llm_provider_order: list[str] | None = None
    # Per-call-site overrides keyed by `PromptedCall.name`, read by skunk's agent loop
    # (prompted_call._resolve_effort / _resolve_model). Empty = every call site uses its
    # own default effort and `llm_model`. The qatfd harness doesn't tune per-call-site,
    # so these default empty — but the agent code path requires the attributes to exist.
    effort_overrides: dict[str, "Effort"] = field(default_factory=dict)
    model_overrides: dict[str, str] = field(default_factory=dict)
    # Per-model vLLM routing: model id -> OpenAI-compatible base URL of the vLLM server that
    # serves it (e.g. {"Qwen/Qwen3-32B": "http://gpu-box:8100/v1"}). Any generation call whose
    # resolved model appears here (exact match — keys must equal the server's
    # --served-model-name) routes to that server; all other models use `llm_provider`. Also
    # read by `embed_query` when emb_provider="vllm" (keyed by `emb_model_id`).
    vllm_base_urls: dict[str, str] = field(default_factory=dict)
    # Optional extra JSON merged into every vLLM chat request body (openai SDK `extra_body`),
    # e.g. {"chat_template_kwargs": {"enable_thinking": false}} to turn off Qwen3-style
    # thinking. None = nothing extra. vLLM-only: OpenRouter calls never send it.
    vllm_extra_body: dict | None = None

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

    # extra steps the agent may take, after its main run, to correct any returned doc_ids that
    # do not name a real document (e.g. a bare filing name missing its page suffix). Kept separate
    # from `agent_max_steps` so a citation fix never eats into the agent's search budget.
    doc_id_correction_steps: int = 3


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
    # LLM model — a full OpenRouter id on "openrouter", the server's --served-model-name
    # on "vllm". (env: SKUNK_LLM_MODEL). Request pacing is a per-model token
    # bucket (`llm:<model>`), paced from `llm_model_rpm` / `llm_default_rpm` — see
    # `_LLMBackend._retry_call`.
    llm_model: str = "google/gemini-3.5-flash"
    # Default LLM provider backend. "openrouter" (default) → OpenRouter chat API (needs
    # OPENROUTER_API_KEY). "vllm" → local vLLM servers; every model must then have a
    # `vllm_base_urls` entry. A model WITH a `vllm_base_urls` entry routes to vLLM
    # regardless of this default, so a mixed run (agent on OpenRouter, semantic filter
    # on a local model) just lists the local models in the map.
    # (env: SKUNK_LLM_PROVIDER)
    llm_provider: Literal["openrouter", "vllm"] = "openrouter"
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
    # Per-model context-window limits (tokens) for judge-request sizing; empty → no model is truncated.
    llm_context_limits: dict[str, int] = field(default_factory=dict)
    # Embedding backend for `LLMClient.embed_query` (vector_search): "openrouter" or
    # "vllm" (a local embedding server, addressed via `vllm_base_urls[emb_model_id]`).
    emb_provider: Literal["openrouter", "vllm"] = "openrouter"
    # Embedding model for vector_search (must match the stored embeddings).
    # (env: SKUNK_EMB_MODEL)
    emb_model_id: str = "qwen/qwen3-embedding-8b"

    # ---- operator knobs -----------------------------------------------------------
    # Compute operator
    compute_max_attempts: int = 3
    # Best-of-N: run this many independent codegen→exec trials per compute call (in
    # parallel) and commit the most frequent outcome. All `NeedsMore` trials pool into a
    # single missing-data candidate; a tie NEVER breaks in favor of missing-data (see
    # `ComputeOp._vote`). 1 = single-trial. (env: SKUNK_COMPUTE_BEST_OF_N)
    compute_best_of_n: int = 5

    # Replan-on-MissingData loop. Total compute invocations ≤ recovery_max_rounds + 1.
    recovery_max_rounds: int = 2

    # Ablation: golden page refs bypass the search-agent retriever (eval runs only); their
    # page text is attached from `clean_page_map_path` so compute reads it like a normal
    # retrieval. (set by the app, e.g. eval_e2e --golden)
    golden_pages: list[PageRef] | None = field(default=None, repr=False)

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


def parse_effort_overrides(raw: str) -> dict[str, "Effort"]:
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


def parse_model_overrides(raw: str) -> dict[str, str]:
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


def parse_csv(raw: str) -> list[str] | None:
    """Parse a comma-separated env var into a list of trimmed entries, or None when
    unset/empty (so the field falls back to its default)."""
    items = [s.strip() for s in raw.split(",") if s.strip()]
    return items or None
