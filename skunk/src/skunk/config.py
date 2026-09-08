"""Centralized library configuration for Skunk."""

from __future__ import annotations

import yaml
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from skunk.common import Effort


@dataclass
class InferenceConfig:
    """Singleton configuration object which configures global inference state."""
    # generation provider for every llm call whose model has no `vllm_base_urls` entry (routing is per call — see `vllm_base_urls` below)
    llm_provider: Literal["openrouter", "vllm"]
    # default model for every llm call
    llm_model: str
    # the client to use for computing embeddings ("vllm" resolves the server URL from `vllm_base_urls[emb_model_id]`)
    emb_provider: Literal["openrouter", "vllm"]
    # default model for computing embeddings
    emb_model_id: str
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
    # default effort for llm calls
    effort: Effort = "medium"
    # Optional OpenRouter provider pin (ignored on the vllm path): an ordered list of provider
    # slugs (e.g. ["parasail"]). When set, generation is routed only to these providers with no
    # fallback, so a specific provider's prompt-cache / pricing is used deterministically. null =
    # let OpenRouter pick. Resolve slugs from the model's Providers tab (e.g. io.net => "io-net").
    llm_provider_order: list[str] | None = None
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


@dataclass
class StorageConfig:
    """Singleton configuration object which configures global storage."""
    # we assume that data is embedded in a chroma vector store and served externally (for now)
    collection_name: str
    chroma_server_host: str
    chroma_server_port: int

    # paths for pdf (rendering) storage
    pdf_dir: str | None = None
    page_renders_dir: str | None = None


@dataclass
class AgentConfig:
    """Configuration for an agent """
    # the agent's human-readable name; used in logging to identify which lines belong to this agent
    name: str
    # unique id used to attribute this agent's token/cost usage on the shared LLMClient's UsageTracker
    # If None, each agent instance will generate a fresh uuid4
    agent_id: str = field(kw_only=True)
    # model for the agent to use; if None, falls back to InferenceConfig.llm_model
    llm_model: str | None = None
    # maximum number of steps taken by the agent
    max_steps: int = 20
    # maximum number of step failures across the agent trajectory; failures do not count
    # against the step budget, but once we reach this limit we terminate to prevent the
    # agent from failing repeatedly
    max_misfires: int = 5
    # the number steps before the max_steps boundary at which we issue a warning to the agent
    warn_steps_remaining: int = 3
    # Per-step sampling temperature, threaded to `LLMClient.acall()`.
    # 1.0 per Gemini 3.x guidance: thinking-enabled calls below 1.0 can trap the
    # model in a degenerate reasoning loop that burns the whole output budget (https://ai.google.dev/gemini-api/docs/gemini-3)
    temperature: float = 1.0
    # reasoning effort for the agent's model; if None, falls back to InferenceConfig.effort
    effort: Effort | None = None
    # Per-step generation caps, threaded to `LLMClient.acall()`.
    # Both None → provider defaults (uncapped output, no wall-clock timeout),
    # preserving behaviour for every agent that doesn't opt in. `SearchAgent` sets
    # these to bound runaway generations and to cap genuinely hung requests.
    max_output_tokens: int | None = None
    request_timeout_s: float | None = None
    # Extra imports authorized inside the per-step code sandbox. Default: none
    # (tool calls only). Compute-oriented agents (e.g. the task solver) widen
    # this to allow numpy / scipy / statistics / ... in their python steps.
    # A tuple (not a list): class-level mutable defaults are shared across every
    # instance, so an in-place append would leak between agents.
    authorized_imports: Sequence[str] = ()
    # cost budget in dollars for the agent; None means no budget limit (default)
    cost_budget: float | None = None
    # latency budget in seconds for the agent; None means no latency limit (default)
    latency_budget: float | None = None
    # fraction of the context limit above which the agent is restricted from making
    # tool calls which do not reclaim context
    context_hard_safety_frac: float = 0.8
    # fraction of the context limit above which the agent is warned that it should begin
    # reclaiming context if possible or return a final answer
    context_soft_safety_frac: float = 0.6

    @classmethod
    def from_yaml(cls, path: str) -> AgentConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    # TODO: move into AgentConfig to enforce this on construction; also check that safety fractions make sense
    # # sanity check config inputs
    # assert config.cost_budget is None or config.cost_budget > 0.0
    # assert config.latency_budget is None or config.latency_budget > 0.0

@dataclass
class LookupAgentConfig(AgentConfig):
    """Additional configuration and default overrides for the LookupAgent."""
    name: str = "lookup_agent"
    # specify set of active lookup tools by name (see `lookup_tools._REGISTRY`); None → all tools.
    lookup_tools: list[str] | None = None
    # override default max steps for the LookupAgent
    max_steps: int = 4
    # override default warn steps remaining
    warn_steps_remaining: int = 1
    # override per-llm call generation caps for the LookupAgent
    max_output_tokens: int | None = 8192
    request_timeout_s: float | None = 150.0
    # default set of authorized imports for the LookupAgent
    authorized_imports: Sequence[str] = field(default_factory=lambda: ["math", "statistics", "numpy", "pandas", "json"])


@dataclass
class SearchAgentConfig(AgentConfig):
    """Additional configuration and default overrides for the SearchAgent."""
    name: str = "search_agent"
    # boolean switches determining which tools the SearchAgent has access to
    include_search_corpus: bool = True
    include_grep_corpus: bool = True
    include_semantic_filter: bool = False
    # the number of chunks to print headers for when the agent fetches data into the WorkingSet
    chunks_per_summary: int = 10
    # the number of workers to use to process semantic filter tool calls in parallel
    semantic_filter_max_workers: int = 16
    # the maximum number of documents that can be processed by a single semantic filter tool call
    semantic_filter_max_candidate_docs: int = 1000
    # the fraction of the semantic filter model's context window that can be used to fit document text
    semantic_filter_context_safety_frac: float = 0.9
    # maximum output tokens for each semantic_filter judge call. The reply is a single TRUE/FALSE
    # token and judge reasoning is disabled (`semantic_filter_disable_reasoning`), so a tiny cap
    # suffices. If reasoning is re-enabled, raise this too — a reasoning judge that hits the cap
    # returns finish_reason=length with empty content, which retries+backs-off and destroys
    # throughput. Threaded into `SemanticFilterTool`.
    semantic_filter_max_output_tokens: int = 4
    # Disable reasoning/thinking on the semantic_filter judge calls (OpenRouter
    # `reasoning={"enabled": false}`): the verdict is one token, so thinking is pure cost.
    # Set False for judge models that mandate reasoning (they 400 on disabled reasoning) —
    # and then raise `semantic_filter_max_output_tokens` to give the judge headroom.
    semantic_filter_disable_reasoning: bool = True
    # Model for the semantic_filter's per-candidate judge calls; None => `llm_model` (the agent
    # model). Set it to a cheaper model to run the (token-heavy) candidate filtering on the cheap
    # model while the search agent itself stays on `llm_model` — the filtered-out candidates never
    # enter the agent model's context, so this drives cost down without touching the agent's reasoning.
    semantic_filter_llm_model: str | None = None
    # OpenRouter provider order (no fallback) for the semantic_filter judge calls only. None => use
    # the client-wide `llm_provider_order`. Lets the judge model route to specific providers (e.g.
    # [akashml, parasail]) while the agent model (which may be a different family, e.g. a Google
    # model that those providers don't serve) stays unpinned.
    semantic_filter_provider_order: list[str] | None = None
    # extra steps the agent may take, after its main run, to correct any returned doc_ids that
    # do not name a real document (e.g. a bare filing name missing its page suffix). Kept separate
    # from `max_steps` so a citation fix never eats into the agent's search budget.
    doc_id_correction_steps: int = 3
    # turn off the intermediate collection used by each Working Set (ablation flag)
    working_set_collection_off: bool = False
    # turn off Working Set id tracking for inclusion / exclusion filters (ablation flag)
    id_tracking_off: bool = False
    # retrieve related working sets before running the agent
    fetch_related_working_sets: bool = False
    # override per-llm call generation caps number for the SearchAgent
    max_output_tokens: int | None = 4096
    request_timeout_s: float | None = 120.0


# --------------------------------------------------------------------------------
# Skunk configuration.
# --------------------------------------------------------------------------------


@dataclass
class SkunkConfig:
    # config for the retrieval operator (search agent)
    search: SearchAgentConfig
    # config for the lookup operator (lookup agent)
    lookup: LookupAgentConfig
    # config for global inference
    inference: InferenceConfig
    # config for global storage
    storage: StorageConfig
