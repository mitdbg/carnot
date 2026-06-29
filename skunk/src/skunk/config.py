"""Centralized configuration. All tuning constants live here, overridable via the
environment (env var noted next to each field). Construct via `SkunkConfig.from_env()`."""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
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
class RAGLLMConfig(SystemConfig):
    # number of chunks for the vector search to return
    top_k: int | None = None

    def __post_init__(self) -> None:
        if self.top_k is None:
            raise ValueError(
                "RAGLLMConfig.top_k is unset (null); set it explicitly, e.g. "
                "`systems.top_k=20` on the command line."
            )


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


@dataclass
class QATFDSearchAgentConfig(SearchAgentConfig):
    # Mirrors the system hierarchy (QATFDSearchAgentSystem subclasses SearchAgentSystem): the
    # qatfd variant reuses the full search-agent config (agent_mode, step/token budgets, ...) and
    # only adds a semantic-filter tool, which needs no extra config beyond agent_model_id (base).
    pass


#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

# Default corpus locations (overridable via env / explicit construction — see the
# `parsed_json_dir` / `pdf_dir` fields below). Homed here so `SkunkConfig` is the single
# source of truth for where the corpus lives; `corpus.py` resolves through it.
_DEFAULT_PARSED_JSON_DIR = (
    Path.home() / "Desktop/officeqa/treasury_bulletins_parsed/jsons"
)
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
    # Best-of-N: run this many independent codegen→exec trials per compute call (in
    # parallel) and commit the most frequent answer (ties broken arbitrarily). MissingData
    # outcomes abstain from the vote — a NeedsMore is returned only when EVERY trial
    # signals it. 1 = single-trial (today's behavior). (env: SKUNK_COMPUTE_BEST_OF_N)
    compute_best_of_n: int = 5
    # Bypass the QuestionExplainer's per-question selection and inject the ENTIRE
    # PRECOMPUTED_CONCEPTS catalog into every compute `## Concept references` block (skips
    # the selection LLM call). Default OFF (the explainer selects only the relevant
    # entries); flip per-run to A/B selection vs. full-dump on compute accuracy.
    # (env: SKUNK_PRECOMPUTED_CONCEPT_REFS=1)
    compute_precomputed_concept_refs: bool = False

    # Replan-on-MissingData loop. Total compute invocations ≤ recovery_max_rounds + 1.
    recovery_max_rounds: int = 2

    # Corpus directories. `parsed_json_dir` holds the per-bulletin parsed-JSON the text
    # accessors read (cleaner than PyMuPDF text on scanned pages); `pdf_dir` holds the
    # source bulletin PDFs used for page rendering. `corpus.py` resolves through these.
    # (env: OFFICEQA_PARSED_JSON_DIR, OFFICEQA_PDF_DIR)
    parsed_json_dir: Path = field(default_factory=lambda: _DEFAULT_PARSED_JSON_DIR)
    pdf_dir: Path = field(default_factory=lambda: _DEFAULT_PDF_DIR)

    # Optional pre-rendered page-PNG cache (`<dir>/<stem>_<page>.png`). When set, page rendering
    # (`render_page_b64`, hence `view_figure`) serves from here instead of rasterizing the PDF —
    # the competition-latency win for corpora with a warmed render cache (e.g. DAIS
    # `~/dais/page_renders`). None → always render on the fly. (env: SKUNK_PAGE_RENDERS_DIR)
    page_renders_dir: Path | None = None

    # Prompt overrides YAML — corpus blurbs, few-shots, lessons. (env: SKUNK_PROMPT_OVERRIDES)
    prompt_overrides_path: str = "config/prompts/us_receipts_expenditures.yaml"

    # Ablation: golden page refs bypass the retrieve operator (eval runs only).
    golden_pages: list[PageRef] | None = field(default=None, repr=False)

    # Retrieve dispatch: "search_agent" (iterative ChromaDB + LLM loop) is the sole
    # backend; golden_pages is a separate eval bypass. (env: SKUNK_RETRIEVER)
    retriever: Literal["search_agent"] = "search_agent"

    # Search-agent corpus artifacts (built offline; agent fails fast if missing).
    # (env: SKUNK_CHROMADB_DIR, SKUNK_CHROMADB_COLLECTION, SKUNK_CLEAN_PAGE_MAP)
    chromadb_dir: str = "cache/chromadb"
    chromadb_collection: str = "treasury_pages"
    clean_page_map_path: str = "cache/clean_page_map.json"

    # ChromaDB server (HttpClient) the read paths connect to. The embedded PersistentClient
    # deadlocks under 15-way in-process concurrency; the server owns ChromaDB's concurrency.
    # Launch it over `chromadb_dir` with `scripts/run_chroma_server.sh`.
    # (env: SKUNK_CHROMA_SERVER_HOST, SKUNK_CHROMA_SERVER_PORT)
    chroma_server_host: str = "127.0.0.1"
    chroma_server_port: int = 8001

    # Embedding model for vector_search (must match the stored embeddings).
    # (env: SKUNK_EMB_MODEL)
    emb_model_id: str = "gemini-embedding-001"

    # Per-question search-agent budget. (env: SKUNK_AGENT_MAX_STEPS, SKUNK_AGENT_MAX_PAGES_PER_TOOL_CALL)
    agent_max_steps: int = 20
    agent_max_pages_per_tool_call: int = 20

    # Hard cap on NON-progressing attempts (parse-misfires — prose with no runnable action
    # block — and exec-machinery failures) before the agent aborts. These attempts do NOT
    # count against `agent_max_steps`, so genuine search progress always gets its full step
    # budget; this knob just stops a model that never emits a valid action from looping
    # forever. Total LLM calls per run are bounded by agent_max_steps + agent_max_misfires.
    # (env: SKUNK_AGENT_MAX_MISFIRES)
    agent_max_misfires: int = 6

    # Hard cap on the rendered size of a single `grep_corpus` observation, in *tokens*
    # (`grep_corpus` defaults to limit=None = "every matching chunk", so one broad pattern
    # could dump ~200K+ tokens into the context in one shot — that ballooned a request past
    # the model's input ceiling and 400'd it repeatedly). Over-budget hits are dropped with a
    # note telling the agent to narrow its pattern / pass `limit`. Token→char conversion uses
    # ~4 chars/token (see GrepCorpusTool). (env: SKUNK_GREP_MAX_OUTPUT_TOKENS)
    grep_max_output_tokens: int = 200_000

    # Hard cap on the rendered size of a single `read_document` observation, in *chars*
    # (`read_document` dumps the full cleaned text of up to `agent_max_pages_per_tool_call`
    # pages; on the dense Treasury tables one over-broad call could otherwise push a request
    # past the model's ~1M-token input ceiling and 400 it). Over-budget output is truncated
    # with a note telling the agent to read fewer doc_ids per call. Char-based (not tokens) to
    # avoid the chars/token estimate error on numeric tables.
    # (env: SKUNK_READ_DOCUMENT_MAX_OUTPUT_CHARS)
    read_document_max_output_chars: int = 400_000

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
    lookup_max_steps: int = 4
    # Active lookup tools by name (see `lookup_tools._REGISTRY`); None → all tools.
    # (env: SKUNK_LOOKUP_TOOLS — comma-separated, e.g. "fetch_fred,tavily_search")
    lookup_tools: list[str] | None = None

    # Agent-loop chat model. None → `llm_model`. (env: SKUNK_AGENT_MODEL)
    agent_model_id: str | None = None

    # Page-index semantic filter: a single coarse pass over each year-filtered page's
    # metadata summary, scored once against every active branch target at once (a B×K
    # true/false matrix, one row per page, one column per target). The filter runs on a
    # cheaper model, resolved through the standard per-call model-override registry under
    # key "semfilter" (seeded in `__post_init__`) — so it's tuned like any other call-site
    # rather than via a dedicated field. NOTE: with the flat-block filter this is the max
    # CONTENT BLOCKS per call (pages are exploded into blocks and packed page-coherently up to
    # this cap), not pages. 32: the flat shape is far less batch-sensitive than the old nested
    # page objects, so a wide cap keeps call count / latency low without losing recall.
    # (env: SKUNK_SEMFILTER_MODEL, SKUNK_SEMFILTER_BATCH)
    semfilter_batch_size: int = 32

    # Per-LLM-call caps for the external-lookup agent's turns (named for the retired
    # selection agent that shared them). Mirrors the search agent: without a combined
    # thinking+visible cap, Flash thrashed to ~63K thinking tokens / ~285s per step
    # and emitted no parseable tool call (parse-retry death spiral). Selection
    # itself (skunk.block_select) is bounded PromptedCalls, not an agent loop.
    # (env: SKUNK_SELECT_AGENT_MAX_OUTPUT_TOKENS, SKUNK_SELECT_AGENT_TIMEOUT_S)
    select_agent_max_output_tokens: int = 8192
    select_agent_request_timeout_s: float = 150.0

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

    # Build: the `vision_rescan` stage always re-reads `parse_broken` pages (mangled parses). Pages
    # flagged `has_unparsed_graphics` that are CHART-ONLY (a chart/figure with no table on the page,
    # so its data is otherwise lost) are re-read only when this is on. That set is the bulk of the
    # vision working set (~6k pages corpus-wide) and low-value for table-centric queries, so chart
    # re-reading is an opt-in phase, default off. (env: SKUNK_VISION_RESCAN_CHARTS=1)
    vision_rescan_charts: bool = False

    # Human-in-the-loop: route specific sub-tasks to a person instead of (or after) the
    # model. Three independent toggles, all default OFF (zero behavior change when unset);
    # the policy/channel wiring lives in `human.py` and is enforced at the orchestrator
    # dispatch seam, so no operator or planner code changes when these flip. Transport is
    # chosen by handler presence: under the competition server these route through the async
    # broker/web UI (non-blocking — a worker resolves each request); for a local CLI run with
    # no handler they fall back to the blocking console, so target a handful of UIDs, not a sweep.
    # - human_figure: for `visual_only` retrieve branches (chart/figure questions the
    #   vision model reads unreliably), show the human the rendered page(s) + the model's
    #   candidate and take their answer. (env: SKUNK_HUMAN_FIGURE=1)
    # - human_verify_extract: for non-visual extractions, the human confirms/corrects the
    #   extracted value(s) against the rendered source page(s). (env: SKUNK_HUMAN_VERIFY_EXTRACT=1)
    # - human_lookup: every lookup_external branch is performed by the human instead of the
    #   LookupAgent. (env: SKUNK_HUMAN_LOOKUP=1)
    human_figure: bool = False
    human_verify_extract: bool = False
    human_lookup: bool = False

    def __post_init__(self) -> None:
        # Route per-stage models through the override registry so `PromptedCall` resolves them
        # like every other call-site. Defaulted here unless a run pins them explicitly
        # (SKUNK_MODEL_OVERRIDES=stage=… or, for the filter, SKUNK_SEMFILTER_MODEL). Efforts
        # come from each call-site's `default_effort` (compute=high, planner=high, replanner=high; extract=medium),
        # overridable via SKUNK_EFFORT_OVERRIDES.
        # - semfilter: the cheap coarse filter runs on flash-lite.
        # - extract.{text,vision}: flash, medium thinking. Pro is the stronger read
        #   on dense scanned tables but its 8M input-tok/min quota + 380s latency tails choke the
        #   parallel select-agent fan-out; pin Pro back per-run via SKUNK_MODEL_OVERRIDES. The
        #   default path is the `text` tier; `vision` is the fallback.
        # - compute.codegen: flash — codegen/reasoning over the extracted values (high thinking).
        # - data_prep.codegen: 3.1 Pro at medium thinking — cleaning/coalescing/unioning the value
        #   pool; unioning a multi-page table needs Pro to merge every row (Flash truncated the
        #   hand-written value dict ~halfway).
        # - replanner: 3.1 Pro at high thinking — same as the planner; revising a failed plan
        #   needs the same decomposition quality as the initial plan.
        # - planner: 3.1 Pro at high thinking. The initial plan's decomposition quality
        #   (branch coverage, operator routing, period fidelity) is worth the per-question
        #   cost — the flash planner systematically dropped/misrouted branches that Pro/high
        #   gets right (plan-probe 2026-06-14: 19/20 known-bad dev plans fixed).
        # Everything else (toc_pick, …) runs on the base `llm_model` (flash).
        self.model_overrides.setdefault("semfilter", "gemini-3.1-flash-lite")
        self.model_overrides.setdefault("extract.text", "gemini-3.5-flash")
        self.model_overrides.setdefault("extract.vision", "gemini-3.5-flash")
        self.model_overrides.setdefault("compute.codegen", "gemini-3.1-pro-preview")
        self.model_overrides.setdefault("planner", "gemini-3.1-pro-preview")
        self.model_overrides.setdefault("replanner", "gemini-3.1-pro-preview")
        self.model_overrides.setdefault("data_prep.codegen", "gemini-3.1-pro-preview")

    @classmethod
    def from_yaml(cls, path: str) -> SkunkConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_env(cls) -> SkunkConfig:
        model_overrides = _parse_model_overrides(
            os.environ.get("SKUNK_MODEL_OVERRIDES", "")
        )
        # SKUNK_SEMFILTER_MODEL is a convenience knob for the "semfilter" override;
        # an explicit SKUNK_MODEL_OVERRIDES=semfilter=… wins, and the hardcoded
        # default (`__post_init__`) fills in if neither is set.
        sem_model = os.environ.get("SKUNK_SEMFILTER_MODEL")
        if sem_model:
            model_overrides.setdefault("semfilter", sem_model)
        return cls(
            llm_model=os.environ.get("SKUNK_LLM_MODEL", "gemini-3.5-flash"),
            llm_provider=os.environ.get("SKUNK_LLM_PROVIDER", "genai"),  # type: ignore[arg-type]
            effort_overrides=_parse_effort_overrides(
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
            semfilter_batch_size=int(os.environ.get("SKUNK_SEMFILTER_BATCH", "32")),
            select_agent_max_output_tokens=int(
                os.environ.get("SKUNK_SELECT_AGENT_MAX_OUTPUT_TOKENS", "8192")
            ),
            select_agent_request_timeout_s=float(
                os.environ.get("SKUNK_SELECT_AGENT_TIMEOUT_S", "150")
            ),
            extract_max_output_tokens=int(
                os.environ.get("SKUNK_EXTRACT_MAX_OUTPUT_TOKENS", "8192")
            ),
            extract_request_timeout_s=float(
                os.environ.get("SKUNK_EXTRACT_TIMEOUT_S", "150")
            ),
            compute_precomputed_concept_refs=os.environ.get(
                "SKUNK_PRECOMPUTED_CONCEPT_REFS", "0"
            )
            not in ("", "0"),
            extract_vision_only=os.environ.get("SKUNK_EXTRACT_VISION_ONLY", "0")
            not in ("", "0"),
            vision_rescan_charts=os.environ.get("SKUNK_VISION_RESCAN_CHARTS", "")
            not in ("", "0"),
            human_figure=os.environ.get("SKUNK_HUMAN_FIGURE", "0") not in ("", "0"),
            human_verify_extract=os.environ.get("SKUNK_HUMAN_VERIFY_EXTRACT", "0")
            not in ("", "0"),
            human_lookup=os.environ.get("SKUNK_HUMAN_LOOKUP", "0") not in ("", "0"),
            retriever=os.environ.get("SKUNK_RETRIEVER", "search_agent"),  # type: ignore[arg-type]
            chromadb_dir=os.environ.get("SKUNK_CHROMADB_DIR", "cache/chromadb"),
            chromadb_collection=os.environ.get(
                "SKUNK_CHROMADB_COLLECTION", "treasury_pages"
            ),
            clean_page_map_path=os.environ.get(
                "SKUNK_CLEAN_PAGE_MAP", "cache/clean_page_map.json"
            ),
            chroma_server_host=os.environ.get("SKUNK_CHROMA_SERVER_HOST", "127.0.0.1"),
            chroma_server_port=int(os.environ.get("SKUNK_CHROMA_SERVER_PORT", "8001")),
            emb_model_id=os.environ.get("SKUNK_EMB_MODEL", "gemini-embedding-001"),
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
