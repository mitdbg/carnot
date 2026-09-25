"""Centralized configuration for experiments."""

from __future__ import annotations

import uuid
import yaml
from dataclasses import dataclass, field
from typing import cast, Literal
from omegaconf import DictConfig, OmegaConf
from qatfd.constants import BROWSECOMP_PLUS, FINANCE_BENCH, FRESHSTACK, OFFICE_QA, OFFICE_QA_SYNTH, QAMPARI, TREC_BIOGEN
from skunk.config import AgentConfig, SearchAgentConfig, StorageConfig

# ---------------------------------------------------------------------------
# agent configuration
# ---------------------------------------------------------------------------

@dataclass
class BootstrapConfig(AgentConfig):
    # the number of workers to use to process semantic map tool calls in parallel
    semantic_map_max_workers: int = 16
    # the maximum number of chunks that can be processed by a single semantic map tool call
    semantic_map_max_candidate_chunks: int = 10_000
    # the fraction of the semantic map model's context window that can be used to fit document text
    semantic_map_context_frac: float = 0.9
    # maximum output tokens for each semantic map judge call
    semantic_map_max_output_tokens: int = 4096
    # disable reasoning/thinking on the semantic map judge calls
    semantic_map_disable_reasoning: bool = True
    # model for the semantic map's per-candidate judge calls; None => `llm_model` (the agent
    # model). Set it to a cheaper model to run the (token-heavy) candidate mapping on a cheap model
    semantic_map_llm_model: str | None = None
    # OpenRouter provider order (no fallback) for the semantic map judge calls only. None => use
    # the client-wide `llm_provider_order`. Lets the judge model route to specific providers (e.g. [akashml, parasail])
    semantic_map_provider_order: list[str] | None = None
    # maximum number of thread for executing non-semantic maps
    map_max_workers: int = 32
    # maximum number of requests that can go to different chromadb collections in parallel
    max_parallel_chroma_queries: int = 16
    # maximum number of chunks one create/add/copy/merge collection tool call may move
    max_copy_chunks: int = 100_000
    # search / grep results are rendered as a summary: the first N chunks per collection, each
    # truncated to M characters (the full result set is still what gets added to a collection)
    search_preview_chunks: int = 10
    search_preview_chars: int = 400
    # number of sample rows rendered per collection after a map / semantic_map call
    map_preview_samples: int = 5
    # cost budget in dollars for the agent; None means no budget limit (default)
    cost_budget: float | None = None
    # latency budget in seconds for the agent; None means no latency limit (default)
    latency_budget: float | None = None


@dataclass
class EnrichConfig(AgentConfig):
    # the number of most recent queries (handled by the system) shown to the agent as the query workload
    max_previous_queries: int = 20
    # the number of workers to use to process semantic map tool calls in parallel
    semantic_map_max_workers: int = 16
    # the maximum number of chunks that can be processed by a single semantic map tool call
    semantic_map_max_candidate_chunks: int = 10_000
    # the fraction of the semantic map model's context window that can be used to fit document text
    semantic_map_context_frac: float = 0.9
    # maximum output tokens for each semantic map judge call
    semantic_map_max_output_tokens: int = 4096
    # disable reasoning/thinking on the semantic map judge calls
    semantic_map_disable_reasoning: bool = True
    # model for the semantic map's per-candidate judge calls; None => `llm_model` (the agent
    # model). Set it to a cheaper model to run the (token-heavy) candidate mapping on a cheap model
    semantic_map_llm_model: str | None = None
    # OpenRouter provider order (no fallback) for the semantic map judge calls only. None => use
    # the client-wide `llm_provider_order`. Lets the judge model route to specific providers (e.g. [akashml, parasail])
    semantic_map_provider_order: list[str] | None = None
    # maximum number of thread for executing non-semantic maps
    map_max_workers: int = 32
    # maximum number of requests that can go to different chromadb collections in parallel
    max_parallel_chroma_queries: int = 16
    # maximum number of chunks one create/add/copy/merge collection tool call may move
    max_copy_chunks: int = 100_000
    # search / grep results are rendered as a summary: the first N chunks per collection, each
    # truncated to M characters (the full result set is still what gets added to a collection)
    search_preview_chunks: int = 10
    search_preview_chars: int = 400
    # number of sample rows rendered per collection after a map / semantic_map call
    map_preview_samples: int = 5
    # cost budget in dollars for the agent; None means no budget limit (default)
    cost_budget: float | None = None
    # latency budget in seconds for the agent; None means no latency limit (default)
    latency_budget: float | None = None


@dataclass
class RAGLLMConfig(AgentConfig):
    # number of chunks for the vector search to return
    top_k: int | None = None

    def __post_init__(self) -> None:
        if self.top_k is None:
            raise ValueError(
                "RAGLLMConfig.top_k is unset (null); set it explicitly, e.g. "
                "`systems.top_k=20` on the command line."
            )


@dataclass(kw_only=True)
class QATFDSearchAgentConfig(SearchAgentConfig):
    # whether to enrich working sets with additional metadata;
    # - null means we do not perform enrichment
    # - "before" means we enrich as a pre-processing step before any queries arrive (BootstrapAgent)
    # - "after" means we enrich after queries arrive (EnrichAgent, every enrich_query_batch_size questions)
    # - "both" means we bootstrap before the first query AND enrich after every batch of queries
    enrich_working_sets: Literal[None, "before", "after", "both"] = None
    # the batch size (in number of queries) to enrich working sets if enrich_working_sets="after"
    enrich_query_batch_size: int | None = None
    # hide the non-bootstrap and non-enrich working sets from the SearchAgent and clear them after a bootstrap/enrich agent runs
    hide_and_clear_working_sets: bool = True
    # configuration for a BootstrapAgent to create initial working sets
    bootstrap_config: BootstrapConfig
    # configuration for an EnrichAgent to enrich existing working sets
    enrich_config: EnrichConfig

    def __post_init__(self) -> None:
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        # hydra hands the nested agent configs over as plain dicts (OmegaConf.to_container); build the
        # dataclasses so their fields (and defaults) apply. `name` is not a yaml key: derive it.
        if isinstance(self.bootstrap_config, dict):
            self.bootstrap_config = BootstrapConfig(**{"name": "bootstrap", **self.bootstrap_config})
        if isinstance(self.enrich_config, dict):
            self.enrich_config = EnrichConfig(**{"name": "enrich", **self.enrich_config})


@dataclass(kw_only=True)
class CodexConfig(AgentConfig):
    # path to the config.toml for the Codex agent
    codex_config_toml: str
    # path to the AGENTS.md for the Codex agent
    codex_agents_md: str
    # path to the example answer schema json file
    codex_answer_schema_file: str
    # whether or not to have codex resume its previous session for each question
    session_resume: bool
    # whether or not to have codex use shell tools and write to AGENTS.md
    codex_shell: bool
    # list of tools the Codex agent may use
    enabled_tools: list[str]
    # URL of the MCP server
    mcp_url: str
    # style of corpus interaction for the Codex system; one of "tools", "dci", or "both"
    corpus_interaction: Literal["tools", "dci", "both"] = "tools"
    # whether we are running questions in parallel or in sequence (copied at runtime from ExperimentConfig)
    run_mode: Literal["parallel", "sequential"]
    # token threshold at which Codex auto-compacts the thread; None keeps Codex's default
    # (90% of the model context window). Codex clamps any value to that 90% ceiling.
    auto_compact_token_limit: int | None = None
    # used when codex_shell is True in order to place the scratch directory outside of the benchmarks directory
    codex_scratch_dir: str | None = None


# ---------------------------------------------------------------------------
# General experiment configuration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    # label prefix for the run dir
    run_name: str
    # the benchmark split to run (dev|test)
    split: Literal["dev", "test"] = "dev"
    # number of questions to run
    sample: int | None = None
    # RNG seed for `sample` selection; set it to draw the same subset across systems (null = nondeterministic)
    sample_seed: int | None = None
    # RNG seed for shuffling qids to test different orderings of the questions in sequential runs
    shuffle_seed: int | None = None
    # shuffle at the level of question GROUPS rather than single questions: the name of a `Question.meta`
    # key (e.g. `loi_idx` for officeqa_synth's lines of inquiry); questions sharing a value stay together, in
    # their original order, and only the order of the groups is drawn from `shuffle_seed`. null = per-question.
    shuffle_group_key: str | None = None
    # specific qids to run (overrides split/sample)
    qids: list[str] | None = None
    # question-level concurrency
    workers: int = 32
    # whether to stream logs to the console (in addition to files)
    console: bool = False
    # resume an interrupted run: path to its existing run dir (the one holding results.jsonl /
    # traces). Questions already recorded in results.jsonl are skipped; the rest are (re)run into
    # the same dir. Reuse the SAME benchmark/system/split/sample/seed so the question set matches.
    resume_dir: str | None = None
    # the mode to run the experiment questions in (parallel|sequential)
    run_mode: Literal["parallel", "sequential"] = "parallel"
    # the analytics_id which uniquely identifies a set of inference calls in OpenRouter;
    # used by Codex to associate inference with solving specific questions
    analytics_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


# --------------------------------------------------------------------------------
# Benchmark-specific configuration
# --------------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    # name of the benchmark
    name: str
    # storage for the benchmark's data
    storage: StorageConfig
    # path (under qatfd/benchmarks/) to the dev/test split JSON — {"dev": [qids], "test": [qids]}. The runner selects
    # `experiments.split` from these lists (dev is explicit, not test's complement). null => no split
    # (dev = everything, test = empty). Generated by scripts/make_splits.py.
    splits_path: str | None
    # path to the config.toml for the Codex agent
    codex_config_toml: str
    # path to the AGENTS.md for the Codex agent
    codex_agents_md: str
    # path to the example answer schema json file
    codex_answer_schema_file: str
    # url for mcp server
    mcp_url: str

    @classmethod
    def from_yaml(cls, path: str) -> BenchmarkConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


@dataclass
class OfficeQAConfig(BenchmarkConfig):
    # path to the benchmark questions and answers
    csv_path: str
    # document map containing clean page text
    clean_page_map_path: str


@dataclass
class OfficeQASynthConfig(OfficeQAConfig):
    # path (under qatfd/benchmarks/, or absolute) to the synthetic qa_pairs.json
    qa_pairs_path: str
    # llm used by the nugget-completion judge: a synthetic answer is a LIST of nugget strings (often several,
    # often short sentences), so the score is nugget recall (KARL-style), not the officeqa exact-answer scorer.
    judge_model: str
    # weight given to a `partial_support` nugget in the recall score (full support = 1.0).
    partial_credit: float = 0.0
    # wall-clock cap (s) on one judge request; a request the provider never answers otherwise hangs until the
    # upstream proxy drops it (~100 min seen on OpenRouter). null = no cap.
    judge_timeout_s: float | None = 300.0
    # include the seed records (idx == null: the original OfficeQA dev questions the lines of inquiry were
    # grown from)? They carry a single string answer and are already covered by the `officeqa` benchmark,
    # so by default only the generated follow-ups are evaluated.
    include_seeds: bool = False


@dataclass
class BrowseCompPlusConfig(BenchmarkConfig):
    # file containing questions for BrowseComp-Plus
    bcp_questions: str
    # glob for metadata files containing doc_id -> source info (e.g. URL) for BrowseComp-Plus documents
    bcp_metadata_glob: str
    # the llm to use for judging system outputs on BrowseComp-Plus
    judge_model: str


@dataclass
class TrecBiogenConfig(BenchmarkConfig):
    # 2025 BioGen Task A JSON: 40 questions + expert reference answers w/cited PMIDs. Task A's
    # expert answers are the 2024-edition answers re-released against the 2025 corpus.
    task_a_path: str
    # the llm used to judge nugget-completion (gold nuggets vs the system answer, KARL D.1 prompt)
    judge_model: str
    # JSON of official gold nuggets keyed by qa_id (organizer/BioACE `baseline_labels.json`: a
    # list of {meta_data.qa_id, answer:[{nuggets:[...]}]}, flattened per question).
    # NOTE: the official nuggets are finer-grained (~24.6/q) than KARL's consolidated set (~7.1/q),
    # so the absolute nugget completion number is NOT directly comparable to KARL's reported 85.0.
    nuggets_path: str
    # weight given to a `partial_support` nugget in the recall score (full support = 1.0).
    partial_credit: float = 0.0
    # If >1, the 26.8M-abstract corpus is split across N per-rank Chroma collections named
    # f"{collection_name}_r{i}", each built in its OWN chroma dir (its own chroma.sqlite3). Separate
    # dirs are essential at build time: collections in ONE dir share one chroma.sqlite3, so it still
    # hits chromadb 1.5.x's ~24M-row metadata-segment scale wall. At read time the benchmark opens all
    # N shard collections and merges query/get across them (MergedCollection). 1 => single collection.
    chroma_server_num_shards: int = 1
    # Per-shard chroma SERVER ports (len == chroma_server_num_shards); REQUIRED when num_shards > 1.
    # Shard i is served by a warm chroma server at (chroma_server_host, ports[i]) over that shard's dir.
    # The benchmark connects via one HttpClient per shard (MergedClient) and queries them in PARALLEL —
    # HNSW stays resident across runs (no cold load). Launch servers via scripts/run_chroma_server.sh.
    chroma_server_shard_ports: list[int] | None = None


@dataclass
class QampariConfig(BenchmarkConfig):
    # QAMPARI test JSONL (`test_data.jsonl`): the 1000-question KARL eval set over Wikipedia. Each
    # record is (qid, question_text, answer_list[]) where every answer carries `answer_text`,
    # `aliases`, and `proof[]` (each proof has `found_in_url` = the supporting Wikipedia article).
    # This is the held-out TEST split (all 1000 qids); run it with experiments.split=test.
    questions_path: str
    # the llm used by the nugget-completion judge (each gold answer entity is one nugget; the answer
    # is graded by entity recall, mirroring KARL's nugget-based completion for QAMPARI). The passage
    # text + article title are served from the Chroma collection at eval time (the full ~25.9M-chunk
    # corpus is too large for an in-RAM doc map), so no metadata-glob config is needed.
    judge_model: str
    # dev JSONL: 50 questions sampled (seed 0) from QAMPARI's train_data — DISJOINT from the test set
    # (which is the full KARL 1000, so dev never leaks into test). Loaded ALONGSIDE questions_path so
    # experiments.split=dev resolves with no per-run override; both splits share the one Wikipedia
    # index (the split is purely at the question level). Same record schema as questions_path.
    # Generated by scripts/make_splits.py. null => no dev extract (the dev split finds no questions).
    dev_questions_path: str | None = None
    # weight given to a `partial_support` nugget in the recall score (full support = 1.0).
    partial_credit: float = 0.0


@dataclass
class FinanceBenchConfig(BenchmarkConfig):
    # FinanceBench open-source JSONL: 150 questions over SEC filings (financebench_id / question /
    # answer / evidence[{doc_name, evidence_page_num, ...}]).
    questions_path: str
    # glob for the embedding-job metadata files (metadata_rank*.json: unique_element_id ->
    # {doc_name, page_num, page_key, cleaned, element_id}), used to rebuild the page-text doc map.
    fb_metadata_glob: str
    # the llm used to judge answer correctness (single-nugget YES/NO judge, KARL-style; see judge.py).
    judge_model: str


@dataclass
class FreshstackConfig(BenchmarkConfig):
    # the llm used by the nugget-completion judge (each gold nugget is one GPT-4o decompositional
    # fact; the answer is graded by nugget recall, mirroring KARL's nugget-based completion).
    judge_model: str
    # FreshStack topic = which (corpus, queries) pair to run. Each topic has its OWN corpus and so
    # its OWN Chroma collection (unlike QAMPARI's shared index): `langchain` is KARL's FreshStack and
    # the held-out TEST set (203 q / 49,514 docs); `laravel` is the closest-sized DEV set (184 q /
    # 52,351 docs). To run dev, override `benchmarks.topic=laravel` — it re-derives the data paths and
    # collection below. (Others: angular, godot, yolo.)
    topic: str = "langchain"
    # base dir under qatfd/benchmarks/ holding each topic's data at `{data_dir}/{topic}/{corpus,queries}.jsonl`.
    data_dir: str = "freshstack"
    # FreshStack queries JSONL; null => derived from topic as `{data_dir}/{topic}/queries.jsonl`. Each
    # record is (query_id, query_title, query_text, nuggets[]) where every nugget carries `text`
    # (a decompositional fact) and `relevant_corpus_ids` (the supporting corpus doc ids).
    questions_path: str | None = None
    # FreshStack corpus JSONL; null => derived from topic as `{data_dir}/{topic}/corpus.jsonl`. Each
    # record is (_id, text, metadata); read directly into the doc map (_id -> text) — the corpus is
    # small enough to hold in RAM and the file is the source of truth for the embedded text.
    corpus_path: str | None = None
    # weight given to a `partial_support` nugget in the recall score (full support = 1.0).
    partial_credit: float = 0.0


# ---------------------------------------------------------------------------
# configuration factories
# ---------------------------------------------------------------------------


def benchmark_config_factory(cfg: DictConfig) -> BenchmarkConfig:
    bench_cfg = cast(dict, OmegaConf.to_container(cfg.benchmarks, resolve=True))
    bench_cfg["storage"] = StorageConfig(
        collection_name=bench_cfg.pop("collection_name"),
        chroma_server_host=bench_cfg.pop("chroma_server_host"),
        chroma_server_port=bench_cfg.pop("chroma_server_port"),
        pdf_dir=bench_cfg.pop("pdf_dir", None),
    )
    if bench_cfg["name"] == OFFICE_QA:
        return OfficeQAConfig(**bench_cfg)
    elif bench_cfg["name"] == OFFICE_QA_SYNTH:
        return OfficeQASynthConfig(**bench_cfg)
    elif bench_cfg["name"] == BROWSECOMP_PLUS:
        return BrowseCompPlusConfig(**bench_cfg)
    elif bench_cfg["name"] == TREC_BIOGEN:
        return TrecBiogenConfig(**bench_cfg)
    elif bench_cfg["name"] == FINANCE_BENCH:
        return FinanceBenchConfig(**bench_cfg)
    elif bench_cfg["name"] == QAMPARI:
        return QampariConfig(**bench_cfg)
    elif bench_cfg["name"] == FRESHSTACK:
        return FreshstackConfig(**bench_cfg)
    else:
        raise ValueError(f"unknown benchmark {bench_cfg['name']!r}")


def system_config_factory(cfg: DictConfig) -> tuple[AgentConfig | None, AgentConfig | None, AgentConfig | None]:
    system_cfg = cast(dict, OmegaConf.to_container(cfg.systems, resolve=True))
    if system_cfg["name"] == "rag_llm":
        retrieve_cfg = RAGLLMConfig(**system_cfg["retrieve"])
        compute_cfg = AgentConfig(**system_cfg["compute"])
        return retrieve_cfg, compute_cfg, None
    elif system_cfg["name"] == "search_agent":
        retrieve_cfg = QATFDSearchAgentConfig(**system_cfg["retrieve"])
        compute_cfg = AgentConfig(**system_cfg["compute"])
        return retrieve_cfg, compute_cfg, None
    elif system_cfg["name"] == "codex":
        codex_cfg = CodexConfig(**system_cfg)
        return None, None, codex_cfg
    else:
        raise ValueError(f"unknown system {system_cfg['name']!r}")
