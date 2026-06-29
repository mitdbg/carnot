"""Centralized configuration for experiments."""

from __future__ import annotations

import yaml
from dataclasses import dataclass
from typing import cast, Literal
from omegaconf import DictConfig, OmegaConf
from qatfd.constants import BROWSECOMP_PLUS, FINANCE_BENCH, FRESHSTACK, OFFICE_QA, QAMPARI, TREC_BIOGEN
from skunk.config import QATFDSearchAgentConfig, RAGLLMConfig, SearchAgentConfig, SystemConfig

# ---------------------------------------------------------------------------
# General experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # the benchmark split to run (dev/test)
    split: Literal["dev", "test"] = "dev"
    # number of questions to run
    sample: int | None = None
    # RNG seed for `sample` selection; set it to draw the same subset across systems (null = nondeterministic)
    seed: int | None = None
    # specific qids to run (overrides split/sample)
    qids: list[str] | None = None
    # question-level concurrency
    workers: int = 32
    # label prefix for the run dir
    run_name: str = ""
    # whether to stream logs to the console (in addition to files)
    console: bool = False

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
    # directory for chromadb (the embedded PersistentClient store, used when chromadb_host is null)
    chromadb_dir: str
    # chroma collection with benchmark embeddings
    chromadb_collection: str
    # ChromaDB server (HttpClient) to connect reads to. null host => embedded PersistentClient over
    # chromadb_dir; set host to use a long-lived warm server (see skunk/scripts/run_chroma_server.sh).
    chromadb_host: str | None
    chromadb_port: int
    # path (under skunk/) to a benchmark's prompt-overrides YAML; null = no benchmark-specific prompt notes.
    prompts_path: str | None
    # path (under skunk/) to the corpus PDFs, enabling the SearchAgent's view_figure tool; null = no figure tool.
    pdf_dir: str | None

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
    # path to json list of test set UIDs
    test_set_uids_path: str

@dataclass
class BrowseCompPlusConfig(BenchmarkConfig):
    # file containing questions for BrowseComp-Plus
    bcp_questions: str
    # glob for metadata files containing doc_id -> source info (e.g. URL) for BrowseComp-Plus documents
    bcp_metadata_glob: str
    # file containing the qids in the BrowseComp-Plus test set (as defined in KARLBench)
    bcp_test_ids_path: str
    # the llm to use for judging system outputs on BrowseComp-Plus
    judge_model: str

@dataclass
class TrecBiogenConfig(BenchmarkConfig):
    # 2025 BioGen Task A JSON: 40 questions + expert reference answers w/cited PMIDs. Task A's
    # expert answers are the 2024-edition answers re-released against the 2025 corpus.
    task_a_path: str
    # glob for metadata files mapping PMID -> abstract text (produced by the embedding job)
    biogen_metadata_glob: str
    # the llm used to judge nugget-completion (gold nuggets vs the system answer, KARL D.1 prompt)
    judge_model: str
    # JSON of official gold nuggets keyed by qa_id (organizer/BioACE `baseline_labels.json`: a
    # list of {meta_data.qa_id, answer:[{nuggets:[...]}]}, flattened per question).
    # NOTE: the official nuggets are finer-grained (~24.6/q) than KARL's consolidated set (~7.1/q),
    # so the absolute nugget completion number is NOT directly comparable to KARL's reported 85.0.
    nuggets_path: str
    # optional JSON of held-out test qids; null => ALL 40 questions are the (held-out) test set,
    # so the benchmark is run with experiments.split=test (the whole set = the comparison).
    test_ids_path: str | None = None
    # weight given to a `partial_support` nugget in the recall score (full support = 1.0).
    partial_credit: float = 0.0
    # If >1, the 26.8M-abstract corpus is split across N per-rank Chroma collections named
    # f"{chromadb_collection}_r{i}", each in its OWN chroma dir (its own chroma.sqlite3). The
    # benchmark opens all N and merges query/get across them (MergedCollection). Separate dirs are
    # essential: collections in ONE dir share one chroma.sqlite3, so it still hits chromadb 1.5.x's
    # ~24M-row metadata-segment scale wall. 1 => single collection.
    chromadb_num_shards: int = 1
    # Optional explicit per-shard chroma dirs (len == chromadb_num_shards). If null, shard i is read
    # from f"{chromadb_dir}/r{i}". Use it to point shards at independent dirs (or mix: some shards in
    # one dir, others elsewhere — e.g. to salvage already-built shards without a full rebuild).
    chromadb_shard_dirs: list[str] | None = None

@dataclass
class QampariConfig(BenchmarkConfig):
    # QAMPARI test JSONL (`test_data.jsonl`): 1000 multi-answer questions over Wikipedia. Each record
    # is (qid, question_text, answer_list[]) where every answer carries `answer_text`, `aliases`, and
    # `proof[]` (each proof has `found_in_url` = the supporting Wikipedia article).
    questions_path: str
    # the llm used by the nugget-completion judge (each gold answer entity is one nugget; the answer
    # is graded by entity recall, mirroring KARL's nugget-based completion for QAMPARI). The passage
    # text + article title are served from the Chroma collection at eval time (the full ~25.9M-chunk
    # corpus is too large for an in-RAM doc map), so no metadata-glob config is needed.
    judge_model: str
    # optional JSON of held-out test qids ({"query_ids": [...]}); null => ALL 1000 questions are the
    # (held-out) test set, so the benchmark is run with experiments.split=test (the whole set = the comparison).
    test_ids_path: str | None = None
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
    # optional JSON of held-out test qids ({"query_ids": [...]}); null => ALL 150 questions are the
    # (held-out) test set, so the benchmark is run with experiments.split=test (the whole set = the comparison).
    test_ids_path: str | None = None

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
    # base dir under skunk/ holding each topic's data at `{data_dir}/{topic}/{corpus,queries}.jsonl`.
    data_dir: str = "freshstack"
    # FreshStack queries JSONL; null => derived from topic as `{data_dir}/{topic}/queries.jsonl`. Each
    # record is (query_id, query_title, query_text, nuggets[]) where every nugget carries `text`
    # (a decompositional fact) and `relevant_corpus_ids` (the supporting corpus doc ids).
    questions_path: str | None = None
    # FreshStack corpus JSONL; null => derived from topic as `{data_dir}/{topic}/corpus.jsonl`. Each
    # record is (_id, text, metadata); read directly into the doc map (_id -> text) — the corpus is
    # small enough to hold in RAM and the file is the source of truth for the embedded text.
    corpus_path: str | None = None
    # optional JSON of held-out test qids ({"query_ids": [...]}); null => ALL of the topic's questions
    # are the (held-out) test set, so the benchmark is run with experiments.split=test.
    test_ids_path: str | None = None
    # weight given to a `partial_support` nugget in the recall score (full support = 1.0).
    partial_credit: float = 0.0

# ---------------------------------------------------------------------------
# configuration factories
# ---------------------------------------------------------------------------

def benchmark_config_factory(cfg: DictConfig) -> BenchmarkConfig:
    bench_cfg = cast(dict, OmegaConf.to_container(cfg.benchmarks, resolve=True))
    if bench_cfg["name"] == OFFICE_QA:
        return OfficeQAConfig(**bench_cfg)
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


def system_config_factory(cfg: DictConfig) -> SystemConfig:
    system_cfg = cast(dict, OmegaConf.to_container(cfg.systems, resolve=True))
    if system_cfg["name"] == "rag_llm":
        return RAGLLMConfig(**system_cfg)
    elif system_cfg["name"] == "search_agent":
        return SearchAgentConfig(**system_cfg)
    elif system_cfg["name"] == "qatfd_search_agent":
        return QATFDSearchAgentConfig(**system_cfg)
    else:
        raise ValueError(f"unknown system {system_cfg['name']!r}")
