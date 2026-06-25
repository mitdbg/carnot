"""Centralized configuration for experiments."""

from __future__ import annotations

import yaml
from dataclasses import dataclass
from typing import cast, Literal
from omegaconf import DictConfig, OmegaConf
from qatfd.constants import BROWSECOMP_PLUS, FINANCE_BENCH, OFFICE_QA, QAMPARI, TREC_BIOGEN
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
    # directory for chromadb
    chromadb_dir: str
    # chroma collection with benchmark embeddings
    chromadb_collection: str
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

@dataclass
class QampariConfig(BenchmarkConfig):
    # QAMPARI test JSONL (`test_data.jsonl`): 1000 multi-answer questions over Wikipedia. Each record
    # is (qid, question_text, answer_list[]) where every answer carries `answer_text`, `aliases`, and
    # `proof[]` (each proof has `found_in_url` = the supporting Wikipedia article).
    questions_path: str
    # glob for the embedding-job metadata files (metadata_rank*.json: chunk_id -> {cleaned, title,
    # page_id, url, element_id}), used to rebuild the passage-text doc map + the chunk->article map
    # that collapses retrieved chunk_ids to Wikipedia articles for doc-recall.
    qampari_metadata_glob: str
    # the llm used by the nugget-completion judge (each gold answer entity is one nugget; the answer
    # is graded by entity recall, mirroring KARL's nugget-based completion for QAMPARI).
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
