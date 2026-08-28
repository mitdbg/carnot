"""The `Benchmark` abstraction.

A Benchmark knows how to (1) load its questions, (2) build the retrieval resources
(a ChromaDB collection + a doc_id->text map) that systems run against, and (3)
score a predicted answer against gold. Subclasses customize each step; the runner
treats every benchmark uniformly.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from chromadb import Collection

from skunk.common import ExecutionContext
from skunk.storage.document_map import DocumentMap

from qatfd.config import BenchmarkConfig
from qatfd.paths import resolve_under_benchmarks
from qatfd.types import Question


def doc_recall(retrieved: list[str] | None, gold: list[str]) -> float:
    """Fraction of gold ids present anywhere in the retrieved set (no @k cutoff).
    The generic building block for the per-benchmark `recall_metrics` below."""
    if not gold or retrieved is None:
        return 0.0
    g = set(gold)
    return len(g & set(retrieved)) / len(g)


@dataclass
class BenchmarkResources:
    """Read-only retrieval substrate, built ONCE per run and shared (read-only)
    across all per-question workers. Mirrors skunk RetrieveOp's
    (collection, document_map) pair, named so systems read fields explicitly."""

    chroma_collection: Collection  # chromadb Collection (vector index over chunks)
    document_map: DocumentMap  # doc_id -> full text (for read_document / answer step)
    answer_format_hint: str = ""  # guidance on the format of the final answer (for the compute agent)
    compute_objective: str = ""  # one sentence on what the final answer is graded on (for the search agent)
    corpus_details: str | None = None  # information about the nature of the corpus, including available metadata fields


class Benchmark(ABC):
    name: str  # registry key, e.g. "officeqa"

    # hint appended to the final-answer system prompt so the generated answer is in
    # the form the scorer expects. OfficeQA overrides to demand a terse numeric form.
    answer_format_hint: str = (
        "Answer concisely with just the factual answer (a short span or value), no explanation."
    )

    # One sentence appended to the search agent's briefing telling it what the final answer will
    # be graded on, so it can calibrate how broadly to retrieve. This default is applied for
    # exact-answer benchmarks
    compute_objective: str = (
        "the final answer to this question will be graded on exact-answer correctness."
    )

    # description of the corpus and any available metadata fields
    corpus_details: str | None = None

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._resources: BenchmarkResources | None = None

    @abstractmethod
    def load_questions(self) -> list[Question]:
        """All questions in the benchmark (the runner applies the dev/test split)."""

    def _chroma_client(self):
        """The Chroma client — always the long-lived warm server (HttpClient) at
        (`chroma_server_host`, `chroma_server_port`). Collections are only ever read through a server.
        Centralized so every benchmark connects identically; see skunk/scripts/run_chroma_server.sh to
        warm one over the corpus's chroma dir."""
        cfg = self.config
        from skunk.chroma_client import make_chroma_client

        return make_chroma_client(cfg.storage.chroma_server_host, cfg.storage.chroma_server_port)

    def _chroma_where(self) -> str:
        cfg = self.config
        return f"server {cfg.storage.chroma_server_host}:{cfg.storage.chroma_server_port}"

    def _open_chroma_collection(self) -> Collection:
        """Open the single configured Chroma collection (`collection_name`)."""
        client = self._chroma_client()
        try:
            return client.get_collection(name=self.config.storage.collection_name)
        except Exception as e:
            raise RuntimeError(
                f"chroma collection {self.config.storage.collection_name!r} not found ({self._chroma_where()})."
            ) from e

    @abstractmethod
    def _build_resources(self) -> BenchmarkResources:
        """Open the chroma collection + build the document_map. Called once, cached by `get_resources`."""

    def get_resources(self) -> BenchmarkResources:
        """Cached accessor for the (collection, document_map) substrate."""
        if self._resources is None:
            self._resources = self._build_resources()
        return self._resources

    @abstractmethod
    async def score(self, question: Question, predicted: str, ctx: ExecutionContext) -> dict:
        """Grade `predicted` against gold. Returns at least
        {"score": float in [0,1], "scorer": <name>}; may add {"judge_rationale": ...}.
        Binary benchmarks return 0.0/1.0; graded ones (nugget-completion) a fraction.
        Async so LLM-judge benchmarks can await; deterministic ones just return."""

    def _load_splits(self) -> dict[str, set[str]] | None:
        """The dev/test split lists from `splits_path` ({"dev": [...], "test": [...]}), cached.
        None when no split file is configured. Generated by scripts/make_splits.py."""
        if getattr(self, "_splits_cache", "unset") == "unset":
            path = self.config.splits_path
            path = str(resolve_under_benchmarks(path)) if path else None
            if path and os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                self._splits_cache: dict[str, set[str]] | None = {
                    "dev": {str(q) for q in data.get("dev", [])},
                    "test": {str(q) for q in data.get("test", [])},
                }
            else:
                self._splits_cache = None
        return self._splits_cache

    def dev_qids(self) -> set[str] | None:
        """Explicit dev-split qids from the split file. None => no split file, so the runner falls
        back to 'everything not in test' (legacy behavior)."""
        splits = self._load_splits()
        return splits["dev"] if splits else None

    def test_qids(self) -> set[str]:
        """Held-out test split from the split file (empty when no split file is configured)."""
        splits = self._load_splits()
        return splits["test"] if splits else set()

    def recall_metrics(self, retrieved: list[str] | None, question: Question) -> dict[str, float]:
        """Per-benchmark retrieval-recall metrics, keyed by the column name they get
        in report.csv. Each benchmark decides the granularity that's meaningful for
        it (e.g. OfficeQA reports page- and document-level; BrowseComp-Plus reports
        gold- and evidence-document recall). Kept a pure function of the retrieved
        ids + the question (no resources) so the same logic can recompute metrics
        offline from an already-written report.csv. Default: one generic doc-recall."""
        return {"doc_recall": doc_recall(retrieved, question.gold_docs)}
