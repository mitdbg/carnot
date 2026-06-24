"""The `Benchmark` abstraction.

A Benchmark knows how to (1) load its questions, (2) build the retrieval resources
(a ChromaDB collection + a doc_id->text map) that systems run against, and (3)
score a predicted answer against gold. Subclasses customize each step; the runner
treats every benchmark uniformly.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

from skunk.prompted_call import PromptOverride, load_prompt_overrides

from qatfd.config import BenchmarkConfig
from qatfd.types import Question


class DocumentMap(Protocol):
    """The `doc_id -> full text` lookup the systems need (read_document / answer context). A real
    dict (OfficeQA / BrowseComp-Plus / FinanceBench) or a lazy chroma-backed mapping (TREC-BioGen,
    to avoid holding 26.8M abstracts in RAM) — the systems only do keyed lookups, never iterate it.

    Params are positional-only (`/`) so a plain `dict[str, str]` — whose `get`/`__getitem__` are
    positional-only in typeshed — structurally satisfies the protocol, same as the lazy mapping."""

    def get(self, doc_id: str, default: Any = None, /) -> Any: ...
    def __getitem__(self, doc_id: str, /) -> str: ...
    def __contains__(self, doc_id: object, /) -> bool: ...


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

    chroma_collection: Any  # chromadb Collection (vector index over chunks)
    document_map: DocumentMap  # doc_id -> full text (for read_document / answer step)
    config: BenchmarkConfig
    answer_format_hint: str = ""
    prompt_overrides: tuple[PromptOverride, ...] = field(default_factory=tuple)
    pdf_dir: str | None = None


class Benchmark(ABC):
    name: str  # registry key, e.g. "officeqa"

    # hint appended to the final-answer system prompt so the generated answer is in
    # the form the scorer expects. OfficeQA overrides to demand a terse numeric form.
    answer_format_hint: str = (
        "Answer concisely with just the factual answer (a short span or value), no explanation."
    )

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._resources: BenchmarkResources | None = None

    @abstractmethod
    def load_questions(self) -> list[Question]:
        """All questions in the benchmark (the runner applies the dev/test split)."""

    @abstractmethod
    def _build_resources(self) -> BenchmarkResources:
        """Open the chroma collection + build the document_map. Called once, cached
        by `get_resources`."""

    def get_resources(self) -> BenchmarkResources:
        """Cached accessor for the (collection, document_map) substrate."""
        if self._resources is None:
            res = self._build_resources()
            res.answer_format_hint = self.answer_format_hint
            path = self.config.prompts_path
            res.prompt_overrides = load_prompt_overrides(path) if path and os.path.exists(path) else ()
            res.pdf_dir = self.config.pdf_dir
            self._resources = res
        return self._resources

    @abstractmethod
    async def score(self, question: Question, predicted: str, ctx) -> dict:
        """Grade `predicted` against gold. Returns at least
        {"score": float in [0,1], "scorer": <name>}; may add {"judge_rationale": ...}.
        Binary benchmarks return 0.0/1.0; graded ones (nugget-completion) a fraction.
        Async so LLM-judge benchmarks can await; deterministic ones just return."""

    def test_qids(self) -> set[str]:
        """Held-out test split (excluded from the default `dev` split). Empty = no split."""
        return set()

    def recall_metrics(self, retrieved: list[str] | None, question: Question) -> dict[str, float]:
        """Per-benchmark retrieval-recall metrics, keyed by the column name they get
        in report.csv. Each benchmark decides the granularity that's meaningful for
        it (e.g. OfficeQA reports page- and document-level; BrowseComp-Plus reports
        gold- and evidence-document recall). Kept a pure function of the retrieved
        ids + the question (no resources) so the same logic can recompute metrics
        offline from an already-written report.csv. Default: one generic doc-recall."""
        return {"doc_recall": doc_recall(retrieved, question.gold_docs)}
