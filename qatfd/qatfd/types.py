"""Shared dataclasses passed between benchmarks, systems, and the runner."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Question:
    """One benchmark question. `qid` is the general per-question id (OfficeQA's
    `uid` / BrowseComp-Plus's `query_id` both map onto it)."""

    qid: str
    text: str
    gold: str
    gold_docs: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class Retrieved:
    """Output of a system's `retrieve()` step. `direct_answer` is set only by
    direct-answer agents (which produce the answer themselves); for retrieve-only
    systems it stays None and a downstream `compute()` generates the answer."""

    doc_ids: list[str]
    direct_answer: str | None = None
    context: str | None = None


@dataclass
class AnswerOutput:
    """What `System.answer()` returns. `retrieved_doc_ids` decouples systems from
    scoring (the benchmark computes doc-recall from it)."""

    answer: str
    retrieved_doc_ids: list[str] | None = None


@dataclass
class Result:
    """One row of report.csv. Superset of skunk eval_e2e's REPORT_FIELDS."""

    benchmark: str
    system: str
    qid: str
    question: str
    predicted: str
    gold: str
    score: float
    scorer: str
    recall_metrics: dict[str, float]
    retrieved_docs: str
    gold_docs: str
    failed: bool
    reason: str
    judge_rationale: str = ""
    wall_s: float = 0.0
    cost: float = 0.0
    total_cache_input_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    embed_tokens: int = 0
    embed_calls: int = 0
    embed_cost: float = 0.0


# Fixed CSV columns. The per-benchmark recall-metric columns are dynamic and get
# inserted just before `retrieved_docs` (i.e. right after `scorer`) at write time.
REPORT_FIELDS = [
    "benchmark",
    "system",
    "qid",
    "question",
    "predicted",
    "gold",
    "score",
    "scorer",
    "retrieved_docs",
    "gold_docs",
    "failed",
    "reason",
    "judge_rationale",
    "wall_s",
    "cost",
    "total_cache_input_tokens",
    "total_input_tokens",
    "total_output_tokens",
    "embed_tokens",
    "embed_calls",
    "embed_cost",
]


def report_columns(rows: list[Result]) -> list[str]:
    """Final CSV column order for a run: the fixed REPORT_FIELDS with each row's
    recall-metric keys inserted after `scorer` (union across rows, first-seen
    order — so a single-benchmark run gets that benchmark's metrics in order)."""
    metric_keys: list[str] = []
    for r in rows:
        for k in r.recall_metrics:
            if k not in metric_keys:
                metric_keys.append(k)
    insert_at = REPORT_FIELDS.index("retrieved_docs")
    return REPORT_FIELDS[:insert_at] + metric_keys + REPORT_FIELDS[insert_at:]


def result_to_row(r: Result) -> dict:
    """Flatten a Result into a CSV row dict, expanding `recall_metrics` into its
    own columns."""
    row = {k: v for k, v in r.__dict__.items() if k != "recall_metrics"}
    row.update(r.recall_metrics)
    return row
