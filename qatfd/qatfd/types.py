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
    """Output of a system's `retrieve()` step.

    `terminate_state` is the retrieval agent's `MultiTurnAgent._terminate_state` (why its
    loop stopped: "finished" or a `|`-joined subset of out_of_steps/over_cost_budget/
    over_latency_budget); None for systems whose retrieve runs no such agent (e.g. rag_llm)."""

    doc_ids: list[str]
    context: str | None = None
    terminate_state: str | None = None


@dataclass
class AnswerOutput:
    """What `System.answer()` returns. `retrieved_doc_ids` decouples systems from
    scoring (the benchmark computes doc-recall from it).

    `terminate_state` surfaces the retrieval agent's stop reason to the runner (see
    `Retrieved.terminate_state`), which parses it into the out_of_steps/over_cost_budget/
    over_latency_budget columns. `retrieve_wall_s`/`compute_wall_s` split the answer wall time
    across the two phases (the runner also records the end-to-end `wall_s`).
    
    `session_id` which is a provider-side id for systems whose spend is metered externally.
    In particular, the CodexSystem forwards its thread id to OpenRouter as `session_id`, so
    the runner can query this question's cost/tokens from the analytics API; subagent traffic
    is billed under the parent's id).
    """

    answer: str
    retrieved_doc_ids: list[str] | None = None
    terminate_state: str | None = None
    retrieve_wall_s: float = 0.0
    compute_wall_s: float = 0.0
    session_id: str | None = None


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
    # end-to-end answer() wall time, then its retrieve()/compute() split (see AnswerOutput).
    wall_s: float = 0.0
    retrieve_wall_s: float = 0.0
    compute_wall_s: float = 0.0
    # all-in cost across every caller this question (generation + embeddings), then the system's
    # own spend broken out by phase. Lets a run compare a retrieval method's cost against compute and total.
    cost: float = 0.0
    retrieve_cost: float = 0.0
    compute_cost: float = 0.0
    total_cache_input_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    embed_tokens: int = 0
    embed_calls: int = 0
    embed_cost: float = 0.0
    # why the retrieval agent's loop stopped (parsed from terminate_state); all False when it
    # finished normally or the system runs no budgeted agent.
    out_of_steps: bool = False
    over_cost_budget: bool = False
    over_latency_budget: bool = False


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
    "retrieve_wall_s",
    "compute_wall_s",
    "cost",
    "retrieve_cost",
    "compute_cost",
    "total_cache_input_tokens",
    "total_input_tokens",
    "total_output_tokens",
    "embed_tokens",
    "embed_calls",
    "embed_cost",
    "out_of_steps",
    "over_cost_budget",
    "over_latency_budget",
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
