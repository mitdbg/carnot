"""SearchAgent rollouts evaluated by document- and chunk-level retrieval metrics.

For each synthetic QA pair that survives de-duplication, we run ``n``
parallel ``SearchAgent`` rollouts ("trajectories") on the question. The
agent runs *exactly* in its test-time configuration: it commits to a set
of supporting ``doc_id``s via the default ``final_answer(page_keys=[...])``
tool (i.e. ``SearchAgent.retrieve``). This way the rollouts produced here
can later be used to train / evaluate the same agent without any
behaviour drift.

We score each rollout at two granularities:

  * Document level (the agent's native output):
      ``doc_output_recall``       -- |gt_d ∩ output_d| / |gt_d|
      ``doc_output_precision``    -- |gt_d ∩ output_d| / |output_d|
      ``doc_output_f1``           -- harmonic mean
      ``doc_trajectory_recall``   -- |gt_d ∩ trajectory_d| / |gt_d|

  * Chunk level (expanded from the doc-level output by taking *every*
    chunk that lives inside the selected docs):
      ``chunk_output_recall``     -- |gt_c ∩ output_c| / |gt_c|
      ``chunk_output_precision``  -- |gt_c ∩ output_c| / |output_c|
      ``chunk_output_f1``         -- harmonic mean
      ``chunk_trajectory_recall`` -- |gt_c ∩ trajectory_c| / |gt_c|

``trajectory_c`` is every ``chunk_id`` that appeared in any tool
observation across the rollout (so a ground-truth chunk counts as "seen"
even if its parent doc wasn't in the final answer); ``trajectory_d`` is
the parent-doc projection of the same set.

The score used for binarization is ``doc_output_recall``. Binarization +
filtering happen *globally* across all rollouts in a harness run (we need
the population mean), so this module exposes a per-pair "generate +
score" step and a separate "apply threshold + filter" step that operates
on a flat list of rollout records loaded back from disk.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from skunk.datagen.task_solver import (
    TaskSolverConfig,
    run_nugget_completion,
    run_task_solver,
)

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

    from skunk.datagen.harness import QAPair


# ---------------------------------------------------------------------------
# Config + record types
# ---------------------------------------------------------------------------

@dataclass
class RolloutConfig:
    """Static config shared by every rollout call within one harness run."""

    # SearchAgent construction args (mirror harness.generate_one)
    model_id: str
    emb_model_id: str
    document_map: dict[str, str]
    chroma_collection: Collection

    # SearchAgent runtime knobs
    max_steps: int
    max_pages_per_tool_call: int
    special_notes: str

    # Sampling parameters used for rollout generation (mirrors the
    # DEFAULT_ROLLOUT_SAMPLING_PARAMS constant from search_agent.py).  Stored
    # here so they can be copied into each RolloutRecord for full
    # reproducibility of the behaviour policy at train time.
    sampling_params: dict = field(default_factory=dict)

    # Shared Tinker generation backend used by every rollout SearchAgent in a
    # run (a `TinkerBackend`; typed loosely to avoid importing tinker at module
    # load). Rollouts sample directly from Tinker so the captured logprobs are
    # exactly pi_old for the model we fine-tune. Built once by the harness.
    tinker_backend: object | None = None

    # Input context window for the rollout model (tokens).  Used to set the
    # soft pruning warning and hard block thresholds in the agent. Default
    # mirrors the Tinker-enforced 64k max sequence length for the rollout model.
    model_context_window: int = 65_536

    # per-pair rollout parallelism (= "n" in the user spec; default 8)
    n_rollouts: int = 8
    rollout_parallelism: int = 8

    # corpus id maps. ``doc_id_to_chunk_ids`` expands the agent's doc-level
    # final answer into the implied chunk-level set (every chunk that lives
    # inside a selected doc) for the chunk-level metrics.
    # ``chunk_id_to_doc_id`` projects the chunk-level trajectory back onto
    # docs for ``doc_trajectory_recall``. Unknown ids contribute nothing.
    doc_id_to_chunk_ids: dict[str, list[str]] = field(default_factory=dict)
    chunk_id_to_doc_id: dict[str, str] = field(default_factory=dict)

    # SearchAgent factory injected by harness so this module doesn't
    # take a hard dependency on the search package layout. Must return:
    #   (output_doc_ids, trajectory_chunk_ids,
    #    completed, num_steps, error)
    agent_factory: Callable | None = None

    # ---- TaskSolver + NuggetCompletion post-processing -----------------
    # When set, every rollout's _one() additionally runs the TaskSolver
    # (no-tool LLM) on the rollout's output_doc_ids, then the nugget-
    # completion judge against the gold nuggets. Results are written to
    # ``RolloutRecord.task_solver_answer`` / ``eval_score`` / ``eval_labels``.
    task_solver_cfg: TaskSolverConfig | None = None

    # Optional override for the gold nuggets used by nugget completion.
    # If None, ``pair.answer`` (the synthesised QA pair's nugget list) is used.
    # Kept open so we can plug in human-curated nuggets later.
    nuggets_for_pair: Callable[[QAPair], list[str]] | None = None


@dataclass
class RolloutRecord:
    """One trajectory of the SearchAgent on one synthetic question.

    Persisted per-seed in ``{seed}_rollouts.json`` after phase 1. Phase 2
    fills in ``binarized_score``, ``kept``, and ``discard_reason`` and
    writes ``{seed}_rollouts_filtered.json``.
    """

    qa_id: str
    rollout_idx: int

    # raw agent output. The agent commits to ``output_doc_ids``; we expand
    # to ``output_chunk_ids`` by enumerating every chunk that lives inside
    # those docs (using cfg.doc_id_to_chunk_ids). The trajectory is recorded
    # in its native chunk form from the message log and projected to docs.
    output_doc_ids: list[str]
    output_chunk_ids: list[str]
    trajectory_chunk_ids: list[str]
    trajectory_doc_ids: list[str]

    completed: bool
    num_steps: int
    error: str | None
    total_time_sec: float

    # paths to large per-rollout artefacts (kept out of the JSON record itself)
    messages_path: str
    trace_path: str

    # model that generated this rollout and the exact sampling params it used.
    # Together these fully specify the behaviour policy pi_old(a | s) needed
    # for the importance-ratio denominator in GRPO / CISPO.
    rollout_model_id: str = ""
    sampling_params: dict = field(default_factory=dict)

    # document-level metrics. ``doc_output_recall`` is the score used for
    # binarization in phase 2.
    doc_output_recall: float | None = None
    doc_output_precision: float | None = None
    doc_output_f1: float | None = None
    doc_trajectory_recall: float | None = None

    # chunk-level metrics, derived from the doc-level final answer by
    # expanding each selected doc into its full chunk set.
    chunk_output_recall: float | None = None
    chunk_output_precision: float | None = None
    chunk_output_f1: float | None = None
    chunk_trajectory_recall: float | None = None

    # raw counts -- kept for downstream sanity checks / future metric tweaks.
    n_gt_chunks: int = 0
    n_gt_docs: int = 0
    n_output_chunks: int = 0
    n_output_docs: int = 0
    n_trajectory_chunks: int = 0
    n_trajectory_docs: int = 0
    n_chunk_output_hits: int = 0
    n_doc_output_hits: int = 0
    n_chunk_trajectory_hits: int = 0
    n_doc_trajectory_hits: int = 0

    # filled in by phase 2 (apply_threshold_and_filter):
    binarized_score: int | None = None      # 0 or 1
    kept: bool | None = None
    discard_reason: str | None = None       # "all_pass" | "all_fail" | None

    # ---- TaskSolver + NuggetCompletion outputs --------------------------
    # ``task_solver_answer`` is the TaskSolver agent's answer derived from the
    # rollout's output_doc_ids. ``eval_score`` is the nugget-completion
    # judge's score in [0, 1] (strict or non-strict depending on cfg).
    # ``eval_labels`` is the per-nugget judgement list aligned with the
    # gold nuggets.
    task_solver_answer: str | None = None
    task_solver_error: str | None = None
    eval_score: float | None = None
    eval_labels: list[str] | None = None
    eval_error: str | None = None


# ---------------------------------------------------------------------------
# Phase 1: per-pair rollouts + scoring
# ---------------------------------------------------------------------------

def run_single_rollout(
    seed: int,
    pair_idx: int,
    rollout_idx: int,
    pair: QAPair,
    cfg: RolloutConfig,
    trace_dir: str,
    show_output: bool = False,
) -> RolloutRecord:
    """Run and score ONE SearchAgent trajectory on ``pair``.

    This is the atomic unit of rollout work -- a single ``(pair, rollout_idx)``
    trajectory. The harness submits these directly to one shared executor so a
    fast rollout finishing anywhere immediately frees a slot for the next one,
    keeping the Tinker sampling backend saturated across pairs and seeds rather
    than draining to 1-2 in-flight calls at every per-pair barrier.

    The rollout is scored at doc and chunk granularity against ``pair.doc_ids``
    / ``pair.chunk_ids``.  ``cfg.agent_factory(question, trace_path,
    messages_path, show_output)`` must return ``(output_doc_ids,
    trajectory_chunk_ids, completed, num_steps, error)``; the chunk-level
    "output" set is reconstructed here by expanding each output doc_id through
    ``cfg.doc_id_to_chunk_ids``.
    """
    if cfg.agent_factory is None:
        raise ValueError("RolloutConfig.agent_factory must be set by the harness")

    gt_chunks = set(pair.chunk_ids)
    gt_docs = set(pair.doc_ids)

    start = time.perf_counter()
    trace_path = os.path.join(trace_dir, f"{seed}_qa{pair_idx}_r{rollout_idx}_trace.txt")
    messages_path = os.path.join(trace_dir, f"{seed}_qa{pair_idx}_r{rollout_idx}_messages.json")

    output_docs, trajectory_chunks, completed, num_steps, error = cfg.agent_factory(
        question=pair.question,
        trace_path=trace_path,
        messages_path=messages_path,
        show_output=show_output,
    ) # type: ignore

    output_docs = list(output_docs)
    trajectory_chunks = sorted(set(trajectory_chunks))
    output_chunks = _expand_docs_to_chunks(output_docs, cfg.doc_id_to_chunk_ids)
    trajectory_docs = sorted({
        cfg.chunk_id_to_doc_id[c]
        for c in trajectory_chunks
        if c in cfg.chunk_id_to_doc_id
    })

    rec = RolloutRecord(
        qa_id=pair.qa_id,
        rollout_idx=rollout_idx,
        output_doc_ids=output_docs,
        output_chunk_ids=output_chunks,
        trajectory_chunk_ids=trajectory_chunks,
        trajectory_doc_ids=trajectory_docs,
        completed=completed,
        num_steps=num_steps,
        error=error,
        total_time_sec=time.perf_counter() - start,
        rollout_model_id=cfg.model_id,
        sampling_params=cfg.sampling_params,
        messages_path=messages_path,
        trace_path=trace_path,
    )
    _score_record(rec, gt_chunks, gt_docs)

    # TaskSolver + NuggetCompletion post-passes (optional).
    if cfg.task_solver_cfg is not None:
        nuggets = (
            cfg.nuggets_for_pair(pair) if cfg.nuggets_for_pair is not None
            else list(pair.answer)
        )
        answer, ts_err = run_task_solver(
            question=pair.question,
            output_doc_ids=output_docs,
            document_map=cfg.document_map,
            cfg=cfg.task_solver_cfg,
        )
        rec.task_solver_answer = answer
        rec.task_solver_error = ts_err
        if ts_err is None:
            score, labels, ev_err = run_nugget_completion(
                question=pair.question,
                candidate_answer=answer,
                nuggets=nuggets,
                cfg=cfg.task_solver_cfg,
            )
            rec.eval_score = score
            rec.eval_labels = labels
            rec.eval_error = ev_err
        else:
            # solver failed -- skip evaluator; surface the error.
            rec.eval_error = "skipped: task_solver failed"
    return rec


def run_rollouts_for_pair(
    seed: int,
    pair_idx: int,
    pair: QAPair,
    cfg: RolloutConfig,
    trace_dir: str,
    show_output: bool = False,
) -> list[RolloutRecord]:
    """Run ``cfg.n_rollouts`` parallel SearchAgent trajectories on ``pair``.

    Convenience wrapper around :func:`run_single_rollout` for standalone use
    (the harness instead submits individual rollouts to one shared executor so
    there is no per-pair barrier -- see ``_run_rollouts_for_seed``).  Returns a
    list of records ordered by ``rollout_idx``.
    """
    with ThreadPoolExecutor(max_workers=max(1, cfg.rollout_parallelism)) as pool:
        records = list(pool.map(
            lambda i: run_single_rollout(
                seed, pair_idx, i, pair, cfg, trace_dir, show_output,
            ),
            range(cfg.n_rollouts),
        ))

    records.sort(key=lambda r: r.rollout_idx)
    return records


def persist_seed_rollouts(seed: int, records: list[RolloutRecord], path: str) -> None:
    """Write all rollout records for one seed to ``{seed}_rollouts.json``."""
    with open(path, "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)


# ---------------------------------------------------------------------------
# Phase 2: global binarization + all-pass/all-fail filtering
# ---------------------------------------------------------------------------

@dataclass
class FilterStats:
    """Funnel-level statistics for one harness run's rollout filter pass.

    ``threshold`` is the global mean of ``doc_output_recall`` across every
    rollout with a non-None score; rollouts with
    ``doc_output_recall > threshold`` binarize to 1.
    """

    threshold: float
    n_rollouts: int
    n_scored: int
    n_pairs: int
    n_pairs_all_pass: int
    n_pairs_all_fail: int
    n_pairs_kept: int


BinarizationMode = Literal["doc-recall", "eval-score"]


def _score_field_for_mode(mode: BinarizationMode) -> str:
    if mode == "doc-recall":
        return "doc_output_recall"
    if mode == "eval-score":
        return "eval_score"
    raise ValueError(f"unknown binarization_mode: {mode!r}")


def apply_threshold_and_filter(
    records_by_seed: dict[int, list[RolloutRecord]],
    binarization_mode: BinarizationMode = "doc-recall",
) -> tuple[dict[int, list[RolloutRecord]], FilterStats]:
    """Binarize every rollout against the global mean of the chosen score
    field and apply the all-pass / all-fail filter.

    Mutates ``records_by_seed`` in place (also returned).

    ``binarization_mode`` selects the per-rollout score:
      - ``"doc-recall"`` -> ``RolloutRecord.doc_output_recall``
      - ``"eval-score"`` -> ``RolloutRecord.eval_score`` (TaskSolver +
        NuggetCompletion judge; requires ``RolloutConfig.task_solver_cfg``
        to have been set when the rollouts were produced).

    Rollouts whose score is None (e.g. the pair has no ground-truth docs,
    or the eval judge failed) are excluded from the mean; they get
    ``binarized_score = None`` and are treated as neither passing nor
    failing for the all-pass / all-fail filter (we conservatively keep
    the pair unless every *scored* rollout is on one side).

    Note: every call recomputes from scratch, so a pair that was discarded
    on an earlier call (because the then-current threshold made every
    rollout pass or fail) can be re-promoted to ``kept=True`` on a later
    call once new rollouts have shifted the global mean.
    """
    score_field = _score_field_for_mode(binarization_mode)
    flat: list[RolloutRecord] = [r for recs in records_by_seed.values() for r in recs]
    scored = [r for r in flat if getattr(r, score_field) is not None]
    if not scored:
        stats = FilterStats(0.0, len(flat), 0, 0, 0, 0, 0)
        return records_by_seed, stats

    threshold = sum(getattr(r, score_field) for r in scored) / len(scored)

    for r in flat:
        s = getattr(r, score_field)
        if s is None:
            r.binarized_score = None
        else:
            r.binarized_score = int(s > threshold)

    by_pair: dict[str, list[RolloutRecord]] = {}
    for r in flat:
        by_pair.setdefault(r.qa_id, []).append(r)

    n_all_pass = n_all_fail = n_kept = 0
    for _qa_id, rs in by_pair.items():
        bins = [r.binarized_score for r in rs if r.binarized_score is not None]
        all_scored = len(bins) == len(rs) and len(bins) > 0
        if all_scored and all(b == 1 for b in bins):
            reason, kept = "all_pass", False
            n_all_pass += 1
        elif all_scored and all(b == 0 for b in bins):
            reason, kept = "all_fail", False
            n_all_fail += 1
        else:
            reason, kept = None, True
            n_kept += 1
        for r in rs:
            r.kept = kept
            r.discard_reason = reason

    stats = FilterStats(
        threshold=threshold,
        n_rollouts=len(flat),
        n_scored=len(scored),
        n_pairs=len(by_pair),
        n_pairs_all_pass=n_all_pass,
        n_pairs_all_fail=n_all_fail,
        n_pairs_kept=n_kept,
    )
    return records_by_seed, stats


def persist_filtered_seed(
    seed: int,
    records: list[RolloutRecord],
    stats: FilterStats,
    path: str,
) -> None:
    """Write ``{seed}_rollouts_filtered.json`` for one seed.

    The file header repeats the global stats so the funnel can be replayed
    standalone; ``pairs`` is a compact per-question summary suitable for
    quick eyeballing, while ``rollouts`` is the full record list (with all
    doc- and chunk-level metrics intact for downstream analyses).
    """
    by_pair: dict[str, list[RolloutRecord]] = {}
    for r in records:
        by_pair.setdefault(r.qa_id, []).append(r)

    pair_summaries = []
    for qa_id, rs in by_pair.items():
        pair_summaries.append({
            "qa_id": qa_id,
            "doc_output_recall": [r.doc_output_recall for r in rs],
            "doc_output_precision": [r.doc_output_precision for r in rs],
            "doc_output_f1": [r.doc_output_f1 for r in rs],
            "doc_trajectory_recall": [r.doc_trajectory_recall for r in rs],
            "chunk_output_recall": [r.chunk_output_recall for r in rs],
            "chunk_output_precision": [r.chunk_output_precision for r in rs],
            "chunk_output_f1": [r.chunk_output_f1 for r in rs],
            "chunk_trajectory_recall": [r.chunk_trajectory_recall for r in rs],
            "binarized_scores": [r.binarized_score for r in rs],
            "kept": rs[0].kept,
            "discard_reason": rs[0].discard_reason,
        })

    payload = {
        "seed": seed,
        "global_stats": asdict(stats),
        "pairs": pair_summaries,
        "rollouts": [asdict(r) for r in records],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_record(
    rec: RolloutRecord,
    gt_chunks: set[str],
    gt_docs: set[str],
) -> None:
    """Fill in all eight retrieval metrics on ``rec``."""
    chunk_output = set(rec.output_chunk_ids)
    chunk_trajectory = set(rec.trajectory_chunk_ids)
    doc_output = set(rec.output_doc_ids)
    doc_trajectory = set(rec.trajectory_doc_ids)

    rec.n_gt_chunks = len(gt_chunks)
    rec.n_gt_docs = len(gt_docs)
    rec.n_output_chunks = len(chunk_output)
    rec.n_output_docs = len(doc_output)
    rec.n_trajectory_chunks = len(chunk_trajectory)
    rec.n_trajectory_docs = len(doc_trajectory)
    rec.n_chunk_output_hits = len(gt_chunks & chunk_output)
    rec.n_doc_output_hits = len(gt_docs & doc_output)
    rec.n_chunk_trajectory_hits = len(gt_chunks & chunk_trajectory)
    rec.n_doc_trajectory_hits = len(gt_docs & doc_trajectory)

    rec.doc_output_recall = _safe_div(rec.n_doc_output_hits, rec.n_gt_docs)
    rec.doc_output_precision = _safe_div(rec.n_doc_output_hits, rec.n_output_docs)
    rec.doc_output_f1 = _f1(rec.doc_output_precision, rec.doc_output_recall)
    rec.doc_trajectory_recall = _safe_div(rec.n_doc_trajectory_hits, rec.n_gt_docs)

    rec.chunk_output_recall = _safe_div(rec.n_chunk_output_hits, rec.n_gt_chunks)
    rec.chunk_output_precision = _safe_div(rec.n_chunk_output_hits, rec.n_output_chunks)
    rec.chunk_output_f1 = _f1(rec.chunk_output_precision, rec.chunk_output_recall)
    rec.chunk_trajectory_recall = _safe_div(rec.n_chunk_trajectory_hits, rec.n_gt_chunks)


def _safe_div(num: int, denom: int) -> float | None:
    """Return num/denom, or None when the denominator is zero."""
    if denom == 0:
        return None
    return num / denom


def _f1(precision: float | None, recall: float | None) -> float | None:
    """Harmonic mean of precision + recall, or None if either is undefined."""
    if precision is None or recall is None:
        return None
    if (precision + recall) == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _expand_docs_to_chunks(
    doc_ids: list[str] | set[str],
    doc_id_to_chunk_ids: dict[str, list[str]],
) -> list[str]:
    """Return every chunk_id that lives inside any of ``doc_ids``.

    Unknown doc_ids contribute zero chunks. Returned sorted for stable
    on-disk ordering.
    """
    chunks: set[str] = set()
    for did in doc_ids:
        for cid in doc_id_to_chunk_ids.get(did, ()):
            chunks.add(cid)
    return sorted(chunks)


def extract_trajectory_chunk_ids(messages_jsonable: list[dict]) -> list[str]:
    """Collect every ``chunk_id`` that surfaced in any non-system message.

    Walks the ``messages_to_jsonable()`` output: every block with
    ``type == "chunk"`` and a non-empty ``chunk_id`` counts as one
    observation of that chunk. Returns the unique chunk_ids sorted for
    stable ordering on disk.

    This is the "trajectory" used by ``chunk_trajectory_recall`` (and,
    via the doc projection, ``doc_trajectory_recall``): a ground-truth
    chunk is considered "seen" if the agent surfaced it at least once
    during retrieval, regardless of whether its parent doc made it into
    the final answer.
    """
    seen: set[str] = set()
    for msg in messages_jsonable:
        if msg.get("role") == "system":
            continue
        for block in msg.get("blocks", []):
            if block.get("type") != "chunk":
                continue
            cid = block.get("chunk_id")
            if cid:
                seen.add(str(cid))
    return sorted(seen)
