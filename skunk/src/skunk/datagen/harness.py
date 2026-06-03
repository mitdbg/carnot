import argparse
import json
import math
import os
import pathlib
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass

import chromadb
import numpy as np
import pandas as pd
import yaml
from chromadb.api.models.Collection import Collection
from datasets import load_dataset
from jinja2 import Template
from openrouter import OpenRouter

from skunk.datagen.dedup import DedupConfig, dedup_batch
from skunk.datagen.quality_filter import (
    BENCHMARK_QUALITY_FILTER_SYSTEM_PROMPT,
    QualityFilterConfig,
    QualityFilterResult,
    load_quality_funnel,
    persist_quality_funnel,
    run_quality_filter_for_pair,
)
from skunk.datagen.rollout import (
    RolloutConfig,
    RolloutRecord,
    apply_threshold_and_filter,
    extract_trajectory_chunk_ids,
    persist_filtered_seed,
    persist_seed_rollouts,
    run_single_rollout,
)
from skunk.datagen.task_solver import TaskSolverConfig
from skunk.datagen.tinker_cost import resolve_prices, write_run_report
from skunk.logging.tracer import Tracer
from skunk.retrieve.search_agent import (
    BROWSECOMP_PLUS_SPECIAL_NOTES,
    DEFAULT_ROLLOUT_SAMPLING_PARAMS,
    MAX_PAGES_PER_TOOL_CALL,
    MAX_PARALLEL_TOOL_CALLS,
    OFFICEQA_SPECIAL_NOTES,
    SearchAgent,
)
from skunk.retrieve.search_tools import (
    datagen_final_answer,
)
from skunk.retrieve.tinker_backend import (
    DEFAULT_ROLLOUT_MAX_TOKENS,
    DEFAULT_TINKER_BASE_MODEL,
    DEFAULT_TINKER_RENDERER,
    build_tinker_backend,
)

TRACE_DIR = "qa_synthesis_traces"
VAL_FRAC = 0.25
N_SEED_CHUNKS = 10
AGENT_MAX_STEPS = 100
DEFAULT_N_QA_PAIRS = 8
DEFAULT_N_ROLLOUTS = 8
# Size of the single shared rollout pool: total SearchAgent rollouts in flight
# across every pair and seed at once. All rollouts are submitted to this one
# executor (no per-pair barrier), so a finished rollout immediately frees a slot
# for the next, keeping the Tinker sampling backend saturated end-to-end.
DEFAULT_ROLLOUT_CONCURRENCY = 32
# Rollout policy. Rollouts sample directly from Tinker via `tinker_backend`
# (see skunk.retrieve.tinker_backend), so the per-token logprobs captured here
# are exactly pi_old -- the behaviour-policy denominator in the GRPO / CISPO
# importance ratio -- for the model we fine-tune. Sampling from Tinker (rather
# than an fp8 OpenRouter build behind a single pinned provider) removes both the
# single-provider failure mode and the train/inference precision mismatch.
# Tinker enforces a 64k max sequence length for this model even though its
# native context window is far larger (262k). The default base model / renderer
# live in tinker_backend (DEFAULT_TINKER_BASE_MODEL / DEFAULT_TINKER_RENDERER).
DEFAULT_ROLLOUT_CONTEXT_WINDOW = 65_536  # Tinker-enforced 64k max sequence length
OFFICEQA_QUESTIONS_PATH = "officeqa_pro.csv"
OFFICEQA_DATA_PATH = "treasury_bulletins_cleaned/"
BROWSECOMP_PLUS_QUESTIONS_PATH = "browsecomp-plus/browsecomp_plus_decrypted.jsonl"

DATAGEN_PROMPT_FILE = pathlib.Path(__file__).parent / "prompts.yaml"
with DATAGEN_PROMPT_FILE.open() as _f:
    _DATAGEN_PROMPTS = yaml.safe_load(_f)

DATAGEN_SYSTEM_PROMPT: str = _DATAGEN_PROMPTS["datagen_system_prompt"]
OFFICEQA_DATAGEN_GUIDANCE: str = _DATAGEN_PROMPTS["officeqa_datagen_guidance"]

# map each benchmark to the SearchAgent `special_notes` block describing its
# corpus (identifier formats, available metadata fields, example filter
# clauses). Used both when synthesising QA pairs (`generate_one`) and when
# running rollouts so the agent always sees corpus notes that match the
# benchmark it's operating on.
BENCHMARK_SPECIAL_NOTES: dict[str, str] = {
    "officeqa":        OFFICEQA_SPECIAL_NOTES,
    "browsecomp-plus": BROWSECOMP_PLUS_SPECIAL_NOTES,
}

# map each benchmark to its de-duplication judge prompt template.
BENCHMARK_DEDUP_JUDGE_PROMPT: dict[str, str] = {
    "officeqa":        _DATAGEN_PROMPTS["dedup_judge_prompt_officeqa"],
    "browsecomp-plus": _DATAGEN_PROMPTS["dedup_judge_prompt_browsecomp_plus"],
}

# map each benchmark to its dataset-specific question-style / computation guidance.
# Benchmarks not listed here get an empty string (no extra guidance).
BENCHMARK_DATASET_GUIDANCE: dict[str, str] = {
    "officeqa": OFFICEQA_DATAGEN_GUIDANCE,
}

# per-benchmark statistics for the number of relevant chunks per question.
# Values are (mean, std) drawn from Table 2 of the SkunkWorks eval paper.
# Used to sample a target chunk count from N(mean, std) for each generation.
BENCHMARK_CHUNKS_STATS: dict[str, tuple[float, float]] = {
    "officeqa":        (1.8,  1.1),
    "browsecomp-plus": (2.9,  2.0),
}

# ChromaDB's SQLite backend hits a "too many SQL variables" error when fetching
# the entire collection at once. Page through it in chunks of this size instead.
CHROMA_PAGE_SIZE = 5000

# Imports the datagen agent is allowed to use inside its python tool blocks,
# so it can compute statistics (means, regressions, etc.) from raw values it
# retrieves from the corpus rather than just regurgitating pre-computed numbers.
DATAGEN_AUTHORIZED_IMPORTS = ["math", "statistics", "numpy", "scipy", "statsmodels"]


@dataclass
class QAPair:
    qa_id: str
    question: str
    answer: list[str]
    chunk_ids: list[str]
    doc_ids: list[str]


@dataclass
class GenerationResult:
    seed: int
    pairs: list[QAPair]
    target_n_qa_pairs: int
    target_n_chunks_per_pair: int
    completed: bool
    num_steps: int
    error: str | None
    total_time_sec: float


def load_chunk_id_to_doc_id(collection: Collection) -> dict[str, str]:
    """Return a mapping from every chunk_id in `collection` to its `doc_id`.

    ChromaDB's SQLite backend raises "too many SQL variables" when the entire
    collection is fetched in a single call, so we page through it in batches
    of ``CHROMA_PAGE_SIZE`` rows.
    """
    chunk_id_to_doc_id: dict[str, str] = {}
    offset = 0
    while True:
        page = collection.get(include=["metadatas"], limit=CHROMA_PAGE_SIZE, offset=offset)
        page_ids = page["ids"]
        if not page_ids:
            break
        for cid, meta in zip(page_ids, page["metadatas"] or [], strict=True):
            chunk_id_to_doc_id[cid] = str((meta or {}).get("doc_id", cid))
        offset += len(page_ids)
        if len(page_ids) < CHROMA_PAGE_SIZE:
            break
    return chunk_id_to_doc_id


def load_benchmark_questions(benchmark: str) -> list[tuple[str, str]]:
    """Load benchmark question-answer pairs and return them as a list of tuples."""
    if benchmark == "officeqa":
        officeqa_df = pd.read_csv(OFFICEQA_QUESTIONS_PATH)
        val_df = officeqa_df.iloc[:int(len(officeqa_df) * VAL_FRAC)]
        return list(zip(val_df["question"].tolist(), val_df["answer"].tolist(), strict=True))

    elif benchmark == "browsecomp-plus":
        with open(BROWSECOMP_PLUS_QUESTIONS_PATH) as f:
            browsecomp = [json.loads(line) for line in f]
            val_browsecomp = browsecomp[:int(len(browsecomp) * VAL_FRAC)]
            return [(item["query"], item["answer"]) for item in val_browsecomp]

    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")


def load_document_map(benchmark: str) -> dict[str, str]:
    """Load the mapping from document ID to document text."""
    if benchmark == "officeqa":
        with open(f"{OFFICEQA_DATA_PATH}/clean_page_map.json") as f:
            clean_page_map = json.load(f)

        document_map: dict[str, str] = {}
        for doc_id, entry in clean_page_map.items():
            rel_path = entry[0]
            filepath = os.path.join(OFFICEQA_DATA_PATH, os.path.basename(rel_path))
            with open(filepath) as f:
                document_map[doc_id] = f.read()

        return document_map

    elif benchmark == "browsecomp-plus":
        ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
        return {item["docid"]: item["text"] for item in ds}  # type: ignore

    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")


def _sample_n_chunks(rng: random.Random, mean: float, std: float) -> int:
    """Sample a target chunk count from N(mean, std), clamped to at least 1.

    Uses the Box-Muller transform so we only need the stdlib `random` module
    (no numpy dependency required here).
    """
    sample = rng.gauss(mean, std)
    return max(1, math.ceil(sample))


def _sample_seed_chunks(
    rng: random.Random,
    chroma_collection: Collection,
    chunk_id_to_doc_id: dict[str, str],
    n: int,
) -> list[tuple[str, str, str]]:
    """Sample `n` random chunks from the corpus.

    Returns a list of `(chunk_id, doc_id, document_text)` triples.
    """
    sampled_ids = rng.sample(list(chunk_id_to_doc_id.keys()), min(n, len(chunk_id_to_doc_id)))
    result = chroma_collection.get(ids=sampled_ids, include=["documents"])
    ids_out = result.get("ids") or []
    docs_out = result.get("documents") or []
    return [
        (cid, chunk_id_to_doc_id[cid], text or "")
        for cid, text in zip(ids_out, docs_out, strict=True)
    ]


def _build_user_prompt(
    seed_chunks: list[tuple[str, str, str]],
    examples: list[tuple[str, str]],
    n_qa_pairs: int,
    n_chunks: int,
) -> str:
    """Construct the initial user message that seeds the generation task."""
    lines: list[str] = []

    lines.append(
        f"Here are {len(seed_chunks)} randomly sampled chunks from the corpus "
        f"to give you a sense of its breadth. Use them to inspire diverse "
        f"question topics, but you are NOT required to ground your final "
        f"questions in these particular chunks.\n"
    )
    for i, (cid, did, text) in enumerate(seed_chunks, 1):
        lines.append(f"Seed chunk {i} (chunk_id={cid}, doc_id={did}):\n{text}\n")

    lines.append(
        f"\nHere are {len(examples)} example question-answer pairs from a "
        f"benchmark built on this corpus, illustrating the target style and "
        f"difficulty:\n"
    )
    for i, (q, a) in enumerate(examples, 1):
        lines.append(f"Example {i}:\n  Q: {q}\n  A: {a}\n")

    lines.append(
        f"\nNow explore the corpus to synthesise {n_qa_pairs} new, diverse "
        f"question-answer pairs. Each answer must be a list of nuggets (key "
        f"facts essential to the answer), and each pair should be grounded "
        f"in roughly {n_chunks} supporting chunk(s). Call `final_answer(...)` "
        f"exactly once with all {n_qa_pairs} pairs."
    )
    return "\n".join(lines)


def generate_one(
    seed: int,
    model_id: str,
    qa_pairs: list[tuple[str, str]],
    document_map: dict[str, str],
    chroma_collection: Collection,
    chunk_id_to_doc_id: dict[str, str],
    emb_model_id: str,
    show_output: bool,
    trace_dir: str,
    special_notes: str = OFFICEQA_SPECIAL_NOTES,
    dataset_guidance: str = "",
    n_examples: int = 5,
    n_qa_pairs: int = DEFAULT_N_QA_PAIRS,
    n_seed_chunks: int = N_SEED_CHUNKS,
    chunks_mean: float = 3.0,
    chunks_std: float = 2.0,
    service_tier: str | None = None,
) -> GenerationResult:
    """Generate ``n_qa_pairs`` question-answer pairs in a single agent run.

    The target number of supporting chunks per pair is sampled from
    ``N(chunks_mean, chunks_std)`` (clamped to >= 1). Ten random seed chunks
    are included in the initial user message to spur topical diversity.
    """
    start = time.perf_counter()
    rng = random.Random(seed)
    examples = rng.sample(qa_pairs, min(n_examples, len(qa_pairs)))
    n_chunks = _sample_n_chunks(rng, chunks_mean, chunks_std)
    seed_chunks = _sample_seed_chunks(rng, chroma_collection, chunk_id_to_doc_id, n_seed_chunks)

    system_prompt = Template(DATAGEN_SYSTEM_PROMPT).render(
        max_steps=AGENT_MAX_STEPS,
        max_pages=MAX_PAGES_PER_TOOL_CALL,
        max_parallel_tool_calls=MAX_PARALLEL_TOOL_CALLS,
        special_notes=special_notes,
        dataset_guidance=dataset_guidance,
        n_chunks=n_chunks,
        n_qa_pairs=n_qa_pairs,
    )
    user_prompt = _build_user_prompt(seed_chunks, examples, n_qa_pairs, n_chunks)

    os.makedirs(trace_dir, exist_ok=True)
    trace_path = f"{trace_dir}/{seed}_trace.txt"

    with Tracer(trace_path, show_output=show_output) as tracer:
        agent = SearchAgent(
            model_id,
            document_map=document_map,
            chroma_collection=chroma_collection,
            emb_model_id=emb_model_id,
            tracer=tracer,
            max_steps=AGENT_MAX_STEPS,
            max_pages_per_tool_call=MAX_PAGES_PER_TOOL_CALL,
            system_prompt_override=system_prompt,
            final_answer_fn=datagen_final_answer,
            additional_authorized_imports=DATAGEN_AUTHORIZED_IMPORTS,
            service_tier=service_tier,
        )
        raw_pairs = agent.qa_synthesis(user_prompt)

    # persist the full message trajectory alongside the trace.
    messages_path = f"{trace_dir}/{seed}_messages.json"
    with open(messages_path, "w") as f:
        json.dump(agent.messages_to_jsonable(), f, indent=2)

    pairs: list[QAPair] = []
    for i, p in enumerate(raw_pairs):
        chunk_ids = [str(c) for c in p.get("chunk_ids", [])]
        doc_ids = sorted({chunk_id_to_doc_id[c] for c in chunk_ids if c in chunk_id_to_doc_id})
        pairs.append(QAPair(
            qa_id=f"{seed}-{i}",
            question=str(p.get("question", "")),
            answer=[str(n) for n in p.get("answer", [])],
            chunk_ids=chunk_ids,
            doc_ids=doc_ids,
        ))

    # persist the QA pairs for this seed alongside the trace and messages.
    qa_pairs_path = f"{trace_dir}/{seed}_qa_pairs.json"
    with open(qa_pairs_path, "w") as f:
        json.dump([asdict(p) for p in pairs], f, indent=2)

    return GenerationResult(
        seed=seed,
        pairs=pairs,
        target_n_qa_pairs=n_qa_pairs,
        target_n_chunks_per_pair=n_chunks,
        completed=agent._completed,
        num_steps=agent._num_steps,
        error=agent._error,
        total_time_sec=time.perf_counter() - start,
    )


def _run_dedup_for_seed(
    seed: int,
    pairs: list[QAPair],
    dedup_cfg: DedupConfig,
    dedup_path: str,
) -> list[QAPair]:
    """Run dedup for one seed; persist funnel JSON. Return kept pairs."""
    kept, funnel = dedup_batch(pairs, dedup_cfg)
    with open(dedup_path, "w") as f:
        json.dump(
            {
                "seed": seed,
                "n_input": len(pairs),
                "n_kept": len(kept),
                "kept_qa_ids": [p.qa_id for p in kept],
                "funnel": funnel,
            },
            f,
            indent=2,
        )
    return kept


def _load_kept_pairs(qa_pairs_path: str, dedup_path: str) -> list[QAPair]:
    """Load the subset of pairs from `qa_pairs_path` that dedup kept.

    Used by the rollout resume path: when a seed has both qa_pairs.json
    and qa_pairs_dedup.json on disk, we recover the kept set by joining
    on `qa_id` rather than re-running dedup.
    """
    with open(qa_pairs_path) as f:
        all_pairs = [QAPair(**p) for p in json.load(f)]
    with open(dedup_path) as f:
        kept_ids = set(json.load(f).get("kept_qa_ids", []))
    return [p for p in all_pairs if p.qa_id in kept_ids]


def _mean(values: list[float | None]) -> float:
    """Mean of a list of possibly-None floats; returns 0.0 if all are None."""
    arr = np.array([np.nan if v is None else v for v in values], dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return 0.0
    return float(np.mean(arr))


def _make_rollout_agent_factory(
    model_id: str,
    emb_model_id: str,
    document_map: dict[str, str],
    chroma_collection: Collection,
    special_notes: str,
    tinker_backend: object,
    sampling_params: dict | None = None,
    model_context_window: int = DEFAULT_ROLLOUT_CONTEXT_WINDOW,
):
    """Build the `agent_factory` callable consumed by `RolloutConfig`.

    Each invocation constructs a fresh `SearchAgent` configured *exactly*
    as it will be at test time (default system prompt, default
    ``final_answer(page_keys=[...])`` tool) and runs it on ``question``
    via ``SearchAgent.retrieve``. This way the rollouts produced here can
    be used to train / evaluate the same agent without behaviour drift.

    The shared ``tinker_backend`` generates every assistant turn, so the
    ``logprobs`` blocks in ``messages_to_jsonable()`` carry
    ``log pi_old(a | s)`` for every assistant token straight from the Tinker
    policy we fine-tune -- the importance-ratio denominator for GRPO / CISPO.
    ``sampling_params`` is still recorded per-rollout for provenance.

    Returns ``(output_doc_ids, trajectory_chunk_ids, completed, num_steps,
    error)`` for ``rollout.run_rollouts_for_pair`` to score and persist.
    The chunk-level "output" set is reconstructed downstream by expanding
    each output doc_id into its full chunk set.
    """
    def factory(question: str, trace_path: str, messages_path: str, show_output: bool):
        with Tracer(trace_path, show_output=show_output) as tracer:
            agent = SearchAgent(
                model_id,
                document_map=document_map,
                chroma_collection=chroma_collection,
                emb_model_id=emb_model_id,
                tracer=tracer,
                max_steps=AGENT_MAX_STEPS,
                max_pages_per_tool_call=MAX_PAGES_PER_TOOL_CALL,
                model_context_window=model_context_window,
                special_notes=special_notes,
                sampling_params=sampling_params,
                tinker_backend=tinker_backend, # type: ignore
                # These are training rollouts (RL data generation), so enable the
                # soft prune nudge at the effective-context threshold -- it lets
                # the agent prune proactively instead of only when forced at the
                # hard cutoff.
                train=True,
            )
            output_doc_ids = agent.retrieve(question)

        # persist the full message trajectory alongside the trace, then
        # derive the trajectory chunk_ids (every chunk surfaced in any
        # tool observation during the rollout) from the same jsonable
        # form so trajectory metrics are reproducible offline.
        messages_jsonable = agent.messages_to_jsonable()
        with open(messages_path, "w") as f:
            json.dump(messages_jsonable, f, indent=2)
        trajectory_chunk_ids = extract_trajectory_chunk_ids(messages_jsonable)

        return (
            output_doc_ids,
            trajectory_chunk_ids,
            agent._completed,
            agent._num_steps,
            agent._error,
        )

    return factory


def _run_rollouts_for_seed(
    seed: int,
    kept: list[QAPair],
    rollout_cfg: "RolloutConfig",
    trace_dir: str,
    rollout_executor: ThreadPoolExecutor,
    show_output: bool,
) -> list[RolloutRecord]:
    """Phase 1 of the rollout pipeline for one seed's dedup-kept pairs.

    Submits every ``(pair, rollout_idx)`` trajectory of the seed to the SHARED
    ``rollout_executor`` up front -- there is no per-pair barrier, so fast
    rollouts of a later pair start as soon as a slot frees instead of waiting
    on an earlier pair's slow straggler. Because the executor is shared across
    every concurrently-running seed too, the Tinker sampling backend stays
    saturated end-to-end rather than draining to 1-2 in-flight calls at each
    barrier.

    Persists `{seed}_rollouts.json` even when `kept` is empty so the
    resume logic treats the seed as done, and returns the records list
    so the caller can fold them into its in-memory accumulator without
    re-reading the file. Phase 2 (global threshold + filter) runs after
    every seed has produced this file.
    """
    # Submit all trajectories for this seed and demultiplex completions back
    # into per-pair buckets (rollouts of different pairs finish interleaved).
    fut_to_pair_idx: dict = {}
    for pair_idx, pair in enumerate(kept):
        for rollout_idx in range(rollout_cfg.n_rollouts):
            fut = rollout_executor.submit(
                run_single_rollout,
                seed, pair_idx, rollout_idx, pair, rollout_cfg, trace_dir, show_output,
            )
            fut_to_pair_idx[fut] = pair_idx

    records_by_pair: dict[int, list[RolloutRecord]] = {i: [] for i in range(len(kept))}
    for fut in as_completed(fut_to_pair_idx):
        pair_idx = fut_to_pair_idx[fut]
        try:
            records_by_pair[pair_idx].append(fut.result())
        except Exception as e:
            print(f"  rollout ERROR for seed={seed} qa={kept[pair_idx].qa_id}: {e}")

    seed_records: list[RolloutRecord] = []
    for pair_idx, pair in enumerate(kept):
        recs = sorted(records_by_pair[pair_idx], key=lambda r: r.rollout_idx)
        seed_records.extend(recs)
        n_scored = sum(1 for r in recs if r.doc_output_recall is not None)
        print(
            f"  seed={seed} qa_idx={pair_idx} qa_id={pair.qa_id}: "
            f"{len(recs)} rollouts, {n_scored} scored "
            f"(doc recall mean={_mean([r.doc_output_recall for r in recs]):.3f}, "
            f"doc traj recall mean={_mean([r.doc_trajectory_recall for r in recs]):.3f}, "
            f"chunk recall mean={_mean([r.chunk_output_recall for r in recs]):.3f}, "
            f"chunk traj recall mean={_mean([r.chunk_trajectory_recall for r in recs]):.3f})"
        )
    rollouts_path = os.path.join(trace_dir, f"{seed}_rollouts.json")
    persist_seed_rollouts(seed, seed_records, rollouts_path)
    return seed_records


def _qa_pairs_path(trace_dir: str, seed: int) -> str:
    return os.path.join(trace_dir, f"{seed}_qa_pairs.json")


def _dedup_path(trace_dir: str, seed: int) -> str:
    return os.path.join(trace_dir, f"{seed}_qa_pairs_dedup.json")


def _rollouts_path(trace_dir: str, seed: int) -> str:
    return os.path.join(trace_dir, f"{seed}_rollouts.json")


def _quality_path(trace_dir: str, seed: int) -> str:
    return os.path.join(trace_dir, f"{seed}_qa_pairs_quality.json")


def _run_quality_filter_task(
    seed: int,
    pair: QAPair,
    rollouts_for_pair: list[RolloutRecord],
    qf_cfg: QualityFilterConfig,
    trace_dir: str,
    show_output: bool,
) -> tuple[int, QualityFilterResult]:
    """Thread-pool entry point for one QualityFilter run.

    Returns ``(seed, result)`` so the main loop can fold the verdict back
    into ``qf_results_by_seed`` without needing to track the (seed, qa_id)
    mapping per future.
    """
    res = run_quality_filter_for_pair(
        pair=pair,
        rollout_records=rollouts_for_pair,
        cfg=qf_cfg,
        trace_dir=trace_dir,
        show_output=show_output,
    )
    return seed, res


def _schedule_qf_for_newly_passing(
    records_by_seed: dict[int, list[RolloutRecord]],
    kept_pairs_by_seed: dict[int, dict[str, QAPair]],
    qf_results_by_seed: dict[int, dict[str, QualityFilterResult]],
    qf_in_flight: set[str],
    trace_dir: str,
    qf_cfg: QualityFilterConfig,
    show_output: bool,
    pool: ThreadPoolExecutor,
    futures: dict,
) -> None:
    """Schedule QF tasks for any (seed, qa_id) above threshold but unjudged.

    Callers must call ``apply_threshold_and_filter`` before this function so
    that ``r.kept`` on every record reflects the current global threshold.
    """
    for s, recs in records_by_seed.items():
        by_qa: dict[str, list[RolloutRecord]] = {}
        for r in recs:
            by_qa.setdefault(r.qa_id, []).append(r)
        seed_kept_lookup = kept_pairs_by_seed.get(s, {})
        seed_qf = qf_results_by_seed.setdefault(s, {})
        for qa_id, rs in by_qa.items():
            # pair is "kept by threshold" iff at least one of its rollouts
            # has kept=True after the recompute (apply_threshold sets
            # kept identically for all rollouts of a pair).
            if not any(r.kept for r in rs):
                continue
            if qa_id in seed_qf or qa_id in qf_in_flight:
                continue
            pair = seed_kept_lookup.get(qa_id)
            if pair is None:
                # we don't have the QAPair text available -- try a fallback
                # disk load (rare; happens for seeds that were bootstrapped
                # without going through dedup again).
                try:
                    kept_loaded = _load_kept_pairs(
                        _qa_pairs_path(trace_dir, s),
                        _dedup_path(trace_dir, s),
                    )
                    kept_pairs_by_seed[s] = {p.qa_id: p for p in kept_loaded}
                    pair = kept_pairs_by_seed[s].get(qa_id)
                except (FileNotFoundError, json.JSONDecodeError, ValueError):
                    pair = None
            if pair is None:
                print(f"  WARN: cannot find QAPair for seed={s} qa_id={qa_id}; skipping QF")
                continue
            qf_in_flight.add(qa_id)
            futures[pool.submit(
                _run_quality_filter_task,
                s, pair, list(rs), qf_cfg, trace_dir, show_output,
            )] = ("quality_filter", s, qa_id)


def _dedup_task(
    seed: int,
    pairs: list[QAPair],
    dedup_cfg: DedupConfig,
    dedup_path: str,
    dedup_lock: threading.Lock,
) -> list[QAPair]:
    """Thread-pool entry point for dedup: serialises every dedup call.

    ``dedup_batch`` reads + upserts to a shared ``qa_collection`` without
    internal synchronisation, so concurrent invocations would race on the
    "existing pairs" snapshot and the final upsert. The lock makes the
    full read-judge-upsert sequence atomic with respect to other dedup
    tasks while still letting generation and rollout tasks run in parallel.
    """
    with dedup_lock:
        return _run_dedup_for_seed(seed, pairs, dedup_cfg, dedup_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the question-answer synthesis script.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="officeqa",
        choices=["officeqa", "browsecomp-plus"],
        help="benchmark whose QA pairs are used as style examples (default: officeqa)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemini-3.5-flash",
        help="ID of the language model to use for synthesis (default: google/gemini-3.5-flash)",
    )
    parser.add_argument(
        "--emb-model-id",
        type=str,
        default="qwen/qwen3-embedding-8b",
        help="ID of the embedding model (default: qwen/qwen3-embedding-8b)",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=".chromadb",
        help="Directory where ChromaDB stores its data (default: .chromadb)",
    )
    parser.add_argument(
        "--chroma-collection-name",
        type=str,
        default="qwen-v2",
        help="Name of the ChromaDB collection (default: qwen)",
    )
    parser.add_argument(
        "--num-new-seeds",
        type=int,
        default=100,
        help="Number of NEW generation seeds to enqueue, in addition to any "
             "in-progress seeds already on disk in --trace-dir (which will be "
             "resumed regardless). Each seed produces one generation -> dedup "
             "-> rollout -> QF pipeline run. Default: 100.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=8,
        help="Number of parallel generation workers (default: 8)",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=5,
        help="Number of example QA pairs shown to the agent per generation (default: 5)",
    )
    parser.add_argument(
        "--n-qa-pairs",
        type=int,
        default=DEFAULT_N_QA_PAIRS,
        help=f"Number of QA pairs the agent should produce per generation (default: {DEFAULT_N_QA_PAIRS})",
    )
    parser.add_argument(
        "--n-seed-chunks",
        type=int,
        default=N_SEED_CHUNKS,
        help=f"Number of random seed chunks to show the agent per generation (default: {N_SEED_CHUNKS})",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=TRACE_DIR,
        help=f"Directory to save agent traces (default: {TRACE_DIR})",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Stream each agent trace to the terminal.",
    )
    parser.add_argument(
        "--qa-collection-name",
        type=str,
        default="officeqa-qa-embeddings-qwen",
        help="Name of the ChromaDB collection holding pre-computed Q/A embeddings for "
             "the benchmark.",
    )
    parser.add_argument(
        "--judge-model-id",
        type=str,
        default="gpt-5-mini",
        help="ID of the LLM used by the de-duplication judge (default: gpt-5-mini).",
    )
    parser.add_argument(
        "--dedup-top-k",
        type=int,
        default=20,
        help="Number of nearest existing questions / answers to compare against per "
             "synthetic pair during de-duplication (default: 20).",
    )
    parser.add_argument(
        "--dedup-judge-parallelism",
        type=int,
        default=8,
        help="Parallelism for de-duplication judge calls within one batch (default: 8).",
    )
    parser.add_argument(
        "--rollout-tinker-model",
        type=str,
        default=DEFAULT_TINKER_BASE_MODEL,
        help=f"Tinker base_model used to generate rollout trajectories and capture "
             f"pi_old logprobs (default: {DEFAULT_TINKER_BASE_MODEL!r}). This is the policy "
             "you intend to fine-tune -- separate from the datagen model.",
    )
    parser.add_argument(
        "--rollout-renderer",
        type=str,
        default=DEFAULT_TINKER_RENDERER,
        help=f"tinker_cookbook renderer (chat template) matching the rollout model "
             f"(default: {DEFAULT_TINKER_RENDERER!r}; e.g. 'qwen3_disable_thinking' to "
             "disable thinking).",
    )
    parser.add_argument(
        "--rollout-max-tokens",
        type=int,
        default=DEFAULT_ROLLOUT_MAX_TOKENS,
        help=f"Per-turn generation cap for rollouts (default: {DEFAULT_ROLLOUT_MAX_TOKENS}). "
             "Tinker does not stream, so a turn ends at the renderer's turn-terminator "
             "token or this cap -- a runaway backstop, not the primary stop condition.",
    )
    parser.add_argument(
        "--rollout-context-window",
        type=int,
        default=DEFAULT_ROLLOUT_CONTEXT_WINDOW,
        help=f"Input context window (in tokens) for the rollout model "
             f"(default: {DEFAULT_ROLLOUT_CONTEXT_WINDOW}, the Tinker-enforced 64k max "
             "sequence length). Controls when the agent is warned to prune and when "
             "non-prune tool calls are blocked.",
    )
    parser.add_argument(
        "--tinker-prefill-price",
        type=float,
        default=None,
        help="Tinker price per 1M PREFILL (prompt) tokens for the rollout model, in USD. "
             "Overrides the TINKER_PRICING_PER_MTOK table in tinker_cost.py. Used only to "
             "compute the recorded dollar cost; token counts are always recorded.",
    )
    parser.add_argument(
        "--tinker-sample-price",
        type=float,
        default=None,
        help="Tinker price per 1M SAMPLE (generated) tokens for the rollout model, in USD. "
             "Overrides the TINKER_PRICING_PER_MTOK table in tinker_cost.py.",
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=DEFAULT_N_ROLLOUTS,
        help=f"Number of SearchAgent rollouts per kept QA pair (default: {DEFAULT_N_ROLLOUTS}).",
    )
    parser.add_argument(
        "--rollout-parallelism",
        type=int,
        default=DEFAULT_ROLLOUT_CONCURRENCY,
        help="Total number of SearchAgent rollouts run concurrently across ALL "
             "pairs and seeds (the size of the single shared rollout pool). "
             "There is no per-pair barrier: a finished rollout immediately frees "
             "a slot for the next one, keeping the Tinker backend saturated "
             f"instead of draining at each barrier. Default: {DEFAULT_ROLLOUT_CONCURRENCY}.",
    )
    parser.add_argument(
        "--task-solver-model-id",
        type=str,
        default="google/gemini-3.5-flash",
        help="Model used by the per-rollout TaskSolver (no-tool LLM that produces an "
             "answer from the rollout's returned documents). "
             "Default: google/gemini-3.5-flash.",
    )
    parser.add_argument(
        "--nugget-judge-model-id",
        type=str,
        default="google/gemini-3.5-flash",
        help="Model used by the nugget-completion judge that scores the TaskSolver's "
             "answer against the gold nuggets. Default: google/gemini-3.5-flash.",
    )
    parser.add_argument(
        "--task-solver-context-window",
        type=int,
        default=1_000_000,
        help="Input context window (tokens) for the TaskSolver model. When the rendered "
             "document set would exceed this budget, the longest doc is halved in "
             "round-robin fashion until it fits. Default: 1000000.",
    )
    parser.add_argument(
        "--strict-nugget-scoring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strict scoring (no partial credit): score = #support / #nuggets. "
             "Use --no-strict-nugget-scoring to allow 0.5 credit for partial_support.",
    )
    parser.add_argument(
        "--quality-filter-model-id",
        type=str,
        default="google/gemini-3.5-flash",
        help="Model used by the QualityFilter agent (a SearchAgent with corpus tools "
             "that decides whether a QA pair is unambiguous AND correct). "
             "Default: google/gemini-3.5-flash.",
    )
    parser.add_argument(
        "--quality-filter-context-window",
        type=int,
        default=1_000_000,
        help="Input context window (tokens) for the QualityFilter model. Default: 1000000.",
    )
    parser.add_argument(
        "--binarization-mode",
        type=str,
        choices=["doc-recall", "eval-score"],
        default="doc-recall",
        help="Per-rollout score used for global mean-threshold binarization: "
             "`doc-recall` uses doc_output_recall (default); `eval-score` uses the "
             "TaskSolver + NuggetCompletion judge's eval_score.",
    )
    parser.add_argument(
        "--service-tier",
        type=str,
        default="flex",
        help="OpenRouter service tier hint applied to ALL non-rollout calls "
             "(datagen agent, dedup judge, task solver, nugget judge, "
             "quality filter). Pass an empty string to disable. Default: flex.",
    )
    args = parser.parse_args()

    # Timestamp identifying this harness invocation; used for the Tinker cost
    # report filename (~/.tinker-cost/datagen_{run_ts}.json).
    run_ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    # normalise empty-string service-tier flag to None. (Rollouts go through
    # Tinker, which has no service-tier concept.)
    service_tier: str | None = args.service_tier or None

    # load example question-answer pairs from the benchmark
    qa_pairs = load_benchmark_questions(args.benchmark)

    # load mean and std. dev. for number of relevant chunks per question for this benchmark
    chunks_mean, chunks_std = BENCHMARK_CHUNKS_STATS[args.benchmark]

    # load dataset-specific question-style / computation guidance for this benchmark
    dataset_guidance = BENCHMARK_DATASET_GUIDANCE.get(args.benchmark, "")

    # load the SearchAgent corpus notes for this benchmark (used by both the
    # datagen agent and the rollout agent so they see notes matching the corpus).
    special_notes = BENCHMARK_SPECIAL_NOTES[args.benchmark]

    # get mapping from document ID to document text for the benchmark data from ChromaDB
    document_map = load_document_map(args.benchmark)

    # load chromadb collection to ensure it's ready before we start generating questions
    client = chromadb.PersistentClient(path=args.chroma_dir)
    collection = client.get_collection(args.chroma_collection_name)

    # set up de-duplication. The pipeline serialises all dedup work via
    # `dedup_lock` (single writer to qa_collection); generation and rollout
    # tasks remain free to run concurrently in the same thread pool.
    qa_collection = client.get_collection(args.qa_collection_name)
    or_client = OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])
    judge_template = BENCHMARK_DEDUP_JUDGE_PROMPT[args.benchmark]
    dedup_cfg = DedupConfig(
        qa_collection=qa_collection,
        emb_model_id=args.emb_model_id,
        or_client=or_client,
        judge_model_id=args.judge_model_id,
        judge_prompt_template=judge_template,
        top_k=args.dedup_top_k,
        judge_parallelism=args.dedup_judge_parallelism,
        service_tier=service_tier,
    )

    # set up rollouts + retrieval scoring. Each rollout task persists its own
    # `{seed}_rollouts.json`; the global mean threshold + all-pass / all-fail
    # filter (phase 2) is recomputed after every rollout completion to drive
    # the task-queue stopping condition, then once more after all tasks drain
    # to produce the final per-seed funnel files.
    # fetch all chunk_ids and their doc_ids once, so workers can sample seed chunks
    # and derive doc_ids from chunk_ids without any heuristics. The rollout
    # pipeline also reuses this map (and its inverse) to expand the agent's
    # doc-level final answer into a chunk-level set for the chunk-level
    # metrics, and to project the chunk-level trajectory onto docs.
    print(f"Fetching chunk ids from collection {args.chroma_collection_name!r}...")
    chunk_id_to_doc_id = load_chunk_id_to_doc_id(collection)
    print(f"  collection has {len(chunk_id_to_doc_id)} chunks.")

    doc_id_to_chunk_ids: dict[str, list[str]] = {}
    for cid, did in chunk_id_to_doc_id.items():
        doc_id_to_chunk_ids.setdefault(did, []).append(cid)

    # Build the shared Tinker sampling backend once; every rollout SearchAgent
    # in this run reuses it so a single SamplingClient drives all rollouts
    # (enabling server-side batching of concurrent sample() calls).
    print(f"Connecting to Tinker (base_model={args.rollout_tinker_model!r}, "
          f"renderer={args.rollout_renderer!r})...")
    tinker_backend = build_tinker_backend(
        base_model=args.rollout_tinker_model,
        renderer_name=args.rollout_renderer,
        max_tokens=args.rollout_max_tokens,
        context_window=args.rollout_context_window,
    )

    rollout_cfg = RolloutConfig(
        model_id=args.rollout_tinker_model,
        emb_model_id=args.emb_model_id,
        document_map=document_map,
        chroma_collection=collection,
        max_steps=AGENT_MAX_STEPS,
        max_pages_per_tool_call=MAX_PAGES_PER_TOOL_CALL,
        special_notes=special_notes,
        sampling_params=DEFAULT_ROLLOUT_SAMPLING_PARAMS,
        tinker_backend=tinker_backend,
        n_rollouts=args.n_rollouts,
        rollout_parallelism=args.rollout_parallelism,
        doc_id_to_chunk_ids=doc_id_to_chunk_ids,
        chunk_id_to_doc_id=chunk_id_to_doc_id,
        agent_factory=_make_rollout_agent_factory(
            model_id=args.rollout_tinker_model,
            emb_model_id=args.emb_model_id,
            document_map=document_map,
            chroma_collection=collection,
            special_notes=special_notes,
            tinker_backend=tinker_backend,
            sampling_params=DEFAULT_ROLLOUT_SAMPLING_PARAMS,
            model_context_window=args.rollout_context_window,
        ),
        task_solver_cfg=TaskSolverConfig(
            or_client=or_client,
            task_solver_model_id=args.task_solver_model_id,
            nugget_judge_model_id=args.nugget_judge_model_id,
            model_context_window=args.task_solver_context_window,
            strict_nugget_scoring=args.strict_nugget_scoring,
            service_tier=service_tier,
        ),
    )

    # set up the QualityFilter agent. One QualityFilterConfig is shared by
    # every QF task in this run; per-pair state (rollouts, ground truth) is
    # passed into ``run_quality_filter_for_pair`` per call.
    qf_system_prompt = BENCHMARK_QUALITY_FILTER_SYSTEM_PROMPT[args.benchmark]
    qf_cfg = QualityFilterConfig(
        model_id=args.quality_filter_model_id,
        emb_model_id=args.emb_model_id,
        document_map=document_map,
        chroma_collection=collection,
        system_prompt_template=qf_system_prompt,
        special_notes=special_notes,
        max_pages_per_tool_call=MAX_PAGES_PER_TOOL_CALL,
        model_context_window=args.quality_filter_context_window,
        binarization_mode=args.binarization_mode,
        service_tier=service_tier,
    )

    # generate synthetic QA pairs in parallel
    trace_dir = os.path.join(args.trace_dir, args.benchmark)
    os.makedirs(trace_dir, exist_ok=True)

    # ---- bootstrap from any prior partial run ---------------------------
    # scan trace_dir to (1) re-load completed rollouts into the accumulator
    # and (2) queue resumption tasks for seeds stuck mid-pipeline. `next_seed`
    # picks up after the highest seed already on disk so we never collide.
    existing_seeds: set[int] = set()
    for fn in os.listdir(trace_dir):
        if fn.endswith("_qa_pairs.json"):
            existing_seeds.add(int(fn.split("_", 1)[0]))

    records_by_seed: dict[int, list[RolloutRecord]] = {}
    # qf_results_by_seed: seed -> {qa_id -> QualityFilterResult}. Loaded from
    # disk so QF verdicts persist across restarts and we never re-judge a pair
    # that already has a result.
    qf_results_by_seed: dict[int, dict[str, QualityFilterResult]] = {}
    # set of qa_ids currently being judged by an in-flight QF task -- used to
    # deduplicate scheduling when the same pair becomes "kept" again after a
    # later threshold recompute.
    qf_in_flight: set[str] = set()
    # per-seed cache of (seed -> {qa_id -> QAPair}) for kept pairs, populated
    # when dedup finishes (or when bootstrapping a resumed rollout seed). The
    # QF stage needs the QAPair to render the user prompt; rollout records
    # alone don't carry the question / answer text.
    kept_pairs_by_seed: dict[int, dict[str, QAPair]] = {}

    bootstrap_dedup: list[int] = []
    bootstrap_rollout: list[int] = []
    for seed in sorted(existing_seeds):
        if os.path.exists(_rollouts_path(trace_dir, seed)):
            try:
                with open(_rollouts_path(trace_dir, seed)) as f:
                    loaded = [RolloutRecord(**r) for r in json.load(f)]
            except (json.JSONDecodeError, ValueError):
                loaded = []
            if loaded:
                records_by_seed[seed] = loaded
                # also recover kept QAPair lookup table for downstream QF.
                if os.path.exists(_dedup_path(trace_dir, seed)):
                    try:
                        kept = _load_kept_pairs(
                            _qa_pairs_path(trace_dir, seed),
                            _dedup_path(trace_dir, seed),
                        )
                        kept_pairs_by_seed[seed] = {p.qa_id: p for p in kept}
                    except (json.JSONDecodeError, FileNotFoundError, ValueError):
                        pass
            elif os.path.exists(_dedup_path(trace_dir, seed)):
                print(f"  seed={seed}: empty/corrupt rollout file, re-queuing rollout")
                bootstrap_rollout.append(seed)
            else:
                print(f"  seed={seed}: empty/corrupt rollout file (no dedup), re-queuing dedup")
                bootstrap_dedup.append(seed)
        elif os.path.exists(_dedup_path(trace_dir, seed)):
            bootstrap_rollout.append(seed)
        else:
            bootstrap_dedup.append(seed)

        # always try to load any prior QF results, regardless of pipeline stage.
        qf_results_by_seed[seed] = load_quality_funnel(_quality_path(trace_dir, seed))

    # initial bootstrap summary -- threshold pass-count from records on disk.
    n_kept_threshold = 0
    if records_by_seed:
        _, _stats = apply_threshold_and_filter(records_by_seed, args.binarization_mode)
        n_kept_threshold = _stats.n_pairs_kept

    next_seed = max(existing_seeds, default=-1) + 1
    print(
        f"Bootstrap: {len(records_by_seed)} seed(s) with rollouts on disk "
        f"({n_kept_threshold} pass current threshold). Resuming "
        f"{len(bootstrap_dedup)} at dedup, {len(bootstrap_rollout)} at rollout. "
        f"Enqueuing {args.num_new_seeds} new generation seed(s) starting at "
        f"{next_seed}."
    )

    # ---- task-queue main loop -------------------------------------------
    # Each seed flows generate -> dedup -> rollout -> (per-pair quality_filter).
    # After every rollout completion we recompute the global threshold; any
    # pair that newly passes the threshold and has no prior QF result is
    # scheduled as a fresh quality_filter task.
    # dedup_lock serialises dedup tasks (they share a single qa_collection
    # writer); generation, rollout, and QF run freely.
    dedup_lock = threading.Lock()
    futures: dict = {}
    n_gen_completed = 0
    n_gen_succeeded = 0

    pool = ThreadPoolExecutor(max_workers=args.parallelism)
    # Single shared rollout pool. Every seed's `_run_rollouts_for_seed` submits
    # its individual trajectories here rather than into a private per-pair pool,
    # so rollouts from all pairs and all in-flight seeds compete for the same
    # slots and a finished rollout immediately starts the next queued one.
    rollout_executor = ThreadPoolExecutor(max_workers=max(1, args.rollout_parallelism))
    try:
        # schedule resumption tasks first so they reprocess alongside fresh seeds.
        for seed in bootstrap_dedup:
            with open(_qa_pairs_path(trace_dir, seed)) as f:
                pairs = [QAPair(**p) for p in json.load(f)]
            futures[pool.submit(
                _dedup_task, seed, pairs, dedup_cfg, _dedup_path(trace_dir, seed), dedup_lock,
            )] = ("dedup", seed)
        for seed in bootstrap_rollout:
            kept = _load_kept_pairs(_qa_pairs_path(trace_dir, seed), _dedup_path(trace_dir, seed))
            kept_pairs_by_seed[seed] = {p.qa_id: p for p in kept}
            futures[pool.submit(
                _run_rollouts_for_seed, seed, kept, rollout_cfg, trace_dir,
                rollout_executor, args.show_output,
            )] = ("rollout", seed)

        # bootstrap: also try to schedule QF for any pair that's already past
        # threshold but never judged.
        if records_by_seed:
            _schedule_qf_for_newly_passing(
                records_by_seed, kept_pairs_by_seed, qf_results_by_seed,
                qf_in_flight, trace_dir, qf_cfg, args.show_output, pool, futures,
            )

        # enqueue exactly `--num-new-seeds` fresh generation tasks up front.
        # The pool's max_workers caps actual concurrency; the rest queue
        # behind in flight tasks until a slot frees up.
        for _ in range(args.num_new_seeds):
            seed = next_seed
            next_seed += 1
            futures[pool.submit(
                generate_one,
                seed,
                args.model_id,
                qa_pairs,
                document_map,
                collection,
                chunk_id_to_doc_id,
                args.emb_model_id,
                args.show_output,
                trace_dir,
                n_examples=args.n_examples,
                n_qa_pairs=args.n_qa_pairs,
                n_seed_chunks=args.n_seed_chunks,
                chunks_mean=chunks_mean,
                chunks_std=chunks_std,
                dataset_guidance=dataset_guidance,
                special_notes=special_notes,
                service_tier=service_tier,
            )] = ("generate", seed)

        # drain: process completions (which may schedule downstream tasks)
        # until nothing is left in flight.
        while futures:
            done, _pending = wait(list(futures), return_when=FIRST_COMPLETED)
            for fut in done:
                key = futures.pop(fut)
                stage = key[0]
                seed = key[1]
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"  {stage} ERROR for seed={seed}: {e}")
                    if stage == "quality_filter":
                        qf_in_flight.discard(key[2])
                    # drop the task; the slot is freed for new work.
                    continue

                if stage == "generate":
                    n_gen_completed += 1
                    if res.completed:
                        n_gen_succeeded += 1
                    print(
                        f"=== seed={res.seed} generated completed={res.completed} "
                        f"steps={res.num_steps} pairs={len(res.pairs)} ==="
                    )
                    for i, pair in enumerate(res.pairs):
                        print(f"  pair {i}: Q: {pair.question}")
                        print(f"           A: {pair.answer}")
                        print(f"           chunk_ids: {pair.chunk_ids}")
                    if res.error:
                        print(f"  error: {res.error}")
                    futures[pool.submit(
                        _dedup_task, seed, res.pairs, dedup_cfg,
                        _dedup_path(trace_dir, seed), dedup_lock,
                    )] = ("dedup", seed)
                elif stage == "dedup":
                    kept = res
                    kept_pairs_by_seed[seed] = {p.qa_id: p for p in kept}
                    print(f"=== seed={seed} deduped kept={len(kept)} ===")
                    futures[pool.submit(
                        _run_rollouts_for_seed, seed, kept, rollout_cfg, trace_dir,
                        rollout_executor, args.show_output,
                    )] = ("rollout", seed)
                elif stage == "rollout":
                    records_by_seed[seed] = res
                    # recompute global threshold and (re-)schedule QF for any
                    # pair that newly passes. apply_threshold_and_filter is
                    # re-runnable -- a pair that was discarded earlier (e.g.
                    # all-pass at a higher threshold) can become kept now.
                    _, _stats = apply_threshold_and_filter(records_by_seed, args.binarization_mode)
                    _schedule_qf_for_newly_passing(
                        records_by_seed, kept_pairs_by_seed, qf_results_by_seed,
                        qf_in_flight, trace_dir, qf_cfg, args.show_output, pool, futures,
                    )
                    print(
                        f"=== seed={seed} rolled out; threshold={_stats.threshold:.4f} "
                        f"kept_by_threshold={_stats.n_pairs_kept} ==="
                    )
                elif stage == "quality_filter":
                    qa_id = key[2]
                    qf_in_flight.discard(qa_id)
                    _, qf_result = res
                    qf_results_by_seed.setdefault(seed, {})[qa_id] = qf_result
                    persist_quality_funnel(
                        seed,
                        qf_results_by_seed[seed],
                        _quality_path(trace_dir, seed),
                    )
                    print(
                        f"=== seed={seed} QF qa_id={qa_id} valid={qf_result.valid} ==="
                    )

    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        rollout_executor.shutdown(wait=False, cancel_futures=True)
        # Record Tinker sampling spend for this invocation (even on interrupt /
        # error) so we can track cost without dashboard access.
        try:
            prefill_price, sample_price = resolve_prices(
                args.rollout_tinker_model,
                args.tinker_prefill_price,
                args.tinker_sample_price,
            )
            usage = tinker_backend.usage
            cost_path = write_run_report(
                timestamp=run_ts,
                base_model=args.rollout_tinker_model,
                prefill_tokens=usage.prefill_tokens,
                sample_tokens=usage.sample_tokens,
                n_sampling_calls=usage.n_calls,
                prefill_price_per_mtok=prefill_price,
                sample_price_per_mtok=sample_price,
            )
            print(
                f"\nTinker usage: {usage.prefill_tokens:,} prefill + "
                f"{usage.sample_tokens:,} sample tokens over {usage.n_calls:,} calls. "
                f"Cost report -> {cost_path} "
                f"(aggregate with `python -m skunk.datagen.tinker_cost`)."
            )
        except Exception as e:
            print(f"  WARN: failed to write Tinker cost report: {e}")

    print(
        f"\nGeneration: {n_gen_succeeded}/{n_gen_completed} completed successfully "
        f"across {len(records_by_seed)} seeds with rollouts."
    )
    print(f"Output written to {trace_dir}/{{seed}}_qa_pairs.json for each seed.")
    print(f"Dedup funnels written to {trace_dir}/{{seed}}_qa_pairs_dedup.json.")

    # ---- final phase 2: global threshold + per-seed filtered funnel files
    # one more apply_threshold_and_filter pass against the final accumulator
    # so the persisted `_rollouts_filtered.json` files reflect the final
    # threshold (not whatever transient value we had mid-loop).
    if records_by_seed:
        # final QF top-up pass: any pair that's past the final threshold but
        # has no QF result gets one now (synchronously, in a fresh small pool).
        records_by_seed, stats = apply_threshold_and_filter(
            records_by_seed, args.binarization_mode,
        )
        pending_qf: list[tuple[int, QAPair, list[RolloutRecord]]] = []
        for s, recs in records_by_seed.items():
            by_qa: dict[str, list[RolloutRecord]] = {}
            for r in recs:
                by_qa.setdefault(r.qa_id, []).append(r)
            seed_qf = qf_results_by_seed.setdefault(s, {})
            seed_kept_lookup = kept_pairs_by_seed.get(s, {})
            for qa_id, rs in by_qa.items():
                if not any(r.kept for r in rs):
                    continue
                if qa_id in seed_qf:
                    continue
                pair = seed_kept_lookup.get(qa_id)
                if pair is None:
                    try:
                        kept_loaded = _load_kept_pairs(
                            _qa_pairs_path(trace_dir, s), _dedup_path(trace_dir, s),
                        )
                        kept_pairs_by_seed[s] = {p.qa_id: p for p in kept_loaded}
                        pair = kept_pairs_by_seed[s].get(qa_id)
                    except (FileNotFoundError, json.JSONDecodeError, ValueError):
                        pair = None
                if pair is not None:
                    pending_qf.append((s, pair, list(rs)))
        if pending_qf:
            print(f"\nFinal QF top-up: judging {len(pending_qf)} pair(s)...")
            with ThreadPoolExecutor(max_workers=max(1, args.parallelism)) as qf_pool:
                qf_futs = [
                    qf_pool.submit(
                        _run_quality_filter_task,
                        s, pair, rs, qf_cfg, trace_dir, args.show_output,
                    )
                    for (s, pair, rs) in pending_qf
                ]
                for f in qf_futs:
                    try:
                        s, res = f.result()
                    except Exception as e:
                        print(f"  final QF ERROR: {e}")
                        continue
                    qf_results_by_seed.setdefault(s, {})[res.qa_id] = res
                    persist_quality_funnel(
                        s, qf_results_by_seed[s], _quality_path(trace_dir, s),
                    )

        # count pairs that BOTH pass the final threshold AND have a passing
        # QF verdict -- the actual acceptance gate. Counted before the demote
        # pass below so the printed stat matches `validated_by_qf`.
        n_passing = 0
        for s, recs in records_by_seed.items():
            seed_qf = qf_results_by_seed.get(s, {})
            seen: set[str] = set()
            for r in recs:
                if r.qa_id in seen or not r.kept:
                    continue
                seen.add(r.qa_id)
                qfr = seed_qf.get(r.qa_id)
                if qfr is not None and qfr.valid is True:
                    n_passing += 1

        # Demote pairs that lack a passing QF verdict so the on-disk filtered
        # files reflect the final acceptance gate (dedup + pass-rate + QF),
        # not just the pass-rate filter. Pairs with valid=None (QF error) or
        # with no QF result at all are treated the same as valid=False here.
        n_qf_demoted = 0
        for s, recs in records_by_seed.items():
            seed_qf = qf_results_by_seed.get(s, {})
            for r in recs:
                if not r.kept:
                    continue
                qfr = seed_qf.get(r.qa_id)
                if qfr is None:
                    r.kept = False
                    r.discard_reason = "qf_missing"
                    n_qf_demoted += 1
                elif qfr.valid is None:
                    r.kept = False
                    r.discard_reason = "qf_error"
                    n_qf_demoted += 1
                elif not qfr.valid:
                    r.kept = False
                    r.discard_reason = "qf_rejected"
                    n_qf_demoted += 1
        if n_qf_demoted:
            print(f"  QF gate demoted {n_qf_demoted} record(s) to kept=False.")

        print(
            f"\nRollout filter: threshold={stats.threshold:.4f} "
            f"pairs={stats.n_pairs} kept_by_threshold={stats.n_pairs_kept} "
            f"all_pass={stats.n_pairs_all_pass} all_fail={stats.n_pairs_all_fail} "
            f"validated_by_qf={n_passing} "
            f"(scored {stats.n_scored}/{stats.n_rollouts} rollouts; "
            f"binarization_mode={args.binarization_mode})"
        )
        for seed, recs in records_by_seed.items():
            persist_filtered_seed(
                seed,
                recs,
                stats,
                os.path.join(trace_dir, f"{seed}_rollouts_filtered.json"),
            )
        print(f"Rollout funnels written to {trace_dir}/{{seed}}_rollouts_filtered.json.")
        print(f"QF funnels written to {trace_dir}/{{seed}}_qa_pairs_quality.json.")
    else:
        print("\nNo rollouts produced; skipping global threshold + filter step.")
