"""Synthetic QA-pair generation for the DAIS slim corpus (full funnel, no Tinker).

Reuses the OfficeQA datagen *funnel stages* (generate -> dedup -> rollouts ->
quality filter) but adapted for a one-off synthetic **test** set over a new corpus:

  * **Reuses OfficeQA assets** as style anchors: few-shot QA pairs from
    ``officeqa_pro.csv``, the OfficeQA datagen guidance / corpus notes, the dedup
    judge prompt, the quality-filter system prompt, and the OfficeQA chunk-count stats.
  * **Points at the new corpus**: the document map comes from the DAIS
    ``clean_page_map.json`` and the vector store is the served ``.chromadb-dais-slim``
    collection.
  * **Rollouts run on Gemini 3.5 Flash, not Tinker** (we don't import ``harness`` — its
    top-level tinker import is the only Tinker coupling; the stage modules ``dedup`` /
    ``rollout`` / ``quality_filter`` / ``task_solver`` are tinker-free). We capture **no**
    logprobs: rollouts exist only to find QA pairs that are *challenging but answerable*
    (kept iff the agent neither always nor never finds the gold docs), which the quality
    filter then validates.

Output: the validated pairs as a synthetic test-set CSV (``officeqa_pro.csv`` schema)
plus a structured JSON with full provenance.

Usage (ChromaDB server must be serving ``.chromadb-dais-slim`` — see README):
    python3 -m skunk.dais.datagen \\
        --clean-page-map dais_cleaned/clean_page_map.json \\
        --chroma-collection-name dais-slim --num-new-seeds 20
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import json
import math
import os
import pathlib
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import zip_longest

import yaml
from jinja2 import Template
from openrouter import OpenRouter

from skunk.chroma_client import make_chroma_client
from skunk.common import ExecutionContext
from skunk.config import SkunkConfig
from skunk.prompted_call import PromptOverride
from skunk.search_agent.search_agent import SearchAgent

from skunk.datagen.dedup import DedupConfig, dedup_batch
from skunk.datagen.quality_filter import (
    BENCHMARK_QUALITY_FILTER_SYSTEM_PROMPT,
    QualityFilterConfig,
    run_quality_filter_for_pair,
)
from skunk.datagen.rollout import (
    RolloutConfig,
    apply_threshold_and_filter,
    extract_trajectory_chunk_ids,
    persist_seed_rollouts,
    run_single_rollout,
)

# --------------------------------------------------------------------------- constants
MAX_PAGES_PER_TOOL_CALL = 20
AGENT_MAX_STEPS = 100
N_SEED_CHUNKS = 10
VAL_FRAC = 0.25
CHROMA_PAGE_SIZE = 5000
DATAGEN_AUTHORIZED_IMPORTS = ["math", "statistics", "numpy", "scipy", "statsmodels"]
# OfficeQA chunk-count stats (mean, std), reused as the target supporting-chunk
# distribution for the new (OfficeQA-like) corpus.
OFFICEQA_CHUNKS_STATS = (1.8, 1.1)
DEFAULT_GEN_MODEL = "google/gemini-3.5-flash"
DEFAULT_ROLLOUT_MODEL = "google/gemini-3.5-flash"
DEFAULT_EMB_MODEL = "qwen/qwen3-embedding-8b"

# OfficeQA prompt assets, loaded the same way harness.py does (but without importing it).
_PROMPTS_FILE = pathlib.Path(__file__).parent.parent / "datagen" / "prompts.yaml"
with _PROMPTS_FILE.open() as _f:
    _DATAGEN_PROMPTS = yaml.safe_load(_f)
DATAGEN_SYSTEM_PROMPT = _DATAGEN_PROMPTS["datagen_system_prompt"]
DAIS_DATAGEN_GUIDANCE = _DATAGEN_PROMPTS["dais_datagen_guidance"]
DEDUP_JUDGE_PROMPT = _DATAGEN_PROMPTS["dedup_judge_prompt_dais"]

# Corpus notes describing the DAIS schema to the SearchAgent (replaces the OfficeQA
# treasury_bulletin notes, which describe month/quarterly cadence + a {year}_{month}_{page}
# doc_id that don't exist here). Mirrors the OfficeQA notes' shape so the agent knows the
# identifier formats and which metadata fields it can `metadata_filter` on. This corpus is
# ANNUAL: filter on `year` (int) — there is no month.
DAIS_CORPUS_NOTES = """\
The corpus is the U.S. "Combined Statement of Receipts, Expenditures, and Balances of the \
United States Government" (the federal "Account of Receipts and Expenditures") — ANNUAL \
fiscal reports spanning roughly 1793 to 2024. There is no monthly or quarterly cadence: \
each report covers a fiscal year. A single year may be split across multiple source files \
(e.g. modern years are split into per-chapter files), so one `year` can map to many \
`file_id`s. Each source file is split into pages, and each page is treated as one \
"document" (with its own `doc_id`).

### Identifier formats
- `file_id`: the source-file stem, e.g. `"combined_statement__modern__2001__c01"` or \
`"govinfo_receipts__1893__SERIALSET-03108_00_00-002-0256-0000"`.
- `doc_id`: `"<file_id>_<page_id>"` where `page_id` is the 1-indexed page within that file \
(matches the PDF viewer's page number), e.g. `"combined_statement__historical__cs-1872_12"`.
- `chunk_id`: `"<file_id>_<page_id>_<element_id>"` (e.g. `"...cs-1872_12_5"` = the 5th element \
on page 12).
- In your final answer, the `page_keys` list holds `doc_id`s in the format above.

### Available metadata fields (for use with `metadata_filter`)
Every chunk has the following metadata you can filter on:
- `year`: INTEGER fiscal year, e.g. `1872` (supports range filters via `$gte`/`$lte`/`$in`).
- `source`: string era/source family — one of `"historical"`, `"modern"`, `"transition"`, \
`"govinfo_receipts"`.
- `page_id`: integer, the 1-indexed page within its file (matches the PDF viewer).
- `file_id`: string, the source-file stem (see above).
- `doc_id`: string, the full page key (see above).
- `type`: string, the element type (`"text"`, `"title"`, `"table"`, `"section_header"`, ...).

### Example `metadata_filter` clauses
```python
# all pages from fiscal year 1872
{"year": 1872}

# any year in the 1900s, tables only
{"$and": [
    {"year": {"$gte": 1900, "$lte": 1999}},
    {"type": "table"},
]}

# only the modern combined-statement era
{"source": "modern"}
```
"""


# --------------------------------------------------------------------------- data model
@dataclass
class QAPair:
    """Duck-typed twin of ``harness.QAPair`` (the stage modules only import it under
    TYPE_CHECKING, so a local definition with the same fields works everywhere)."""

    qa_id: str
    question: str
    answer: list[str]
    chunk_ids: list[str]
    doc_ids: list[str]


class QASynthAgent(SearchAgent):
    """SearchAgent specialised for QA-pair synthesis (genai backend)."""

    name = "qa_synth"
    authorized_imports = DATAGEN_AUTHORIZED_IMPORTS

    def validate_final_answer(self, payload: object, observations: list[str]) -> str | None:
        if not isinstance(payload, dict) or not isinstance(payload.get("qa_pairs"), list):
            return ('Emit a JSON object {"qa_pairs": [{"question": ..., '
                    '"answer": [...], "chunk_ids": [...]}, ...]}.')
        return None

    async def synth(self, ctx: ExecutionContext, user_prompt: str) -> list[dict]:
        try:
            payload = await self.call(ctx, user_prompt)
        except Exception:
            return []
        return payload.get("qa_pairs", []) if isinstance(payload, dict) else []


# --------------------------------------------------------------------------- corpus maps
def load_chunk_id_to_doc_id(collection) -> dict[str, str]:
    """Map every chunk_id in the collection to its doc_id (paged to dodge SQLite limits)."""
    out: dict[str, str] = {}
    offset = 0
    while True:
        page = collection.get(include=["metadatas"], limit=CHROMA_PAGE_SIZE, offset=offset)
        ids = page["ids"]
        if not ids:
            break
        for cid, meta in zip(ids, page["metadatas"] or [], strict=True):
            out[cid] = str((meta or {}).get("doc_id", cid))
        offset += len(ids)
        if len(ids) < CHROMA_PAGE_SIZE:
            break
    return out


def invert_chunk_map(chunk_id_to_doc_id: dict[str, str]) -> dict[str, list[str]]:
    doc_to_chunks: dict[str, list[str]] = {}
    for cid, did in chunk_id_to_doc_id.items():
        doc_to_chunks.setdefault(did, []).append(cid)
    return doc_to_chunks


def load_document_map(clean_page_map_path: str) -> dict[str, str]:
    """Build doc_id -> cleaned page text from a DAIS ``clean_page_map.json``."""
    with open(clean_page_map_path) as f:
        clean_page_map = json.load(f)
    base = os.path.dirname(clean_page_map_path)
    document_map: dict[str, str] = {}
    for doc_id, entry in clean_page_map.items():
        path = entry[0]
        if not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(base, os.path.basename(path))
        with open(path) as f:
            document_map[doc_id] = f.read()
    return document_map


def load_few_shot_questions(csv_path: str) -> list[tuple[str, str]]:
    """OfficeQA few-shot (question, answer) pairs from its benchmark CSV (val split)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    val = df.iloc[: int(len(df) * VAL_FRAC)]
    return list(zip(val["question"].tolist(), val["answer"].tolist(), strict=True))


# --------------------------------------------------------------------------- generation
def _sample_n_chunks(rng: random.Random, mean: float, std: float) -> int:
    return max(1, math.ceil(rng.gauss(mean, std)))


def _sample_seed_chunks(rng, collection, chunk_id_to_doc_id, n) -> list[tuple[str, str, str]]:
    sampled = rng.sample(list(chunk_id_to_doc_id.keys()), min(n, len(chunk_id_to_doc_id)))
    res = collection.get(ids=sampled, include=["documents"])
    ids_out = res.get("ids") or []
    docs_out = res.get("documents") or []
    return [(cid, chunk_id_to_doc_id[cid], text or "") for cid, text in zip(ids_out, docs_out, strict=True)]


def _build_user_prompt(seed_chunks, examples, n_qa_pairs, n_chunks) -> str:
    lines = [
        f"Here are {len(seed_chunks)} randomly sampled chunks from the corpus "
        f"to give you a sense of its breadth. Use them to inspire diverse "
        f"question topics, but you are NOT required to ground your final "
        f"questions in these particular chunks.\n"
    ]
    for i, (cid, did, text) in enumerate(seed_chunks, 1):
        lines.append(f"Seed chunk {i} (chunk_id={cid}, doc_id={did}):\n{text}\n")
    lines.append(
        f"\nHere are {len(examples)} example question-answer pairs from a "
        f"benchmark built on this corpus, illustrating the target style and difficulty:\n"
    )
    for i, (q, a) in enumerate(examples, 1):
        lines.append(f"Example {i}:\n  Q: {q}\n  A: {a}\n")
    lines.append(
        f"\nNow explore the corpus to synthesise {n_qa_pairs} new, diverse "
        f"question-answer pairs. Each answer must be a list of nuggets (key facts "
        f"essential to the answer), and each pair should be grounded in roughly "
        f"{n_chunks} supporting chunk(s). Call `final_answer(...)` exactly once with "
        f"all {n_qa_pairs} pairs."
    )
    return "\n".join(lines)


def _agent_config(
    model_id: str, emb_model_id: str, provider: str, max_steps: int,
    pdf_dir: str | None = None, page_renders_dir: str | None = None,
) -> SkunkConfig:
    """SearchAgent SkunkConfig. genai uses the bare model id; openrouter the full id. `pdf_dir`
    (+ optional pre-rendered `page_renders_dir`) point the figure-viewing tool at the DAIS PDFs /
    page-render cache so ``view_figure`` works for this corpus."""
    agent_model = model_id if provider == "openrouter" else model_id.removeprefix("google/")
    extra: dict = {}
    if pdf_dir:
        extra["pdf_dir"] = pathlib.Path(pdf_dir)
    if page_renders_dir:
        extra["page_renders_dir"] = pathlib.Path(page_renders_dir)
    return SkunkConfig(
        agent_model_id=agent_model,
        emb_model_id=emb_model_id,
        llm_provider=provider,  # type: ignore[arg-type]
        agent_max_steps=max_steps,
        agent_max_pages_per_tool_call=MAX_PAGES_PER_TOOL_CALL,
        **extra,
    )


def generate_one(
    seed: int,
    model_id: str,
    provider: str,
    examples_pool: list[tuple[str, str]],
    document_map: dict[str, str],
    collection,
    chunk_id_to_doc_id: dict[str, str],
    emb_model_id: str,
    trace_dir: str,
    special_notes: str,
    dataset_guidance: str,
    n_examples: int,
    n_qa_pairs: int,
    show_output: bool,
    pdf_dir: str | None = None,
    page_renders_dir: str | None = None,
) -> list[QAPair]:
    """One synthesis run -> list of QAPair (empty on failure)."""
    rng = random.Random(seed)
    examples = rng.sample(examples_pool, min(n_examples, len(examples_pool)))
    n_chunks = _sample_n_chunks(rng, *OFFICEQA_CHUNKS_STATS)
    seed_chunks = _sample_seed_chunks(rng, collection, chunk_id_to_doc_id, N_SEED_CHUNKS)

    system_prompt = Template(DATAGEN_SYSTEM_PROMPT).render(
        max_steps=AGENT_MAX_STEPS,
        max_pages=MAX_PAGES_PER_TOOL_CALL,
        special_notes=special_notes,
        dataset_guidance=dataset_guidance,
        n_chunks=n_chunks,
        n_qa_pairs=n_qa_pairs,
    )
    user_prompt = _build_user_prompt(seed_chunks, examples, n_qa_pairs, n_chunks)

    os.makedirs(trace_dir, exist_ok=True)
    trace_path = f"{trace_dir}/{seed}_trace.txt"
    config = _agent_config(model_id, emb_model_id, provider, AGENT_MAX_STEPS, pdf_dir, page_renders_dir)
    ctx = ExecutionContext(question=user_prompt, config=config, log_path=trace_path, verbose=show_output)
    try:
        agent = QASynthAgent(
            config=config,
            document_map=document_map,
            chroma_collection=collection,
            system_prompt_override=system_prompt,
        )
        raw_pairs = asyncio.run(agent.synth(ctx, user_prompt))
    finally:
        ctx.close()

    with open(f"{trace_dir}/{seed}_messages.json", "w") as f:
        json.dump(agent.messages_to_jsonable(), f, indent=2)

    pairs: list[QAPair] = []
    for i, p in enumerate(raw_pairs):
        chunk_ids = [str(c) for c in p.get("chunk_ids", [])]
        mapped = [c for c in chunk_ids if c in chunk_id_to_doc_id]
        doc_ids = sorted({chunk_id_to_doc_id[c] for c in mapped})
        # The generator can emit malformed chunk_ids (esp. for govinfo file_ids with many
        # underscores/hyphens). A pair with NO mappable chunk_id has empty provenance and would
        # enter the funnel with a null doc-recall (silently kept) — drop it. Partial misses are
        # fine (the pair still has ≥1 valid supporting doc); just warn so they're visible.
        if not doc_ids:
            print(f"  [datagen] seed {seed} qa {seed}-{i}: dropping — no valid chunk_ids "
                  f"(emitted {chunk_ids!r})")
            continue
        if len(mapped) < len(chunk_ids):
            print(f"  [datagen] seed {seed} qa {seed}-{i}: dropped {len(chunk_ids) - len(mapped)} "
                  f"unmappable chunk_id(s): {[c for c in chunk_ids if c not in chunk_id_to_doc_id]!r}")
        pairs.append(QAPair(
            qa_id=f"{seed}-{i}",
            question=str(p.get("question", "")),
            answer=[str(n) for n in p.get("answer", [])],
            chunk_ids=mapped,
            doc_ids=doc_ids,
        ))
    with open(f"{trace_dir}/{seed}_qa_pairs.json", "w") as f:
        json.dump([asdict(p) for p in pairs], f, indent=2)
    return pairs


# --------------------------------------------------------------------------- rollouts
def make_gemini_rollout_agent_factory(
    model_id: str,
    provider: str,
    emb_model_id: str,
    document_map: dict[str, str],
    collection,
    special_notes: str,
    pdf_dir: str | None = None,
    page_renders_dir: str | None = None,
):
    """``RolloutConfig.agent_factory`` that runs a *plain* SearchAgent (Gemini, no Tinker,
    no logprobs) exactly as it runs at test time — mirrors quality_filter's construction."""
    config = _agent_config(model_id, emb_model_id, provider, AGENT_MAX_STEPS, pdf_dir, page_renders_dir)
    overrides = (
        (PromptOverride(section="corpus", targets=("search_agent",), content=special_notes),)
        if special_notes else ()
    )

    def factory(question: str, trace_path: str, messages_path: str, show_output: bool):
        ctx = ExecutionContext(
            question=question, config=config, prompt_overrides=overrides,
            log_path=trace_path, verbose=show_output,
        )
        agent: SearchAgent | None = None
        output_doc_ids: list[str] = []
        error: str | None = None
        try:
            agent = SearchAgent(
                config=config,
                document_map=document_map,
                chroma_collection=collection,
                generation_backend=None,   # plain genai/openrouter loop, not Tinker
                capture_logprobs=False,     # this set is for evaluation, not RL training
            )
            output_doc_ids = asyncio.run(agent.retrieve(ctx, question))
            completed = True
        except Exception as e:  # noqa: BLE001
            completed = False
            error = f"rollout agent failed: {e}"
        finally:
            ctx.close()

        messages_jsonable = agent.messages_to_jsonable() if agent else []
        with open(messages_path, "w") as f:
            json.dump(messages_jsonable, f, indent=2)
        trajectory_chunk_ids = extract_trajectory_chunk_ids(messages_jsonable)
        num_steps = sum(1 for m in (agent.messages if agent else []) if m["role"] == "assistant")
        return (output_doc_ids, trajectory_chunk_ids, completed, num_steps, error)

    return factory


# --------------------------------------------------------------------------- output
def _write_outputs(valid_pairs: list[tuple[QAPair, str]], out_dir: str) -> None:
    """Write the validated synthetic test set: CSV (officeqa_pro schema) + structured JSON.

    ``valid_pairs`` is a list of ``(pair, qf_reasoning)``.
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "dais_synthetic_test_set.csv")
    json_path = os.path.join(out_dir, "dais_synthetic_test_set.json")

    ordered = sorted(valid_pairs, key=lambda pr: pr[0].qa_id)
    rows = []
    structured = []
    for i, (pair, reasoning) in enumerate(ordered, 1):
        uid = f"DAIS{i:04d}"
        file_ids = sorted({"_".join(d.split("_")[:-1]) for d in pair.doc_ids})
        rows.append({
            "uid": uid,
            "question": pair.question,
            "answer": " ; ".join(pair.answer),
            "source_docs": ",".join(pair.doc_ids),
            "source_files": ",".join(file_ids),
            "difficulty": "synthetic",
        })
        structured.append({
            "uid": uid,
            "qa_id": pair.qa_id,
            "question": pair.question,
            "answer_nuggets": pair.answer,
            "chunk_ids": pair.chunk_ids,
            "doc_ids": pair.doc_ids,
            "source_files": file_ids,
            "quality_filter_reasoning": reasoning,
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["uid", "question", "answer", "source_docs", "source_files", "difficulty"]
        )
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(structured, f, indent=2)
    print(f"Wrote {len(rows)} validated QA pairs to:\n  {csv_path}\n  {json_path}")


# --------------------------------------------------------------------------- resume
def load_deduped_pairs(trace_dir: str, seeds: list[int] | None = None) -> dict[int, dict[str, QAPair]]:
    """Reconstruct the post-dedup kept pairs from a prior run's trace dir.

    For each seed with BOTH ``{seed}_qa_pairs.json`` (full generated pairs) and
    ``{seed}_qa_pairs_dedup.json`` (the dedup decision), rebuild the kept ``QAPair``
    objects by filtering the full pairs to the dedup ``kept_qa_ids``. Lets a killed run
    resume straight into rollouts without regenerating or re-deduping. Seeds that were
    generated but never deduped are skipped (no dedup decision to honour).
    """
    if seeds is None:
        seeds = sorted(
            int(os.path.basename(p).split("_")[0])
            for p in glob.glob(os.path.join(trace_dir, "*_qa_pairs_dedup.json"))
        )
    kept_by_seed: dict[int, dict[str, QAPair]] = {}
    for seed in seeds:
        dedup_path = os.path.join(trace_dir, f"{seed}_qa_pairs_dedup.json")
        pairs_path = os.path.join(trace_dir, f"{seed}_qa_pairs.json")
        if not (os.path.exists(dedup_path) and os.path.exists(pairs_path)):
            continue
        with open(dedup_path) as f:
            kept_ids = set(json.load(f).get("kept_qa_ids", []))
        with open(pairs_path) as f:
            all_pairs = json.load(f)
        kept = {d["qa_id"]: QAPair(**d) for d in all_pairs if d["qa_id"] in kept_ids}
        if kept:
            kept_by_seed[seed] = kept
    return kept_by_seed


# --------------------------------------------------------------------------- ASAP mode
def run_asap_pipeline(
    kept_pairs_by_seed: dict[int, dict[str, QAPair]],
    rollout_cfg,
    qf_cfg,
    trace_dir: str,
    out_dir: str,
    *,
    n_rollouts: int,
    threshold: float,
    rollout_concurrency: int,
    qf_parallelism: int,
    target_passing: int,
    skip_quality_filter: bool,
    show_output: bool,
) -> None:
    """Emergency ASAP funnel: per-pair, fully pipelined, incremental writes.

    Replaces the barriered phases 2-5. Each pair is independent (no global-mean
    threshold): a rollout binarizes to 1 iff ``doc_output_recall > threshold``, and a
    pair is "challenging" iff its scored rollouts DISAGREE (not all 1, not all 0). The
    moment a pair's rollouts finish it is binarized and (unless ``skip_quality_filter``)
    handed to the quality filter; every accepted pair is written to the output CSV/JSON
    immediately, so passing pairs accumulate live and the run can be Ctrl-C'd anytime
    with a valid partial test set. With ``target_passing > 0`` the run stops early once
    that many pairs pass.
    """
    # Flatten pairs, interleaving across seeds so early results are diverse.
    per_seed = {s: list(d.values()) for s, d in kept_pairs_by_seed.items()}
    cols = [[(s, i, p) for i, p in enumerate(ps)] for s, ps in per_seed.items()]
    flat: list[tuple[int, int, QAPair]] = [
        item for group in zip_longest(*cols) for item in group if item is not None
    ]
    print(f"[asap] {len(flat)} pairs -> rollouts({n_rollouts}/pair, fixed threshold "
          f"{threshold}) -> {'(QF skipped) ' if skip_quality_filter else 'quality filter -> '}"
          f"incremental write to {out_dir}"
          f"{f'; stop after {target_passing} passing' if target_passing else ''}.", flush=True)

    lock = threading.Lock()
    valid_pairs: list[tuple[QAPair, str]] = []
    state = {"pairs_done": 0, "challenging": 0, "all_pass": 0, "all_fail": 0,
             "unscored": 0, "passed": 0, "qf_fail": 0}
    stop = threading.Event()
    pair_records: dict[tuple[int, int], list] = {}
    pair_completions: dict[tuple[int, int], int] = {}
    qf_pool = ThreadPoolExecutor(max_workers=max(1, qf_parallelism))
    qf_futs = []

    def _accept(pair: QAPair, reasoning: str) -> None:
        valid_pairs.append((pair, reasoning))
        state["passed"] += 1
        _write_outputs(sorted(valid_pairs, key=lambda pr: pr[0].qa_id), out_dir)
        print(f"  [asap] ✅ accepted {pair.qa_id} (passed={state['passed']}, "
              f"pairs_done={state['pairs_done']}/{len(flat)}, challenging={state['challenging']})",
              flush=True)
        if target_passing and state["passed"] >= target_passing:
            stop.set()

    def _judge(seed: int, pair_idx: int, pair: QAPair, recs: list) -> None:
        if stop.is_set():
            return
        persist_seed_rollouts(seed, recs,
                              os.path.join(trace_dir, f"{seed}_qa{pair_idx}_rollouts.json"))
        bins = [int(r.doc_output_recall > threshold)
                for r in recs if r.doc_output_recall is not None]
        if len(bins) < 2 or all(b == 1 for b in bins) or all(b == 0 for b in bins):
            with lock:
                if not bins:
                    state["unscored"] += 1
                elif bins and all(b == 1 for b in bins):
                    state["all_pass"] += 1
                else:
                    state["all_fail"] += 1
            return
        with lock:
            state["challenging"] += 1
        if skip_quality_filter:
            with lock:
                _accept(pair, "asap: accepted on rollout disagreement (QF skipped)")
            return
        res = run_quality_filter_for_pair(pair, recs, qf_cfg, trace_dir, show_output)
        with lock:
            if res.valid:
                _accept(pair, res.reasoning)
            else:
                state["qf_fail"] += 1

    with ThreadPoolExecutor(max_workers=max(1, rollout_concurrency)) as rpool:
        roll_futs = {}
        for seed, pair_idx, pair in flat:
            pair_records[(seed, pair_idx)] = []
            pair_completions[(seed, pair_idx)] = 0
            for r in range(n_rollouts):
                roll_futs[rpool.submit(run_single_rollout, seed, pair_idx, r, pair,
                                       rollout_cfg, trace_dir, show_output)] = (seed, pair_idx, pair)
        for fut in as_completed(roll_futs):
            seed, pair_idx, pair = roll_futs[fut]
            key = (seed, pair_idx)
            try:
                rec = fut.result()
                pair_records[key].append(rec)
            except Exception as e:  # noqa: BLE001
                print(f"  [asap] rollout error ({pair.qa_id}): {e}")
            with lock:
                pair_completions[key] += 1
                complete = pair_completions[key] == n_rollouts
                if complete:
                    state["pairs_done"] += 1
            if complete and not stop.is_set():
                qf_futs.append(qf_pool.submit(_judge, seed, pair_idx, pair,
                                              list(pair_records[key])))

    for qf in as_completed(qf_futs):
        try:
            qf.result()
        except Exception as e:  # noqa: BLE001
            print(f"  [asap] judge error: {e}")
    qf_pool.shutdown(wait=True)
    print(f"[asap] DONE. accepted={state['passed']} written to {out_dir} | "
          f"challenging={state['challenging']} qf_fail={state['qf_fail']} "
          f"all_pass={state['all_pass']} all_fail={state['all_fail']} "
          f"unscored={state['unscored']} pairs_done={state['pairs_done']}/{len(flat)}")


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic QA test set for the DAIS corpus.")
    parser.add_argument("--clean-page-map", required=True, help="DAIS clean_page_map.json (from clean_corpus).")
    parser.add_argument("--chroma-collection-name", required=True, help="Served corpus collection name.")
    parser.add_argument("--qa-collection-name", default="dais-qa-embeddings-qwen",
                        help="Collection used by dedup for QA Q/A embeddings (created if absent).")
    parser.add_argument("--chroma-host", default=None, help="ChromaDB server host (default: config/env).")
    parser.add_argument("--chroma-port", type=int, default=None, help="ChromaDB server port (default: config/env).")
    parser.add_argument("--few-shot-csv", default="officeqa_pro.csv", help="OfficeQA benchmark CSV (few-shot).")
    parser.add_argument("--out-dir", default="dais_synthetic_qa", help="Output dir for the test set + traces.")
    parser.add_argument("--dais-pdf-dir", default=str(pathlib.Path.home() / "dais" / "pdfs"),
                        help="DAIS source PDFs (<file_id>.pdf) — lets the agents' view_figure render pages.")
    parser.add_argument("--page-renders-dir", default=str(pathlib.Path.home() / "dais" / "page_renders"),
                        help="Pre-rendered page PNGs (from skunk.dais.render_corpus); view_figure serves "
                             "these instead of rasterizing the PDF (fitz fallback if absent).")

    parser.add_argument("--gen-model-id", default=DEFAULT_GEN_MODEL)
    parser.add_argument("--rollout-model-id", default=DEFAULT_ROLLOUT_MODEL)
    parser.add_argument("--qf-model-id", default=DEFAULT_ROLLOUT_MODEL)
    parser.add_argument("--dedup-judge-model-id", default=DEFAULT_GEN_MODEL,
                        help="Dedup judge model (runs via OpenRouter; use a google/ id).")
    parser.add_argument("--emb-model-id", default=DEFAULT_EMB_MODEL)
    parser.add_argument("--provider", choices=["genai", "openrouter"], default="genai",
                        help="LLM provider for generation/rollout/QF agents (default genai; "
                             "switch to openrouter wholesale if Gemini rate-limits persist).")

    parser.add_argument("--num-new-seeds", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--n-qa-pairs", type=int, default=8)
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--n-rollouts", type=int, default=8)
    parser.add_argument("--gen-parallelism", type=int, default=8)
    parser.add_argument("--rollout-concurrency", type=int, default=16)
    parser.add_argument("--qf-parallelism", type=int, default=8)
    parser.add_argument("--resume-from-dedup", action="store_true",
                        help="Skip phase 1 (generate + dedup); reload the deduped pairs already "
                             "written under <out-dir>/traces and resume straight into rollouts. "
                             "Seeds generated but not yet deduped are skipped.")
    parser.add_argument("--asap", action="store_true",
                        help="EMERGENCY mode: per-pair pipelined funnel with a FIXED rollout "
                             "threshold (no global-mean barrier) and incremental writes. Accepted "
                             "pairs are written the moment they pass, so output accumulates live. "
                             "Replaces the barriered phases 2-5.")
    parser.add_argument("--asap-threshold", type=float, default=0.5,
                        help="ASAP fixed binarization threshold on doc_output_recall (a rollout "
                             "'passes' iff recall > this). A pair is kept iff its rollouts disagree.")
    parser.add_argument("--target-passing", type=int, default=0,
                        help="ASAP: stop once this many pairs have passed (0 = process all).")
    parser.add_argument("--skip-quality-filter", action="store_true",
                        help="ASAP: accept any challenging (disagreeing) pair WITHOUT the quality "
                             "filter — fastest, lower precision.")
    parser.add_argument("--show-output", action="store_true")
    args = parser.parse_args()

    base_cfg = SkunkConfig.from_env()
    host = args.chroma_host or base_cfg.chroma_server_host
    port = args.chroma_port or base_cfg.chroma_server_port
    trace_dir = os.path.join(args.out_dir, "traces")
    os.makedirs(trace_dir, exist_ok=True)

    # --- shared resources ---
    client = make_chroma_client(host, port)
    collection = client.get_collection(name=args.chroma_collection_name)
    if not args.resume_from_dedup:  # dedup-only resources; rollouts/QF don't need them
        qa_collection = client.get_or_create_collection(name=args.qa_collection_name)
        or_client = OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])

    print("Loading corpus maps...")
    document_map = load_document_map(args.clean_page_map)
    chunk_id_to_doc_id = load_chunk_id_to_doc_id(collection)
    doc_id_to_chunk_ids = invert_chunk_map(chunk_id_to_doc_id)
    examples_pool = load_few_shot_questions(args.few_shot_csv)
    special_notes = DAIS_CORPUS_NOTES
    print(f"  {len(document_map)} docs, {len(chunk_id_to_doc_id)} chunks, {len(examples_pool)} few-shot examples.")

    if not args.resume_from_dedup:
        dedup_cfg = DedupConfig(
            qa_collection=qa_collection,
            emb_model_id=args.emb_model_id,
            or_client=or_client,
            judge_model_id=args.dedup_judge_model_id,
            judge_prompt_template=DEDUP_JUDGE_PROMPT,
        )
    rollout_cfg = RolloutConfig(
        model_id=args.rollout_model_id,
        emb_model_id=args.emb_model_id,
        document_map=document_map,
        chroma_collection=collection,
        max_steps=AGENT_MAX_STEPS,
        max_pages_per_tool_call=MAX_PAGES_PER_TOOL_CALL,
        special_notes=special_notes,
        n_rollouts=args.n_rollouts,
        rollout_parallelism=args.rollout_concurrency,
        doc_id_to_chunk_ids=doc_id_to_chunk_ids,
        chunk_id_to_doc_id=chunk_id_to_doc_id,
        agent_factory=make_gemini_rollout_agent_factory(
            args.rollout_model_id, args.provider, args.emb_model_id,
            document_map, collection, special_notes,
            args.dais_pdf_dir, args.page_renders_dir,
        ),
        tinker_backend=None,
        task_solver_cfg=None,
    )
    qf_cfg = QualityFilterConfig(
        model_id=args.qf_model_id,
        emb_model_id=args.emb_model_id,
        document_map=document_map,
        chroma_collection=collection,
        system_prompt_template=BENCHMARK_QUALITY_FILTER_SYSTEM_PROMPT["dais"],
        special_notes=special_notes,
        binarization_mode="doc-recall",
        pdf_dir=args.dais_pdf_dir,
        page_renders_dir=args.page_renders_dir,
    )

    if args.resume_from_dedup:
        # --- resume: reload deduped pairs from a prior run, skip phase 1 entirely ---
        kept_pairs_by_seed = load_deduped_pairs(trace_dir)
        seeds = sorted(kept_pairs_by_seed)
        if not kept_pairs_by_seed:
            raise SystemExit(f"--resume-from-dedup: no deduped pairs found under {trace_dir}.")
        n_kept = sum(len(d) for d in kept_pairs_by_seed.values())
        print(f"Resuming from dedup: {n_kept} kept pairs across {len(seeds)} seeds "
              f"(skipped phase 1: generate + dedup).")
    else:
        seeds = list(range(args.start_seed, args.start_seed + args.num_new_seeds))

        # --- phase 1: generate + dedup (parallel across seeds; dedup serialized) ---
        print(f"Phase 1: generate + dedup over {len(seeds)} seeds...")
        dedup_lock = threading.Lock()
        kept_pairs_by_seed: dict[int, dict[str, QAPair]] = {}

        def _gen_and_dedup(seed: int):
            pairs = generate_one(
                seed, args.gen_model_id, args.provider, examples_pool, document_map, collection,
                chunk_id_to_doc_id, args.emb_model_id, trace_dir, special_notes,
                DAIS_DATAGEN_GUIDANCE, args.n_examples, args.n_qa_pairs, args.show_output,
                args.dais_pdf_dir, args.page_renders_dir,
            )
            with dedup_lock:
                kept, funnel = dedup_batch(pairs, dedup_cfg)
            with open(os.path.join(trace_dir, f"{seed}_qa_pairs_dedup.json"), "w") as f:
                json.dump({"seed": seed, "n_input": len(pairs), "n_kept": len(kept),
                           "kept_qa_ids": [p.qa_id for p in kept], "funnel": funnel}, f, indent=2)
            return seed, kept

        with ThreadPoolExecutor(max_workers=max(1, args.gen_parallelism)) as pool:
            for fut in as_completed([pool.submit(_gen_and_dedup, s) for s in seeds]):
                seed, kept = fut.result()
                kept_pairs_by_seed[seed] = {p.qa_id: p for p in kept}
        n_kept = sum(len(d) for d in kept_pairs_by_seed.values())
        print(f"  dedup kept {n_kept} pairs.")

    if args.asap:
        # EMERGENCY: per-pair pipelined funnel + incremental writes (replaces phases 2-5).
        run_asap_pipeline(
            kept_pairs_by_seed, rollout_cfg, qf_cfg, trace_dir, args.out_dir,
            n_rollouts=args.n_rollouts, threshold=args.asap_threshold,
            rollout_concurrency=args.rollout_concurrency, qf_parallelism=args.qf_parallelism,
            target_passing=args.target_passing, skip_quality_filter=args.skip_quality_filter,
            show_output=args.show_output,
        )
        return

    # --- phase 2: rollouts (shared executor, no per-pair barrier) ---
    print(f"Phase 2: {args.n_rollouts} Gemini rollouts/pair...")
    records_by_seed: dict[int, list] = {s: [] for s in seeds}
    with ThreadPoolExecutor(max_workers=max(1, args.rollout_concurrency)) as pool:
        futs = {}
        for seed, kept in kept_pairs_by_seed.items():
            for pair_idx, pair in enumerate(kept.values()):
                for r in range(args.n_rollouts):
                    futs[pool.submit(run_single_rollout, seed, pair_idx, r, pair, rollout_cfg,
                                     trace_dir, args.show_output)] = seed
        for fut in as_completed(futs):
            seed = futs[fut]
            try:
                records_by_seed[seed].append(fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"  rollout error (seed={seed}): {e}")

    # --- phase 3: global threshold + all-pass/all-fail filter => "challenging" pairs ---
    print("Phase 3: binarize + filter (keep challenging pairs)...")
    _, stats = apply_threshold_and_filter(records_by_seed, binarization_mode="doc-recall")
    print(f"  threshold={stats.threshold:.3f} kept={stats.n_pairs_kept} "
          f"all_pass={stats.n_pairs_all_pass} all_fail={stats.n_pairs_all_fail}")

    challenging: list[tuple[int, QAPair, list]] = []
    for seed, recs in records_by_seed.items():
        by_qa: dict[str, list] = {}
        for r in recs:
            by_qa.setdefault(r.qa_id, []).append(r)
        for qa_id, rs in by_qa.items():
            if any(r.kept for r in rs):
                pair = kept_pairs_by_seed.get(seed, {}).get(qa_id)
                if pair is not None:
                    challenging.append((seed, pair, rs))

    # --- phase 4: quality filter (final validity gate) ---
    print(f"Phase 4: quality filter on {len(challenging)} challenging pairs...")
    valid_pairs: list[tuple[QAPair, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.qf_parallelism)) as pool:
        futs = {
            pool.submit(run_quality_filter_for_pair, pair, rs, qf_cfg, trace_dir, args.show_output): pair
            for _seed, pair, rs in challenging
        }
        for fut in as_completed(futs):
            pair = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  QF error (qa_id={pair.qa_id}): {e}")
                continue
            if res.valid:
                valid_pairs.append((pair, res.reasoning))

    # --- phase 5: write the synthetic test set ---
    _write_outputs(valid_pairs, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
