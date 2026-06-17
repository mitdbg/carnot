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
import json
import math
import os
import pathlib
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass

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


def _agent_config(model_id: str, emb_model_id: str, provider: str, max_steps: int) -> SkunkConfig:
    """SearchAgent SkunkConfig. genai uses the bare model id; openrouter the full id."""
    agent_model = model_id if provider == "openrouter" else model_id.removeprefix("google/")
    return SkunkConfig(
        agent_model_id=agent_model,
        emb_model_id=emb_model_id,
        llm_provider=provider,  # type: ignore[arg-type]
        agent_max_steps=max_steps,
        agent_max_pages_per_tool_call=MAX_PAGES_PER_TOOL_CALL,
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
    config = _agent_config(model_id, emb_model_id, provider, AGENT_MAX_STEPS)
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
        doc_ids = sorted({chunk_id_to_doc_id[c] for c in chunk_ids if c in chunk_id_to_doc_id})
        pairs.append(QAPair(
            qa_id=f"{seed}-{i}",
            question=str(p.get("question", "")),
            answer=[str(n) for n in p.get("answer", [])],
            chunk_ids=chunk_ids,
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
):
    """``RolloutConfig.agent_factory`` that runs a *plain* SearchAgent (Gemini, no Tinker,
    no logprobs) exactly as it runs at test time — mirrors quality_filter's construction."""
    config = _agent_config(model_id, emb_model_id, provider, AGENT_MAX_STEPS)
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
    qa_collection = client.get_or_create_collection(name=args.qa_collection_name)
    or_client = OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])

    print("Loading corpus maps...")
    document_map = load_document_map(args.clean_page_map)
    chunk_id_to_doc_id = load_chunk_id_to_doc_id(collection)
    doc_id_to_chunk_ids = invert_chunk_map(chunk_id_to_doc_id)
    examples_pool = load_few_shot_questions(args.few_shot_csv)
    special_notes = DAIS_CORPUS_NOTES
    print(f"  {len(document_map)} docs, {len(chunk_id_to_doc_id)} chunks, {len(examples_pool)} few-shot examples.")

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
    )

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
