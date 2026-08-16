import argparse
import asyncio
import json
import os
import pathlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from chromadb.api.models.Collection import Collection
from jinja2 import Environment, StrictUndefined
from skunk.chroma_client import make_chroma_client
from skunk.common import ExecutionContext
from skunk.config import InferenceConfig, LookupAgentConfig, OrchestratorConfig, SearchAgentConfig, StorageConfig
from skunk.llm_client import LLMClient
from skunk.multi_turn_agent import MultiTurnAgent, Tool
from skunk.search_agent.retrieval_state import RetrievalState
from skunk.search_agent.search_tools import (
    GrepCorpusTool,
    ReadDocumentTool,
    SearchCorpusTool,
    ViewFigureTool,
)
from skunk.storage.document_map import DocumentMap

from dataset_datagen.gen_stats import GenStats, aggregate, events_latency, usage_snapshot

_ENV = Environment(
    autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined
)
LLM_PRICES = {
    "anthropic/claude-haiku-4.5": {"in": 1.0, "out": 5.0, "cached": 0.1},
    "anthropic/claude-sonnet-5": {"in": 2.0, "out": 10.0, "cached": 0.2},
    "openai/gpt-5.6-luna": {"in": 0.2, "out": 1.2, "cached": 0.02},
    "openai/gpt-5.6-terra": {"in": 2.0, "out": 12.0, "cached": 0.2},
    "gemini-3.7-flash": {"in": 0.75, "out": 3.75, "cached": 0.075},
    "gemini-3.6-flash": {"in": 0.75, "out": 3.75, "cached": 0.075},
    "gemini-3.5-flash": {"in": 1.5, "out": 9.0, "cached": 0.15},
    "gemini-3.5-flash-lite": {"in": 0.3, "out": 2.5, "cached": 0.03},
    "gemini-3.1-flash-lite": {"in": 0.25, "out": 1.5, "cached": 0.025},
    "gemini-3.1-pro": {"in": 2.0, "out": 12.0, "cached": 0.20},
    "qwen3-embedding-8b": {"in": 0.01, "out": 0.00, "cached": 0.00},
    "google/gemma-3-12b-it": {"in": 0.05, "out": 0.15, "cached": 0.00},
    "nvidia/nemotron-3.5-lightning": {"in": 0.08, "out": 0.20, "cached": 0.04},
    "qwen/qwen3.6-35b-a3b": {"in": 0.15, "out": 1.0, "cached": 0.05},
    "qwen/qwen3.6-27b": {"in": 0.285, "out": 2.4, "cached": 0.15},
}
LLM_CONTEXT_LIMITS = {
    "anthropic/claude-haiku-4.5": 200000,
    "anthropic/claude-sonnet-5": 1000000,
    "openai/gpt-5.6-luna": 1050000,
    "openai/gpt-5.6-terra": 1050000,
    "gemini-3.7-flash": 1048576,
    "gemini-3.6-flash": 1048576,
    "gemini-3.5-flash": 1048576,
    "gemini-3.5-flash-lite": 1048576,
    "gemini-3.1-flash-lite": 1048576,
    "gemini-3.1-pro": 1048576,
    "nvidia/nemotron-3.5-lightning": 262144,
    "qwen/qwen3.6-35b-a3b": 262144,
    "qwen/qwen3.6-27b": 262144,
}

SOLVER_MODEL_ID = "openai/gpt-5.6-terra"
SOLVER_DUMMY_EMB_MODEL_ID = "qwen3-embedding-8b"
TOP_K_NEIGHBORS = 20

MAX_INQUIRY_TRIES = 5
MAX_QA_PAIR_TRIES = 5

LINE_OF_INQUIRY_LOCK = threading.Lock()
QA_PAIR_LOCK = threading.Lock()

CARNOT_BASE_PATH = pathlib.Path(__file__).parent.parent
OFFICEQA_QUESTIONS_PATH = CARNOT_BASE_PATH / "qatfd/benchmarks/officeqa/officeqa_pro.csv"
OFFICEQA_DEV_SET_PATH = CARNOT_BASE_PATH / "qatfd/benchmarks/officeqa/officeqa_splits.json"
OFFICEQA_CLEAN_PAGE_MAP_PATH = CARNOT_BASE_PATH / "qatfd/benchmarks/officeqa/treasury_bulletins_cleaned/clean_page_map.json"
OFFICEQA_CLEAN_DATA_PATH = CARNOT_BASE_PATH / "qatfd/benchmarks/officeqa/treasury_bulletins_cleaned/"
OFFICEQA_PDF_DIR = CARNOT_BASE_PATH / "qatfd/benchmarks/officeqa/treasury_bulletin_pdfs/"

DATAGEN_PROMPT_FILE = pathlib.Path(__file__).parent / "prompts.yaml"
with DATAGEN_PROMPT_FILE.open() as _f:
    _DATAGEN_PROMPTS = yaml.safe_load(_f)

DATAGEN_LINE_OF_INQUIRY_SYSTEM_PROMPT: str = _DATAGEN_PROMPTS["datagen_line_of_inquiry_system_prompt"]
DATAGEN_LINE_OF_INQUIRY_USER_PROMPT: str = _DATAGEN_PROMPTS["datagen_line_of_inquiry_user_prompt"]
DATAGEN_FOLLOW_UP_QUESTION_SYSTEM_PROMPT: str = _DATAGEN_PROMPTS["datagen_follow_up_question_system_prompt"]
DATAGEN_FOLLOW_UP_QUESTION_USER_PROMPT: str = _DATAGEN_PROMPTS["datagen_follow_up_question_user_prompt"]
DATAGEN_SOLVER_SYSTEM_PROMPT: str = _DATAGEN_PROMPTS["datagen_solver_system_prompt"]
DATAGEN_SOLVER_USER_PROMPT: str = _DATAGEN_PROMPTS["datagen_solver_user_prompt"]
DATAGEN_EQUIVALENCE_SYSTEM_PROMPT: str = _DATAGEN_PROMPTS["datagen_answer_equivalence_system_prompt"]
DATAGEN_EQUIVALENCE_USER_PROMPT: str = _DATAGEN_PROMPTS["datagen_answer_equivalence_user_prompt"]
DATAGEN_OFFICEQA_SPECIAL_NOTES: str = _DATAGEN_PROMPTS["datagen_officeqa_special_notes"]
OFFICEQA_DATAGEN_GUIDANCE: str = _DATAGEN_PROMPTS["officeqa_datagen_guidance"]

DEDUP_LINE_OF_INQUIRY_JUDGE_PROMPT = _DATAGEN_PROMPTS["dedup_line_of_inquiry_judge_prompt"]
DEDUP_QA_PAIR_JUDGE_PROMPT = _DATAGEN_PROMPTS["dedup_qa_pair_judge_prompt"]

# per-benchmark statistics for the number of relevant chunks per question.
# Values are (mean, std) drawn from Table 2 of the KARL paper.
# Used to sample a target chunk count from N(mean, std) for each generation.
BENCHMARK_CHUNKS_STATS: dict[str, tuple[float, float]] = {
    "officeqa":        (1.8,  1.1),
    "browsecomp-plus": (2.9,  2.0),
}

# imports the datagen agent is allowed to use inside its python tool blocks,
# so it can compute statistics (means, regressions, etc.) from raw values it
# retrieves from the corpus rather than just regurgitating pre-computed numbers.
DATAGEN_AUTHORIZED_IMPORTS = ["math", "statistics", "numpy", "scipy", "statsmodels"]

# source_docs URL -> (month, year, page)
_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}
_URL_RE = re.compile(
    r"/(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)"
    r"-(?P<year>\d{4})[^?]*\?page=(?P<page>\d+)",
    re.IGNORECASE,
)

class InquirySynthAgent(MultiTurnAgent):
    """MultiTurnAgent specialised for synthesizing line of inquiry for a given question-answer pair."""

    name = "inquiry_synth_agent"

    def validate_final_answer(self, payload: object, observations: list[str]) -> str | None:
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("line_of_inquiry"), str)
            or not isinstance(payload.get("doc_ids"), list)
        ):
            return 'Emit a JSON object {"line_of_inquiry": ..., "doc_ids": [...]}.'
        return None

class FollowUpQuestionSynthAgent(MultiTurnAgent):
    """MultiTurnAgent specialised for synthesizing follow-up questions for a given question-answer pair."""

    name = "follow_up_question_synth_agent"
    authorized_imports = DATAGEN_AUTHORIZED_IMPORTS

    def validate_final_answer(self, payload: object, observations: list[str]) -> str | None:
        if not isinstance(payload, dict) or not all(k in payload for k in ("question", "answer", "doc_ids")):
            return 'Emit a JSON object {"question": ..., "answer": [...], "doc_ids": [...]}.'
        return None

class SolverAgent(MultiTurnAgent):
    """MultiTurnAgent specialised for solving questions given their groundtruth docs."""

    name = "solver_agent"
    authorized_imports = DATAGEN_AUTHORIZED_IMPORTS

    def validate_final_answer(self, payload: object, observations: list[str]) -> str | None:
        if not isinstance(payload, dict) or "answer" not in payload:
            return 'Emit a JSON object {"answer": [...]}.'
        return None

class EquivalenceAgent(MultiTurnAgent):
    """MultiTurnAgent specialised for assessing whether two answers are equivalent."""

    name = "equivalence_agent"

    def validate_final_answer(self, payload: object, observations: list[str]) -> str | None:
        if not isinstance(payload, dict) or "equal" not in payload:
            return 'Emit a JSON object {"equal": "TRUE" | "FALSE"}.'
        return None

class DuplicateAgent(MultiTurnAgent):
    """MultiTurnAgent specialised for assessing whether lines of inquiry or QA pairs are duplicates of existing ones."""

    name = "duplicate_agent"

    def validate_final_answer(self, payload: object, observations: list[str]) -> str | None:
        if not isinstance(payload, dict) or "duplicate" not in payload or "reasoning" not in payload:
            return 'Emit a JSON object {"duplicate": "ID" | None, "reasoning": "reasoning str" | None}.'
        return None

def get_page_ids(source_docs: str) -> list[str]:
    """Convert source_docs column to list of page ids."""
    keys: list[str] = []
    for m in _URL_RE.finditer(source_docs):
        mm = _MONTH_MAP[m.group("month").lower()]
        keys.append(f"{m.group('year')}_{mm}_{int(m.group('page'))}")

    return keys

def load_questions(benchmark: str, split: str) -> list[dict]:
    if benchmark == "officeqa":
        split_qids = {}
        with open(OFFICEQA_DEV_SET_PATH) as f:
            split_qids = json.load(f)

        qids = split_qids[split]
        df = pd.read_csv(OFFICEQA_QUESTIONS_PATH)
        df = df[df.uid.isin(qids)]
        questions = [
            {"qid": row["uid"], "question": row["question"], "answer": row["answer"], "page_ids": get_page_ids(row["source_docs"])}
            for _, row in df.iterrows()
        ]
        return questions
    else:
        raise Exception("Not implemented")

def load_document_map(benchmark: str) -> dict[str, str]:
    """Load the mapping from document ID to document text."""
    if benchmark == "officeqa":
        with open(OFFICEQA_CLEAN_PAGE_MAP_PATH) as f:
            clean_page_map = json.load(f)

        document_map: dict[str, str] = {}
        for doc_id, entry in clean_page_map.items():
            rel_path = entry[0]
            filepath = os.path.join(OFFICEQA_CLEAN_DATA_PATH, os.path.basename(rel_path))
            with open(filepath) as f:
                document_map[doc_id] = f.read()

        return document_map
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")

def _create_docs_str(document_map: DocumentMap, doc_ids: list[str]) -> str:
    """Create a string containing the text of the documents with the given IDs.

    Unknown ids are skipped rather than raising an exception.
    """
    docs_str = ""
    for doc_id in doc_ids:
        page_text = document_map.get(doc_id)
        if page_text is None:
            continue
        docs_str += f"\n\n=== Document ID: {doc_id} ===\n{page_text}"
    return docs_str

async def _is_inquiry_unique(inquiry: str, question: dict, ctx: ExecutionContext, collection: Collection, lock: threading.Lock) -> tuple[bool, GenStats | None]:
    """Check if the inquiry is unique in the given ChromaDB collection.

    Uses a lock to ensure thread-safe access to the collection.
    """
    with lock:
        # step 1: query closest neighbors by inquiry
        embedding = ctx.llm_client.embed_query(inquiry)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=TOP_K_NEIGHBORS,
            include=["metadatas", "documents"],
        )
        ids = results['ids'][0]
        inquiries = results['documents'][0]
        questions = [m['question'] for m in results['metadatas'][0]]
        answers = [m['answer'] for m in results['metadatas'][0]]
        neighbor_lines_of_inquiry = [
            {"id": _id, "line_of_inquiry": inquiry, "question": question, "answer": answer}
            for _id, inquiry, question, answer in zip(ids, inquiries, questions, answers, strict=True)
        ]

        # for the first line of inquiry, skip duplicate judgement
        duplicate, stats = False, None
        if len(inquiries) > 0:
            # step 2: judge inquiry vs. closest neighbors
            duplicate_agent_id = f"duplicate_agent_{question['qid']}"
            system_prompt = _ENV.from_string(DEDUP_LINE_OF_INQUIRY_JUDGE_PROMPT).render(
                generated_line_of_inquiry=inquiry,
                seed_question=question["question"],
                seed_answer=question["answer"],
                lines_of_inquiry=neighbor_lines_of_inquiry,
            )
            duplicate_agent = DuplicateAgent(
                tools=[],
                max_steps=1,
                max_misfires=3,
                agent_id=duplicate_agent_id,
                system_prompt_override=system_prompt,
            )
            out, stats = await _run_agent(
                duplicate_agent, ctx, "output:\n", kind="line_of_inquiry_dedup", qid=str(question["qid"])
            )
            duplicate = out["duplicate"] is True
            if duplicate:
                stats.error = out["reasoning"]

        # step 3: insert inquiry into database (if it survives judging)
        if not duplicate:
            collection.add(ids=[str(question["id"])], documents=[inquiry], metadatas=[{"question": question["question"], "answer": question["answer"]}])

        is_unique = not duplicate
        return is_unique, stats

async def _is_qa_pair_unique(qa_pair: dict, qid: str, idx: int, solvable: bool, ctx: ExecutionContext, collection: Collection, lock: threading.Lock) -> tuple[bool, GenStats | None]:
    """Check if the qa_pair is unique in the given ChromaDB collection.

    Uses a lock to ensure thread-safe access to the collection.
    """
    # skip LLM check if qa pair is not solvable
    if not solvable:
        return False, None

    synth_qa_pair_id = f"{qid}_{idx}"
    with lock:
            # step 1: query closest neighbors by question
            embedding = ctx.llm_client.embed_query(qa_pair["question"])
            results = collection.query(
                query_embeddings=[embedding],
                n_results=TOP_K_NEIGHBORS,
                where={"kind": "question"},
                include=["metadatas", "documents"],
            )
            ids = results['ids'][0]
            questions = results['documents'][0]
            answers = [m['answer'] for m in results['metadatas'][0]]
            neighbor_qa_pairs = [
                {"qid": _id, "question": question, "answer": answer}
                for _id, question, answer in zip(ids, questions, answers, strict=True)
            ]
    
            # step 2: judge qa-pair vs. closest neighbors
            duplicate_agent_id = f"duplicate_agent_{synth_qa_pair_id}"
            system_prompt = _ENV.from_string(DEDUP_QA_PAIR_JUDGE_PROMPT).render(
                generated_question=qa_pair["question"],
                generated_answer=qa_pair["answer"],
                qa_pairs=neighbor_qa_pairs,
            )
            duplicate_agent = DuplicateAgent(
                tools=[],
                max_steps=1,
                max_misfires=3,
                agent_id=duplicate_agent_id,
                system_prompt_override=system_prompt,
            )
            out, stats = await _run_agent(
                duplicate_agent, ctx, "output:\n", kind="qa_pair_dedup", qid=synth_qa_pair_id
            )
            duplicate = out["duplicate"] is True
            if duplicate:
                stats.error = out["reasoning"]
    
            # step 3: insert qa-pair into database (if it survives judging)
            if not duplicate:
                collection.add(
                    ids=[synth_qa_pair_id],
                    documents=[qa_pair["question"]],
                    metadatas=[{"qa_id": synth_qa_pair_id, "kind": "question", "is_synthetic": True, "answer": qa_pair["answer"]}],
                )
    
            is_unique = not duplicate
            return is_unique, stats

async def _is_solvable(qa_pair: dict, qid: str, idx: int, ctx: ExecutionContext, document_map: DocumentMap) -> tuple[bool, list[GenStats]]:
    """Check whether an agent given the question and the groundtruth documents can solve the question.
    
    The question will be discarded if it cannot be solved with the groundtruth documents.
    """
    # use powerful model for verifying question-answer pairs
    inference_config = InferenceConfig(
        llm_provider="openrouter",
        llm_model=SOLVER_MODEL_ID,
        emb_provider="openrouter",
        emb_model_id=SOLVER_DUMMY_EMB_MODEL_ID,
        llm_max_retries=5,
        llm_retry_initial_delay_s=1.0,
        llm_model_rpm={},
        llm_default_rpm=1000,
        llm_model_tpm={},
        llm_default_tpm=None,
        llm_prices=LLM_PRICES,
        llm_context_limits=LLM_CONTEXT_LIMITS,
    )
    solver_llm_client = LLMClient(config=inference_config)

    # store original llm_client and then swap in solver's
    orig_llm_client = ctx.llm_client
    ctx.llm_client = solver_llm_client

    try:
        # create solver agent
        solver_agent_id = f"solver_agent_{qid}_{idx}"
        system_prompt = _ENV.from_string(DATAGEN_SOLVER_SYSTEM_PROMPT).render(
            max_steps=ctx.config.search.agent_max_steps,
            special_notes=DATAGEN_OFFICEQA_SPECIAL_NOTES,
        )
        solver_agent = SolverAgent(
            tools=[],
            max_steps=ctx.config.search.agent_max_steps,
            max_misfires=ctx.config.search.agent_max_misfires,
            agent_id=solver_agent_id,
            system_prompt_override=system_prompt,
        )

        # run agent to generate solution
        groundtruth_docs = _create_docs_str(document_map, qa_pair["doc_ids"])
        input_str = _ENV.from_string(DATAGEN_SOLVER_USER_PROMPT).render(
            question=qa_pair["question"],
            groundtruth_docs=groundtruth_docs,
        )
        out, stats = await _run_agent(
            solver_agent, ctx, input_str, kind="solver", qid=qid
        )

        # set the original client back; (we can use the original model for equivalence checking
        # because it is a very simple task
        ctx.llm_client = orig_llm_client

        # return True if the answers are syntactically equivalent
        if out["answer"] == qa_pair["answer"]:
            return True, [stats]

        # return True if the answers are semantically equiavlent
        equivalence_agent_id = f"equivalence_agent_{qid}_{idx}"
        equivalence_agent = EquivalenceAgent(
            tools=[],
            max_steps=ctx.config.search.agent_max_steps,
            max_misfires=ctx.config.search.agent_max_misfires,
            agent_id=equivalence_agent_id,
            system_prompt_override=DATAGEN_EQUIVALENCE_SYSTEM_PROMPT,
        )
        input_str = _ENV.from_string(DATAGEN_EQUIVALENCE_USER_PROMPT).render(
            question=qa_pair["question"],
            groundtruth_docs=groundtruth_docs,
            answer=qa_pair["answer"],
            pred_answer=out["answer"],
        )
        out, equiv_stats = await _run_agent(
            equivalence_agent, ctx, input_str, kind="solver", qid=qid
        )

        # TODO: support non-json outputs from agents? I think we have a way to override parse_step() which expects JSON
        return "true" in out["equal"].lower(), [stats, equiv_stats]

    # make sure we set the original llm client back
    finally:
        ctx.llm_client = orig_llm_client

async def _run_agent(
    agent: MultiTurnAgent,
    ctx: ExecutionContext,
    input_str: str,
    *,
    kind: str,
    qid: str,
    idx: int | None = None,
) -> tuple[dict, GenStats]:
    """Run one agent and return its payload plus the cost/latency it incurred.

    Cost comes from the agent's own `UsageTracker` bucket (keyed by its `agent_id`), so it
    is exact even though every agent in the run shares one `LLMClient`. LLM latency is
    recovered from the slice of this question's event stream the agent produced — valid
    because the agents within one question run sequentially on one `ExecutionContext`.
    """
    events_start = len(ctx.events)
    t0 = time.monotonic()
    try:
        out: dict = await agent.call(ctx, input_str)
        error = None
    except Exception as e:  # keep one failed generation from killing the whole question
        out = {}
        error = f"{type(e).__name__}: {e}"
        ctx.emit(f"generation_failed kind={kind} agent_id={agent.agent_id} error={error!r}")
    stats = usage_snapshot(
        ctx.llm_client,
        str(agent.agent_id),
        kind=kind,
        qid=qid,
        idx=idx,
        wall_latency_s=time.monotonic() - t0,
        llm_latency_s=events_latency(ctx.events, events_start),
    )
    stats.terminate_state = "error" if error else agent.terminate_state
    stats.error = error
    return out, stats

async def _generate_line_of_inquiry(
    question: dict, ctx: ExecutionContext, collection: Collection, document_map: DocumentMap
) -> tuple[dict, GenStats]:
    # instantiate inquiry agent. `agent_id` doubles as the UsageTracker bucket key for both
    # the agent's own LLM steps and its tools' embedding calls, so it must be unique.
    inquiry_agent_id = f"inquiry_agent_{question['qid']}"
    state = RetrievalState()
    tools: list[Tool] = [
        SearchCorpusTool(collection, ctx.llm_client, state, ctx, usage_key=inquiry_agent_id),
        GrepCorpusTool(collection, state),
        ReadDocumentTool(document_map, state),
    ]
    if ctx.config.storage.pdf_dir:
        tools.append(ViewFigureTool(document_map, ctx.config.storage.pdf_dir))
    tools_prompt_str = "\n\n".join(t.doc for t in tools)
    system_prompt = _ENV.from_string(DATAGEN_LINE_OF_INQUIRY_SYSTEM_PROMPT).render(
        tools_prompt_str=tools_prompt_str,
        max_steps=ctx.config.search.agent_max_steps,
        special_notes=DATAGEN_OFFICEQA_SPECIAL_NOTES,
    )
    inquiry_agent = InquirySynthAgent(
        tools=tools,
        max_steps=ctx.config.search.agent_max_steps,
        max_misfires=ctx.config.search.agent_max_misfires,
        agent_id=inquiry_agent_id,
        system_prompt_override=system_prompt,
    )
    supporting_docs = _create_docs_str(document_map, question["page_ids"])
    input_str = _ENV.from_string(DATAGEN_LINE_OF_INQUIRY_USER_PROMPT).render(
        question=question["question"],
        answer=question["answer"],
        supporting_docs=supporting_docs,
    )
    out, stats = await _run_agent(
        inquiry_agent, ctx, input_str, kind="line_of_inquiry", qid=str(question["qid"])
    )
    return out, stats

async def _generate_follow_up_question(idx: int, n_docs: int, question: dict, qa_pairs: list[dict], line_of_inquiry: dict, ctx: ExecutionContext, collection: Collection, document_map: DocumentMap) -> tuple[dict, GenStats]:
    # instantiate follow-up question agent
    follow_up_agent_id = f"follow_up_agent_{question['qid']}_{idx}"
    state = RetrievalState()
    tools: list[Tool] = [
        SearchCorpusTool(collection, ctx.llm_client, state, ctx, usage_key=follow_up_agent_id),
        GrepCorpusTool(collection, state),
        ReadDocumentTool(document_map, state),
    ]
    if ctx.config.storage.pdf_dir:
        tools.append(ViewFigureTool(document_map, ctx.config.storage.pdf_dir))
    tools_prompt_str = "\n\n".join(t.doc for t in tools)
    system_prompt = _ENV.from_string(DATAGEN_FOLLOW_UP_QUESTION_SYSTEM_PROMPT).render(
        n_docs=n_docs,
        tools_prompt_str=tools_prompt_str,
        max_steps=ctx.config.search.agent_max_steps,
        special_notes=DATAGEN_OFFICEQA_SPECIAL_NOTES,
    )
    follow_up_question_agent = FollowUpQuestionSynthAgent(
        tools=tools,
        max_steps=ctx.config.search.agent_max_steps,
        max_misfires=ctx.config.search.agent_max_misfires,
        agent_id=follow_up_agent_id,
        system_prompt_override=system_prompt,
    )
    line_of_inquiry_str = line_of_inquiry["line_of_inquiry"]
    relevant_docs = _create_docs_str(document_map, line_of_inquiry["doc_ids"]) + "\n\n"
    input_str = _ENV.from_string(DATAGEN_FOLLOW_UP_QUESTION_USER_PROMPT).render(
        line_of_inquiry=line_of_inquiry_str,
        relevant_docs=relevant_docs,
        qa_pairs=[{"question": q["question"], "answer": q["answer"], "supporting_docs": _create_docs_str(document_map, q["doc_ids"])} for q in qa_pairs],
    )
    return await _run_agent(
        follow_up_question_agent, ctx, input_str,
        kind="follow_up_question", qid=str(question["qid"]), idx=idx,
    )

async def generate_follow_up_questions(
    k: int,
    docs_mean: float,
    docs_std: float,
    question: dict,
    ctx: ExecutionContext,
    collection: Collection,
    document_map: DocumentMap,
    inquiry_collection: Collection,
    qa_collection: Collection,
) -> tuple[list[dict], list[GenStats]]:
    """Generate `k` follow-up QA pairs for one seed question, with their per-generation stats.

    The returned `qa_pairs` lead with the seed question (index 0) so each follow-up sees the
    full chain; `stats` carries one `GenStats` per agent run (the line of inquiry, then one
    per follow-up), in generation order.
    """
    rng = np.random.default_rng(seed=int.from_bytes(str(question["qid"]).encode(), "big"))
    qa_pairs: list[dict] = [{"question": question["question"], "answer": question["answer"], "doc_ids": question["page_ids"]}]
    stats: list[GenStats] = []

    # generate line of inquiry (repeat until it is unique)
    line_of_inquiry, loi_stats = await _generate_line_of_inquiry(question, ctx, collection, document_map)
    stats.append(loi_stats)
    inquiry_is_unique, loi_uniq_stats = await _is_inquiry_unique(line_of_inquiry["line_of_inquiry"], question, ctx, inquiry_collection, LINE_OF_INQUIRY_LOCK)
    if loi_uniq_stats is not None:
        stats.append(loi_uniq_stats)

    inquiry_tries = 1
    while not inquiry_is_unique and inquiry_tries < MAX_INQUIRY_TRIES:
        ctx.emit(f"line_of_inquiry_not_unique qid={question['qid']} line_of_inquiry={line_of_inquiry['line_of_inquiry']!r}")
        line_of_inquiry, loi_stats = await _generate_line_of_inquiry(question, ctx, collection, document_map)
        stats.append(loi_stats)
        inquiry_is_unique, loi_uniq_stats = await _is_inquiry_unique(line_of_inquiry["line_of_inquiry"], question, ctx, inquiry_collection, LINE_OF_INQUIRY_LOCK)
        if loi_uniq_stats is not None:
            stats.append(loi_uniq_stats)
        inquiry_tries += 1

    if inquiry_tries == MAX_INQUIRY_TRIES:
        raise Exception("Exceeded MAX_INQUIRY_TRIES")

    for idx in range(k):
        n_docs = max(1, round(rng.normal(loc=docs_mean, scale=docs_std)))
        qa_pair, qa_stats = await _generate_follow_up_question(idx, n_docs, question, qa_pairs, line_of_inquiry, ctx, collection, document_map)
        stats.append(qa_stats)
        solvable, solver_stats = await _is_solvable(qa_pair, question["qid"], idx, ctx, document_map)
        stats.extend(solver_stats)
        qa_pair_is_unique, qa_uniq_stats = await _is_qa_pair_unique(qa_pair, question["qid"], idx, solvable, ctx, qa_collection, QA_PAIR_LOCK)
        if qa_uniq_stats is not None:
            stats.append(qa_uniq_stats)

        qa_pair_tries = 1
        while (not solvable or not qa_pair_is_unique) and qa_pair_tries < MAX_QA_PAIR_TRIES:
            qa_pair, qa_stats = await _generate_follow_up_question(idx, n_docs, question, qa_pairs, line_of_inquiry, ctx, collection, document_map)
            stats.append(qa_stats)
            solvable, solver_stats = await _is_solvable(qa_pair, question["qid"], idx, ctx, document_map)
            stats.extend(solver_stats)
            qa_pair_is_unique, qa_uniq_stats = await _is_qa_pair_unique(qa_pair, question["qid"], idx, solvable, ctx, qa_collection, QA_PAIR_LOCK)
            if qa_uniq_stats is not None:
                stats.append(qa_uniq_stats)
            qa_pair_tries += 1

        if qa_pair_tries == MAX_QA_PAIR_TRIES:
            raise Exception("Exceeded MAX_QA_PAIR_TRIES")

        qa_pairs.append({
            "qid": question["qid"],
            "idx": idx,
            "question": qa_pair["question"],
            "answer": qa_pair["answer"],
            "doc_ids": qa_pair["doc_ids"],
            "n_docs_target": n_docs,
        })

    return qa_pairs, stats

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the question-answer synthesis script.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="officeqa",
        choices=["officeqa", "trec-biogen"],
        help="the benchmark to generate additional questions for (default: officeqa)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="the split of the benchmark to generate follow-up questions for (default: dev)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="openai/gpt-5.6-luna",
        help="ID of the language model to use for synthesis (default: openai/gpt-5.6-luna)",
    )
    parser.add_argument(
        "--emb-model-id",
        type=str,
        default="qwen/qwen3-embedding-8b",
        help="ID of the embedding model (default: qwen/qwen3-embedding-8b)",
    )
    parser.add_argument(
        "--chroma-host",
        type=str,
        default=os.environ.get("SKUNK_CHROMA_SERVER_HOST", "127.0.0.1"),
        help="ChromaDB server host (default: 127.0.0.1). Start it with scripts/run_chroma_server.sh",
    )
    parser.add_argument(
        "--chroma-port",
        type=int,
        default=int(os.environ.get("SKUNK_CHROMA_SERVER_PORT", "8001")),
        help="ChromaDB server port (default: 8001)",
    )
    parser.add_argument(
        "--qa-inquiry-chroma-host",
        type=str,
        default=os.environ.get("QA_INQUIRY_CHROMA_SERVER_HOST", "127.0.0.1"),
        help="ChromaDB server host holding question/answer embeddings (used by dedup)",
    )
    parser.add_argument(
        "--qa-inquiry-chroma-port",
        type=str,
        default=os.environ.get("QA_INQUIRY_CHROMA_SERVER_PORT", "8002"),
        help="ChromaDB server port holding question/answer embeddings (used by dedup)",
    )
    parser.add_argument(
        "--chroma-collection-name",
        type=str,
        default="officeqa-qwen-8b",
        help="Name of the ChromaDB collection (default: officeqa-qwen-8b)",
    )
    parser.add_argument(
        "--qa-collection-name",
        type=str,
        default="officeqa-qa-embeddings-qwen-8b",
        help="Name of the ChromaDB collection holding question/answer embeddings; used by dedup (default: officeqa-qa-embeddings-qwen-8b)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="the number of follow-up questions to generate for each real benchmark question",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="the number of questions to generate for concurrently",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only generate for the first N questions of the split (smoke tests)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(pathlib.Path(__file__).parent / "output"),
        help="directory to write the run's outputs into (a per-run subdirectory is created)",
    )
    return parser.parse_args()


def _run_question(
    k: int,
    docs_mean: float,
    docs_std: float,
    question: dict,
    config: OrchestratorConfig,
    llm_client: LLMClient,
    collection: Collection,
    document_map: DocumentMap,
    inquiry_collection: Collection,
    qa_collection: Collection,
    run_dir: pathlib.Path,
) -> tuple[dict, list[GenStats]]:
    """Generate one seed question's follow-ups, on this worker thread's own event loop.

    Called from a `ThreadPoolExecutor` worker; `asyncio.run` gives this question a private
    event loop, matching skunk's documented execution model (see `common._step_frame`:
    "one loop per worker thread") and `qatfd/runner.py`. What makes that safe is that the
    shared state is all thread-safe by construction: the rate limiters and the TPM budget
    guard their token math with a `threading.Lock` and expose both a blocking `acquire()`
    and a loop-yielding `acquire_async()` over ONE bucket, so the caps hold across loops;
    `UsageTracker` is likewise lock-guarded. The one type that is NOT shareable is
    `ExecutionContext` (per-question by contract — unsynchronized event list, step-index
    allocator and log file), so each question builds its own.

    Returns `(record, stats)` rather than persisting them: `main` writes both from the
    single collecting thread as each future lands, so incremental persistence needs no
    cross-thread synchronization.
    """
    qid = str(question["qid"])
    ctx = ExecutionContext(
        question=question["question"],
        config=config,
        document_map=document_map,
        uid=qid,
        llm_client=llm_client,
        chroma_collection=collection,
        log_path=str(run_dir / "traces" / f"{qid}.jsonl"),
    )
    t0 = time.monotonic()
    try:
        qa_pairs, stats = asyncio.run(
            generate_follow_up_questions(
                k, docs_mean, docs_std, question, ctx, collection, document_map, inquiry_collection, qa_collection,
            )
        )
    except Exception as e:  # a failed question must not sink the rest of the run
        print(f"  WARN: question {qid} failed: {type(e).__name__}: {e}")
        qa_pairs, stats = [], []
    finally:
        ctx.close()

    record = {
        "qid": qid,
        "seed_question": question["question"],
        "seed_answer": question["answer"],
        "qa_pairs": qa_pairs,
        "wall_latency_s": round(time.monotonic() - t0, 3),
        "cost_usd": sum(s.cost_usd for s in stats),
        "n_generations": len(stats),
        "n_errors": sum(1 for s in stats if s.error is not None),
    }
    return record, stats


def _question_finished(question: dict, records_path: Path) -> bool:
    """Returns True if the question has had a full set of synthetic follow-up questions generated."""
    with open(records_path) as f:
        for line in f:
            data = json.loads(line)
            if data['qid'] == question['qid']:
                return True
    return False


def main() -> None:
    # parse arguments
    args = parse_arguments()

    # load the set of questions
    questions = load_questions(args.benchmark, args.split)
    if args.limit is not None:
        questions = questions[: args.limit]

    # get mapping from document ID to document text for the benchmark data from ChromaDB
    document_map = load_document_map(args.benchmark)

    # load collections for benchmark and question answer embeddings
    client = make_chroma_client(args.chroma_host, args.chroma_port)
    collection = client.get_collection(args.chroma_collection_name)
    qa_inquiry_client = make_chroma_client(args.qa_inquiry_chroma_host, args.qa_inquiry_chroma_port)
    qa_collection = qa_inquiry_client.get_collection(args.qa_collection_name)
    inquiry_collection = qa_inquiry_client.get_or_create_collection(f"{args.qa_collection_name}_inquiry")

    # get the mean and std for the number of relevant chunks per question for this benchmark
    assert args.benchmark in BENCHMARK_CHUNKS_STATS, f"no chunk stats for benchmark {args.benchmark}"
    docs_mean, docs_std = BENCHMARK_CHUNKS_STATS[args.benchmark]

    # set the pdf_dir based on the benchmark (if applicable)
    pdf_dir = None
    if args.benchmark == "officeqa":
        pdf_dir = str(OFFICEQA_PDF_DIR)

    # create an OrchestratorConfig and InferenceConfig for the LLM client
    inference_config = InferenceConfig(
        llm_provider="openrouter",
        llm_model=args.model_id,
        emb_provider="openrouter",
        emb_model_id=args.emb_model_id,
        llm_max_retries=5,
        llm_retry_initial_delay_s=1.0,
        llm_model_rpm={},
        llm_default_rpm=1000,
        llm_model_tpm={},
        llm_default_tpm=None,
        llm_prices=LLM_PRICES,
        llm_context_limits=LLM_CONTEXT_LIMITS,
    )
    config = OrchestratorConfig(
        search=SearchAgentConfig(name=""),
        lookup=LookupAgentConfig(name=""),
        inference=inference_config,
        storage=StorageConfig(args.chroma_collection_name, args.chroma_host, args.chroma_port, pdf_dir=pdf_dir),
    )

    # initialize llm client
    llm_client = LLMClient(config=inference_config)

    # per-run output directory; results and stats stream into it as questions complete
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pathlib.Path(args.output_dir) / f"follow_up_questions_{args.benchmark}_{args.split}_{ts}"
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    with (run_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # One worker thread per question, each running its own event loop (`_run_question`),
    # `max_workers` questions in flight at a time. The work is I/O-bound — LLM calls, chroma
    # reads — so the GIL is released for essentially all of it.
    #
    # Each question's record + stats are written the moment its future lands, so an
    # interruption the per-question `except` cannot catch (e.g. Ctrl-C) still persists
    # the completed questions.
    assert args.max_workers >= 1, "max-workers must be >= 1"
    qid_to_result: dict[str, tuple[dict, list[GenStats]]] = {}
    t0 = time.monotonic()
    with (
        (run_dir / "records.jsonl").open("w", encoding="utf-8") as rf,
        (run_dir / "generation_stats.jsonl").open("w", encoding="utf-8") as sf,
        ThreadPoolExecutor(max_workers=args.max_workers) as pool,
    ):
        futures = [
            pool.submit(
                _run_question, args.k, docs_mean, docs_std, q, config, llm_client,
                collection, document_map, inquiry_collection, qa_collection, run_dir,
            )
            for q in questions
            if not _question_finished(q, run_dir / "records.jsonl")
        ]
        for fut in as_completed(futures):
            record, stats = fut.result()
            rf.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
            rf.flush()
            for s in stats:
                sf.write(json.dumps(s.to_json(), default=str, ensure_ascii=False) + "\n")
            sf.flush()
            qid_to_result[record["qid"]] = (record, stats)
    elapsed_s = time.monotonic() - t0

    # re-order into the original split order (`as_completed` yields in completion order)
    results = [qid_to_result[str(q["qid"])] for q in questions if str(q["qid"]) in qid_to_result]

    # write the initial + synthetic qa pairs
    all_qa_pairs = [qa_pair for record, _ in results for qa_pair in record["qa_pairs"]]
    with (run_dir / "qa_pairs.json").open("w") as f:
        json.dump(all_qa_pairs, f, indent=2)

    # roll the per-generation stats up into a run-level usage summary. `llm_client.usage`
    # is the authoritative run total; the aggregate is the same number reached by summing
    # the per-agent buckets, so a mismatch means some LLM call was made off-key.
    all_stats: list[GenStats] = [s for _, stats in results for s in stats]
    summary = aggregate(all_stats)
    summary["run"] = {
        "benchmark": args.benchmark,
        "split": args.split,
        "model_id": args.model_id,
        "emb_model_id": args.emb_model_id,
        "k": args.k,
        "n_questions": len(questions),
        "n_qa_pairs": len(all_qa_pairs),
        "elapsed_s": round(elapsed_s, 3),
        "tracker_total_cost_usd": llm_client.usage.cost(),
        "tracker_embed_cost_usd": llm_client.usage.embed_cost(),
        "tracker_input_tokens": llm_client.usage.total_input_tokens,
        "tracker_output_tokens": llm_client.usage.total_output_tokens,
        "tracker_cached_tokens": llm_client.usage.total_cached_tokens,
        "tracker_embed_tokens": llm_client.usage.total_embed_tokens,
    }
    with (run_dir / "usage_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"wrote {len(all_qa_pairs)} qa pairs from {len(questions)} questions to {run_dir} "
        f"(${summary['run']['tracker_total_cost_usd']:.2f}, {elapsed_s / 60:.1f} min)"
    )


if __name__ == "__main__":
    main()


# TODO: add solve logic and measure fraction of solve(s) per question
# TODO: run 1
