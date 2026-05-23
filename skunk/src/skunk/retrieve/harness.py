from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
import pandas as pd
from chromadb.api.models.Collection import Collection

from skunk.logging.tracer import Tracer
from skunk.retrieve.search_agent import OFFICEQA_SPECIAL_NOTES, SearchAgent
from skunk.utils.officeqa_eval import source_docs_to_page_keys, source_files_to_year_months

MODEL_ID = "google/gemini-3-flash-preview"

TRACE_DIR = "search_agent_traces"
BULLETINS_DIR = "treasury_bulletins_cleaned"

# validation set: the 34 UIDs that completed in v2_search_agent_traces.
VALIDATION_UIDS = [
    "UID0001", "UID0003", "UID0004", "UID0005", "UID0007",
    "UID0009", "UID0010", "UID0012", "UID0013", "UID0015",
    "UID0017", "UID0018", "UID0019", "UID0022", "UID0025",
    "UID0027", "UID0028", "UID0029", "UID0030", "UID0031",
    "UID0032", "UID0035", "UID0036", "UID0037", "UID0039",
    "UID0042", "UID0044", "UID0049", "UID0050", "UID0053",
    "UID0055", "UID0056", "UID0057", "UID0058",
]


def _run_one(
    row: dict,
    model_id: str,
    clean_page_map: dict,
    chroma_collection: Collection,
    emb_model_id: str,
    show_output: bool,
    trace_dir: str,
) -> tuple[str, dict]:
    """Run the search agent for a single question and return an analysis dict."""
    start_time = time.perf_counter()
    uid = row["uid"]
    question = row["question"]
    trace_path = f"{trace_dir}/{uid}_trace.txt"

    with Tracer(trace_path, show_output=show_output) as tracer:
        agent = SearchAgent(
            model_id,
            clean_page_map=clean_page_map,
            chroma_collection=chroma_collection,
            emb_model_id=emb_model_id,
            tracer=tracer,
            special_notes=OFFICEQA_SPECIAL_NOTES,
        )
        page_keys = agent.retrieve(question)

    # Compute accuracy against ground truth.
    source_page_keys = source_docs_to_page_keys(str(row.get("source_docs", "")))
    source_year_months = source_files_to_year_months(str(row.get("source_files", "")))

    correct_page = 0.0
    correct_document = 0.0
    if agent._completed and page_keys:
        if source_page_keys:
            correct_page = sum(1 for k in source_page_keys if k in page_keys) / len(source_page_keys)
        if source_year_months:
            def _key_ym(k: str) -> tuple[str, str] | None:
                parts = k.split("_")
                return (parts[0], parts[1]) if len(parts) >= 2 else None
            final_yms = {_key_ym(k) for k in page_keys} - {None}
            correct_document = sum(1 for ym in source_year_months if ym in final_yms) / len(source_year_months)

    # persist the full message history for later debugging.
    messages_path = f"{trace_dir}/{uid}_messages.json"
    with open(messages_path, "w") as f:
        json.dump(agent.messages_to_jsonable(), f, indent=2)

    analysis = {
        "uid": uid,
        "completed": agent._completed,
        "correct_page": correct_page,
        "correct_document": correct_document,
        "num_steps": agent._num_steps,
        "error": agent._error,
        "page_keys": page_keys,
        "total_time_sec": time.perf_counter() - start_time,
    }
    return uid, analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the treasury bulletin search agent.")
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=TRACE_DIR,
        help=f"Directory to save agent traces (default: {TRACE_DIR})",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Stream the agent trace to the terminal in addition to the trace file.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=4,
        help="Number of questions to process in parallel (default: 4).",
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
        default="gemini",
        help="Name of the ChromaDB collection to use for embeddings (default: gemini-embedding-2)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help=f"ID of the language model to use (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--emb-model-id",
        type=str,
        default="google/gemini-embedding-2-preview",
        help="ID of the embedding model to use (default: google/gemini-embedding-2-preview)",
    )
    parser.add_argument(
        "--uid",
        type=str,
        default=None,
        help="Run on a single UID for a smoke test (streams logs to stdout). Must be in VALIDATION_UIDS.",
    )
    args = parser.parse_args()

    # If --uid is set, force a smoke-test configuration: single UID, sequential, stream output.
    if args.uid is not None:
        if args.uid not in VALIDATION_UIDS:
            parser.error(f"--uid {args.uid} is not in VALIDATION_UIDS")
        args.parallelism = 1
        args.show_output = True

    # step 0: initialize directories for agent traces
    os.makedirs(args.trace_dir, exist_ok=True)

    # step 1: load questions
    officeqa_df = pd.read_csv("officeqa_pro.csv")

    # step 2: load mapping from "year-month-page_id" --> [clean page text file path, sorted elements order]
    with open(f"{BULLETINS_DIR}/clean_page_map.json") as f:
        clean_page_map = json.load(f)

    # step 2.5: load chromadb collection to ensure it's ready before we start processing questions
    client = chromadb.PersistentClient(path=args.chroma_dir)
    collection = client.get_collection(args.chroma_collection_name)

    # step 3: restrict to the validation set (or the single --uid smoke-test target),
    # then filter out questions whose traces already exist.
    target_uids = {args.uid} if args.uid is not None else set(VALIDATION_UIDS)
    validation_df = officeqa_df[officeqa_df["uid"].isin(target_uids)]
    missing = target_uids - set(validation_df["uid"])
    if missing:
        print(f"WARNING: {len(missing)} target UID(s) not found in officeqa_pro.csv: {sorted(missing)}")

    rows = [
        row
        for _, row in validation_df.iterrows()
        if not os.path.exists(f"{args.trace_dir}/{row['uid']}_trace.txt")
    ]
    skipped = len(validation_df) - len(rows)
    if skipped:
        print(f"Skipping {skipped} question(s) with existing traces.")

    # step 4: run questions (debug: sequential loop with pdb breakpoint)
    with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        futures = {
            pool.submit(_run_one, row, args.model_id, clean_page_map, collection, args.emb_model_id, args.show_output, args.trace_dir): row # type: ignore
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            uid = row["uid"]
            try:
                uid, analysis = future.result()
            except Exception as e:
                print(f"ERROR for UID {uid}: {e}")
                analysis = {
                    "uid": uid,
                    "completed": False,
                    "correct_page": 0.0,
                    "correct_document": 0.0,
                    "num_steps": 0,
                    "error": str(e),
                    "page_keys": [],
                }
            with open(f"{args.trace_dir}/{uid}_analysis.json", "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"============= END OF TRACE: {uid} ==================")
            print(f"UID: {uid}")
            print(f"Question: {row['question']}")
            print(f"Analysis: {json.dumps(analysis, indent=2)}")
            print(f"Ground Truth Source Docs: {row['source_docs']}")
            print(f"Ground Truth Source Files: {row['source_files']}")
            print("===========================================")
