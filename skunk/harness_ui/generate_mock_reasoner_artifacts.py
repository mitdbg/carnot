"""Run the real Skunk reasoner and capture replay artifacts for the 1x5 GUI."""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skunk_reasoner import solve_with_trace

HERE = Path(__file__).resolve().parent
QUESTIONS_FILE = HERE / "questions/officeqa_1x5_questions.json"
OUTPUT_DIR = HERE / "cache"
OUTPUT_FILE = OUTPUT_DIR / "mock_reasoner_artifacts_officeqa_1x5.pkl"


def _write_artifact(artifact: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_file = OUTPUT_FILE.with_suffix(f"{OUTPUT_FILE.suffix}.tmp")
    with temporary_file.open("wb") as artifact_file:
        pickle.dump(artifact, artifact_file)
        artifact_file.flush()
        os.fsync(artifact_file.fileno())
    os.replace(temporary_file, OUTPUT_FILE)


async def _run_question(
    round_num: int,
    question: dict[str, Any],
    artifact: dict[str, Any],
    cache_lock: asyncio.Lock,
) -> None:
    prompt = question["prompt"]
    started_at = time.perf_counter()
    print(f"Running round {round_num} / {question['question_id']}...")
    try:
        answer, events = await solve_with_trace(prompt)
        result = {
            "round_num": round_num,
            "question_id": question["question_id"],
            "canonical_answer": question.get("canonical_answer"),
            "answer": answer.answer,
            "events": events,
            "elapsed_s": round(time.perf_counter() - started_at, 3),
            "error": None,
        }
    except Exception as error:
        result = {
            "round_num": round_num,
            "question_id": question["question_id"],
            "canonical_answer": question.get("canonical_answer"),
            "answer": "",
            "events": [],
            "elapsed_s": round(time.perf_counter() - started_at, 3),
            "error": f"{type(error).__name__}: {error}",
        }
        print(f"Failed {question['question_id']}: {error}")

    async with cache_lock:
        artifact["results"][prompt] = result
        _write_artifact(artifact)


async def generate_artifacts() -> dict[str, Any]:
    questions_payload = json.loads(QUESTIONS_FILE.read_text())
    artifact = {
        "format_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "questions_file": str(QUESTIONS_FILE),
        "results": {},
    }
    cache_lock = asyncio.Lock()
    workers = [
        _run_question(round_payload["round_num"], question, artifact, cache_lock)
        for round_payload in questions_payload["rounds"]
        for question in round_payload["questions"]
    ]
    await asyncio.gather(*workers)
    return artifact


if __name__ == "__main__":
    generated_artifact = asyncio.run(generate_artifacts())
    print(f"Wrote {len(generated_artifact['results'])} executions to {OUTPUT_FILE}")
