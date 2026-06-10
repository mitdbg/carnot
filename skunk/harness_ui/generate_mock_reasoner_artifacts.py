"""Run the real Skunk reasoner and capture replay artifacts for the 1x5 GUI."""

from __future__ import annotations

import asyncio
import json
import pickle
import time
from datetime import UTC, datetime
from pathlib import Path

from skunk_reasoner import solve_with_trace

HERE = Path(__file__).resolve().parent
QUESTIONS_FILE = HERE / "questions/officeqa_1x5_questions.json"
OUTPUT_DIR = HERE / "cache"
OUTPUT_FILE = OUTPUT_DIR / "mock_reasoner_artifacts_officeqa_1x5.pkl"

questions_payload = json.loads(QUESTIONS_FILE.read_text())
results = {}

for round_payload in questions_payload["rounds"]:
    for question in round_payload["questions"]:
        prompt = question["prompt"]
        started_at = time.perf_counter()
        print(f"Running round {round_payload['round_num']} / {question['question_id']}...")
        try:
            answer, events = asyncio.run(solve_with_trace(prompt))
            results[prompt] = {
                "round_num": round_payload["round_num"],
                "question_id": question["question_id"],
                "canonical_answer": question.get("canonical_answer"),
                "answer": answer.answer,
                "events": events,
                "elapsed_s": round(time.perf_counter() - started_at, 3),
                "error": None,
            }
        except Exception as error:
            results[prompt] = {
                "round_num": round_payload["round_num"],
                "question_id": question["question_id"],
                "canonical_answer": question.get("canonical_answer"),
                "answer": "",
                "events": [],
                "elapsed_s": round(time.perf_counter() - started_at, 3),
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"Failed {question['question_id']}: {error}")

artifact = {
    "format_version": 2,
    "generated_at": datetime.now(UTC).isoformat(),
    "questions_file": str(QUESTIONS_FILE),
    "results": results,
}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with OUTPUT_FILE.open("wb") as artifact_file:
    pickle.dump(artifact, artifact_file)
print(f"Wrote {len(results)} executions to {OUTPUT_FILE}")
