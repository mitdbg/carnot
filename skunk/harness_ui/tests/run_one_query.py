"""Run a single OfficeQA query through the Skunk pipeline."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

QUERY_ID = "UID0001"

SKUNK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SKUNK_ROOT.parent

env_path = SKUNK_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

SKUNK_SRC = SKUNK_ROOT / "src"
if str(SKUNK_SRC) not in sys.path:
    sys.path.insert(0, str(SKUNK_SRC))

QUESTIONS_FILE = SKUNK_ROOT / "harness_ui" / "questions" / "officeqa_3x5_questions.json"
with QUESTIONS_FILE.open(encoding="utf-8") as f:
    questions_data = json.load(f)

prompt = None
canonical_answer = None
for round_ in questions_data["rounds"]:
    for question in round_.get("questions", []):
        if question["question_id"] == QUERY_ID:
            prompt = question["prompt"]
            canonical_answer = question["canonical_answer"]
            break
    if prompt is not None:
        break

if prompt is None:
    raise KeyError(f"QUERY_ID {QUERY_ID!r} not found in {QUESTIONS_FILE}")

os.environ.setdefault(
    "OFFICEQA_PARSED_JSON_DIR",
    str(REPO_ROOT / "data/officeqa/treasury_bulletins_parsed/jsons"),
)
os.environ.setdefault(
    "OFFICEQA_PDF_DIR",
    str(REPO_ROOT / "data/officeqa/treasury_bulletin_pdfs"),
)

from skunk import (  # noqa: E402
    MissingData,
    Orchestrator,
    SkunkConfig,
    StepFailed,
    load_prompt_overrides,
)

config = SkunkConfig.from_env()
overrides_path = Path(config.prompt_overrides_path)
if not overrides_path.is_absolute():
    overrides_path = SKUNK_ROOT / config.prompt_overrides_path
prompt_overrides = load_prompt_overrides(overrides_path) if overrides_path.exists() else ()

orch = None
started_at = time.perf_counter()
try:
    orch = Orchestrator(
        prompt,
        config=config,
        prompt_overrides=prompt_overrides,
        uid=QUERY_ID,
        verbose=True,
    )
    answer = asyncio.run(orch.execute())

    print(f"Query ID: {QUERY_ID}")
    print(f"Prompt: {prompt}")
    print(f"Canonical answer: {canonical_answer}")
    print(f"Returned answer: {answer}")
except (MissingData, StepFailed) as e:
    print(f"Skunk pipeline failed: {e}", file=sys.stderr)
    raise
finally:
    if orch is not None:
        elapsed_s = time.perf_counter() - started_at
        replanning_steps = sum(
            1
            for event in orch.ctx.events
            if event.get("kind") == "plan"
            and event.get("data", {}).get("label") == "replan"
        )
        print(f"Elapsed time: {elapsed_s:.2f} seconds")
        print(f"Replanning steps: {replanning_steps}")
        orch.ctx.close()
