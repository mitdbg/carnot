"""Tests for generate_mock_reasoner_artifacts."""

import asyncio
import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

import generate_mock_reasoner_artifacts as generator


@pytest.fixture
def questions_file(tmp_path: Path) -> Path:
    """Create a temporary questions JSON file with at least three questions."""
    questions_data = {
        "rounds": [
            {
                "round_num": 1,
                "questions": [
                    {"question_id": "1", "prompt": "Question 1?"},
                    {"question_id": "2", "prompt": "Question 2?", "canonical_answer": "A"},
                ],
            },
            {
                "round_num": 2,
                "questions": [
                    {"question_id": "3", "prompt": "Question 3?"},
                ],
            },
        ]
    }
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(questions_data))
    return path


def test_generate_artifacts(monkeypatch: pytest.MonkeyPatch, questions_file: Path) -> None:
    active = [0]
    max_active = [0]

    async def tracking_solve_with_trace(prompt: str):
        active[0] += 1
        max_active[0] = max(max_active[0], active[0])
        try:
            await asyncio.sleep(0.05)
            return SimpleNamespace(answer=f"Answer to: {prompt}"), [{"event": "done"}]
        finally:
            active[0] -= 1

    snapshots: list[dict] = []

    def tracking_write_artifact(artifact: dict) -> None:
        snapshots.append(copy.deepcopy(artifact))

    monkeypatch.setattr(generator, "QUESTIONS_FILE", questions_file)
    monkeypatch.setattr(generator, "solve_with_trace", tracking_solve_with_trace)
    monkeypatch.setattr(generator, "_write_artifact", tracking_write_artifact)

    artifact = asyncio.run(generator.generate_artifacts())

    assert max_active[0] == 3
    assert len(snapshots) == 3
    assert [len(snapshot["results"]) for snapshot in snapshots] == [1, 2, 3]

    assert len(artifact["results"]) == 3
    for prompt in ["Question 1?", "Question 2?", "Question 3?"]:
        assert prompt in artifact["results"]
        assert artifact["results"][prompt]["answer"] == f"Answer to: {prompt}"


def test_write_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "cache"
    output_file = output_dir / "test_artifact.pkl"
    monkeypatch.setattr(generator, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(generator, "OUTPUT_FILE", output_file)

    artifact = {"key": "value", "nested": [1, 2, 3]}
    generator._write_artifact(artifact)

    with output_file.open("rb") as artifact_file:
        assert pickle.load(artifact_file) == artifact
    assert not output_file.with_suffix(f"{output_file.suffix}.tmp").exists()
