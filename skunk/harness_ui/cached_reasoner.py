"""Disk-backed reasoner that replays captured Skunk executions."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from cup_kit.agent_runtime import AgentAnswer
from skunk_reasoner import SkunkReasoner

DEFAULT_ARTIFACT = "cache/mock_reasoner_artifacts_officeqa_1x5.pkl"
_artifact_cache: dict[str, Any] | None = None


def _load_artifact() -> dict[str, Any]:
    global _artifact_cache
    if _artifact_cache is None:
        artifact_path = Path(os.environ.get("SKUNK_MOCK_ARTIFACT", DEFAULT_ARTIFACT))
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Mock reasoner artifact not found: {artifact_path}. "
                "Run generate_mock_reasoner_artifacts.py first."
            )
        with artifact_path.open("rb") as artifact_file:
            loaded = pickle.load(artifact_file)
        if not isinstance(loaded, dict) or not isinstance(loaded.get("results"), dict):
            raise ValueError(f"Invalid mock reasoner artifact: {artifact_path}")
        _artifact_cache = loaded
    return _artifact_cache


class CachedSkunkReasoner(SkunkReasoner):
    async def execute(self, prompt: str) -> tuple[str, list[dict]]:
        artifact = _load_artifact()
        results = artifact["results"]
        matched_prompt = prompt if prompt in results else None
        if matched_prompt is None:
            matches = [
                cached_prompt
                for cached_prompt in results
                if prompt.startswith(cached_prompt + "\n\n")
            ]
            if matches:
                matched_prompt = max(matches, key=len)
        if matched_prompt is None:
            raise KeyError("No cached Skunk execution matches this prompt")

        result = results[matched_prompt]
        if result.get("error"):
            raise RuntimeError(f"Cached Skunk execution failed: {result['error']}")
        events = result.get("events")
        if not isinstance(events, list):
            raise ValueError("Cached Skunk execution has no event trace")
        return str(result["answer"]), events


async def solve(prompt: str) -> AgentAnswer:
    return await CachedSkunkReasoner().solve(prompt)


async def solve_with_trace(prompt: str) -> tuple[AgentAnswer, list[dict]]:
    return await CachedSkunkReasoner().solve_with_trace(prompt)
