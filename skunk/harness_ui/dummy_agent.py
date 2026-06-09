"""Dummy reasoner for testing the local console."""

from __future__ import annotations

import asyncio
import random

from cup_kit.agent_runtime import AgentAnswer


async def solve(prompt: str) -> AgentAnswer:
    delay_s = random.uniform(1.0, 5.0)
    await asyncio.sleep(delay_s)
    answer = str(random.randint(0, 1_000_000))
    return AgentAnswer(
        answer=answer,
        reasoning=(
            f"Dummy agent test run. It received a prompt with {len(prompt)} characters, "
            f"waited {delay_s:.2f} seconds to simulate computation, and returned a "
            f"random numeric answer: {answer}. This reasoning text is intentionally "
            "long enough to satisfy the Cup API validation floor for agent submissions."
        ),
        source_docs=["dummy_agent.py"],
    )
