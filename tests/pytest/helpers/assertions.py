"""Reusable assertion helpers for the Carnot test suite.

These helpers are shared across test files to avoid duplication.
Import them directly::

    from helpers.assertions import assert_agent_did_not_hit_max_steps
"""

from __future__ import annotations

from typing import Any

from carnot.agents.base import BaseAgent
from carnot.agents.planner import Planner


def assert_agent_did_not_hit_max_steps(agent: BaseAgent) -> None:
    """Assert that *agent* completed before its step budget.

    Checks the agent's memory for an ``AgentMaxStepsError`` in the last
    recorded step.  Simple tests should never exhaust the step budget.

    Requires:
        - *agent* has a ``memory`` attribute with a ``steps`` list.
        - *agent* has a ``max_steps`` attribute (int).

    Raises:
        AssertionError: if the last step's ``error`` is an
        ``AgentMaxStepsError``.
    """
    from carnot.agents.utils import AgentMaxStepsError

    if agent.memory.steps:
        last_step = agent.memory.steps[-1]
        error = getattr(last_step, "error", None)
        assert not isinstance(error, AgentMaxStepsError), (
            f"Agent hit max_steps limit ({agent.max_steps}). "
            "This task should complete in fewer steps."
        )


def assert_planner_did_not_hit_max_steps(planner: Planner, result: Any) -> None:
    """Assert that the planner did not exhaust its step budget.

    Checks two conditions:
    1. The *result* string is not the sentinel "max steps reached" message.
    2. If the planner exposes ``planning_memory``, the last step does not
       contain an ``AgentMaxStepsError``.

    Requires:
        - *planner* has a ``max_steps`` attribute (int).

    Raises:
        AssertionError: if either condition indicates the planner hit its
        step limit.
    """
    from carnot.agents.utils import AgentMaxStepsError

    max_steps_message = (
        "The agent did not return a final answer within the maximum number of steps."
    )
    assert result != max_steps_message, (
        f"Planner hit max_steps limit ({planner.max_steps}). "
        "This simple task should complete in fewer steps."
    )

    # also check planning_memory if present (used by some planner variants).
    planning_memory = getattr(planner, "planning_memory", None)
    if planning_memory and planning_memory.steps:
        last_step = planning_memory.steps[-1]
        error = getattr(last_step, "error", None)
        assert not isinstance(error, AgentMaxStepsError), (
            f"Planner hit max_steps: {error}"
        )


def assert_accuracy_above(
    predicted: list[dict],
    ground_truth: list[dict],
    pred_key: str,
    gt_key: str,
    threshold: float = 0.8,
) -> None:
    """Assert that predicted values match ground truth above *threshold*.

    Comparison is case-insensitive on the string representation of values.

    Requires:
        - *predicted* and *ground_truth* have the same length.
        - Each dict contains its respective key.
        - 0.0 <= *threshold* <= 1.0.

    Raises:
        AssertionError: if accuracy < *threshold*.
    """
    assert len(predicted) == len(ground_truth), (
        f"Length mismatch: predicted={len(predicted)}, "
        f"ground_truth={len(ground_truth)}"
    )
    correct = sum(
        1
        for p, gt in zip(predicted, ground_truth, strict=True)
        if str(p[pred_key]).lower() == str(gt[gt_key]).lower()
    )
    accuracy = correct / len(predicted)
    assert accuracy >= threshold, (
        f"Accuracy {accuracy:.2%} is below threshold {threshold:.2%}"
    )
