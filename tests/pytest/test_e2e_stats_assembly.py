"""End-to-end stats assembly tests (mocked LLM).

These tests verify that ``Execution.plan()`` and ``Execution.run()``
correctly assemble ``PhaseStats`` and ``ExecutionStats`` from the
per-operator ``OperatorStats`` collected during execution.

All tests are Tier 2 (mocked) — they validate the *wiring* between
planning stats collection, execution stats assembly, and the progress
event enrichment without making any real LLM calls.
"""

from __future__ import annotations

import pytest
from helpers.mock_utils import msg_text

from carnot.core.models import ExecutionStats, OperatorStats
from carnot.data.dataset import Dataset
from carnot.execution.execution import Execution
from carnot.execution.progress import ExecutionProgress, PlanningProgress

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_LLM_CONFIG = {"OPENAI_API_KEY": "test-key-not-real"}


def _wrap_planner_code(code: str) -> str:
    """Wrap Python *code* in a fenced block with a Thought preamble."""
    return f"Thought: generating plan\n```python\n{code}\n```"


def _wrap_paraphrase(text: str) -> str:
    """Wrap *text* in the plan-tags the Planner expects for paraphrasing."""
    return f"<begin_plan>\n{text}\n<end_plan>"


def _make_completion_response(content: str):
    """Create a mock litellm completion response."""
    from fixtures.mocks import _make_completion_response as _make

    return _make(content)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def animals() -> Dataset:
    return Dataset(
        name="Animals",
        annotation="A dataset of animals.",
        items=[
            {"animal": "giraffe", "class": "mammal"},
            {"animal": "anaconda", "class": "reptile"},
            {"animal": "salmon", "class": "fish"},
            {"animal": "elephant", "class": "mammal"},
            {"animal": "toucan", "class": "bird"},
        ],
    )


# ---------------------------------------------------------------------------
# Test: Execution.run() returns ExecutionStats
# ---------------------------------------------------------------------------


class TestRunReturnsExecutionStats:
    """Verify that ``Execution.run()`` returns a 3-tuple with stats."""

    def test_run_returns_three_tuple(self, mock_litellm, animals):
        """``run()`` returns ``(items, answer_str, ExecutionStats)``."""

        plan_code = (
            'ds = datasets["Animals"]\n'
            'ds = ds.sem_filter("the animal is a mammal")\n'
            "final_answer(ds.serialize())"
        )
        paraphrase_text = "Filter Animals to keep only mammals."
        mammals = {"giraffe", "elephant"}

        call_count = {"n": 0}
        phase = {"state": "planning"}

        def handler(model, messages, **kw):
            if phase["state"] == "planning":
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return _make_completion_response(_wrap_planner_code(plan_code))
                return _make_completion_response(_wrap_paraphrase(paraphrase_text))

            text = msg_text(messages)

            # ReasoningOperator
            if "final_answer" in text.lower() or "code" in model.lower():
                code = (
                    'items = input_datasets[list(input_datasets.keys())[-1]].items\n'
                    'final_answer({"final_items": items, "final_answer_str": "Mammals."})'
                )
                return _make_completion_response(f"```python\n{code}\n```")

            # SemFilter
            is_mammal = any(m in text for m in mammals)
            answer = "TRUE" if is_mammal else "FALSE"
            return _make_completion_response(f"```text\n{answer}\n```")

        mock_litellm.set_completion_handler(handler)

        execution = Execution(
            query="Find all mammals",
            datasets=[animals],
            llm_config=_LLM_CONFIG,
        )

        nl_plan, logical_plan = execution.plan()
        phase["state"] = "running"
        execution._plan = logical_plan

        items, answer_str, stats = execution.run()

        # Basic return type checks
        assert isinstance(items, list)
        assert isinstance(answer_str, str)
        assert isinstance(stats, ExecutionStats)

        execution.planner.cleanup()

    def test_execution_stats_has_planning_phase(self, mock_litellm, animals):
        """``stats.planning`` is populated after ``plan()`` + ``run()``."""

        plan_code = (
            'ds = datasets["Animals"]\n'
            'ds = ds.sem_filter("the animal is a mammal")\n'
            "final_answer(ds.serialize())"
        )
        paraphrase_text = "Filter mammals."
        mammals = {"giraffe", "elephant"}

        call_count = {"n": 0}
        phase = {"state": "planning"}

        def handler(model, messages, **kw):
            if phase["state"] == "planning":
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return _make_completion_response(_wrap_planner_code(plan_code))
                return _make_completion_response(_wrap_paraphrase(paraphrase_text))

            text = msg_text(messages)
            if "final_answer" in text.lower() or "code" in model.lower():
                code = (
                    'items = input_datasets[list(input_datasets.keys())[-1]].items\n'
                    'final_answer({"final_items": items, "final_answer_str": "Done."})'
                )
                return _make_completion_response(f"```python\n{code}\n```")

            is_mammal = any(m in text for m in mammals)
            answer = "TRUE" if is_mammal else "FALSE"
            return _make_completion_response(f"```text\n{answer}\n```")

        mock_litellm.set_completion_handler(handler)

        execution = Execution(
            query="Find all mammals",
            datasets=[animals],
            llm_config=_LLM_CONFIG,
        )

        nl_plan, logical_plan = execution.plan()
        phase["state"] = "running"
        execution._plan = logical_plan

        items, answer_str, stats = execution.run()

        # Planning phase should have stats
        assert stats.planning.phase == "planning"
        assert stats.planning.wall_clock_secs > 0.0

        # Should have at least a Planner OperatorStats
        planner_ops = [
            op for op in stats.planning.operator_stats
            if op.operator_name == "Planner"
        ]
        assert len(planner_ops) == 1, (
            f"Expected 1 Planner OperatorStats, got {len(planner_ops)}"
        )

        # Planner should have LLM calls (at least plan + paraphrase)
        planner_stats = planner_ops[0]
        assert len(planner_stats.llm_calls) >= 2, (
            f"Expected >= 2 planner LLM calls, got {len(planner_stats.llm_calls)}"
        )

        execution.planner.cleanup()

    def test_execution_stats_has_execution_phase(self, mock_litellm, animals):
        """``stats.execution`` contains operator stats from ``run()``."""

        plan_code = (
            'ds = datasets["Animals"]\n'
            'ds = ds.sem_filter("the animal is a mammal")\n'
            "final_answer(ds.serialize())"
        )
        paraphrase_text = "Filter mammals."
        mammals = {"giraffe", "elephant"}

        call_count = {"n": 0}
        phase = {"state": "planning"}

        def handler(model, messages, **kw):
            if phase["state"] == "planning":
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return _make_completion_response(_wrap_planner_code(plan_code))
                return _make_completion_response(_wrap_paraphrase(paraphrase_text))

            text = msg_text(messages)
            if "final_answer" in text.lower() or "code" in model.lower():
                code = (
                    'items = input_datasets[list(input_datasets.keys())[-1]].items\n'
                    'final_answer({"final_items": items, "final_answer_str": "Done."})'
                )
                return _make_completion_response(f"```python\n{code}\n```")

            is_mammal = any(m in text for m in mammals)
            answer = "TRUE" if is_mammal else "FALSE"
            return _make_completion_response(f"```text\n{answer}\n```")

        mock_litellm.set_completion_handler(handler)

        execution = Execution(
            query="Find all mammals",
            datasets=[animals],
            llm_config=_LLM_CONFIG,
        )

        nl_plan, logical_plan = execution.plan()
        phase["state"] = "running"
        execution._plan = logical_plan

        items, answer_str, stats = execution.run()

        # Execution phase should have stats
        assert stats.execution.phase == "execution"
        assert stats.execution.wall_clock_secs > 0.0

        # Should have operator stats (SemFilter + Reasoning)
        op_names = [op.operator_name for op in stats.execution.operator_stats]
        assert "SemFilter" in op_names, f"Expected SemFilter, got {op_names}"
        assert "Reasoning" in op_names, f"Expected Reasoning, got {op_names}"

        # Each operator should have LLM calls
        for op in stats.execution.operator_stats:
            assert len(op.llm_calls) > 0, (
                f"{op.operator_name} has no LLM calls"
            )

        execution.planner.cleanup()

    def test_execution_stats_totals(self, mock_litellm, animals):
        """``stats.total_cost_usd`` sums planning + execution costs."""

        plan_code = (
            'ds = datasets["Animals"]\n'
            'ds = ds.sem_filter("the animal is a mammal")\n'
            "final_answer(ds.serialize())"
        )
        paraphrase_text = "Filter mammals."
        mammals = {"giraffe", "elephant"}

        call_count = {"n": 0}
        phase = {"state": "planning"}

        def handler(model, messages, **kw):
            if phase["state"] == "planning":
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return _make_completion_response(_wrap_planner_code(plan_code))
                return _make_completion_response(_wrap_paraphrase(paraphrase_text))

            text = msg_text(messages)
            if "final_answer" in text.lower() or "code" in model.lower():
                code = (
                    'items = input_datasets[list(input_datasets.keys())[-1]].items\n'
                    'final_answer({"final_items": items, "final_answer_str": "Done."})'
                )
                return _make_completion_response(f"```python\n{code}\n```")

            is_mammal = any(m in text for m in mammals)
            answer = "TRUE" if is_mammal else "FALSE"
            return _make_completion_response(f"```text\n{answer}\n```")

        mock_litellm.set_completion_handler(handler)

        execution = Execution(
            query="Find all mammals",
            datasets=[animals],
            llm_config=_LLM_CONFIG,
        )

        nl_plan, logical_plan = execution.plan()
        phase["state"] = "running"
        execution._plan = logical_plan

        items, answer_str, stats = execution.run()

        # Total cost should equal sum of phases
        assert stats.total_cost_usd == pytest.approx(
            stats.planning.total_cost_usd + stats.execution.total_cost_usd
        )

        # Both phases should have non-zero cost (mock returns 0.001 per call)
        assert stats.planning.total_cost_usd > 0.0
        assert stats.execution.total_cost_usd > 0.0

        # Total wall clock is sum of phases
        assert stats.total_wall_clock_secs == pytest.approx(
            stats.planning.wall_clock_secs + stats.execution.wall_clock_secs
        )

        # Token counts should be non-zero
        assert stats.total_input_tokens > 0
        assert stats.total_output_tokens > 0

        execution.planner.cleanup()

    def test_to_summary_dict_on_full_lifecycle(self, mock_litellm, animals):
        """``stats.to_summary_dict()`` produces a valid JSON-friendly dict."""

        plan_code = (
            'ds = datasets["Animals"]\n'
            'ds = ds.sem_filter("the animal is a mammal")\n'
            "final_answer(ds.serialize())"
        )
        paraphrase_text = "Filter mammals."
        mammals = {"giraffe", "elephant"}

        call_count = {"n": 0}
        phase = {"state": "planning"}

        def handler(model, messages, **kw):
            if phase["state"] == "planning":
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return _make_completion_response(_wrap_planner_code(plan_code))
                return _make_completion_response(_wrap_paraphrase(paraphrase_text))

            text = msg_text(messages)
            if "final_answer" in text.lower() or "code" in model.lower():
                code = (
                    'items = input_datasets[list(input_datasets.keys())[-1]].items\n'
                    'final_answer({"final_items": items, "final_answer_str": "Done."})'
                )
                return _make_completion_response(f"```python\n{code}\n```")

            is_mammal = any(m in text for m in mammals)
            answer = "TRUE" if is_mammal else "FALSE"
            return _make_completion_response(f"```text\n{answer}\n```")

        mock_litellm.set_completion_handler(handler)

        execution = Execution(
            query="Find all mammals",
            datasets=[animals],
            llm_config=_LLM_CONFIG,
        )

        nl_plan, logical_plan = execution.plan()
        phase["state"] = "running"
        execution._plan = logical_plan

        items, answer_str, stats = execution.run()

        summary = stats.to_summary_dict()

        # Top-level keys
        assert "total_cost_usd" in summary
        assert "total_wall_clock_secs" in summary
        assert "total_input_tokens" in summary
        assert "total_output_tokens" in summary
        assert "planning" in summary
        assert "execution" in summary
        assert summary["query"] == "Find all mammals"

        # Planning phase has operator breakdowns
        assert len(summary["planning"]["operator_stats"]) >= 1
        assert summary["planning"]["total_cost_usd"] > 0

        # Execution phase has operator breakdowns
        assert len(summary["execution"]["operator_stats"]) >= 2  # SemFilter + Reasoning
        for op_dict in summary["execution"]["operator_stats"]:
            assert "operator_name" in op_dict
            assert "total_cost_usd" in op_dict
            assert "total_llm_calls" in op_dict

        execution.planner.cleanup()


# ---------------------------------------------------------------------------
# Test: run() pushes OperatorStats on completion events via progress queue
# ---------------------------------------------------------------------------


class TestRunProgressStats:
    """Verify that ``run()`` enriches completion events with stats."""

    def test_completed_events_carry_operator_stats(self, mock_litellm, animals):
        """After each operator, the 'Completed' event has ``operator_stats``."""

        import queue

        plan_code = (
            'ds = datasets["Animals"]\n'
            'ds = ds.sem_filter("the animal is a mammal")\n'
            "final_answer(ds.serialize())"
        )
        paraphrase_text = "Filter mammals."
        mammals = {"giraffe", "elephant"}

        call_count = {"n": 0}
        phase = {"state": "planning"}

        def handler(model, messages, **kw):
            if phase["state"] == "planning":
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return _make_completion_response(_wrap_planner_code(plan_code))
                return _make_completion_response(_wrap_paraphrase(paraphrase_text))

            text = msg_text(messages)
            if "final_answer" in text.lower() or "code" in model.lower():
                code = (
                    'items = input_datasets[list(input_datasets.keys())[-1]].items\n'
                    'final_answer({"final_items": items, "final_answer_str": "Done."})'
                )
                return _make_completion_response(f"```python\n{code}\n```")

            is_mammal = any(m in text for m in mammals)
            answer = "TRUE" if is_mammal else "FALSE"
            return _make_completion_response(f"```text\n{answer}\n```")

        mock_litellm.set_completion_handler(handler)

        execution = Execution(
            query="Find all mammals",
            datasets=[animals],
            llm_config=_LLM_CONFIG,
        )

        nl_plan, logical_plan = execution.plan()
        phase["state"] = "running"
        execution._plan = logical_plan

        progress_queue: queue.Queue = queue.Queue()
        items, answer_str, stats = execution.run(progress_queue=progress_queue)
        assert isinstance(stats, ExecutionStats)

        # Drain all progress events from the queue
        events: list[ExecutionProgress] = []
        while not progress_queue.empty():
            event_dict = progress_queue.get_nowait()
            events.append(ExecutionProgress(**event_dict))

        # Find "Completed" events (they contain operator_stats)
        completed = [e for e in events if "Completed" in e.message or "complete" in e.message.lower()]

        # At least: dataset completed, SemFilter completed, Reasoning completed
        stats_events = [e for e in completed if e.operator_stats is not None]
        assert len(stats_events) >= 1, (
            f"Expected at least 1 event with operator_stats. "
            f"Completed events: {[(e.message, e.operator_stats) for e in completed]}"
        )

        # The last completed event should be Reasoning
        last_stats_event = stats_events[-1]
        assert last_stats_event.operator_stats is not None
        assert isinstance(last_stats_event.operator_stats, OperatorStats)

        execution.planner.cleanup()

    def test_run_returns_three_tuple(self, mock_litellm, animals):
        """``run()`` return value is a 3-tuple with ``ExecutionStats``."""

        plan_code = (
            'ds = datasets["Animals"]\n'
            'ds = ds.sem_filter("the animal is a mammal")\n'
            "final_answer(ds.serialize())"
        )
        paraphrase_text = "Filter mammals."
        mammals = {"giraffe", "elephant"}

        call_count = {"n": 0}
        phase = {"state": "planning"}

        def handler(model, messages, **kw):
            if phase["state"] == "planning":
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return _make_completion_response(_wrap_planner_code(plan_code))
                return _make_completion_response(_wrap_paraphrase(paraphrase_text))

            text = msg_text(messages)
            if "final_answer" in text.lower() or "code" in model.lower():
                code = (
                    'items = input_datasets[list(input_datasets.keys())[-1]].items\n'
                    'final_answer({"final_items": items, "final_answer_str": "Done."})'
                )
                return _make_completion_response(f"```python\n{code}\n```")

            is_mammal = any(m in text for m in mammals)
            answer = "TRUE" if is_mammal else "FALSE"
            return _make_completion_response(f"```text\n{answer}\n```")

        mock_litellm.set_completion_handler(handler)

        execution = Execution(
            query="Find all mammals",
            datasets=[animals],
            llm_config=_LLM_CONFIG,
        )

        nl_plan, logical_plan = execution.plan()
        phase["state"] = "running"
        execution._plan = logical_plan

        items, answer_str, stats = execution.run()

        assert isinstance(items, list)
        assert isinstance(answer_str, str)
        assert isinstance(stats, ExecutionStats)
        assert stats.execution.phase == "execution"
        assert len(stats.execution.operator_stats) >= 2

        execution.planner.cleanup()


# ---------------------------------------------------------------------------
# Test: plan() pushes step_cost_usd on final event via progress queue
# ---------------------------------------------------------------------------


class TestPlanProgressCost:
    """Verify that ``plan()`` reports cost on the final event."""

    def test_final_event_has_step_cost(self, mock_litellm, animals):
        """The last ``PlanningProgress`` event includes ``step_cost_usd``."""

        plan_code = (
            'ds = datasets["Animals"]\n'
            'ds = ds.sem_filter("the animal is a mammal")\n'
            "final_answer(ds.serialize())"
        )
        paraphrase_text = "Filter mammals."

        call_count = {"n": 0}

        def handler(model, messages, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_completion_response(_wrap_planner_code(plan_code))
            return _make_completion_response(_wrap_paraphrase(paraphrase_text))

        mock_litellm.set_completion_handler(handler)

        execution = Execution(
            query="Find all mammals",
            datasets=[animals],
            llm_config=_LLM_CONFIG,
        )

        import queue
        progress_queue: queue.Queue = queue.Queue()
        nl_plan, logical_plan = execution.plan(progress_queue=progress_queue)

        # Drain all progress events from the queue
        events: list[PlanningProgress] = []
        while not progress_queue.empty():
            event_dict = progress_queue.get_nowait()
            events.append(PlanningProgress(**event_dict))

        assert nl_plan is not None
        assert logical_plan is not None

        # Find the final "Plan summary complete" event
        final_events = [
            e for e in events
            if "complete" in e.message.lower()
        ]
        assert len(final_events) >= 1

        last_event = final_events[-1]
        assert last_event.step_cost_usd is not None
        assert last_event.step_cost_usd > 0.0, (
            f"Expected positive step cost, got {last_event.step_cost_usd}"
        )

        execution.planner.cleanup()


# ---------------------------------------------------------------------------
# Test: run() without prior plan() still works (empty planning stats)
# ---------------------------------------------------------------------------


class TestRunWithoutPlan:
    """Verify graceful handling when ``run()`` is called without ``plan()``."""

    def test_run_without_plan_has_empty_planning_stats(self, mock_litellm, animals):
        """When ``plan()`` is skipped, ``stats.planning`` has no operators."""

        mammals = {"giraffe", "elephant"}

        def handler(model, messages, **kw):
            text = msg_text(messages)
            if "final_answer" in text.lower() or "code" in model.lower():
                code = (
                    'items = input_datasets[list(input_datasets.keys())[-1]].items\n'
                    'final_answer({"final_items": items, "final_answer_str": "Done."})'
                )
                return _make_completion_response(f"```python\n{code}\n```")

            is_mammal = any(m in text for m in mammals)
            answer = "TRUE" if is_mammal else "FALSE"
            return _make_completion_response(f"```text\n{answer}\n```")

        mock_litellm.set_completion_handler(handler)

        # Manually set a plan, skip plan()
        plan = {
            "name": "filtered_animals",
            "output_dataset_id": "filtered_animals",
            "params": {
                "operator": "SemanticFilter",
                "condition": "the animal is a mammal",
            },
            "parents": [
                {
                    "name": "Animals",
                    "output_dataset_id": "Animals",
                    "params": {},
                    "parents": [],
                }
            ],
        }

        execution = Execution(
            query="Find all mammals",
            datasets=[animals],
            plan=plan,
            llm_config=_LLM_CONFIG,
        )

        items, answer_str, stats = execution.run()

        # Planning stats should be default (empty)
        assert stats.planning.phase == "planning"
        assert len(stats.planning.operator_stats) == 0
        assert stats.planning.total_cost_usd == 0.0

        # Execution stats should still be populated
        assert len(stats.execution.operator_stats) >= 2
        assert stats.execution.total_cost_usd > 0.0
