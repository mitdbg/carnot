"""End-to-end query lifecycle tests (mocked LLM).

These tests exercise the full ``Execution.plan() → Execution.run()`` pipeline
with a mocked LLM.  Each test feeds deterministic code-block responses to the
Planner (so it produces a known logical plan) and deterministic operator
responses (so physical execution yields predictable items).

Three scenarios:

1. **Single-dataset query** — ``sem_filter`` on a small animal dataset.
2. **Multi-dataset join query** — ``sem_join`` on animals × sounds.
3. **Index-aware search query** — ``sem_topk`` on an indexed dataset.

All tests are Tier 2 (mocked) — they validate the *wiring* between
planning, operator dispatch, and result assembly without making any real
LLM calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from helpers.mock_utils import msg_text

from carnot.data.dataset import Dataset
from carnot.execution.execution import Execution

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


def _make_completion_response(content: str) -> MagicMock:
    """Create a mock litellm completion response."""
    from fixtures.mocks import _make_completion_response as _make

    return _make(content)


# ---------------------------------------------------------------------------
# Scenario 1 — Single-dataset filter
# ---------------------------------------------------------------------------


class TestE2ESingleDatasetFilter:
    """Plan → run a ``sem_filter`` on a small animal dataset."""

    @pytest.fixture()
    def animals(self) -> Dataset:
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

    def test_plan_and_run_sem_filter(self, mock_litellm, animals):
        """Full lifecycle: plan produces a sem_filter, run executes it,
        result contains only mammals."""

        plan_code = (
            'ds = datasets["Animals"]\n'
            'ds = ds.sem_filter("the animal is a mammal")\n'
            "final_answer(ds.serialize())"
        )
        paraphrase_text = "Filter Animals to keep only mammals."
        mammals = {"giraffe", "elephant"}

        call_count = {"n": 0}
        phase = {"state": "planning"}

        def phased_handler(model, messages, **kw):
            if phase["state"] == "planning":
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return _make_completion_response(_wrap_planner_code(plan_code))
                return _make_completion_response(_wrap_paraphrase(paraphrase_text))

            # Operator phase
            text = msg_text(messages)

            # ReasoningOperator (final answer)
            if "final_answer" in text.lower() or "code" in model.lower():
                code = (
                    'items = input_datasets[list(input_datasets.keys())[-1]].items\n'
                    'final_answer({"final_items": items, "final_answer_str": "Filtered mammals."})'
                )
                return _make_completion_response(f"```python\n{code}\n```")

            # SemFilter operator calls
            is_mammal = any(m in text for m in mammals)
            answer = "TRUE" if is_mammal else "FALSE"
            return _make_completion_response(f"```text\n{answer}\n```")

        mock_litellm.set_completion_handler(phased_handler)

        execution = Execution(
            query="Find all mammals",
            datasets=[animals],
            llm_config=_LLM_CONFIG,
        )

        # Plan phase
        nl_plan, logical_plan = execution.plan()

        # Switch to operator phase
        phase["state"] = "running"

        # Validate plan structure
        assert isinstance(logical_plan, dict)
        assert "params" in logical_plan
        assert logical_plan["params"].get("operator") == "SemanticFilter"
        assert logical_plan["params"].get("condition") == "the animal is a mammal"
        assert len(logical_plan["parents"]) == 1
        assert logical_plan["parents"][0]["name"] == "Animals"

        # Validate paraphrase
        assert isinstance(nl_plan, str)
        assert "mammals" in nl_plan.lower()

        # Run phase
        execution._plan = logical_plan
        items, answer_str = execution.run()

        # Only mammals should remain
        animal_names = {item["animal"] for item in items if isinstance(item, dict)}
        assert animal_names == mammals, f"Expected {mammals}, got {animal_names}"

        execution.planner.cleanup()


# ---------------------------------------------------------------------------
# Scenario 2 — Multi-dataset join
# ---------------------------------------------------------------------------


class TestE2EMultiDatasetJoin:
    """Plan → run a ``sem_join`` on animals × sounds."""

    @pytest.fixture()
    def animals(self) -> Dataset:
        return Dataset(
            name="Animals",
            annotation="Animals dataset.",
            items=[
                {"animal": "cow"},
                {"animal": "dog"},
                {"animal": "cat"},
            ],
        )

    @pytest.fixture()
    def sounds(self) -> Dataset:
        return Dataset(
            name="Sounds",
            annotation="Sounds dataset.",
            items=[
                {"sound": "moo"},
                {"sound": "woof"},
                {"sound": "meow"},
            ],
        )

    def test_plan_and_run_sem_join(self, mock_litellm, animals, sounds):
        """Full lifecycle: plan produces a sem_join, run matches animals to
        their correct sounds."""

        # The correct matches
        matches = {
            ("cow", "moo"): True,
            ("dog", "woof"): True,
            ("cat", "meow"): True,
        }

        plan_code = (
            'animals = datasets["Animals"]\n'
            'sounds = datasets["Sounds"]\n'
            'joined = animals.sem_join(sounds, "the animal makes the given sound")\n'
            "final_answer(joined.serialize())"
        )
        paraphrase_text = "Join Animals with Sounds where the animal makes the sound."

        call_count = {"n": 0}
        phase = {"state": "planning"}

        def handler(model, messages, **kw):
            if phase["state"] == "planning":
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return _make_completion_response(_wrap_planner_code(plan_code))
                return _make_completion_response(_wrap_paraphrase(paraphrase_text))

            text = msg_text(messages)

            # ReasoningOperator (final answer)
            if "final_answer" in text.lower() and ("input_datasets" in text or "code" in text.lower()):
                code = (
                    'items = input_datasets[list(input_datasets.keys())[-1]].items\n'
                    'final_answer({"final_items": items, "final_answer_str": "Joined."})'
                )
                return _make_completion_response(f"```python\n{code}\n```")

            # SemJoin operator: return TRUE if the animal-sound pair is correct.
            # Only inspect the last message (which contains the specific pair).
            pair_text = msg_text([messages[-1]])
            is_match = False
            for (animal, sound), _ in matches.items():
                if animal in pair_text and sound in pair_text:
                    is_match = True
                    break
            answer = "TRUE" if is_match else "FALSE"
            return _make_completion_response(f"```text\n{answer}\n```")

        mock_litellm.set_completion_handler(handler)

        execution = Execution(
            query="Match animals to their sounds",
            datasets=[animals, sounds],
            llm_config=_LLM_CONFIG,
        )

        nl_plan, logical_plan = execution.plan()
        phase["state"] = "running"

        # Validate plan structure
        assert logical_plan["params"].get("operator") == "SemanticJoin"
        assert len(logical_plan["parents"]) == 2
        parent_names = {p["name"] for p in logical_plan["parents"]}
        assert parent_names == {"Animals", "Sounds"}

        # Run
        execution._plan = logical_plan
        items, answer_str = execution.run()

        # Should have 3 matched pairs
        assert len(items) == 3, f"Expected 3 joined rows, got {len(items)}"

        # Verify each match is correct
        for item in items:
            if isinstance(item, dict) and "animal" in item and "sound" in item:
                animal = item["animal"]
                sound = item["sound"]
                assert (animal, sound) in matches, (
                    f"Unexpected join pair: ({animal}, {sound})"
                )

        execution.planner.cleanup()


# ---------------------------------------------------------------------------
# Scenario 3 — Index-aware top-k search
# ---------------------------------------------------------------------------


class TestE2EIndexAwareTopK:
    """Plan → run a ``sem_topk`` on a dataset with a pre-built index."""

    @pytest.fixture()
    def indexed_animals(self, mock_litellm) -> Dataset:
        """Animals dataset with a mock 'chroma' index pre-built."""
        items = [
            {"animal": "giraffe", "habitat": "savanna"},
            {"animal": "penguin", "habitat": "antarctic"},
            {"animal": "kangaroo", "habitat": "outback"},
            {"animal": "polar bear", "habitat": "arctic"},
            {"animal": "camel", "habitat": "desert"},
        ]
        ds = Dataset(
            name="Animals",
            annotation="Animals with habitats.",
            items=items,
        )

        # Create a mock index that returns items based on keyword matching
        mock_index = MagicMock()
        mock_index.search = MagicMock(
            side_effect=lambda query, k: [
                item for item in items
                if "cold" in query.lower()
                and item["habitat"] in ("antarctic", "arctic")
            ][:k]
        )
        ds.indices["chroma"] = mock_index

        return ds

    def test_plan_and_run_sem_topk(self, mock_litellm, indexed_animals):
        """Full lifecycle: plan produces a sem_topk, run retrieves items
        via the index."""

        plan_code = (
            'ds = datasets["Animals"]\n'
            'ds = ds.sem_topk(index_name="chroma", search_str="animals that live in cold climates", k=2)\n'
            "final_answer(ds.serialize())"
        )
        paraphrase_text = "Search for cold-climate animals using the chroma index."

        call_count = {"n": 0}
        phase = {"state": "planning"}

        def handler(model, messages, **kw):
            if phase["state"] == "planning":
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return _make_completion_response(_wrap_planner_code(plan_code))
                return _make_completion_response(_wrap_paraphrase(paraphrase_text))

            # ReasoningOperator (only LLM call during run for sem_topk)
            code = (
                'items = input_datasets[list(input_datasets.keys())[-1]].items\n'
                'final_answer({"final_items": items, "final_answer_str": "Cold-climate animals."})'
            )
            return _make_completion_response(f"```python\n{code}\n```")

        mock_litellm.set_completion_handler(handler)

        execution = Execution(
            query="Find animals that live in cold climates",
            datasets=[indexed_animals],
            llm_config=_LLM_CONFIG,
        )

        nl_plan, logical_plan = execution.plan()
        phase["state"] = "running"

        # Validate plan structure
        assert logical_plan["params"].get("operator") == "SemanticTopK"
        assert logical_plan["params"]["k"] == 2
        assert "cold" in logical_plan["params"]["search_str"].lower()
        assert len(logical_plan["parents"]) == 1
        assert logical_plan["parents"][0]["name"] == "Animals"

        # Run
        execution._plan = logical_plan
        items, answer_str = execution.run()

        # The mock index returns penguin + polar bear for "cold" queries
        assert len(items) >= 1, f"Expected cold-climate animals, got {items}"
        habitats = {item["habitat"] for item in items if isinstance(item, dict)}
        assert habitats <= {"antarctic", "arctic"}, (
            f"Expected cold habitats, got {habitats}"
        )

        # Verify the mock index was actually called
        indexed_animals.indices["chroma"].search.assert_called_once()

        execution.planner.cleanup()
