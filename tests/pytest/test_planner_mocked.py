"""Tier 2 mocked tests for ``Planner.generate_logical_plan`` and
``Planner.paraphrase_plan``.

These tests verify that the Planner produces structurally valid plans
**without** contacting a real LLM.  The ``mock_litellm`` fixture
(from ``fixtures/mocks.py``) intercepts ``litellm.completion`` so we
can feed deterministic code-block responses and assert on the
resulting plan dictionaries.

Contracts under test (from ``Planner`` docstrings):
    - ``generate_logical_plan`` returns a ``dict`` with keys
      ``name``, ``dataset_id``, ``params``, ``parents``.
    - ``paraphrase_plan`` returns a ``str``.
    - The Planner respects ``max_steps`` and yields a
      ``FinalAnswerStep`` at the end of the stream.
"""

from __future__ import annotations

import json

import pytest
from helpers.mock_utils import msg_text

from carnot.agents.models import LiteLLMModel
from carnot.agents.planner import Planner
from carnot.data.dataset import Dataset
from carnot.data.item import DataItem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Required keys for every node produced by ``Dataset.serialize()``.
_PLAN_KEYS = {"name", "dataset_id", "params", "parents"}


def _wrap_code(code: str) -> str:
    """Wrap *code* in the markdown fenced block the Planner expects."""
    return f"Thought: mock thought\n```python\n{code}\n```"


def _make_movie_dataset() -> Dataset:
    """Minimal movie dataset used across all tests in this module."""
    return Dataset(
        name="Movies",
        annotation="A simple movie dataset",
        items=[
            DataItem.from_dict({"id": "inception", "title": "Inception", "genre": "Sci-Fi", "rating": 8.8}),
            DataItem.from_dict({"id": "prestige", "title": "The Prestige", "genre": "Mystery", "rating": 8.5}),
        ],
    )


def _validate_plan_node(node: dict) -> None:
    """Recursively assert that *node* has the required plan structure."""
    assert isinstance(node, dict), f"Expected dict, got {type(node)}"
    missing = _PLAN_KEYS - set(node.keys())
    assert not missing, f"Plan node missing keys: {missing}"
    assert isinstance(node["parents"], list)
    for parent in node["parents"]:
        _validate_plan_node(parent)


def _make_planner(mock_llm_config: dict, datasets: list[Dataset]) -> Planner:
    """Create a ``Planner`` wired to the mock LLM."""
    model = LiteLLMModel(
        model_id="gpt-4o-mini",
        api_key=mock_llm_config["OPENAI_API_KEY"],
    )
    return Planner(datasets=datasets, tools={}, model=model)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def movies() -> Dataset:
    return _make_movie_dataset()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlannerMocked:
    """Mocked Planner tests — no real LLM calls."""

    # -- Basic plan generation ------------------------------------------------

    def test_leaf_plan_returns_valid_structure(self, mock_litellm, mock_llm_config, movies):
        """A plan that simply serialises the raw dataset must have all
        required keys and an empty ``parents`` list."""
        from fixtures.mocks import _make_completion_response

        def handler(model, messages, **kw):
            return _make_completion_response(
                _wrap_code('ds = datasets["Movies"]\nfinal_answer(ds.serialize())')
            )

        mock_litellm.set_completion_handler(handler)

        planner = _make_planner(mock_llm_config, [movies])
        plan = planner.generate_logical_plan(query="List all movies")
        plan = plan.serialize()

        _validate_plan_node(plan)
        assert plan["name"] == "Movies"
        assert plan["parents"] == []
        planner.cleanup()

    def test_sem_filter_plan(self, mock_litellm, mock_llm_config, movies):
        """A ``sem_filter`` plan should have one parent (the leaf dataset)
        and operator params indicating a SemanticFilter."""
        from fixtures.mocks import _make_completion_response

        code = (
            'ds = datasets["Movies"]\n'
            'ds = ds.sem_filter("the movie is a science fiction film")\n'
            "final_answer(ds.serialize())"
        )

        def handler(model, messages, **kw):
            return _make_completion_response(_wrap_code(code))

        mock_litellm.set_completion_handler(handler)

        planner = _make_planner(mock_llm_config, [movies])
        plan = planner.generate_logical_plan(query="Find sci-fi movies")
        plan = plan.serialize()

        _validate_plan_node(plan)
        assert plan["params"].get("operator") == "SemanticFilter"
        assert len(plan["parents"]) == 1
        assert plan["parents"][0]["name"] == "Movies"
        planner.cleanup()

    def test_sem_map_plan(self, mock_litellm, mock_llm_config, movies):
        """A ``sem_map`` plan should record the output field and have one parent."""
        from fixtures.mocks import _make_completion_response

        code = (
            'ds = datasets["Movies"]\n'
            'ds = ds.sem_map("genre_class", str, "broad genre classification")\n'
            "final_answer(ds.serialize())"
        )

        def handler(model, messages, **kw):
            return _make_completion_response(_wrap_code(code))

        mock_litellm.set_completion_handler(handler)

        planner = _make_planner(mock_llm_config, [movies])
        plan = planner.generate_logical_plan(query="Classify genres")
        plan = plan.serialize()

        _validate_plan_node(plan)
        assert plan["params"].get("operator") == "SemanticMap"
        assert len(plan["parents"]) == 1
        planner.cleanup()

    def test_chained_operators(self, mock_litellm, mock_llm_config, movies):
        """Chaining two operators (filter → filter) should produce a two-deep plan tree."""
        from fixtures.mocks import _make_completion_response

        code = (
            'ds = datasets["Movies"]\n'
            'ds = ds.sem_filter("rating > 8.5")\n'
            'ds = ds.sem_filter("genre is science fiction")\n'
            "final_answer(ds.serialize())"
        )

        def handler(model, messages, **kw):
            return _make_completion_response(_wrap_code(code))

        mock_litellm.set_completion_handler(handler)

        planner = _make_planner(mock_llm_config, [movies])
        plan = planner.generate_logical_plan(query="Average rating of top movies")
        plan = plan.serialize()

        _validate_plan_node(plan)
        # Root should be the second filter, with one parent (the first filter)
        assert plan["params"].get("operator") == "SemanticFilter"
        assert len(plan["parents"]) == 1
        filter_node = plan["parents"][0]
        assert filter_node["params"].get("operator") == "SemanticFilter"
        # The first filter's parent should be the leaf dataset
        assert len(filter_node["parents"]) == 1
        assert filter_node["parents"][0]["name"] == "Movies"
        planner.cleanup()

    # -- Plan structure validation -------------------------------------------

    def test_plan_is_json_serialisable(self, mock_litellm, mock_llm_config, movies):
        """The plan dict returned by ``generate_logical_plan`` must be
        JSON-serialisable (no non-primitive types)."""
        from fixtures.mocks import _make_completion_response

        code = (
            'ds = datasets["Movies"]\n'
            'ds = ds.sem_filter("sci-fi")\n'
            "final_answer(ds.serialize())"
        )

        def handler(model, messages, **kw):
            return _make_completion_response(_wrap_code(code))

        mock_litellm.set_completion_handler(handler)

        planner = _make_planner(mock_llm_config, [movies])
        plan = planner.generate_logical_plan(query="sci-fi movies")
        plan = plan.serialize()

        json_str = json.dumps(plan)
        assert json.loads(json_str) == plan
        planner.cleanup()

    def test_plan_preserves_dataset_name(self, mock_litellm, mock_llm_config, movies):
        """Leaf nodes in the plan should carry the original dataset name."""
        from fixtures.mocks import _make_completion_response

        code = (
            'ds = datasets["Movies"]\n'
            'ds = ds.sem_filter("any condition")\n'
            "final_answer(ds.serialize())"
        )

        def handler(model, messages, **kw):
            return _make_completion_response(_wrap_code(code))

        mock_litellm.set_completion_handler(handler)

        planner = _make_planner(mock_llm_config, [movies])
        plan = planner.generate_logical_plan(query="some query")
        plan = plan.serialize()

        # Walk to the leaf
        leaf = plan["parents"][0]
        assert leaf["name"] == "Movies"
        assert leaf["parents"] == []
        planner.cleanup()

    # -- Multi-step (data-discovery → plan) ---------------------------------

    def test_multi_step_data_discovery_then_plan(self, mock_litellm, mock_llm_config, movies):
        """When the first LLM call produces a ``data_discovery()`` call and
        a subsequent call produces the ``final_answer()`` plan, the Planner
        should iterate correctly and return a valid plan.

        The data_discovery agent itself will also call ``litellm.completion``
        so we route responses by inspecting the message context.
        """
        from fixtures.mocks import _make_completion_response

        # Code for step 1: call data_discovery (no final_answer)
        discovery_code = (
            'report = data_discovery("What is the schema of the Movies dataset?")\n'
            "print(report)"
        )
        # Code for step 2: use the report and build the plan
        plan_code = (
            'ds = datasets["Movies"]\n'
            'ds = ds.sem_filter("the movie is science fiction")\n'
            "final_answer(ds.serialize())"
        )

        planner_call = {"n": 0}

        def handler(model, messages, **kw):
            all_text = msg_text(messages)

            # DataDiscoveryAgent internal call: the system message mentions
            # "data discovery" or "data_discovery" and the task mentions schema.
            if "data_discovery" in all_text and "You're a helpful agent" in all_text:
                return _make_completion_response(
                    _wrap_code('final_answer("Fields: id, title, genre, rating")')
                )

            # Planner calls
            planner_call["n"] += 1
            if planner_call["n"] <= 1:
                # Planner step 1: call data_discovery
                return _make_completion_response(_wrap_code(discovery_code))
            # Planner step 2+: build the plan
            return _make_completion_response(_wrap_code(plan_code))

        mock_litellm.set_completion_handler(handler)

        planner = _make_planner(mock_llm_config, [movies])
        plan = planner.generate_logical_plan(query="Find sci-fi movies")
        plan = plan.serialize()

        _validate_plan_node(plan)
        assert plan["params"].get("operator") == "SemanticFilter"
        planner.cleanup()

    # -- LLM interaction checks -----------------------------------------------

    def test_completion_called_at_least_once(self, mock_litellm, mock_llm_config, movies):
        """``litellm.completion`` must be called at least once during planning."""
        from fixtures.mocks import _make_completion_response

        def handler(model, messages, **kw):
            return _make_completion_response(
                _wrap_code('ds = datasets["Movies"]\nfinal_answer(ds.serialize())')
            )

        mock_litellm.set_completion_handler(handler)

        planner = _make_planner(mock_llm_config, [movies])
        planner.generate_logical_plan(query="list movies")

        assert len(mock_litellm.completion_calls) >= 1
        planner.cleanup()

    def test_query_appears_in_messages(self, mock_litellm, mock_llm_config, movies):
        """The user's query should be present in the messages sent to the LLM."""
        from fixtures.mocks import _make_completion_response

        def handler(model, messages, **kw):
            return _make_completion_response(
                _wrap_code('ds = datasets["Movies"]\nfinal_answer(ds.serialize())')
            )

        mock_litellm.set_completion_handler(handler)

        query = "Find movies directed by Nolan"
        planner = _make_planner(mock_llm_config, [movies])
        planner.generate_logical_plan(query=query)

        # Check the first completion call's messages contain the query
        _, messages, _ = mock_litellm.completion_calls[0]
        all_text = msg_text(messages)
        assert "Nolan" in all_text or "Find movies" in all_text
        planner.cleanup()

    def test_dataset_info_in_system_messages(self, mock_litellm, mock_llm_config, movies):
        """The dataset name and annotation should appear in the messages
        so the LLM knows what data is available."""
        from fixtures.mocks import _make_completion_response

        def handler(model, messages, **kw):
            return _make_completion_response(
                _wrap_code('ds = datasets["Movies"]\nfinal_answer(ds.serialize())')
            )

        mock_litellm.set_completion_handler(handler)

        planner = _make_planner(mock_llm_config, [movies])
        planner.generate_logical_plan(query="anything")

        _, messages, _ = mock_litellm.completion_calls[0]
        all_text = msg_text(messages)
        assert "Movies" in all_text
        planner.cleanup()

    # -- Paraphrase -----------------------------------------------------------

    def test_paraphrase_returns_string(self, mock_litellm, mock_llm_config, movies):
        """``paraphrase_plan`` should return a non-empty ``str``
        when the LLM responds with a plan description inside the expected
        tags."""
        from fixtures.mocks import _make_completion_response

        # Code that the planning phase LLM returns
        plan_code = (
            'ds = datasets["Movies"]\n'
            'ds = ds.sem_filter("sci-fi")\n'
            "final_answer(ds.serialize())"
        )
        # Text that the paraphrase phase LLM returns (inside plan tags)
        paraphrase_text = (
            "<begin_plan>\n"
            "1. Load the Movies dataset.\n"
            "2. Filter to keep only science fiction films.\n"
            "<end_plan>"
        )

        def handler(model, messages, **kw):
            all_text = msg_text(messages)
            # The paraphrase system prompt contains "translate" and the
            # serialised logical plan.  Use that to distinguish phases.
            if "translate" in all_text.lower() or "paraphrase" in all_text.lower():
                return _make_completion_response(paraphrase_text)
            return _make_completion_response(_wrap_code(plan_code))

        mock_litellm.set_completion_handler(handler)

        planner = _make_planner(mock_llm_config, [movies])
        plan = planner.generate_logical_plan(query="Find sci-fi movies")

        nl = planner.paraphrase_plan(
            query="Find sci-fi movies",
            logical_plan=plan.serialize(),
        )

        assert isinstance(nl, str)
        assert len(nl) > 0
        # The paraphrase text should contain something from our canned response
        assert "science fiction" in nl.lower() or "filter" in nl.lower()
        planner.cleanup()

    # -- Max-steps safety ----------------------------------------------------

    def test_max_steps_produces_fallback_answer(self, mock_litellm, mock_llm_config, movies):
        """If the LLM never calls ``final_answer``, the Planner should
        stop after ``max_steps`` and return a fallback string (not hang)."""
        from fixtures.mocks import _make_completion_response

        # Always return code that does NOT call final_answer
        def handler(model, messages, **kw):
            return _make_completion_response(
                _wrap_code('print("still thinking...")')
            )

        mock_litellm.set_completion_handler(handler)

        planner = _make_planner(mock_llm_config, [movies])
        plan = planner.generate_logical_plan(query="anything")

        # The planner should have exhausted its steps and returned a
        # fallback message string (not a dict).
        assert isinstance(plan, str)
        assert "maximum" in plan.lower() or "did not return" in plan.lower()
        planner.cleanup()
