"""
Unit tests for logical planning functionality.

Tests cover:
1. Data discovery agent functionality
2. Logical plan generation (code-based)
3. Plan paraphrasing to natural language
"""
import json

import pytest
from helpers.assertions import (
    assert_agent_did_not_hit_max_steps,
    assert_planner_did_not_hit_max_steps,
)

from carnot.agents.data_discovery import DataDiscoveryAgent
from carnot.agents.models import LiteLLMModel
from carnot.agents.planner import Planner

pytestmark = pytest.mark.llm


class TestDataDiscoveryAgent:
    """Tests for the DataDiscoveryAgent which handles data discovery."""

    def test_data_discovery_basic_exploration(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test that the DataDiscoveryAgent can explore a simple dataset and generate a report.
        """
        model = LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        
        discovery_agent = DataDiscoveryAgent(
            datasets=[simple_movie_dataset],
            model=model,
        )
        
        report = discovery_agent.run(
            task="What is the schema of the Movies dataset? What fields are available?"
        )
        
        # Verify agent didn't hit max_steps
        assert_agent_did_not_hit_max_steps(discovery_agent)
        
        # Verify report structure
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Report should mention the dataset name
        assert "Movies" in report or "movies" in report.lower()
        
        # Report should mention some data-related findings
        assert any(keyword in report.lower() for keyword in [
            "rating", "field", "column", "attribute", "schema", "data", "found"
        ])
        
        discovery_agent.cleanup()

    def test_data_discovery_code_execution(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test that data discovery executes Python code to explore CSV-based datasets.
        
        This test verifies that the agent:
        1. Loads datasets into the Python executor
        2. Executes code to examine schemas
        3. Generates a report about the data structure
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        model = LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        
        discovery_agent = DataDiscoveryAgent(
            datasets=[movies_dataset, reviews_dataset],
            model=model,
        )
        
        report = discovery_agent.run(
            task="Examine the schema of both datasets. Which dataset is relevant for finding review scores?"
        )
        
        # Verify agent didn't hit max_steps
        assert_agent_did_not_hit_max_steps(discovery_agent)
        
        # Verify report is generated
        assert isinstance(report, str)
        assert len(report) > 100  # Should be a substantial report
        
        # Report should mention both datasets
        assert "Movies" in report or "movies" in report.lower()
        assert "Reviews" in report or "reviews" in report.lower()
        
        # Report should discuss data exploration
        assert any(keyword in report.lower() for keyword in [
            "field", "column", "attribute", "schema", "structure", 
            "examined", "explored", "found", "contains"
        ])
        
        discovery_agent.cleanup()


class TestLogicalPlanGeneration:
    """Tests for code-based logical plan generation."""

    def test_logical_plan_generation_simple(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test that a logical plan can be generated from a simple query.
        """
        query = "What is the average rating of all movies?"
        
        planner = Planner(
            datasets=[simple_movie_dataset],
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Generate logical plan (code-based)
        logical_plan = planner.generate_logical_plan(
            query=query,
            conversation=None,
        )
        
        # Verify planner didn't hit max_steps
        assert_planner_did_not_hit_max_steps(planner, logical_plan)
        
        # Verify plan structure
        assert isinstance(logical_plan, dict)
        assert "name" in logical_plan
        assert "output_dataset_id" in logical_plan
        assert "params" in logical_plan
        assert "parents" in logical_plan
        
        planner.cleanup()

    def test_logical_plan_with_managed_agent(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test that logical plan generation can use the managed DataDiscoveryAgent.
        
        The Planner should be able to call its managed agent to discover
        relevant data before building the logical plan.
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        
        query = "Find the movie with the highest average review score"
        
        planner = Planner(
            datasets=[movies_dataset, reviews_dataset],
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Verify the managed agent is available
        assert "data_discovery" in planner.managed_agents
        
        # Generate logical plan - the planner can use its managed agent internally
        logical_plan = planner.generate_logical_plan(
            query=query,
            conversation=None,
        )
        
        # Verify planner didn't hit max_steps
        assert_planner_did_not_hit_max_steps(planner, logical_plan)
        
        # Verify plan structure
        assert isinstance(logical_plan, dict)
        assert "name" in logical_plan
        assert "params" in logical_plan
        
        planner.cleanup()


class TestPlanParaphrasing:
    """Tests for paraphrasing logical plans to natural language."""

    def test_paraphrase_simple_logical_plan(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test paraphrasing of a logical plan to natural language.
        """
        query = "What is the average rating of all movies?"
        
        planner = Planner(
            datasets=[simple_movie_dataset],
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Generate logical plan
        logical_plan = planner.generate_logical_plan(
            query=query,
            conversation=None,
        )
        
        # Verify planner didn't hit max_steps for logical plan
        assert_planner_did_not_hit_max_steps(planner, logical_plan)
        
        # Paraphrase to natural language
        nl_plan = planner.paraphrase_logical_plan(
            query=query,
            logical_plan=logical_plan,
            conversation=None,
        )
        
        # Verify planner didn't hit max_steps for paraphrase
        assert_planner_did_not_hit_max_steps(planner, nl_plan)
        
        # Verify NL plan structure
        assert isinstance(nl_plan, str)
        assert len(nl_plan) > 0
        
        # NL plan should mention the query goal
        assert any(keyword in nl_plan.lower() for keyword in [
            "rating", "average", "mean", "movie"
        ])
        
        planner.cleanup()

    def test_logical_plan_structure_validity(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test that the logical plan has valid structure and can be serialized.
        """
        query = "Find all sci-fi movies"
        
        planner = Planner(
            datasets=[simple_movie_dataset],
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        logical_plan = planner.generate_logical_plan(
            query=query,
            conversation=None,
        )
        
        # Verify planner didn't hit max_steps
        assert_planner_did_not_hit_max_steps(planner, logical_plan)
        
        # Verify structure
        assert isinstance(logical_plan, dict)
        
        # Should be JSON-serializable
        json_str = json.dumps(logical_plan)
        assert len(json_str) > 0
        
        # Can be deserialized
        reloaded = json.loads(json_str)
        assert reloaded == logical_plan
        
        # Verify nested structure
        assert isinstance(logical_plan["parents"], list)
        
        # Recursively verify all parents have the same structure
        def validate_plan_node(node):
            assert "name" in node
            assert "output_dataset_id" in node
            assert "params" in node
            assert "parents" in node
            for parent in node["parents"]:
                validate_plan_node(parent)
        
        validate_plan_node(logical_plan)
        
        planner.cleanup()

    def test_plan_with_code_operator(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test that plans use Code operator for schema-based lookups.
        
        This test ensures that when the planner discovers schema information,
        it uses code() operations to access specific fields.
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        
        query = "What is the genre of the movie 'Inception'?"
        
        planner = Planner(
            datasets=[movies_dataset, reviews_dataset],
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Generate plan - the planner can use its managed agent to discover schema
        logical_plan = planner.generate_logical_plan(
            query=query,
            conversation=None,
        )
        
        # Verify planner didn't hit max_steps
        assert_planner_did_not_hit_max_steps(planner, logical_plan)
        
        # Verify it's a valid plan structure
        assert isinstance(logical_plan, dict)
        assert "name" in logical_plan
        assert "params" in logical_plan
        
        # The plan should use a Code operator (or potentially filter + code)
        def find_code_operator(plan_dict):
            """Recursively search for Code operator in plan."""
            if plan_dict.get("params", {}).get("operator") == "Code":
                return True
            return any(find_code_operator(parent) for parent in plan_dict.get("parents", []))
        
        # For this type of query, we expect a Code operator somewhere in the plan
        has_code_op = find_code_operator(logical_plan)
        assert has_code_op, "Expected Code operator in plan for schema-based lookup"
        
        planner.cleanup()


class TestEndToEndPlanning:
    """End-to-end tests of the complete planning pipeline."""

    def test_full_planning_pipeline(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test the complete planning pipeline:
        1. Logical plan generation (with managed agent for discovery)
        2. Plan paraphrasing to natural language
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        
        query = "Find the movie with the highest average review score"
        
        planner = Planner(
            datasets=[movies_dataset, reviews_dataset],
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Step 1: Generate the logical plan (code-based)
        # The planner can call its managed DataDiscoveryAgent during this phase
        logical_plan = planner.generate_logical_plan(
            query=query,
            conversation=None,
        )
        
        # Verify planner didn't hit max_steps for logical plan
        assert_planner_did_not_hit_max_steps(planner, logical_plan)
        
        # Verify logical plan
        assert isinstance(logical_plan, dict)
        assert "name" in logical_plan
        assert "params" in logical_plan
        assert "operator" in logical_plan["params"]
        
        # For this query, we expect multiple steps (join, aggregate, etc.)
        def count_operators(plan_dict):
            count = 1  # Count this node
            for parent in plan_dict.get("parents", []):
                count += count_operators(parent)
            return count
        
        total_ops = count_operators(logical_plan)
        # Should have at least 2 operators for this complex query
        assert total_ops >= 2, f"Expected multi-step plan but got {total_ops} operators"
        
        # Step 2: Paraphrase to natural language
        nl_plan = planner.paraphrase_logical_plan(
            query=query,
            logical_plan=logical_plan,
            conversation=None,
        )
        
        # Verify planner didn't hit max_steps for paraphrase
        assert_planner_did_not_hit_max_steps(planner, nl_plan)
        
        assert isinstance(nl_plan, str)
        assert len(nl_plan) > 50
        
        planner.cleanup()

    def test_planning_with_code_preference(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test that the planner prefers code operations when schema information is available.
        
        After using the DataDiscoveryAgent to learn the schema, the planner should use 
        code() operations to access fields rather than expensive semantic operations.
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        
        # This query should use code to look up the 'director' field
        query = "Who directed the movie 'Inception'?"
        
        planner = Planner(
            datasets=[movies_dataset, reviews_dataset],
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Generate logical plan - planner can use managed agent to discover schema
        logical_plan = planner.generate_logical_plan(
            query=query,
            conversation=None,
        )
        
        # Verify planner didn't hit max_steps
        assert_planner_did_not_hit_max_steps(planner, logical_plan)
        
        # Extract all operators used in the plan
        def extract_operators(plan_dict):
            operators = [plan_dict.get("params", {}).get("operator", plan_dict["name"])]
            for parent in plan_dict.get("parents", []):
                operators.extend(extract_operators(parent))
            return operators
        
        operators = extract_operators(logical_plan)
        
        # For a simple field lookup query, we expect Code operator
        # (possibly with a filter first to find the right movie)
        assert "Code" in operators, f"Expected Code operator for field lookup, got: {operators}"
        
        planner.cleanup()
