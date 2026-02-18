"""
Unit tests for logical planning functionality.

Tests cover:
1. Data discovery report generation
2. Natural language plan generation
3. Serialized logical plan compilation
"""
import json
import os

import pytest

from carnot.agents.models import LiteLLMModel
from carnot.agents.planner import Planner


class TestDataDiscovery:
    """Tests for the data discovery phase of query planning."""

    def test_data_discovery_basic_exploration(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test that data discovery can explore a simple dataset and generate a report.
        """
        query = "What is the average rating of all movies?"
        
        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        report = planner.search_for_relevant_data(
            query=query,
            datasets=[simple_movie_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=None,
        )
        
        # Verify report structure
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Report should mention the dataset name
        assert "Movies" in report or "movies" in report.lower()
        
        # Report should mention some data-related findings
        # (it should discuss schema, fields, or sample data)
        assert any(keyword in report.lower() for keyword in [
            "rating", "field", "column", "attribute", "schema", "data", "found"
        ])
        
        planner.cleanup()

    def test_data_discovery_code_execution(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test that data discovery executes Python code to explore CSV-based datasets.
        
        This test verifies that the planner:
        1. Loads datasets into the Python executor
        2. Executes code to examine schemas
        3. Generates a report about the data structure
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        
        query = "Find the movie with the highest average review score"
        
        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        report = planner.search_for_relevant_data(
            query=query,
            datasets=[movies_dataset, reviews_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=None,
        )
        
        # Verify report is generated
        assert isinstance(report, str)
        assert len(report) > 100  # Should be a substantial report
        
        # Report should mention both datasets
        assert "Movies" in report or "movies" in report.lower()
        assert "Reviews" in report or "reviews" in report.lower()
        
        # Report should discuss data exploration
        # The planner should have examined the schema or structure
        assert any(keyword in report.lower() for keyword in [
            "field", "column", "attribute", "schema", "structure", 
            "examined", "explored", "found", "contains"
        ])
        
        planner.cleanup()


class TestNaturalLanguagePlan:
    """Tests for natural language plan generation."""

    def test_nl_plan_generation_simple(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test that a natural language plan can be generated from a simple query.
        """
        query = "What is the average rating of all movies?"
        
        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Generate a simple discovery report (or skip for this test)
        nl_plan = planner.generate_logical_plan(
            query=query,
            datasets=[simple_movie_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=None,
            data_discovery_report=None  # Can be None for simple cases
        )
        
        # Verify plan structure
        assert isinstance(nl_plan, str)
        assert len(nl_plan) > 0
        
        # Plan should mention the query goal
        assert any(keyword in nl_plan.lower() for keyword in [
            "rating", "average", "mean"
        ])
        
        # Plan should mention the dataset
        assert "Movies" in nl_plan or "movies" in nl_plan.lower()
        
        planner.cleanup()

    def test_nl_plan_with_discovery_report(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test that natural language plan generation incorporates data discovery findings.
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        
        query = "Find the movie with the highest average review score"
        
        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # First perform data discovery
        discovery_report = planner.search_for_relevant_data(
            query=query,
            datasets=[movies_dataset, reviews_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=None,
        )
        
        # Then generate plan with discovery report
        nl_plan = planner.generate_logical_plan(
            query=query,
            datasets=[movies_dataset, reviews_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=None,
            data_discovery_report=discovery_report
        )
        
        # Verify plan
        assert isinstance(nl_plan, str)
        assert len(nl_plan) > 50
        
        # Plan should reference both datasets or the join/relationship
        # The specific content depends on the discovery findings
        assert len(nl_plan.split()) > 10  # Should be a meaningful plan
        
        planner.cleanup()


class TestLogicalPlanCompilation:
    """Tests for logical plan compilation to serialized format."""

    def test_compile_simple_logical_plan(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test compilation of a simple natural language plan to a logical plan structure.
        """
        query = "What is the average rating of all movies?"
        
        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Generate NL plan
        nl_plan = planner.generate_logical_plan(
            query=query,
            datasets=[simple_movie_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=None,
            data_discovery_report=None
        )
        
        # Compile to logical plan
        compiled_plan = planner.compile_logical_plan(
            query=query,
            datasets=[simple_movie_dataset],
            nl_plan=nl_plan,
            data_discovery_report=None
        )
        
        # Verify compiled plan structure
        assert isinstance(compiled_plan, dict)
        assert "name" in compiled_plan
        assert "output_dataset_id" in compiled_plan
        assert "params" in compiled_plan
        assert "parents" in compiled_plan
        
        # Verify params contain operator info
        params = compiled_plan["params"]
        assert "operator" in params
        
        planner.cleanup()

    def test_compile_plan_with_code_operator(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test that compilation generates a plan using Code operator for schema-based lookups.
        
        This test ensures that when the discovery report reveals schema information,
        the compiled plan uses code() operations to access specific fields.
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        
        query = "What is the genre of the movie 'Inception'?"
        
        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Perform discovery
        discovery_report = planner.search_for_relevant_data(
            query=query,
            datasets=[movies_dataset, reviews_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=None,
        )
        
        # Generate NL plan
        nl_plan = planner.generate_logical_plan(
            query=query,
            datasets=[movies_dataset, reviews_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=None,
            data_discovery_report=discovery_report
        )
        
        # Compile plan
        compiled_plan = planner.compile_logical_plan(
            query=query,
            datasets=[movies_dataset, reviews_dataset],
            nl_plan=nl_plan,
            data_discovery_report=discovery_report
        )
        
        # Verify it's a valid plan structure
        assert isinstance(compiled_plan, dict)
        assert "name" in compiled_plan
        assert "params" in compiled_plan
        
        # The plan should use a Code operator (or potentially filter + code)
        # We verify by checking the operator type in the plan
        def find_code_operator(plan_dict):
            """Recursively search for Code operator in plan."""
            if plan_dict.get("params", {}).get("operator") == "Code":
                return True
            return any(find_code_operator(parent) for parent in plan_dict.get("parents", []))
        
        # For this type of query, we expect a Code operator somewhere in the plan
        has_code_op = find_code_operator(compiled_plan)
        assert has_code_op, "Expected Code operator in plan for schema-based lookup"
        
        planner.cleanup()

    def test_compile_plan_structure_validity(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test that the compiled plan has valid structure and can be serialized.
        """
        query = "Find all sci-fi movies"
        
        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        nl_plan = planner.generate_logical_plan(
            query=query,
            datasets=[simple_movie_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=None,
            data_discovery_report=None
        )
        
        compiled_plan = planner.compile_logical_plan(
            query=query,
            datasets=[simple_movie_dataset],
            nl_plan=nl_plan,
            data_discovery_report=None
        )
        
        # Verify structure
        assert isinstance(compiled_plan, dict)
        
        # Should be JSON-serializable
        json_str = json.dumps(compiled_plan)
        assert len(json_str) > 0
        
        # Can be deserialized
        reloaded = json.loads(json_str)
        assert reloaded == compiled_plan
        
        # Verify nested structure
        assert isinstance(compiled_plan["parents"], list)
        
        # Recursively verify all parents have the same structure
        def validate_plan_node(node):
            assert "name" in node
            assert "output_dataset_id" in node
            assert "params" in node
            assert "parents" in node
            for parent in node["parents"]:
                validate_plan_node(parent)
        
        validate_plan_node(compiled_plan)
        
        planner.cleanup()


class TestEndToEndPlanning:
    """End-to-end tests of the complete planning pipeline."""

    def test_full_planning_pipeline(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test the complete planning pipeline:
        1. Data discovery
        2. NL plan generation
        3. Logical plan compilation
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        
        query = "Find the movie with the highest average review score"
        
        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Step 1: Data discovery
        discovery_report = planner.search_for_relevant_data(
            query=query,
            datasets=[movies_dataset, reviews_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=None,
        )
        
        assert isinstance(discovery_report, str)
        assert len(discovery_report) > 50
        
        # Step 2: Generate NL plan
        nl_plan = planner.generate_logical_plan(
            query=query,
            datasets=[movies_dataset, reviews_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=None,
            data_discovery_report=discovery_report
        )
        
        assert isinstance(nl_plan, str)
        assert len(nl_plan) > 50
        
        # Step 3: Compile logical plan
        compiled_plan = planner.compile_logical_plan(
            query=query,
            datasets=[movies_dataset, reviews_dataset],
            nl_plan=nl_plan,
            data_discovery_report=discovery_report
        )
        
        # Verify final compiled plan
        assert isinstance(compiled_plan, dict)
        assert "name" in compiled_plan
        assert "params" in compiled_plan
        assert "operator" in compiled_plan["params"]
        
        # Verify it's a complete plan (has parent operations)
        # For this query, we expect multiple steps (join, aggregate, etc.)
        def count_operators(plan_dict):
            count = 1  # Count this node
            for parent in plan_dict.get("parents", []):
                count += count_operators(parent)
            return count
        
        total_ops = count_operators(compiled_plan)
        # Should have at least 2 operators for this complex query
        assert total_ops >= 2, f"Expected multi-step plan but got {total_ops} operators"
        
        planner.cleanup()

    def test_planning_with_code_preference(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test that the planner prefers code operations when schema information is available.
        
        After data discovery reveals the schema, the planner should use code()
        operations to access fields rather than expensive semantic operations.
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        
        # This query should use code to look up the 'director' field
        query = "Who directed the movie 'Inception'?"
        
        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Discovery should reveal the schema
        discovery_report = planner.search_for_relevant_data(
            query=query,
            datasets=[movies_dataset, reviews_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=None,
        )
        
        nl_plan = planner.generate_logical_plan(
            query=query,
            datasets=[movies_dataset, reviews_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=None,
            data_discovery_report=discovery_report
        )
        
        compiled_plan = planner.compile_logical_plan(
            query=query,
            datasets=[movies_dataset, reviews_dataset],
            nl_plan=nl_plan,
            data_discovery_report=discovery_report
        )
        
        # Extract all operators used in the plan
        def extract_operators(plan_dict):
            operators = [plan_dict.get("params", {}).get("operator", plan_dict["name"])]
            for parent in plan_dict.get("parents", []):
                operators.extend(extract_operators(parent))
            return operators
        
        operators = extract_operators(compiled_plan)
        
        # For a simple field lookup query, we expect Code operator
        # (possibly with a filter first to find the right movie)
        assert "Code" in operators, f"Expected Code operator for field lookup, got: {operators}"
        
        planner.cleanup()


# Mark tests that require API key
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
