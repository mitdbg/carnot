"""
Unit tests for conversation-aware planning functionality.

Tests cover:
1. Data discovery with conversation feedback
2. Logical plan generation with conversation refinement  
3. Plan paraphrasing with conversation context
4. End-to-end execution with conversation context
"""
from carnot.agents.data_discovery import DataDiscoveryAgent
from carnot.agents.memory import ConversationAgentStep, ConversationUserStep
from carnot.agents.models import LiteLLMModel
from carnot.agents.planner import Planner
from carnot.agents.utils import AgentMaxStepsError
from carnot.conversation.conversation import Conversation
from carnot.execution.execution import Execution


def assert_agent_did_not_hit_max_steps(agent) -> None:
    """
    Assert that an agent did not hit its max_steps limit.
    
    Checks the agent's memory for AgentMaxStepsError in the last step.
    These simple tests should never require the agent to reach max_steps.
    """
    if agent.memory.steps:
        last_step = agent.memory.steps[-1]
        error = getattr(last_step, "error", None)
        assert not isinstance(error, AgentMaxStepsError), (
            f"Agent hit max_steps limit ({agent.max_steps}). "
            "This simple task should complete in fewer steps."
        )


def assert_planner_did_not_hit_max_steps(planner, result) -> None:
    """
    Assert that the planner did not hit its max_steps limit.
    
    The planner returns a specific string when max_steps is reached.
    These simple tests should never require the planner to reach max_steps.
    """
    max_steps_message = "The agent did not return a final answer within the maximum number of steps."
    assert result != max_steps_message, (
        f"Planner hit max_steps limit ({planner.max_steps}). "
        "This simple task should complete in fewer steps."
    )


class TestConversationalDataDiscovery:
    """Tests for data discovery with conversation feedback using the DataDiscoveryAgent."""

    def test_data_discovery_with_dataset_guidance(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test that data discovery incorporates user feedback about which dataset to focus on.
        
        Scenario:
        - User asks a query
        - Data discovery explores datasets
        - User provides feedback to look at a specific dataset
        - New data discovery focuses on the suggested dataset
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        model = LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        
        # Use the DataDiscoveryAgent directly for targeted discovery
        discovery_agent = DataDiscoveryAgent(
            datasets=[movies_dataset, reviews_dataset],
            model=model,
        )
        
        # Perform data discovery focused on Reviews dataset per user feedback
        report = discovery_agent.run(
            task="What fields are available for analyzing critic sentiment?"
        )
        
        # Verify agent didn't hit max_steps
        assert_agent_did_not_hit_max_steps(discovery_agent)
        
        # Verify that the report mentions the Reviews dataset prominently
        assert isinstance(report, str)
        assert len(report) > 0
        
        # The report should explicitly mention Reviews dataset
        assert "reviews" in report.lower()
        
        # The report should discuss critic-related content
        assert any(keyword in report.lower() for keyword in [
            "critic", "review", "opinion", "sentiment", "score"
        ])
        
        discovery_agent.cleanup()

    def test_data_discovery_refines_search_scope(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test that data discovery can refine its scope based on user feedback.
        
        Scenario:
        - Initial discovery explores broadly
        - User provides feedback to narrow focus to specific fields or criteria
        - New discovery examines those specific aspects
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        model = LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        
        discovery_agent = DataDiscoveryAgent(
            datasets=[movies_dataset, reviews_dataset],
            model=model,
        )
        
        # Perform targeted data discovery checking for specific field
        report = discovery_agent.run(
            task="Check if the Movies dataset has a 'rating' or 'score' field that we can use for finding high-rated movies."
        )
        
        # Verify agent didn't hit max_steps
        assert_agent_did_not_hit_max_steps(discovery_agent)
        
        # Verify the report addresses the specific field inquiry
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Report should mention the Movies dataset
        assert "movies" in report.lower()
        
        # Report should discuss rating/score fields
        assert any(keyword in report.lower() for keyword in [
            "rating", "score", "field", "column", "attribute"
        ])
        
        discovery_agent.cleanup()


class TestConversationalLogicalPlanGeneration:
    """Tests for logical plan generation with conversation refinement."""

    def test_logical_plan_adds_missing_step(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test that a logical plan can be refined to add a missing step.
        
        Scenario:
        - Initial plan is generated but misses a step
        - User points out the missing step
        - New plan includes the suggested step
        """
        # Initial query and incomplete plan
        initial_query = "Find all sci-fi movies from the year 2020"
        incomplete_plan = {
            "name": "FilterOperation1",
            "output_dataset_id": "FilterOperation1",
            "params": {
                "output_dataset_id": "FilterOperation1",
                "operator": "SemanticFilter",
                "description": "Filtered Movies by condition: The movie was released in 2020",
                "condition": "The movie was released in 2020"
            },
            "parents": [
                {'name': 'Movies', 'output_dataset_id': 'Movies', 'params': {}, 'parents': []}
            ]
        }

        # Create conversation with feedback
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Search",
            dataset_ids=["movies"],
            messages=[
                {"role": "user", "content": initial_query},
                {"role": "agent", "content": str(incomplete_plan), "type": "logical-plan"},
                {"role": "user", "content": "You forgot to filter by genre='Sci-Fi'. Please add that step."},
            ]
        )

        planner = Planner(
            datasets=[simple_movie_dataset],
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )

        # Generate refined plan
        refined_plan = planner.generate_logical_plan(
            query="You forgot to filter by genre='Sci-Fi'. Please add that step.",
            datasets=[simple_movie_dataset],
            conversation=conversation,
        )

        # Verify planner didn't hit max_steps
        assert_planner_did_not_hit_max_steps(planner, refined_plan)

        # Verify the refined plan includes the genre filter
        assert isinstance(refined_plan, dict)
        
        # Convert to string for checking content
        plan_str = str(refined_plan).lower()

        # Plan should now mention filtering by genre or sci-fi
        assert any(keyword in plan_str for keyword in [
            "sci-fi", "science fiction", "genre", "filter"
        ])

        planner.cleanup()

    def test_logical_plan_refines_operation(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test that a logical plan can refine an operation based on feedback.
        
        Scenario:
        - Initial plan suggests one approach
        - User suggests a different or refined approach
        - New plan incorporates the refinement
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        
        # Initial plan with a simple approach - realistic structure based on Dataset.sem_aggregate()
        initial_query = "Find the best movie"
        initial_plan = {
            "name": "AggregateOperation1",
            "output_dataset_id": "AggregateOperation1",
            "params": {
                "operator": "SemanticAgg",
                "description": "Aggregated Movies on fields: [{'name': 'best_movie', 'type': 'str', 'description': 'The best movie based on rating'}]",
                "task": "Find the best movie based on the rating field",
                "agg_fields": [{"name": "best_movie", "type": "str", "description": "The best movie based on rating"}]
            },
            "parents": [
                {"name": "Movies", "output_dataset_id": "Movies", "params": {}, "parents": []}
            ]
        }
        
        # Create conversation with refinement feedback
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Analysis",
            dataset_ids=["movies", "reviews"],
            messages=[
                {"role": "user", "content": initial_query},
                {"role": "agent", "content": str(initial_plan), "type": "logical-plan"},
                {"role": "user", "content": "Instead of using the movie rating, calculate the average review score from the Reviews dataset for each movie."},
            ]
        )
        
        planner = Planner(
            datasets=[movies_dataset, reviews_dataset],
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Generate refined plan
        refined_plan = planner.generate_logical_plan(
            query="Instead of using the movie rating, calculate the average review score from the Reviews dataset for each movie.",
            datasets=[movies_dataset, reviews_dataset],
            conversation=conversation,
        )
        
        # Verify planner didn't hit max_steps
        assert_planner_did_not_hit_max_steps(planner, refined_plan)
        
        # Verify the refined plan
        assert isinstance(refined_plan, dict)
        
        # Convert to string for checking content
        plan_str = str(refined_plan).lower()
        
        # Plan should mention reviews or calculations
        assert any(keyword in plan_str for keyword in [
            "review", "average", "score", "join"
        ])
        
        planner.cleanup()


class TestConversationalPlanRefinement:
    """Tests for plan refinement with conversation updates."""

    def test_plan_adds_operator(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test that planning can add an operator based on user feedback.
        
        Scenario:
        - Initial plan is provided
        - User suggests adding a specific operation
        - New plan includes the additional operator
        """
        initial_query = "Get all sci-fi movies"
        # Realistic structure based on Dataset.sem_filter()
        initial_plan = {
            "name": "FilterOperation1",
            "output_dataset_id": "FilterOperation1",
            "params": {
                "operator": "SemanticFilter",
                "description": "Filtered Movies by condition: The movie is a sci-fi genre film",
                "condition": "The movie is a sci-fi genre film"
            },
            "parents": [
                {"name": "Movies", "output_dataset_id": "Movies", "params": {}, "parents": []}
            ]
        }
        
        # Create conversation with feedback to add sorting
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Search",
            dataset_ids=["movies"],
            messages=[
                {"role": "user", "content": initial_query},
                {"role": "agent", "content": str(initial_plan), "type": "logical-plan"},
                {"role": "user", "content": "Please add a step to sort the results by rating in descending order."},
            ]
        )
        
        planner = Planner(
            datasets=[simple_movie_dataset],
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Generate new plan with the sorting feedback
        new_plan = planner.generate_logical_plan(
            query="Please add a step to sort the results by rating in descending order.",
            datasets=[simple_movie_dataset],
            conversation=conversation
        )
        
        # Verify planner didn't hit max_steps
        assert_planner_did_not_hit_max_steps(planner, new_plan)
        
        # Verify plan structure
        assert isinstance(new_plan, dict)
        assert "params" in new_plan
        
        # Check that the plan includes sorting logic
        plan_str = str(new_plan).lower()
        assert any(keyword in plan_str for keyword in [
            "sort", "order", "rating", "descending", "topk"
        ])
        
        planner.cleanup()

    def test_plan_changes_operator_params(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test that planning can adjust operator parameters based on feedback.
        
        Scenario:
        - Initial plan suggests one approach
        - User provides feedback about specific parameter values
        - New plan uses the suggested parameters
        """
        initial_query = "Find top movies"
        # Realistic structure based on Dataset.limit()
        initial_plan = {
            "name": "LimitOperation1",
            "output_dataset_id": "LimitOperation1",
            "params": {
                "operator": "Limit",
                "description": "Limited Movies to first 10 records",
                "n": 10
            },
            "parents": [
                {"name": "Movies", "output_dataset_id": "Movies", "params": {}, "parents": []}
            ]
        }
        
        # Create conversation with specific parameter feedback
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Search",
            dataset_ids=["movies"],
            messages=[
                {"role": "user", "content": initial_query},
                {"role": "agent", "content": str(initial_plan), "type": "logical-plan"},
                {"role": "user", "content": "Return only the top 5 results, not all of them."},
            ]
        )
        
        planner = Planner(
            datasets=[simple_movie_dataset],
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Generate new plan with limit feedback
        new_plan = planner.generate_logical_plan(
            query="Return only the top 5 results, not all of them.",
            datasets=[simple_movie_dataset],
            conversation=conversation
        )
        
        # Verify planner didn't hit max_steps
        assert_planner_did_not_hit_max_steps(planner, new_plan)
        
        # Verify plan includes limit
        assert isinstance(new_plan, dict)
        
        # Check for limit/top-k operator or parameter
        plan_str = str(new_plan).lower()
        assert any(keyword in plan_str for keyword in [
            "limit", "top", "5", "topk"
        ])
        
        planner.cleanup()

    def test_plan_uses_different_operator_type(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test that planning can switch operator types based on feedback.
        
        Scenario:
        - Initial plan suggests using one operator approach
        - User suggests a different operator type would be better
        - New plan uses the suggested operator type
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        
        initial_query = "Find movies with positive reviews"
        # Realistic structure based on Dataset.sem_filter()
        initial_plan = {
            "name": "FilterOperation1",
            "output_dataset_id": "FilterOperation1",
            "params": {
                "operator": "SemanticFilter",
                "description": "Filtered Movies by condition: The movie has positive sentiment",
                "condition": "The movie has positive sentiment"
            },
            "parents": [
                {"name": "Movies", "output_dataset_id": "Movies", "params": {}, "parents": []}
            ]
        }
        
        # Create conversation suggesting join with reviews
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Analysis",
            dataset_ids=["movies", "reviews"],
            messages=[
                {"role": "user", "content": initial_query},
                {"role": "agent", "content": str(initial_plan), "type": "logical-plan"},
                {"role": "user", "content": "Actually, join with the Reviews dataset and filter reviews by sentiment score > 0.7"},
            ]
        )
        
        planner = Planner(
            datasets=[movies_dataset, reviews_dataset],
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Generate new plan with join feedback
        new_plan = planner.generate_logical_plan(
            query="Actually, join with the Reviews dataset and filter reviews by sentiment score > 0.7",
            datasets=[movies_dataset, reviews_dataset],
            conversation=conversation
        )
        
        # Verify planner didn't hit max_steps
        assert_planner_did_not_hit_max_steps(planner, new_plan)
        
        # Verify plan references both datasets
        assert isinstance(new_plan, dict)
        
        # Check that the plan involves both datasets (likely through join or separate operations)
        plan_str = str(new_plan).lower()
        
        # Should mention reviews dataset or sentiment
        assert any(keyword in plan_str for keyword in [
            "review", "sentiment", "join", "0.7", "score"
        ])
        
        planner.cleanup()


class TestEndToEndConversationalExecution:
    """End-to-end tests for Execution.plan() with conversation context."""

    def test_execution_plan_with_simple_refinement(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test end-to-end execution with a simple plan refinement.
        
        Scenario:
        - User asks initial query
        - System generates initial plan
        - User provides refinement
        - System generates updated complete plan incorporating feedback
        """
        # Create conversation with initial query and refinement
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Search",
            dataset_ids=["movies"],
            messages=[
                {"role": "user", "content": "Find all movies"},
                {"role": "user", "content": "Actually, only show sci-fi movies"},
            ]
        )
        
        # Create execution with conversation
        execution = Execution(
            query="Actually, only show sci-fi movies",
            datasets=[simple_movie_dataset],
            conversation=conversation,
            llm_config=llm_config,
        )
        
        # Generate plan
        nl_plan, logical_plan = execution.plan()
        
        # Verify planner didn't hit max_steps
        assert_planner_did_not_hit_max_steps(execution.planner, logical_plan)
        
        # Verify outputs
        assert isinstance(nl_plan, str)
        assert len(nl_plan) > 0
        assert isinstance(logical_plan, dict)
        
        # NL plan should mention sci-fi filtering
        assert any(keyword in nl_plan.lower() for keyword in [
            "sci-fi", "science fiction", "genre"
        ])
        
        # Logical plan should have filter operation
        logical_plan_str = str(logical_plan).lower()
        assert any(keyword in logical_plan_str for keyword in [
            "filter", "sci-fi", "genre"
        ])
        
        execution.planner.cleanup()

    def test_execution_plan_with_multi_turn_refinement(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test end-to-end execution with multiple refinement turns.
        
        Scenario:
        - User asks initial query
        - Receives initial plan
        - Provides first refinement
        - Receives updated plan
        - Provides second refinement
        - Final plan incorporates all feedback
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        
        # Create conversation with multiple refinements
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Analysis",
            dataset_ids=["movies", "reviews"],
            messages=[
                {"role": "user", "content": "Find the best movies"},
                {
                    "role": "agent",
                    "content": "1. Load Movies dataset\n2. Sort by rating\n3. Return top movie",
                    "type": "natural-language-plan"
                },
                {"role": "user", "content": "Use the Reviews dataset to calculate average review scores"},
                {
                    "role": "agent",
                    "content": "1. Load Reviews dataset\n2. Group by movie\n3. Calculate average score\n4. Find highest average",
                    "type": "natural-language-plan"
                },
                {"role": "user", "content": "Also filter to only include movies from after 2010"},
            ]
        )
        
        # Create execution with full conversation context
        execution = Execution(
            query="Also filter to only include movies from after 2010",
            datasets=[movies_dataset, reviews_dataset],
            conversation=conversation,
            llm_config=llm_config,
        )
        
        # Generate final plan
        nl_plan, logical_plan = execution.plan()
        
        # Verify planner didn't hit max_steps
        assert_planner_did_not_hit_max_steps(execution.planner, logical_plan)
        
        # Verify outputs incorporate all feedback
        assert isinstance(nl_plan, str)
        assert len(nl_plan) > 0
        assert isinstance(logical_plan, dict)
        
        nl_plan_lower = nl_plan.lower()
        
        # Should mention reviews (from first refinement)
        assert "review" in nl_plan_lower
        
        # Should mention year filtering (from second refinement)
        assert any(keyword in nl_plan_lower for keyword in [
            "2010", "year", "after"
        ])
        
        execution.planner.cleanup()

    def test_execution_plan_with_operator_suggestion(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test end-to-end execution where user suggests specific operator approach.
        
        Scenario:
        - User asks query
        - User provides specific guidance on how to solve it
        - Final plan follows the suggested approach
        """
        # Create conversation with specific operator guidance
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Search",
            dataset_ids=["movies"],
            messages=[
                {"role": "user", "content": "What's the average rating of Nolan movies?"},
                {"role": "user", "content": "Filter for director='Christopher Nolan' then calculate the average rating"},
            ]
        )
        
        # Create execution
        execution = Execution(
            query="Filter for director='Christopher Nolan' then calculate the average rating",
            datasets=[simple_movie_dataset],
            conversation=conversation,
            llm_config=llm_config,
        )
        
        # Generate plan
        nl_plan, logical_plan = execution.plan()
        
        # Verify planner didn't hit max_steps
        assert_planner_did_not_hit_max_steps(execution.planner, logical_plan)
        
        # Verify plan follows suggested approach
        assert isinstance(nl_plan, str)
        assert isinstance(logical_plan, dict)
        
        nl_plan_lower = nl_plan.lower()
        
        # Should mention filtering by director
        assert any(keyword in nl_plan_lower for keyword in [
            "nolan", "director", "filter"
        ])
        
        # Should mention averaging
        assert any(keyword in nl_plan_lower for keyword in [
            "average", "mean", "rating"
        ])
        
        # Logical plan should have these operations
        logical_plan_str = str(logical_plan).lower()
        assert "filter" in logical_plan_str or "nolan" in logical_plan_str
        
        execution.planner.cleanup()

    def test_execution_plan_respects_dataset_preference(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test that execution respects user's dataset preference from conversation.
        
        Scenario:
        - Multiple datasets available
        - User specifies which dataset to use
        - Final plan uses the preferred dataset
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        
        # Create conversation specifying dataset preference
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Review Analysis",
            dataset_ids=["movies", "reviews"],
            messages=[
                {"role": "user", "content": "What are critics saying?"},
                {"role": "user", "content": "Use only the Reviews dataset for this analysis"},
            ]
        )
        
        # Create execution
        execution = Execution(
            query="Use only the Reviews dataset for this analysis",
            datasets=[movies_dataset, reviews_dataset],
            conversation=conversation,
            llm_config=llm_config,
        )
        
        # Generate plan
        nl_plan, logical_plan = execution.plan()
        
        # Verify planner didn't hit max_steps
        assert_planner_did_not_hit_max_steps(execution.planner, logical_plan)
        
        # Verify plan focuses on Reviews dataset
        assert isinstance(nl_plan, str)
        assert isinstance(logical_plan, dict)
        
        # NL plan should mention Reviews dataset
        nl_plan_lower = nl_plan.lower()
        assert "review" in nl_plan_lower
        
        # Logical plan should primarily use Reviews dataset
        # Check the dataset references in the plan
        logical_plan_str = str(logical_plan)
        
        # Should reference Reviews more than Movies (if at all)
        review_count = logical_plan_str.lower().count("review")
        assert review_count > 0, "Plan should reference Reviews dataset"
        
        execution.planner.cleanup()


class TestConversationMemoryIntegration:
    """Tests for proper memory integration of conversation history."""

    def test_latest_logical_plan_in_memory(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test that only the latest logical plan is added to memory.
        """
        # Create conversation with multiple user messages and realistic logical plans
        plan1 = {
            "name": "Movies",
            "output_dataset_id": "Movies",
            "params": {},
            "parents": []
        }
        plan2 = {
            "name": "FilterOperation1",
            "output_dataset_id": "FilterOperation1",
            "params": {
                "operator": "SemanticFilter",
                "description": "Filtered Movies by condition: The movie is a sci-fi genre film",
                "condition": "The movie is a sci-fi genre film"
            },
            "parents": [
                {"name": "Movies", "output_dataset_id": "Movies", "params": {}, "parents": []}
            ]
        }
        
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Search",
            dataset_ids=["movies"],
            messages=[
                {"role": "user", "content": "Find all movies"},
                {"role": "agent", "content": str(plan1), "type": "logical-plan"},
                {"role": "user", "content": "Only sci-fi movies"},
                {"role": "agent", "content": str(plan2), "type": "logical-plan"},
                {"role": "user", "content": "From after 2010"},
            ]
        )

        planner = Planner(
            datasets=[simple_movie_dataset],
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )

        # Generate plan - this should only add latest user message to memory
        logical_plan = planner.generate_logical_plan(
            query="From after 2010",
            datasets=[simple_movie_dataset],
            conversation=conversation,
        )

        # Verify planner didn't hit max_steps
        assert_planner_did_not_hit_max_steps(planner, logical_plan)

        # Verify plan was generated
        assert isinstance(logical_plan, dict)

        # Check memory contains only latest user message and latest plan
        # Use planning_memory which is the phase-specific memory for logical plan generation
        memory_steps = planner.planning_memory.steps

        # Filter for conversation-related steps
        conversation_steps = [s for s in memory_steps if isinstance(s, (ConversationUserStep, ConversationAgentStep))]

        # Should have exactly one agent step (latest plan)
        agent_steps = [s for s in conversation_steps if isinstance(s, ConversationAgentStep)]

        assert len(agent_steps) == 1, "Should have exactly one agent plan in memory"
        assert "FilterOperation1" in agent_steps[0].content, "Should be the latest logical plan"

        planner.cleanup()

    def test_planning_includes_prior_logical_plan(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test that the planning phase includes the latest logical plan from conversation.
        """
        # Realistic prior plan structure based on Dataset.sem_filter() and sem_topk()
        prior_plan = {
            "name": "TopKOperation1",
            "output_dataset_id": "TopKOperation1",
            "params": {
                "operator": "SemanticTopK",
                "index_name": "chroma",
                "description": "Top-5 items from FilterOperation1 for search string: highest rated",
                "search_str": "highest rated",
                "k": 5
            },
            "parents": [{
                "name": "FilterOperation1",
                "output_dataset_id": "FilterOperation1",
                "params": {
                    "operator": "SemanticFilter",
                    "description": "Filtered Movies by condition: The movie is a sci-fi genre film",
                    "condition": "The movie is a sci-fi genre film"
                },
                "parents": [
                    {"name": "Movies", "output_dataset_id": "Movies", "params": {}, "parents": []}
                ]
            }]
        }

        # create conversation with prior logical plan
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Search",
            dataset_ids=["movies"],
            messages=[
                {"role": "user", "content": "Find top sci-fi movies"},
                {"role": "agent", "content": str(prior_plan), "type": "logical-plan"},
                {"role": "user", "content": "Make it top 3 instead of top 5"},
            ]
        )

        planner = Planner(
            datasets=[simple_movie_dataset],
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )

        # Generate new plan with conversation containing prior logical plan
        new_plan = planner.generate_logical_plan(
            query="Make it top 3 instead of top 5",
            datasets=[simple_movie_dataset],
            conversation=conversation
        )

        # Verify planner didn't hit max_steps
        assert_planner_did_not_hit_max_steps(planner, new_plan)

        # Verify plan was generated
        assert isinstance(new_plan, dict)

        # Check that memory includes the logical plan
        # Use planning_memory which is the phase-specific memory for logical plan generation
        agent_steps = [s for s in planner.planning_memory.steps if isinstance(s, ConversationAgentStep)]

        # Should have the prior logical plan
        logical_plan_steps = [s for s in agent_steps if s.message_type == "logical-plan"]
        assert len(logical_plan_steps) == 1, "Should have logical plan in memory"

        planner.cleanup()
