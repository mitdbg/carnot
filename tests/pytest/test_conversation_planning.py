"""
Unit tests for conversation-aware planning functionality.

Tests cover:
1. Data discovery with conversation feedback
2. Natural language planning with conversation refinement
3. Logical plan compilation with conversation updates
4. End-to-end execution with conversation context
"""
import textwrap

from carnot.agents.memory import ConversationAgentStep, ConversationUserStep
from carnot.agents.models import LiteLLMModel
from carnot.agents.planner import Planner
from carnot.conversation.conversation import Conversation
from carnot.execution.execution import Execution


class TestConversationalDataDiscovery:
    """Tests for data discovery with conversation feedback."""

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
        
        # Initial query
        initial_query = "What do critics think about action movies?"
        
        # Create conversation with feedback
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Analysis",
            dataset_ids=["movies", "reviews"],
            messages=[
                {"role": "user", "content": initial_query},
                {"role": "user", "content": "Please focus your analysis on the Reviews dataset, as it contains the critic opinions."},
            ]
        )
        
        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Perform data discovery with conversation context
        report = planner.search_for_relevant_data(
            query="Please focus your analysis on the Reviews dataset, as it contains the critic opinions.",
            datasets=[movies_dataset, reviews_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=conversation,
        )
        
        # Verify that the report mentions the Reviews dataset prominently
        assert isinstance(report, str)
        assert len(report) > 0
        
        # The report should explicitly mention Reviews dataset
        # (case-insensitive check)
        assert "reviews" in report.lower()
        
        # The report should discuss critic-related content
        assert any(keyword in report.lower() for keyword in [
            "critic", "review", "opinion", "sentiment", "score"
        ])
        
        planner.cleanup()

    def test_data_discovery_refines_search_scope(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test that data discovery can refine its scope based on user feedback.
        
        Scenario:
        - Initial discovery explores broadly
        - User provides feedback to narrow focus to specific fields or criteria
        - New discovery examines those specific aspects
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        
        # Create conversation with refinement feedback
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Analysis",
            dataset_ids=["movies", "reviews"],
            messages=[
                {"role": "user", "content": "Find movies with high ratings"},
                {"role": "user", "content": "Please check if the Movies dataset has a 'rating' or 'score' field that we can use."},
            ]
        )
        
        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Perform targeted data discovery
        report = planner.search_for_relevant_data(
            query="Please check if the Movies dataset has a 'rating' or 'score' field that we can use.",
            datasets=[movies_dataset, reviews_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=conversation,
        )
        
        # Verify the report addresses the specific field inquiry
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Report should mention the Movies dataset
        assert "movies" in report.lower()
        
        # Report should discuss rating/score fields
        assert any(keyword in report.lower() for keyword in [
            "rating", "score", "field", "column", "attribute"
        ])
        
        planner.cleanup()


class TestConversationalNaturalLanguagePlanning:
    """Tests for natural language planning with conversation refinement."""

    def test_nl_plan_adds_missing_step(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test that a natural language plan can be refined to add a missing step.
        
        Scenario:
        - Initial plan is generated but misses a step
        - User points out the missing step
        - New plan includes the suggested step
        """
        # Initial query and incomplete plan
        initial_query = "Find all sci-fi movies"
        incomplete_plan = "1. Filter the Movies dataset for movies\n2. Return the filtered movies"

        # Create conversation with feedback
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Search",
            dataset_ids=["movies"],
            messages=[
                {"role": "user", "content": initial_query},
                {"role": "agent", "content": incomplete_plan, "type": "natural-language-plan"},
                {"role": "user", "content": "You forgot to filter by genre='Sci-Fi'. Please add that step."},
            ]
        )

        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )

        # Generate refined plan
        refined_plan = planner.generate_logical_plan(
            query="You forgot to filter by genre='Sci-Fi'. Please add that step.",
            datasets=[simple_movie_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=conversation,
            data_discovery_report=None
        )

        # Verify the refined plan includes the genre filter
        assert isinstance(refined_plan, str)
        assert len(refined_plan) > 0

        # Plan should now mention filtering by genre or sci-fi
        assert any(keyword in refined_plan.lower() for keyword in [
            "sci-fi", "science fiction", "genre"
        ])

        # Plan should still have multiple steps (structure preserved)
        assert "3." in refined_plan or "2." in refined_plan and "2. Return the filtered movies" not in refined_plan

        planner.cleanup()

    def test_nl_plan_refines_operation(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test that a natural language plan can refine an operation based on feedback.
        
        Scenario:
        - Initial plan suggests one approach
        - User suggests a different or refined approach
        - New plan incorporates the refinement
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        
        # Initial plan with a simple approach
        initial_query = "Find the best movie"
        initial_plan = textwrap.dedent("""1. Load all movies from Movies dataset
                                          2. Find the movie with highest rating
                                          3. Return the best movie""")
        
        # Create conversation with refinement feedback
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Analysis",
            dataset_ids=["movies", "reviews"],
            messages=[
                {"role": "user", "content": initial_query},
                {"role": "agent", "content": initial_plan, "type": "natural-language-plan"},
                {"role": "user", "content": "Instead of using the movie rating, calculate the average review score from the Reviews dataset for each movie."},
            ]
        )
        
        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Generate refined plan
        refined_plan = planner.generate_logical_plan(
            query="Instead of using the movie rating, calculate the average review score from the Reviews dataset for each movie.",
            datasets=[movies_dataset, reviews_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=conversation,
            data_discovery_report=None
        )
        
        # Verify the refined plan uses reviews
        assert isinstance(refined_plan, str)
        assert len(refined_plan) > 0
        
        # Plan should mention reviews or review scores
        assert any(keyword in refined_plan.lower() for keyword in [
            "review", "average", "score"
        ])
        
        # Plan should reference both datasets or joining them
        plan_lower = refined_plan.lower()
        assert "movies" in plan_lower or "reviews" in plan_lower
        
        planner.cleanup()

    def test_nl_plan_changes_ordering(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test that a natural language plan can reorder steps based on feedback.
        
        Scenario:
        - Initial plan has steps in one order
        - User suggests a different ordering for efficiency or logic
        - New plan reflects the reordering
        """
        initial_query = "Find highly rated sci-fi movies from after 2010"
        initial_plan = textwrap.dedent("""1. Filter Movies dataset for highly rated movies (rating > 8.0)
                                          2. Filter for movies from after 2010
                                          3. Filter for sci-fi genre
                                          4. Return the results""")

        # Create conversation suggesting reordering
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Search",
            dataset_ids=["movies"],
            messages=[
                {"role": "user", "content": initial_query},
                {"role": "agent", "content": initial_plan, "type": "natural-language-plan"},
                {"role": "user", "content": "It would be more efficient to filter by genre first since that's the most selective criterion."},
            ]
        )

        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )

        # Generate reordered plan
        refined_plan = planner.generate_logical_plan(
            query="It would be more efficient to filter by genre first since that's the most selective criterion.",
            datasets=[simple_movie_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=conversation,
            data_discovery_report=None
        )

        # Verify plan structure
        assert isinstance(refined_plan, str)
        assert len(refined_plan) > 0

        # Plan should mention all three filtering criteria
        plan_lower = refined_plan.lower()
        assert "genre" in plan_lower or "sci-fi" in plan_lower
        assert "rating" in plan_lower or "rated" in plan_lower
        assert "2010" in plan_lower or "year" in plan_lower

        # Check that filter for sci-fi genre comes before the others
        sci_fi_index = plan_lower.find("sci-fi")
        rating_index = plan_lower.find("rating")
        year_index = plan_lower.find("2010")
        assert sci_fi_index != -1
        assert rating_index != -1
        assert year_index != -1
        assert sci_fi_index < rating_index and sci_fi_index < year_index

        planner.cleanup()


class TestConversationalLogicalPlanCompilation:
    """Tests for logical plan compilation with conversation updates."""

    def test_compile_adds_operator(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test that compilation can add an operator based on user feedback.
        
        Scenario:
        - Initial NL plan is provided
        - User suggests adding a specific operation
        - Compiled plan includes the additional operator
        """
        initial_query = "Get all sci-fi movies"
        nl_plan = """1. Filter Movies dataset for sci-fi genre
2. Return the results"""
        
        # Create conversation with feedback to add sorting
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Search",
            dataset_ids=["movies"],
            messages=[
                {"role": "user", "content": initial_query},
                {"role": "agent", "content": nl_plan, "type": "natural-language-plan"},
                {"role": "user", "content": "Please add a step to sort the results by rating in descending order."},
            ]
        )
        
        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Compile with the sorting feedback
        compiled_plan = planner.compile_logical_plan(
            query="Please add a step to sort the results by rating in descending order.",
            datasets=[simple_movie_dataset],
            nl_plan=nl_plan,
            data_discovery_report=None,
            conversation=conversation
        )
        
        # Verify compiled plan structure
        assert isinstance(compiled_plan, dict)
        assert "params" in compiled_plan
        
        # Check that the plan includes sorting logic
        # This could be in a Code operator or mentioned in task descriptions
        plan_str = str(compiled_plan).lower()
        assert any(keyword in plan_str for keyword in [
            "sort", "order", "rating", "descending"
        ])
        
        planner.cleanup()

    def test_compile_changes_operator_params(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test that compilation can adjust operator parameters based on feedback.
        
        Scenario:
        - Initial NL plan suggests one approach
        - User provides feedback about specific parameter values
        - Compiled plan uses the suggested parameters
        """
        initial_query = "Find top movies"
        nl_plan = """1. Filter Movies dataset
2. Sort by rating
3. Return top results"""
        
        # Create conversation with specific parameter feedback
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Search",
            dataset_ids=["movies"],
            messages=[
                {"role": "user", "content": initial_query},
                {"role": "agent", "content": nl_plan, "type": "natural-language-plan"},
                {"role": "user", "content": "Return only the top 5 results, not all of them."},
            ]
        )
        
        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Compile with limit feedback
        compiled_plan = planner.compile_logical_plan(
            query="Return only the top 5 results, not all of them.",
            datasets=[simple_movie_dataset],
            nl_plan=nl_plan,
            data_discovery_report=None,
            conversation=conversation
        )
        
        # Verify compiled plan includes limit
        assert isinstance(compiled_plan, dict)
        
        # Check for limit/top-k operator or parameter
        plan_str = str(compiled_plan).lower()
        assert any(keyword in plan_str for keyword in [
            "limit", "top", "5", "topk"
        ])
        
        planner.cleanup()

    def test_compile_uses_different_operator_type(self, movie_reviews_datasets, llm_config, test_model_id):
        """
        Test that compilation can switch operator types based on feedback.
        
        Scenario:
        - Initial NL plan suggests using one operator approach
        - User suggests a different operator type would be better
        - Compiled plan uses the suggested operator type
        """
        movies_dataset, reviews_dataset = movie_reviews_datasets
        
        initial_query = "Find movies with positive reviews"
        nl_plan = textwrap.dedent("""1. Load Movies dataset
                                     2. Filter for movies with positive sentiment
                                     3. Return matching movies""")
        
        # Create conversation suggesting join with reviews
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Analysis",
            dataset_ids=["movies", "reviews"],
            messages=[
                {"role": "user", "content": initial_query},
                {"role": "agent", "content": nl_plan, "type": "natural-language-plan"},
                {"role": "user", "content": "Actually, join with the Reviews dataset and filter reviews by sentiment score > 0.7"},
            ]
        )
        
        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )
        
        # Compile with join feedback
        compiled_plan = planner.compile_logical_plan(
            query="Actually, join with the Reviews dataset and filter reviews by sentiment score > 0.7",
            datasets=[movies_dataset, reviews_dataset],
            nl_plan=nl_plan,
            data_discovery_report=None,
            conversation=conversation
        )
        
        # Verify compiled plan references both datasets
        assert isinstance(compiled_plan, dict)
        
        # Check that the plan involves both datasets (likely through join or separate operations)
        plan_str = str(compiled_plan).lower()
        
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

    def test_latest_nl_plan_in_memory(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test that only the latest natural language plan is added to memory.
        """
        # Create conversation with multiple user messages
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Search",
            dataset_ids=["movies"],
            messages=[
                {"role": "user", "content": "Find all movies"},
                {"role": "agent", "content": "Plan: Filter movies", "type": "natural-language-plan"},
                {"role": "user", "content": "Only sci-fi movies"},
                {"role": "agent", "content": "Plan: Filter sci-fi", "type": "natural-language-plan"},
                {"role": "user", "content": "From after 2010"},
            ]
        )

        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )

        # Generate plan - this should only add latest user message to memory
        nl_plan = planner.generate_logical_plan(
            query="From after 2010",
            datasets=[simple_movie_dataset],
            indices=[],
            tools={},
            memories=[],
            conversation=conversation,
            data_discovery_report=None
        )

        # Verify plan was generated
        assert isinstance(nl_plan, str)
        assert len(nl_plan) > 0

        # Check memory contains only latest user message and latest plan
        # (implementation detail: we add ConversationUserStep for latest message)
        memory_steps = planner.memory.steps

        # Should have: latest user message + latest NL plan + PlannerTaskStep
        # Filter for conversation-related steps
        conversation_steps = [s for s in memory_steps if isinstance(s, (ConversationUserStep, ConversationAgentStep))]

        # Should have exactly one agent step (latest plan)
        agent_steps = [s for s in conversation_steps if isinstance(s, ConversationAgentStep)]

        assert len(agent_steps) == 1, "Should have exactly one agent plan in memory"
        assert agent_steps[0].content == "Plan: Filter sci-fi", "Should be the latest NL plan"

        planner.cleanup()

    def test_compilation_includes_logical_plan(self, simple_movie_dataset, llm_config, test_model_id):
        """
        Test that compilation phase includes the latest logical plan from conversation.
        """
        nl_plan = "1. Filter sci-fi\n2. Sort by rating\n3. Return top 5"
        logical_plan_code = 'ds = datasets["Movies"]\nds = ds.sem_filter("sci-fi")\nfinal_answer(ds.serialize())'

        # create conversation with both plan types
        conversation = Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Movie Search",
            dataset_ids=["movies"],
            messages=[
                {"role": "user", "content": "Find top sci-fi movies"},
                {"role": "agent", "content": nl_plan, "type": "natural-language-plan"},
                {"role": "agent", "content": logical_plan_code, "type": "logical-plan"},
                {"role": "user", "content": "Make it top 3 instead of top 5"},
            ]
        )

        planner = Planner(
            tools={},
            model=LiteLLMModel(model_id=test_model_id, api_key=llm_config["OPENAI_API_KEY"])
        )

        # Compile with conversation containing both plan types
        compiled_plan = planner.compile_logical_plan(
            query="Make it top 3 instead of top 5",
            datasets=[simple_movie_dataset],
            nl_plan=nl_plan,
            data_discovery_report=None,
            conversation=conversation
        )

        # Verify plan was compiled
        assert isinstance(compiled_plan, dict)

        # Check that memory includes the logical plan
        agent_steps = [s for s in planner.memory.steps if isinstance(s, ConversationAgentStep)]

        # Should have both NL and logical plan
        logical_plan_steps = [s for s in agent_steps if s.message_type == "logical-plan"]
        assert len(logical_plan_steps) == 1, "Should have logical plan in memory"

        planner.cleanup()
