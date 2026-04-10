"""
Tests to verify the conversation integration with the Planner class.

This includes:
1. Basic Conversation functionality tests
2. Conversation variable accessibility in Python executor
3. State consistency across planning phases
"""
import pytest

from carnot.agents.memory import ConversationAgentStep, ConversationUserStep
from carnot.conversation.conversation import Conversation
from carnot.data.dataset import DataItem, Dataset


def test_conversation_basic():
    """Test basic Conversation functionality."""
    # Create a conversation with some messages
    conversation = Conversation(
        user_id="test_user",
        session_id="test_session",
        title="Test Conversation",
        dataset_ids=["dataset1", "dataset2"],
        messages=[]
    )

    # Add some messages
    conversation.add_message("user", "What are the top movies from 2020?")
    conversation.add_message("agent", "1. Filter for movies from 2020\n2. Apply semantic map to compute rating for each movie\n3. Return top-10 movies based on rating", message_type="natural-language-plan")
    conversation.add_message("user", "Actually, can you filter by action movies only?")
    
    # Test get_latest_user_message
    latest_user_msg = conversation.get_latest_user_message()
    assert latest_user_msg is not None
    assert latest_user_msg.content == "Actually, can you filter by action movies only?"
    
    # Test get_latest_agent_plan
    latest_plan = conversation.get_latest_agent_plan("natural-language-plan")
    assert latest_plan is not None
    assert latest_plan.message_type == "natural-language-plan"
    assert "1. Filter for movies from 2020" in latest_plan.content
    
    # Test to_dict_list
    dict_list = conversation.to_dict_list()
    assert len(dict_list) == 3
    assert dict_list[0]["role"] == "user"
    assert dict_list[1]["role"] == "agent"
    assert dict_list[1]["type"] == "natural-language-plan"
    assert dict_list[2]["role"] == "user"

def test_conversation_to_memory_steps():
    """Test conversion to MemoryStep objects."""
    # Create a conversation
    conversation = Conversation(
        user_id="test_user",
        session_id="test_session",
        title="Test Conversation",
        dataset_ids=["dataset1"],
        messages=[
            {"role": "user", "content": "Find top action movies"},
            {"role": "agent", "content": "1. Filter by action genre\n2. Sort by rating", "type": "natural-language-plan"},
            {"role": "user", "content": "What about comedies?"},
        ]
    )

    # Convert to memory steps
    memory_steps = conversation.to_memory_steps()

    # Verify types
    assert isinstance(memory_steps[0], ConversationUserStep)
    assert isinstance(memory_steps[1], ConversationAgentStep)
    assert isinstance(memory_steps[2], ConversationUserStep)

    # Verify content
    assert memory_steps[0].content == "Find top action movies"
    assert memory_steps[1].content == "1. Filter by action genre\n2. Sort by rating"
    assert memory_steps[1].message_type == "natural-language-plan"

    # Test to_messages conversion
    messages = memory_steps[0].to_messages()
    assert len(messages) == 1
    assert messages[0].role.value == "user"


def test_conversation_condense():
    """Test conversation condensation."""
    conversation = Conversation(
        user_id="test_user",
        session_id="test_session",
        title="Test",
        dataset_ids=[],
        messages=[
            {"role": "user", "content": "First question"},
            {"role": "agent", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]
    )
    condensed = conversation.condense("test query")
    print(f"  Condensed conversation:\n{condensed}\n")
    assert "First question" in condensed
    assert "Second question" in condensed


# ==================================================================================
# Conversation Variable Accessibility Tests
# ==================================================================================


class TestConversationVariableAccessibility:
    """Tests for conversation variable accessibility in Python executor."""

    @pytest.fixture
    def simple_dataset(self):
        """Create a simple test dataset."""
        items = [
            DataItem.from_dict({"idx": 0, "contents": "Test item 1"}),
            DataItem.from_dict({"idx": 1, "contents": "Test item 2"}),
            DataItem.from_dict({"idx": 2, "contents": "Test item 3"}),
        ]
        return Dataset(name="Test Dataset", items=items)

    @pytest.fixture
    def conversation_with_history(self):
        """Create a conversation with multiple messages."""
        return Conversation(
            user_id="test_user",
            session_id="test_session",
            title="Test Conversation with History",
            dataset_ids=["1"],
            messages=[
                {"role": "user", "content": "First query about the data"},
                {
                    "role": "agent",
                    "content": "I will analyze the data...",
                    "type": "natural-language-plan",
                },
                {
                    "role": "agent",
                    "content": '{"operators": [...]}',
                    "type": "logical-plan",
                },
                {"role": "user", "content": "Follow-up query with refinement"},
            ],
        )

    def test_conversation_serialization_format(self, conversation_with_history):
        """
        Test that conversation serializes to the correct dictionary format.
        
        The conversation should serialize to a dict with:
        - user_id, session_id, dataset_ids, messages keys
        - messages as a list of dicts with role, content, and optional type
        """
        messages = conversation_with_history.to_dict_list()

        # Check messages structure
        assert isinstance(messages, list)
        assert len(messages) == 4

        # Verify each message has required fields
        for msg in messages:
            assert "role" in msg, "Message missing 'role'"
            assert "content" in msg, "Message missing 'content'"
            assert msg["role"] in ["user", "agent"], f"Invalid role: {msg['role']}"

        # Verify type field is present where expected
        agent_messages = [msg for msg in messages if msg["role"] == "agent"]
        assert all("type" in msg for msg in agent_messages), "Agent messages should have type field"

    def test_conversation_to_dict_list(self, conversation_with_history):
        """
        Test that we can extract just the messages list for the Python executor.
        
        This is what gets passed as the 'conversation' variable to the executor.
        """
        messages = conversation_with_history.to_dict_list()

        # Verify it's a list of dicts
        assert isinstance(messages, list)
        assert all(isinstance(msg, dict) for msg in messages)

        # Verify message contents
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "First query about the data"
        assert messages[1]["role"] == "agent"
        assert messages[1]["type"] == "natural-language-plan"
        assert messages[1]["content"] == "I will analyze the data..."
        assert messages[2]["role"] == "agent"
        assert messages[2]["type"] == "logical-plan"
        assert messages[2]["content"] == '{"operators": [...]}'
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "Follow-up query with refinement"
