"""
Tests to verify the conversation integration with the Planner class.
"""
from carnot.agents.memory import ConversationAgentStep, ConversationUserStep
from carnot.conversation.conversation import Conversation


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
