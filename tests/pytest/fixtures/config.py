import os

import pytest


@pytest.fixture
def llm_config():
    """
    LLM configuration for tests. Uses environment variables.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set in environment")
    
    return {"OPENAI_API_KEY": api_key}


@pytest.fixture
def test_model_id():
    """
    Model ID to use for testing (LLM for text generation).
    """
    return "openai/gpt-5-mini"


@pytest.fixture
def test_embedding_model_id():
    """
    Embedding model ID to use for testing (for semantic similarity/topk operations).
    """
    return "openai/text-embedding-3-small"
