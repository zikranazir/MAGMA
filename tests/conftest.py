"""Shared test fixtures for the agent test suite."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from agent.config import MemoryConfig, Settings


@pytest.fixture()
def test_settings() -> Settings:
    """Create a Settings instance with memory disabled to avoid file I/O."""
    return Settings(
        memory=MemoryConfig(enabled=False, backend="memory"),
    )


@pytest.fixture()
def mock_chat_model() -> MagicMock:
    """Return a mock BaseChatModel that returns a canned AIMessage.

    The mock supports both sync ``invoke`` and async ``ainvoke`` so it can
    be used in place of a real LLM without making network calls.
    """
    model = MagicMock()
    canned_response = AIMessage(content="Hello from the mock LLM!")

    model.invoke.return_value = canned_response
    model.ainvoke = AsyncMock(return_value=canned_response)
    model.bind_tools.return_value = model  # bind_tools returns same mock

    return model
