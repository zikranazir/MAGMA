"""Tests for the configuration system (agent.config)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.config import (
    LLMConfig,
    LoggingConfig,
    MemoryConfig,
    ServerConfig,
    Settings,
    get_chat_model,
)

# ---------------------------------------------------------------------------
# Sub-config defaults
# ---------------------------------------------------------------------------

class TestLLMConfig:
    def test_defaults(self) -> None:
        cfg = LLMConfig()
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o-mini"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096


class TestMemoryConfig:
    def test_defaults(self) -> None:
        cfg = MemoryConfig()
        assert cfg.enabled is True
        assert cfg.backend == "sqlite"
        assert cfg.sqlite_path == "./checkpoints.db"


class TestServerConfig:
    def test_defaults(self) -> None:
        cfg = ServerConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000


class TestLoggingConfig:
    def test_defaults(self) -> None:
        cfg = LoggingConfig()
        assert cfg.level == "INFO"
        assert cfg.format == "json"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class TestSettings:
    def test_loads_with_defaults(self, test_settings: Settings) -> None:
        """Settings should instantiate with sensible defaults."""
        assert test_settings.llm.provider == "openai"
        assert test_settings.memory.enabled is False
        assert test_settings.openai_api_key is None
        assert test_settings.anthropic_api_key is None
        assert test_settings.google_api_key is None

    def test_nested_sub_configs_present(self, test_settings: Settings) -> None:
        assert isinstance(test_settings.llm, LLMConfig)
        assert isinstance(test_settings.memory, MemoryConfig)
        assert isinstance(test_settings.server, ServerConfig)
        assert isinstance(test_settings.logging, LoggingConfig)


# ---------------------------------------------------------------------------
# get_chat_model
# ---------------------------------------------------------------------------

class TestGetChatModel:
    def test_unknown_provider_raises(self, test_settings: Settings) -> None:
        """get_chat_model should raise ValueError for an unsupported provider."""
        # Force an invalid provider value via object.__setattr__ since
        # LLMConfig is a Pydantic model with validation.
        object.__setattr__(test_settings.llm, "provider", "unsupported")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_chat_model(test_settings)

    def test_openai_provider(self, test_settings: Settings) -> None:
        """get_chat_model should import and return a ChatOpenAI instance."""
        test_settings.llm.provider = "openai"  # type: ignore[assignment]
        fake_cls = type("FakeChatOpenAI", (), {"__init__": lambda self, **kw: None})

        with (
            patch("agent.config.ChatOpenAI", fake_cls, create=True),
            patch.dict(
                "sys.modules",
                {"langchain_openai": type("mod", (), {"ChatOpenAI": fake_cls})()},
            ),
        ):
            model = get_chat_model(test_settings)
            assert isinstance(model, fake_cls)
