"""Configuration system for the LangGraph agent boilerplate."""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

# ---------------------------------------------------------------------------
# Sub-config models
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: Literal["openai", "anthropic", "google"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096


class MemoryConfig(BaseModel):
    """Checkpointer / memory configuration."""

    enabled: bool = True
    backend: Literal["sqlite", "memory"] = "sqlite"
    sqlite_path: str = "./checkpoints.db"


class ServerConfig(BaseModel):
    """FastAPI server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"


# ---------------------------------------------------------------------------
# Aggregated settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Application settings loaded from env vars, .env file, and config.yaml.

    Priority (highest to lowest):
        1. Environment variables
        2. .env file
        3. config.yaml
        4. Field defaults
    """

    llm: LLMConfig = LLMConfig()
    memory: MemoryConfig = MemoryConfig()
    server: ServerConfig = ServerConfig()
    logging: LoggingConfig = LoggingConfig()

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    model_config = {
        "env_file": ".env",
        "yaml_file": "config.yaml",
        "env_nested_delimiter": "__",
    }

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customise settings source priority.

        Order: init > env vars > .env file > yaml file > defaults.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


# ---------------------------------------------------------------------------
# Chat model factory
# ---------------------------------------------------------------------------

def get_chat_model(settings: Settings) -> BaseChatModel:
    """Return a LangChain chat model based on the current settings.

    Provider-specific packages are imported lazily so only the selected
    provider's dependency is required at runtime.

    Raises:
        ValueError: If the configured provider is not supported.
    """
    llm = settings.llm

    if llm.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=llm.model,
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
            api_key=settings.openai_api_key,
        )

    if llm.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=llm.model,
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
            api_key=settings.anthropic_api_key,
        )

    if llm.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=llm.model,
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
            google_api_key=settings.google_api_key,
        )

    raise ValueError(f"Unknown LLM provider: {llm.provider!r}")
