# Agent Boilerplate Design Spec

## Purpose

A production-ready starter kit for building AI agents with LangGraph. Supports multiple LLM providers (OpenAI, Anthropic, Google) out of the box. Uses `uv` for package management. Designed to let developers clone, configure, and start building custom agent workflows immediately.

## Tech Stack

- **Python 3.12+**
- **LangGraph** — graph-based agent orchestration
- **LangChain ChatModel** — multi-provider LLM abstraction (`langchain-openai`, `langchain-anthropic`, `langchain-google-genai`)
- **langgraph-checkpoint-sqlite** — async-compatible SQLite checkpointing
- **pydantic-settings[yaml]** — typed configuration with YAML support
- **FastAPI** — HTTP API server
- **uv** — package manager
- **pytest / ruff / mypy / pre-commit** — dev tooling
- **Docker** — containerization
- **Python stdlib logging** — structured logging

## Project Structure

```
agent-boilerplate/
├── pyproject.toml
├── uv.lock
├── .env.example
├── config.yaml
├── Dockerfile
├── docker-compose.yaml
├── Makefile
├── ruff.toml
├── .pre-commit-config.yaml
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── __main__.py        # Entry point: python -m agent
│       ├── py.typed            # PEP 561 marker
│       ├── config.py          # Settings classes, LLM factory
│       ├── logging.py         # Logging configuration
│       ├── graph.py           # StateGraph definition
│       ├── state.py           # AgentState TypedDict
│       ├── tools.py           # Placeholder tool(s)
│       ├── nodes.py           # Node function implementations
│       └── server.py          # FastAPI app
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_graph.py
│   └── test_server.py
└── README.md
```

## Configuration System

### `.env.example` (secrets)

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

# Optional: LangSmith tracing
# LANGCHAIN_TRACING_V2=true
# LANGSMITH_API_KEY=
```

### `config.yaml` (settings)

```yaml
llm:
  provider: openai          # openai | anthropic | google
  model: gpt-4o-mini
  temperature: 0.7
  max_tokens: 4096

memory:
  enabled: true
  backend: sqlite           # sqlite | memory
  sqlite_path: ./checkpoints.db

server:
  host: 0.0.0.0
  port: 8000

logging:
  level: INFO               # DEBUG | INFO | WARNING | ERROR
  format: json              # json | text
```

### `config.py`

- `LLMConfig` — provider, model, temperature, max_tokens
- `MemoryConfig` — enabled, backend, sqlite_path
- `ServerConfig` — host, port
- `LoggingConfig` — level, format
- `Settings(BaseSettings)` — aggregates all config sections. Uses `pydantic-settings[yaml]` with `YamlConfigSettingsSource` as a custom settings source to load `config.yaml`. API keys come from `.env` via the default env source.
- `get_chat_model(config: LLMConfig) -> BaseChatModel` — factory function that returns the right ChatModel based on provider. Called once at startup, not per-request.

## LangGraph Core

### `state.py`

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

### `graph.py`

Builds a `StateGraph[AgentState]` with:
- **`agent` node** — invokes the LLM (pre-bound with tools) with state messages
- **`tools` node** — executes any tool calls from the LLM response
- **Conditional edge** from `agent`: if tool calls present -> `tools` node, else -> `END`
- Edge from `tools` -> `agent` (loop back)
- Compiled with `AsyncSqliteSaver` (from `langgraph-checkpoint-sqlite`) or `MemorySaver` based on config

The `build_graph(settings: Settings) -> CompiledGraph` function:
1. Instantiates the ChatModel once via `get_chat_model()`
2. Binds tools to the model
3. Creates node functions that close over the bound model
4. Compiles the graph with the appropriate checkpointer
5. Returns the compiled graph ready for invocation

### `nodes.py`

- `make_agent_node(model: BaseChatModel)` — returns a node function that uses the pre-configured model. Model is instantiated once, not per-invocation.
- `tools_node` — uses LangGraph's `ToolNode` to execute tool calls

### `tools.py`

- Single placeholder `@tool` function (e.g., `get_current_time`) as an example
- Clear comments showing how to add more tools

## FastAPI Server

### `server.py`

Endpoints:
- `POST /invoke` — accepts `{"messages": [...], "thread_id": "..."}`, runs graph to completion, returns final message. If `thread_id` is omitted, generates a UUID.
- `POST /stream` — same input, returns SSE stream using LangGraph's `.astream_events()`. Events include: `on_chat_model_stream` (token deltas), `on_tool_start`, `on_tool_end`, and final response.
- `GET /health` — returns `{"status": "ok"}`

Error handling:
- LLM API errors (rate limits, timeouts) are caught and returned as structured JSON error responses with appropriate HTTP status codes
- LangChain's built-in `max_retries` on ChatModel is configured (default: 2)
- Streaming errors send an SSE error event before closing the connection

Middleware:
- `CORSMiddleware` included with configurable allowed origins (defaults to `["*"]` for development)

Uses FastAPI `lifespan` to:
- Load config and initialize logging
- Build and compile the graph once (including model instantiation)
- Share compiled graph instance across requests

### `__main__.py`

Entry point for `python -m agent`:
```python
import uvicorn
from agent.config import Settings
from agent.server import create_app

settings = Settings()
app = create_app(settings)
uvicorn.run(app, host=settings.server.host, port=settings.server.port)
```

Also registered as a script in `pyproject.toml` `[project.scripts]`: `agent = "agent.__main__:main"`

## Logging

### `logging.py`

- `setup_logging(config: LoggingConfig)` — configures Python stdlib logging
- Two formatters: JSON (for production, using `json` stdlib) and text (for development)
- Configurable log level from config.yaml
- Applied at startup in server lifespan and importable for CLI usage

## Docker

### `Dockerfile`

- Multi-stage build: builder (uv install) + runtime
- Based on `python:3.12-slim`
- Copies only necessary files
- Runs with non-root user
- Exposes port 8000

### `docker-compose.yaml`

- Single `agent` service
- Mounts `.env` and `config.yaml`
- Volume for SQLite checkpoints persistence
- Health check configured

## Dev Tooling

### `pyproject.toml`

- Project metadata + dependencies
- Optional dependency groups: `[dev]` (pytest, ruff, mypy, pre-commit, pytest-asyncio, httpx), provider groups (`[openai]`, `[anthropic]`, `[google]`, `[all]`)
- mypy and pytest config sections

### `ruff.toml`

- Line length: 88
- Target: Python 3.12
- Select: E, F, I, UP, B, SIM
- Format enabled

### `.pre-commit-config.yaml`

- ruff check + fix
- ruff format
- mypy

### `Makefile`

```makefile
install:       uv sync --all-extras
test:          uv run pytest
lint:          uv run ruff check src/ tests/
format:        uv run ruff format src/ tests/
typecheck:     uv run mypy src/
serve:         uv run python -m agent
docker:        docker compose up --build
```

### Tests

- `conftest.py` — shared fixtures (mock LLM, test config, async client)
- `test_config.py` — tests config loading with defaults, env overrides, and YAML
- `test_graph.py` — tests that the graph compiles and runs with a mock LLM
- `test_server.py` — tests FastAPI endpoints using `httpx.AsyncClient`

## Verification Plan

1. `make install` — installs without errors
2. `make test` — all tests pass
3. `make lint` — no linting errors
4. `make typecheck` — no type errors
5. `cp .env.example .env` + fill in one key + `make serve` — FastAPI server starts
6. `curl localhost:8000/health` — returns 200
7. `make docker` — builds and starts successfully
