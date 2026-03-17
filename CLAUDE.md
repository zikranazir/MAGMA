# CLAUDE.md — AI Assistant Guide for MAGMA

MAGMA is a production-ready boilerplate for multi-agent AI systems built on LangGraph. This file provides guidance for AI assistants working in this codebase.

---

## Project Overview

MAGMA provides a FastAPI server that auto-discovers and serves multiple LangGraph agents. Each agent is isolated in its own directory under `src/agent/agents/`, follows a consistent 4-file convention, and is exposed via HTTP endpoints.

**Key capabilities:**
- Multi-agent support with auto-discovery
- Multi-LLM provider support (OpenAI, Anthropic, Google)
- Streaming via Server-Sent Events
- Conversation memory via SQLite or in-memory checkpointing
- Type-safe configuration via Pydantic + YAML

---

## Repository Structure

```
MAGMA/
├── src/agent/                  # Main source code (installable package)
│   ├── __main__.py             # Entry point: python -m agent
│   ├── config.py               # Settings (Pydantic) + LLM factory
│   ├── logging.py              # Structured logging (JSON/text)
│   ├── server.py               # FastAPI app factory + routes
│   ├── core/
│   │   ├── registry.py         # Agent auto-discovery
│   │   └── state.py            # BaseAgentState (TypedDict)
│   └── agents/
│       ├── agent1/             # Example agent 1
│       └── agent2/             # Example agent 2
├── tests/                      # Pytest tests (mirrors src structure)
├── docs/superpowers/specs/     # Design specifications
├── config.yaml                 # Application configuration
├── pyproject.toml              # Dependencies, build, tool config
├── ruff.toml                   # Linter/formatter config
├── .pre-commit-config.yaml     # Git hooks
├── Makefile                    # Development commands
├── Dockerfile                  # Multi-stage production build
└── docker-compose.yaml         # Local Docker orchestration
```

---

## Technology Stack

| Category | Tool |
|----------|------|
| Agent orchestration | LangGraph |
| LLM abstraction | LangChain |
| HTTP server | FastAPI + Uvicorn |
| Configuration | Pydantic + YAML |
| Checkpointing | langgraph-checkpoint-sqlite |
| Package manager | UV (not pip) |
| Python version | 3.12+ |
| Linting | Ruff |
| Type checking | Mypy (strict) |
| Testing | Pytest + pytest-asyncio |
| Streaming | sse-starlette (SSE) |

---

## Development Commands

Always use `uv` for running commands, never plain `python` or `pip`:

```bash
make install    # uv sync --all-extras
make test       # uv run pytest
make lint       # uv run ruff check src/ tests/
make format     # uv run ruff format src/ tests/
make typecheck  # uv run mypy src/
make serve      # uv run python -m agent
make docker     # docker compose up --build
```

---

## Agent Convention

Every agent lives in `src/agent/agents/<name>/` and MUST follow this 4-file structure:

```
agents/<name>/
├── __init__.py
├── state.py    # Extends BaseAgentState
├── tools.py    # @tool functions + ALL_TOOLS list
├── nodes.py    # make_agent_node() and make_tool_node() factories
└── graph.py    # build_graph(settings: Settings) -> CompiledGraph
```

### state.py
```python
from agent.core.state import BaseAgentState

class MyAgentState(BaseAgentState):
    """State for my_agent."""
```

### tools.py
```python
from langchain_core.tools import tool

@tool
def my_tool() -> str:
    """Tool description for the LLM."""
    return "result"

ALL_TOOLS = [my_tool]
```

### nodes.py
```python
from langgraph.prebuilt import ToolNode

def make_agent_node(model, tools):
    model_with_tools = model.bind_tools(tools)
    def agent_node(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}
    return agent_node

def make_tool_node(tools):
    return ToolNode(tools)
```

### graph.py
```python
from langgraph.graph import END, StateGraph
from agent.config import Settings, get_chat_model
from .state import MyAgentState
from .tools import ALL_TOOLS
from .nodes import make_agent_node, make_tool_node

def should_continue(state: MyAgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

def build_graph(settings: Settings):
    model = get_chat_model(settings)
    agent_node = make_agent_node(model, ALL_TOOLS)
    tool_node = make_tool_node(ALL_TOOLS)

    graph = StateGraph(MyAgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    # Checkpointer selection
    if settings.memory.enabled:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()  # or AsyncSqliteSaver for SQLite
    else:
        checkpointer = None

    return graph.compile(checkpointer=checkpointer)
```

The registry (`core/registry.py`) auto-discovers agents by scanning `agents/` and importing `build_graph` from each subfolder. **No manual registration required.**

---

## Configuration System

Configuration priority (highest to lowest):
1. Environment variables
2. `.env` file
3. `config.yaml`
4. Pydantic defaults

### config.yaml structure
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

### Required environment variables
```bash
# At least one required:
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=...
```

Copy `.env.example` to `.env` and fill in the appropriate API key.

---

## FastAPI Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status + list of loaded agents |
| POST | `/{agent_name}/invoke` | Synchronous agent call |
| POST | `/{agent_name}/stream` | Streaming response via SSE |

### Request body (invoke and stream)
```json
{
  "messages": [{"role": "user", "content": "Hello"}],
  "thread_id": "optional-conversation-id"
}
```

If `thread_id` is omitted, a UUID is auto-generated. Use the same `thread_id` across requests to maintain conversation context.

### SSE stream event types
- `token` — LLM output token delta
- `tool_start` — Tool invocation started
- `tool_end` — Tool execution result
- `error` — Exception during processing

---

## Code Quality Requirements

All code must pass:
- `ruff check` — Lint rules: E, F, I, UP, B, SIM; line length 88
- `ruff format` — Consistent formatting
- `mypy src/` — Strict type checking with pydantic plugin

### Type annotation conventions
- Use `TypedDict` for LangGraph state types (not dataclasses or Pydantic models)
- All public functions must have full type annotations
- Use `from __future__ import annotations` when needed for forward refs
- Avoid `Any` unless unavoidable; document why when used

### Import conventions
- LLM provider packages (`langchain-openai`, etc.) use **lazy imports** inside functions to avoid requiring all providers to be installed. Follow this pattern in `config.py:get_chat_model()`.

---

## Testing Conventions

```
tests/
├── conftest.py         # Shared fixtures (test_settings, mock_chat_model)
├── test_config.py      # Configuration system tests
├── test_graph.py       # Graph construction tests
└── test_server.py      # FastAPI endpoint tests (async)
```

- Tests are async by default (`asyncio_mode = "auto"`)
- Use `mock_chat_model` fixture from `conftest.py` to avoid real LLM calls
- Use `test_settings` fixture for `Settings` with `memory.enabled = False`
- HTTP tests use `httpx.AsyncClient` with the app as transport
- Mock LLM providers via `unittest.mock.patch` or monkeypatching

Example test pattern:
```python
import pytest
from httpx import AsyncClient
from agent.server import create_app

async def test_health(test_settings):
    app = create_app(test_settings)
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
```

---

## Adding a New Agent

1. Create `src/agent/agents/<your_agent_name>/`
2. Add `__init__.py`, `state.py`, `tools.py`, `nodes.py`, `graph.py` following the convention above
3. The registry auto-discovers it — no other files need changing
4. Add tests in `tests/test_<your_agent_name>.py`
5. Verify with `make test && make typecheck && make lint`

---

## Docker

The Dockerfile uses a **multi-stage build**:
- **Builder**: `ghcr.io/astral-sh/uv:latest` installs deps into `.venv`
- **Runtime**: `python:3.12-slim` runs as non-root `agent` user

```bash
# Build and run with docker compose
make docker

# Or manually
docker build -t magma .
docker run -p 8000:8000 --env-file .env magma
```

The docker-compose mounts `config.yaml` as read-only and persists checkpoints via a named volume.

---

## Common Pitfalls

- **Do not** use `pip install` or `pip run`; always use `uv sync` / `uv run`
- **Do not** add agents to a registry manually; the auto-discovery handles it
- **Do not** import LLM provider packages at module top-level; use lazy imports
- **Do not** modify `BaseAgentState` directly; extend it in your agent's `state.py`
- **Do not** use `memory.MemorySaver` in production; configure SQLite for persistence
- `thread_id` must be passed in `config={"configurable": {"thread_id": ...}}` when invoking graphs with checkpointers
