# MAGMA

**Multi-Agent Graph Management Architecture**

A production-ready boilerplate for building multi-agent AI systems with [LangGraph](https://github.com/langchain-ai/langgraph). Drop in your agents, configure your LLM, ship.

## Features

- Multi-agent support — add agents by dropping a folder, auto-discovered at startup
- Multi-LLM provider support (OpenAI, Anthropic, Google)
- FastAPI server with per-agent invoke and streaming endpoints
- Conversation memory via LangGraph checkpointing (SQLite/in-memory)
- Structured logging (JSON/text)
- Docker support
- Full dev tooling (ruff, mypy, pytest, pre-commit)

## Quick Start

```bash
# Clone
git clone <repo-url>
cd magma

# Install (requires uv)
make install

# Configure
cp .env.example .env
# Edit .env with your API key(s)

# Run
make serve
```

Server starts at `http://localhost:8000`.

## Adding an Agent

Create a folder under `src/agent/agents/` — it's auto-discovered at startup:

```
src/agent/agents/
└── my_agent/
    ├── __init__.py
    ├── state.py    # extend BaseAgentState
    ├── tools.py    # define @tool functions + ALL_TOOLS list
    ├── nodes.py    # make_agent_node, make_tool_node
    └── graph.py    # build_graph(settings) -> CompiledGraph
```

The agent is immediately available at `/my_agent/invoke` and `/my_agent/stream`.

See `src/agent/agents/agent1/` and `agent2/` for working examples.

## API Endpoints

### `GET /health`

Returns server status and list of loaded agents.

```json
{"status": "ok", "agents": ["agent1", "agent2"]}
```

### `POST /{agent_name}/invoke`

Send messages to an agent and get a complete response.

```bash
curl -X POST http://localhost:8000/agent1/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "thread_id": "my-session"
  }'
```

```json
{"response": "Hello! How can I help?", "thread_id": "my-session"}
```

### `POST /{agent_name}/stream`

Same input, streams response via Server-Sent Events (SSE).

```bash
curl -N -X POST http://localhost:8000/agent1/stream \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

Events: `on_chat_model_stream` (token deltas), `on_tool_start`, `on_tool_end`, `error`.

## Configuration

**API keys** in `.env`:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
```

**Settings** in `config.yaml`:

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

## LLM Providers

Install only the provider(s) you need:

```bash
uv sync --extra openai      # OpenAI
uv sync --extra anthropic   # Anthropic
uv sync --extra google      # Google
uv sync --extra all         # All providers
```

| Provider  | `config.yaml` value | Models (examples)          |
|-----------|---------------------|----------------------------|
| OpenAI    | `openai`            | `gpt-4o`, `gpt-4o-mini`   |
| Anthropic | `anthropic`         | `claude-sonnet-4-20250514` |
| Google    | `google`            | `gemini-2.0-flash`         |

## Project Structure

```
magma/
├── src/agent/
│   ├── config.py              # Settings (env + yaml)
│   ├── logging.py             # Structured logging
│   ├── server.py              # FastAPI — dynamic agent routing
│   ├── __main__.py            # Entry point
│   │
│   ├── core/
│   │   ├── state.py           # BaseAgentState
│   │   └── registry.py        # Auto-discover agents
│   │
│   └── agents/
│       ├── agent1/            # Example agent 1
│       │   ├── state.py
│       │   ├── tools.py
│       │   ├── nodes.py
│       │   └── graph.py
│       └── agent2/            # Example agent 2
│           ├── state.py
│           ├── tools.py
│           ├── nodes.py
│           └── graph.py
│
├── tests/
├── config.yaml
├── .env.example
├── Dockerfile
├── docker-compose.yaml
├── Makefile
└── pyproject.toml
```

## Development

```bash
make test       # Run tests
make lint       # Lint with ruff
make format     # Format with ruff
make typecheck  # Type-check with mypy
make docker     # Run via Docker
```

## License

MIT
