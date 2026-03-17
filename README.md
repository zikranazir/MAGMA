# Agent Boilerplate

A production-ready starter kit for building AI agents with [LangGraph](https://github.com/langchain-ai/langgraph). Multi-LLM support (OpenAI, Anthropic, Google).

## Features

- Flexible LangGraph agent architecture
- Multi-LLM provider support (OpenAI, Anthropic, Google)
- FastAPI server with invoke and streaming endpoints
- Conversation memory via LangGraph checkpointing (SQLite/in-memory)
- Structured logging (JSON/text)
- Docker support
- Full dev tooling (ruff, mypy, pytest, pre-commit)

## Quick Start

```bash
# Clone
git clone <repo-url>
cd agent-boilerplate

# Install (requires uv)
make install

# Configure
cp .env.example .env
# Edit .env with your API key(s)

# Run
make serve
```

The server starts at `http://localhost:8000`.

## Configuration

**API keys** go in `.env`:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
```

**Agent settings** go in `config.yaml`:

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
uv sync --extra openai      # OpenAI only
uv sync --extra anthropic   # Anthropic only
uv sync --extra google      # Google only
uv sync --extra all         # All providers
```

| Provider  | `config.yaml` value | Models (examples)            |
|-----------|---------------------|------------------------------|
| OpenAI    | `openai`            | `gpt-4o`, `gpt-4o-mini`     |
| Anthropic | `anthropic`         | `claude-sonnet-4-20250514`|
| Google    | `google`            | `gemini-2.0-flash`          |

## API Endpoints

### `GET /health`

Returns server status.

### `POST /invoke`

Send a message and get a complete response.

```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "thread_id": "my-session"}'
```

### `POST /stream`

Send a message and receive a streamed response via Server-Sent Events (SSE).

```bash
curl -N -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "thread_id": "my-session"}'
```

## Development

```bash
make test       # Run tests
make lint       # Lint with ruff
make format     # Format with ruff
make typecheck  # Type-check with mypy
```

## Docker

```bash
make docker
# or
docker compose up --build
```

## Project Structure

```
agent-boilerplate/
├── src/agent/
│   ├── __init__.py
│   ├── __main__.py        # Entry point
│   ├── config.py          # Settings (env + yaml)
│   ├── graph.py           # LangGraph agent graph
│   ├── logging.py         # Structured logging setup
│   ├── nodes.py           # Graph node functions
│   ├── server.py          # FastAPI server
│   ├── state.py           # Agent state schema
│   └── tools.py           # Tool definitions
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_graph.py
│   └── test_server.py
├── config.yaml
├── .env.example
├── Dockerfile
├── docker-compose.yaml
├── Makefile
├── pyproject.toml
└── ruff.toml
```

## License

MIT
