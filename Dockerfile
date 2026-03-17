# === Builder stage ===
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock* ./
COPY config.yaml ./

RUN uv sync --frozen --no-dev

# === Runtime stage ===
FROM python:3.12-slim

RUN useradd --create-home agent

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ /app/src/
COPY config.yaml /app/config.yaml

ENV PATH="/app/.venv/bin:$PATH"

USER agent

EXPOSE 8000

CMD ["python", "-m", "agent"]
