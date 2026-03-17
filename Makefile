.PHONY: install test lint format typecheck serve docker

install:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

serve:
	uv run python -m agent

docker:
	docker compose up --build
