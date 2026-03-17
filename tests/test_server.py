"""Tests for the FastAPI server (agent.server)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

from agent.config import Settings
from agent.server import create_app


@pytest.fixture()
def mock_graph() -> MagicMock:
    """Return a mock compiled graph with a canned ainvoke response."""
    graph = MagicMock()
    graph.ainvoke = AsyncMock(
        return_value={
            "messages": [AIMessage(content="Mock response from graph")],
        }
    )
    return graph


@pytest.fixture()
async def client(
    test_settings: Settings, mock_graph: MagicMock
) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client with mocked graph."""
    with (
        patch("agent.server.build_graph", return_value=mock_graph),
        patch("agent.server.setup_logging"),
    ):
        app = create_app(test_settings)
        # Manually set graph on state since lifespan may not run with ASGITransport
        app.state.graph = mock_graph
        transport = ASGITransport(app=app)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.mark.asyncio
async def test_health_returns_ok(client: AsyncClient) -> None:
    """GET /health should return 200 with status ok."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_invoke_returns_response(
    client: AsyncClient, mock_graph: MagicMock
) -> None:
    """POST /invoke should return the AI response and a thread_id."""
    resp = await client.post(
        "/invoke",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "thread_id": "test-thread-123",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["response"] == "Mock response from graph"
    assert data["thread_id"] == "test-thread-123"
    mock_graph.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_invoke_generates_thread_id(client: AsyncClient) -> None:
    """POST /invoke without thread_id should auto-generate one."""
    resp = await client.post(
        "/invoke",
        json={"messages": [{"role": "user", "content": "Hi"}]},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert "thread_id" in data
    assert len(data["thread_id"]) > 0


@pytest.mark.asyncio
async def test_invoke_handles_error(
    client: AsyncClient, mock_graph: MagicMock
) -> None:
    """POST /invoke should return 500 when the graph raises."""
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

    resp = await client.post(
        "/invoke",
        json={"messages": [{"role": "user", "content": "Fail"}]},
    )

    assert resp.status_code == 500
    data = resp.json()
    assert "error" in data
