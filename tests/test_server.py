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
    """Create an async test client with mocked agents."""
    mock_agents = {"agent1": mock_graph}
    with (
        patch("agent.server.load_all_agents", return_value=mock_agents),
        patch("agent.server.setup_logging"),
    ):
        app = create_app(test_settings)
        # Manually set agents on state since lifespan may not run with ASGITransport
        app.state.agents = mock_agents
        transport = ASGITransport(app=app)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.mark.asyncio
async def test_health_returns_ok(client: AsyncClient) -> None:
    """GET /health should return 200 with status ok and agents list."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "agents" in data


@pytest.mark.asyncio
async def test_invoke_returns_response(
    client: AsyncClient, mock_graph: MagicMock
) -> None:
    """POST /agent1/invoke should return the AI response and a thread_id."""
    resp = await client.post(
        "/agent1/invoke",
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
    """POST /agent1/invoke without thread_id should auto-generate one."""
    resp = await client.post(
        "/agent1/invoke",
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
    """POST /agent1/invoke should return 500 when the graph raises."""
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

    resp = await client.post(
        "/agent1/invoke",
        json={"messages": [{"role": "user", "content": "Fail"}]},
    )

    assert resp.status_code == 500
    data = resp.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_invoke_unknown_agent_returns_404(client: AsyncClient) -> None:
    """POST /unknown/invoke should return 404."""
    resp = await client.post(
        "/unknown/invoke",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert resp.status_code == 404
    data = resp.json()
    assert "error" in data
