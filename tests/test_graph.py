"""Tests for graph construction (agent.graph)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langgraph.graph.state import CompiledStateGraph

from agent.config import Settings
from agent.graph import build_graph, should_continue
from agent.state import AgentState


class TestShouldContinue:
    def test_returns_tools_when_tool_calls_present(self) -> None:
        msg = MagicMock()
        msg.tool_calls = [{"name": "some_tool"}]
        state: AgentState = {"messages": [msg]}
        assert should_continue(state) == "tools"

    def test_returns_end_when_no_tool_calls(self) -> None:
        msg = MagicMock()
        msg.tool_calls = []
        state: AgentState = {"messages": [msg]}
        assert should_continue(state) == "__end__"

    def test_returns_end_when_no_tool_calls_attr(self) -> None:
        msg = MagicMock(spec=[])  # no attributes at all
        state: AgentState = {"messages": [msg]}
        assert should_continue(state) == "__end__"


class TestBuildGraph:
    def test_returns_compiled_graph(
        self, test_settings: Settings, mock_chat_model: MagicMock
    ) -> None:
        """build_graph should return a CompiledStateGraph."""
        with patch("agent.graph.get_chat_model", return_value=mock_chat_model):
            graph = build_graph(test_settings)

        assert isinstance(graph, CompiledStateGraph)

    def test_graph_has_expected_nodes(
        self, test_settings: Settings, mock_chat_model: MagicMock
    ) -> None:
        """The compiled graph should contain 'agent' and 'tools' nodes."""
        with patch("agent.graph.get_chat_model", return_value=mock_chat_model):
            graph = build_graph(test_settings)

        node_names = set(graph.nodes.keys())
        assert "agent" in node_names
        assert "tools" in node_names
