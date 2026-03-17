"""Graph node factories for agent2."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt import ToolNode

from agent.agents.agent2.state import Agent2State


def make_agent_node(model: BaseChatModel, tools: list[Any]) -> Callable[..., Any]:
    """Create an agent node that invokes the chat model.

    The model is bound to the supplied tools **once** (inside this factory),
    so the binding cost is not repeated on every graph invocation.

    Args:
        model: A LangChain chat model instance.
        tools: List of LangChain tools to bind to the model.

    Returns:
        A node function compatible with LangGraph's ``StateGraph``.
    """
    bound_model = model.bind_tools(tools) if tools else model

    def agent_node(state: Agent2State) -> dict[str, Any]:
        response = bound_model.invoke(state["messages"])
        return {"messages": [response]}

    return agent_node


def make_tool_node(tools: list[Any]) -> ToolNode:
    """Create a tool-execution node from the given tools.

    Args:
        tools: List of LangChain tools.

    Returns:
        A ``ToolNode`` instance that can execute tool calls.
    """
    return ToolNode(tools)
