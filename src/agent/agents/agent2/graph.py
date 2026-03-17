"""LangGraph graph construction for agent2."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from agent.agents.agent2.nodes import make_agent_node, make_tool_node
from agent.agents.agent2.state import Agent2State
from agent.agents.agent2.tools import ALL_TOOLS
from agent.config import Settings, get_chat_model


def should_continue(state: Agent2State) -> str:
    """Decide whether the agent should call tools or finish.

    Returns ``"tools"`` if the last message contains tool calls,
    otherwise returns ``END`` to terminate the graph.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def build_graph(settings: Settings) -> Any:
    """Build and compile the LangGraph agent graph for agent2.

    Args:
        settings: Application settings (LLM config, memory config, etc.).

    Returns:
        A compiled ``StateGraph`` ready to be invoked.
    """
    # --- Model & nodes ------------------------------------------------
    model = get_chat_model(settings)
    agent_node = make_agent_node(model, ALL_TOOLS)
    tool_node = make_tool_node(ALL_TOOLS)

    # --- Graph --------------------------------------------------------
    graph = StateGraph(Agent2State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    # --- Checkpointer -------------------------------------------------
    checkpointer: Any = None

    if settings.memory.enabled:
        if settings.memory.backend == "memory":
            from langgraph.checkpoint.memory import MemorySaver

            checkpointer = MemorySaver()

        elif settings.memory.backend == "sqlite":
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            checkpointer = AsyncSqliteSaver.from_conn_string(
                settings.memory.sqlite_path,
            )

    return graph.compile(checkpointer=checkpointer)
