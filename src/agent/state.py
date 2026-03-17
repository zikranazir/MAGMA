"""Agent state definition for the LangGraph agent."""

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State object passed between nodes in the agent graph.

    Attributes:
        messages: Conversation message history, managed by LangGraph's
            ``add_messages`` reducer so that new messages are appended
            rather than replacing the list.
    """

    messages: Annotated[list[BaseMessage], add_messages]
