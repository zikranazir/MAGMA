"""Tool definitions for the LangGraph agent.

Add new tools by defining functions decorated with ``@tool`` and
appending them to the ``ALL_TOOLS`` list at the bottom of this file.
"""

from datetime import datetime, timezone

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Get the current UTC time."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# To add more tools, define them above and include them in ALL_TOOLS.
#
# Example:
#
#   @tool
#   def search_web(query: str) -> str:
#       """Search the web for a query."""
#       ...
#
# Then add `search_web` to the list below.
# ---------------------------------------------------------------------------

ALL_TOOLS: list = [
    get_current_time,
]
