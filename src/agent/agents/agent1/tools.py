"""Tool definitions for agent1.

Add new tools by defining functions decorated with ``@tool`` and
appending them to the ``ALL_TOOLS`` list at the bottom of this file.
"""

from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Get the current UTC time."""
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# To add more tools, define them above and include them in ALL_TOOLS.
# ---------------------------------------------------------------------------

ALL_TOOLS: list[Any] = [
    get_current_time,
]
