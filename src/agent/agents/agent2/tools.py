"""Tool definitions for agent2.

Add new tools by defining functions decorated with ``@tool`` and
appending them to the ``ALL_TOOLS`` list at the bottom of this file.
"""

import random
from typing import Any

from langchain_core.tools import tool


@tool
def get_random_number() -> int:
    """Get a random integer between 1 and 100 (inclusive)."""
    return random.randint(1, 100)


# ---------------------------------------------------------------------------
# To add more tools, define them above and include them in ALL_TOOLS.
# ---------------------------------------------------------------------------

ALL_TOOLS: list[Any] = [
    get_random_number,
]
