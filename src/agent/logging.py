"""Logging setup for the LangGraph agent."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import LoggingConfig


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON objects for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": datetime.fromtimestamp(
                    record.created, tz=UTC
                ).isoformat(),
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
            }
        )


_TEXT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(config: LoggingConfig) -> None:
    """Configure stdlib logging based on the provided config.

    This function is idempotent — it clears existing handlers on the root
    logger before adding a new ``StreamHandler`` to stdout.

    Args:
        config: A ``LoggingConfig`` instance with ``level`` and ``format`` fields.
    """
    root = logging.getLogger()

    # Clear existing handlers for idempotency.
    root.handlers.clear()

    # Choose formatter.
    if config.format == "json":
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(_TEXT_FORMAT)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root.setLevel(config.level)
    root.addHandler(handler)
