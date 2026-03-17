"""Agent registry — discovers and builds all agents in the agents/ directory."""

import logging
from importlib import import_module
from pathlib import Path
from typing import Any

from agent.config import Settings


def load_all_agents(settings: Settings) -> dict[str, Any]:
    """Discover and build all agents in the agents/ directory.

    Convention: each agent module must expose a build_graph(settings) function.
    """
    agents: dict[str, Any] = {}
    agents_dir = Path(__file__).parent.parent / "agents"

    for agent_dir in sorted(agents_dir.iterdir()):
        if not agent_dir.is_dir() or agent_dir.name.startswith("_"):
            continue
        try:
            module = import_module(f"agent.agents.{agent_dir.name}.graph")
            agents[agent_dir.name] = module.build_graph(settings)
        except Exception as e:
            logging.getLogger(__name__).error(
                "Failed to load agent %s: %s", agent_dir.name, e
            )

    return agents
