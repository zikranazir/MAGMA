"""Entry point for running the agent server via ``python -m agent``."""

import uvicorn

from agent.config import Settings
from agent.server import create_app


def main() -> None:
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
