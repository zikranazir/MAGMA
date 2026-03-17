"""FastAPI server for the LangGraph agent."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from langchain_core.messages import AIMessage, HumanMessage

from agent.config import Settings
from agent.graph import build_graph
from agent.logging import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class InvokeRequest(BaseModel):
    """Request body for the /invoke endpoint."""

    messages: list[dict]  # Each dict has "role" and "content"
    thread_id: str | None = None


class InvokeResponse(BaseModel):
    """Response body for the /invoke endpoint."""

    response: str
    thread_id: str


class StreamRequest(BaseModel):
    """Request body for the /stream endpoint."""

    messages: list[dict]  # Each dict has "role" and "content"
    thread_id: str | None = None


class ErrorResponse(BaseModel):
    """Error response body."""

    error: str
    detail: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert_messages(raw_messages: list[dict]) -> list[HumanMessage | AIMessage]:
    """Convert plain dicts to LangChain message objects."""
    converted: list[HumanMessage | AIMessage] = []
    for msg in raw_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            # Default to HumanMessage for unknown roles
            converted.append(HumanMessage(content=content))
    return converted


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(settings: Settings) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Application settings.

    Returns:
        A configured FastAPI instance.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        setup_logging(settings.logging)
        logger.info("Building agent graph...")
        app.state.graph = build_graph(settings)
        logger.info("Agent graph ready.")
        yield

    app = FastAPI(title="LangGraph Agent", lifespan=lifespan)

    # -- CORS middleware ----------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Endpoints ----------------------------------------------------------

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/invoke", response_model=InvokeResponse)
    async def invoke(body: InvokeRequest) -> InvokeResponse | JSONResponse:
        try:
            thread_id = body.thread_id or str(uuid.uuid4())
            messages = _convert_messages(body.messages)
            config = {"configurable": {"thread_id": thread_id}}

            result = await app.state.graph.ainvoke(
                {"messages": messages},
                config=config,
            )

            ai_message = result["messages"][-1]
            return InvokeResponse(
                response=ai_message.content,
                thread_id=thread_id,
            )
        except Exception as exc:
            logger.exception("Error during invoke")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=str(exc),
                    detail=None,
                ).model_dump(),
            )

    @app.post("/stream")
    async def stream(body: StreamRequest) -> EventSourceResponse:
        thread_id = body.thread_id or str(uuid.uuid4())
        messages = _convert_messages(body.messages)
        config = {"configurable": {"thread_id": thread_id}}

        async def event_generator() -> AsyncGenerator[dict, None]:
            try:
                async for event in app.state.graph.astream_events(
                    {"messages": messages},
                    config=config,
                    version="v2",
                ):
                    kind = event.get("event")

                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            yield {
                                "event": "on_chat_model_stream",
                                "data": json.dumps({"token": content, "thread_id": thread_id}),
                            }

                    elif kind == "on_tool_start":
                        yield {
                            "event": "on_tool_start",
                            "data": json.dumps({
                                "tool": event.get("name", ""),
                                "input": event["data"].get("input", {}),
                                "thread_id": thread_id,
                            }),
                        }

                    elif kind == "on_tool_end":
                        yield {
                            "event": "on_tool_end",
                            "data": json.dumps({
                                "tool": event.get("name", ""),
                                "output": str(event["data"].get("output", "")),
                                "thread_id": thread_id,
                            }),
                        }

            except Exception as exc:
                logger.exception("Error during stream")
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(exc)}),
                }

        return EventSourceResponse(event_generator())

    return app
