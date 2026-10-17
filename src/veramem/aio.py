"""Async facade for asyncio applications (FastAPI, LangGraph, agent loops).

Every call runs the synchronous implementation in a worker thread. That is safe:
all database access is serialised by the store's lock, and model calls are I/O bound.

    from veramem import AsyncMemory
    mem = AsyncMemory()
    await mem.add("I moved to Lisbon", user_id="priya")
    answer = await mem.answer("Where does Priya live?", user_id="priya")
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any

from .memory import Memory

_ASYNC_METHODS = (
    "add",
    "search",
    "answer",
    "get",
    "get_all",
    "evidence",
    "history",
    "update",
    "delete",
    "forget",
    "feedback",
    "learn",
    "consolidate",
    "check_action",
    "export_trajectories",
    "share",
    "set_shareable",
    "grant_agent",
    "revoke_agent",
    "grant_resource",
    "revoke_resource",
    "join_team",
    "leave_team",
    "verify",
    "export",
    "import_",
    "reindex",
)


class AsyncMemory:
    """Same API as `Memory`, with awaitable methods."""

    def __init__(self, *args: Any, **kwargs: Any):
        self.sync = Memory(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.sync, name)
        if name in _ASYNC_METHODS:

            @functools.wraps(attr)
            async def call(*a: Any, **kw: Any) -> Any:
                return await asyncio.to_thread(attr, *a, **kw)

            return call
        return attr

    def info(self) -> dict[str, Any]:
        return self.sync.info()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.sync.close)

    async def __aenter__(self) -> AsyncMemory:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
