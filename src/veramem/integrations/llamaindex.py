"""LlamaIndex adapters: a retriever and a ``Memory`` block backed by VeraMem.

pip install veramem llama-index-core

retriever = VeraMemRetriever(mem, user_id="priya")
nodes = retriever.retrieve("where does priya live?")

from llama_index.core.memory import Memory as LIMemory
block = VeraMemMemoryBlock(memory=mem, user_id="priya")
chat_memory = LIMemory.from_defaults(session_id="s1", memory_blocks=[block])
"""

from __future__ import annotations

import asyncio
from typing import Any

try:
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from llama_index.core.bridge.pydantic import Field, PrivateAttr, field_validator
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.memory.memory import BaseMemoryBlock
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ImportError as _e:  # pragma: no cover - exercised only without the extra
    from ._common import missing

    raise missing("llama-index-core", "llamaindex", _e) from _e

from ..memory import Memory
from ._common import hit_metadata, require_user

__all__ = ["VeraMemRetriever", "VeraMemMemoryBlock", "chat_role"]

_ROLES = {
    MessageRole.USER: "user",
    MessageRole.ASSISTANT: "assistant",
    MessageRole.MODEL: "assistant",
    MessageRole.CHATBOT: "assistant",
    MessageRole.TOOL: "tool",
    MessageRole.FUNCTION: "tool",
    MessageRole.SYSTEM: "system",
    MessageRole.DEVELOPER: "system",
}


def chat_role(message: ChatMessage) -> str:
    """VeraMem role for a LlamaIndex message; unknown roles become ``tool`` (Unendorsed)."""
    return _ROLES.get(MessageRole(message.role), "tool")


class VeraMemRetriever(BaseRetriever):
    """VeraMem hybrid search as a LlamaIndex retriever returning ``NodeWithScore``.

    Node ids are VeraMem memory ids; node metadata has ``id``, ``authority``,
    ``kind``, ``date``, ``trace_id``, ``granularity`` and ``score``.
    """

    def __init__(
        self,
        memory: Memory,
        *,
        user_id: str,
        agent_id: str | None = None,
        limit: int = 8,
        include_raw: bool = True,
        callback_manager: CallbackManager | None = None,
        verbose: bool = False,
    ) -> None:
        self.memory = memory
        self.user_id = require_user(user_id)
        self.agent_id = agent_id
        self.limit = limit
        self.include_raw = include_raw
        self.last_trace_id: str | None = None
        super().__init__(callback_manager=callback_manager, verbose=verbose)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        r = self.memory.search(
            query_bundle.query_str,
            user_id=self.user_id,
            agent_id=self.agent_id,
            limit=self.limit,
            include_raw=self.include_raw,
        )
        self.last_trace_id = r.trace_id
        return [
            NodeWithScore(
                node=TextNode(
                    id_=h.memory.id, text=h.memory.text, metadata=hit_metadata(h, r.trace_id)
                ),
                score=h.score,
            )
            for h in r.hits
        ]

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return await asyncio.to_thread(self._retrieve, query_bundle)

    def feedback(
        self, reward: float, *, cited: list[str] | None = None, trace_id: str | None = None
    ) -> dict[str, Any]:
        """Report how well the last (or a given) retrieval worked: reward in [-1, 1]."""
        tid = trace_id or self.last_trace_id
        if tid is None:
            raise ValueError("no retrieval has been run yet")
        return self.memory.feedback(
            tid, reward, cited=cited, user_id=self.user_id, agent_id=self.agent_id
        )


class VeraMemMemoryBlock(BaseMemoryBlock[str]):
    """Long-term memory block for ``llama_index.core.memory.Memory``.

    ``aget`` searches VeraMem with the latest user message and returns a cited
    context block; ``aput`` stores messages flushed from short-term memory (roles
    mapped user/assistant/tool/system, so authority follows the speaker).
    """

    name: str = Field(default="veramem", description="The name of the memory block.")
    memory: Any = Field(exclude=True, description="A veramem.Memory instance.")
    user_id: str = Field(description="VeraMem user this block is bound to.")
    agent_id: str | None = Field(default=None)
    limit: int = Field(default=8, description="Memories to inject per turn.")
    infer: bool = Field(default=True, description="Extract memories with the Memory's LLM.")

    _last_trace_id: str | None = PrivateAttr(default=None)

    @field_validator("user_id")
    @classmethod
    def _user(cls, v: str) -> str:
        return require_user(v)

    @property
    def last_trace_id(self) -> str | None:
        return self._last_trace_id

    async def _aget(self, messages: list[ChatMessage] | None = None, **block_kwargs: Any) -> str:
        query = next(
            (
                m.content
                for m in reversed(messages or [])
                if MessageRole(m.role) == MessageRole.USER and m.content
            ),
            None,
        )
        if not query:
            return ""
        mem: Memory = self.memory
        r = await asyncio.to_thread(
            mem.search, query, user_id=self.user_id, agent_id=self.agent_id, limit=self.limit
        )
        self._last_trace_id = r.trace_id
        return r.context()

    async def _aput(self, messages: list[ChatMessage]) -> None:
        batch: list[dict[str, Any]] = []
        session_id = None
        for m in messages:
            if not m.content or not str(m.content).strip():
                continue
            session_id = session_id or m.additional_kwargs.get("session_id")
            batch.append({"role": chat_role(m), "content": str(m.content)})
        if not batch:
            return
        mem: Memory = self.memory
        await asyncio.to_thread(
            mem.add,
            batch,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=session_id,
            infer=self.infer,
        )
