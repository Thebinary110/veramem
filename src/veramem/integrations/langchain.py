"""LangChain adapters: a retriever and a chat message history backed by VeraMem.

pip install "veramem[langchain]"

retriever = VeraMemRetriever(memory=mem, user_id="priya")
docs = retriever.invoke("where does priya live?")
retriever.feedback(1.0)              # outcome for the last search

history = VeraMemChatMessageHistory(mem, user_id="priya", session_id="s1")
# use with RunnableWithMessageHistory(chain, lambda sid: ..., ...)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.documents import Document
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        ChatMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.retrievers import BaseRetriever
    from pydantic import PrivateAttr, field_validator
except ImportError as _e:  # pragma: no cover - exercised only without the extra
    from ._common import missing

    raise missing("langchain-core", "langchain", _e) from _e

from ..memory import Memory
from ..types import SourceType, now
from ._common import hit_metadata, require_user

__all__ = ["VeraMemRetriever", "VeraMemChatMessageHistory", "message_role"]

_ROLE_BY_TYPE = {
    "human": "user",
    "ai": "assistant",
    "tool": "tool",
    "function": "tool",
    "system": "system",
}


def message_role(message: BaseMessage) -> str:
    """VeraMem role for a LangChain message: human->user, ai->assistant, tool->tool.

    Anything unrecognised is stored as ``tool`` (Unendorsed), never as the user.
    """
    if isinstance(message, ChatMessage):
        r = message.role.lower()
        return {"human": "user", "ai": "assistant"}.get(
            r, r if r in ("user", "assistant", "tool", "system") else "tool"
        )
    return _ROLE_BY_TYPE.get(message.type, "tool")


class VeraMemRetriever(BaseRetriever):
    """Hybrid, access-controlled VeraMem search as a LangChain retriever.

    Each ``Document`` has the memory text as ``page_content`` and ``metadata`` with
    ``id``, ``authority``, ``kind``, ``date``, ``trace_id``, ``granularity`` and
    ``score``. The trace of the most recent search is kept in ``last_trace_id`` so
    you can report the outcome with ``feedback()``.
    """

    memory: Any
    user_id: str
    agent_id: str | None = None
    limit: int = 8
    include_raw: bool = True

    _last_trace_id: str | None = PrivateAttr(default=None)

    @field_validator("user_id")
    @classmethod
    def _user(cls, v: str) -> str:
        return require_user(v)

    @property
    def last_trace_id(self) -> str | None:
        return self._last_trace_id

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        mem: Memory = self.memory
        r = mem.search(
            query,
            user_id=self.user_id,
            agent_id=self.agent_id,
            limit=self.limit,
            include_raw=self.include_raw,
        )
        self._last_trace_id = r.trace_id
        return [
            Document(page_content=h.memory.text, metadata=hit_metadata(h, r.trace_id))
            for h in r.hits
        ]

    def feedback(
        self, reward: float, *, cited: list[str] | None = None, trace_id: str | None = None
    ) -> dict[str, Any]:
        """Report how well the last (or a given) search worked: reward in [-1, 1]."""
        tid = trace_id or self._last_trace_id
        if tid is None:
            raise ValueError("no search has been run yet")
        mem: Memory = self.memory
        return mem.feedback(tid, reward, cited=cited, user_id=self.user_id, agent_id=self.agent_id)


class VeraMemChatMessageHistory(BaseChatMessageHistory):
    """Chat history for one (user, session), stored as VeraMem episodes.

    ``add_messages`` writes through ``Memory.add`` (so an LLM-configured Memory
    extracts long-term memories from the conversation), and ``messages`` returns the
    most recent ``k`` turns of this session.

    Tool messages are stored (as Unendorsed evidence) but not replayed by default:
    the assistant tool-call that preceded them isn't stored, and most chat APIs
    reject a tool result without it. Set ``include_tool_messages=True`` to get them.

    ``clear()`` hides earlier turns from ``messages``; it does not erase anything
    from long-term memory. Use ``Memory.forget`` for erasure.
    """

    def __init__(
        self,
        memory: Memory,
        *,
        user_id: str,
        session_id: str,
        agent_id: str | None = None,
        k: int = 50,
        infer: bool = True,
        include_tool_messages: bool = False,
    ) -> None:
        self.memory = memory
        self.user_id = require_user(user_id)
        self.session_id = session_id
        self.agent_id = agent_id
        self.k = k
        self.infer = infer
        self.include_tool_messages = include_tool_messages

    @property
    def _cleared_key(self) -> str:
        return f"langchain.history.cleared:{self.user_id}:{self.agent_id or ''}:{self.session_id}"

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        cutoff = float(self.memory.store.meta_get(self._cleared_key) or 0.0)
        eps = self.memory.store.episodes(
            "e.user_id IS ? AND e.agent_id IS ? AND e.session_id=? AND e.occurred_at>?",
            (self.user_id, self.agent_id, self.session_id, cutoff),
        )
        out: list[BaseMessage] = []
        for ep in eps:
            src = ep.source
            if src is SourceType.USER:
                out.append(HumanMessage(content=ep.content))
            elif src is SourceType.ASSISTANT:
                out.append(AIMessage(content=ep.content))
            elif src is SourceType.SYSTEM:
                out.append(SystemMessage(content=ep.content))
            elif src is SourceType.TOOL:
                if self.include_tool_messages:
                    out.append(ToolMessage(content=ep.content, tool_call_id=ep.actor or "tool"))
            else:
                out.append(ChatMessage(role=src.value, content=ep.content))
        return out[-self.k :] if self.k else out

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        batch = []
        for m in messages:
            text = m.text
            if not text.strip():
                continue  # e.g. an AI message that only carries tool calls
            role = message_role(m)
            item: dict[str, Any] = {"role": role, "content": text}
            if role == "tool":
                item["name"] = getattr(m, "tool_call_id", None) or m.name or "tool"
            elif m.name:
                item["name"] = m.name
            batch.append(item)
        if batch:
            self.memory.add(
                batch,
                user_id=self.user_id,
                agent_id=self.agent_id,
                session_id=self.session_id,
                infer=self.infer,
            )

    def clear(self) -> None:
        self.memory.store.meta_set(self._cleared_key, repr(now()))
