"""CrewAI adapters: a memory ``StorageBackend`` and agent tools backed by VeraMem.

    pip install veramem crewai

    from crewai import Agent, Crew
    from crewai.memory import Memory as CrewMemory

    # 1) VeraMem as the storage behind CrewAI's own memory system
    crew = Crew(..., memory=CrewMemory(storage=VeraMemStorage(mem, user_id="priya")))

    # 2) VeraMem's own recall/feedback loop as agent tools
    agent = Agent(..., tools=veramem_tools(mem, user_id="priya"))

``VeraMemStorage`` implements ``crewai.memory.storage.backend.StorageBackend``. Every
CrewAI record becomes one VeraMem memory owned by the bound user: hashed raw
evidence, a ledger entry and an authority label. CrewAI records are written by
CrewAI's LLM analysis of task output, so they are stored as the agent's claims
(``role="assistant"``, Attested) unless you choose otherwise.

What it cannot do exactly: CrewAI's protocol passes a *precomputed query embedding*
(from CrewAI's embedder) rather than query text, so VeraMem's hybrid text+vector+
outcome ranking can't run inside ``search``. The adapter keeps CrewAI's embedding
with each memory and ranks by cosine similarity, like CrewAI's LanceDB backend.
For VeraMem-ranked recall with outcome feedback, give agents ``veramem_tools``.
Deletes retract (kept for audit, hidden from retrieval) unless ``hard_delete=True``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

try:
    from crewai.memory.types import MemoryRecord, ScopeInfo
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field
except ImportError as _e:  # pragma: no cover - exercised only without the extra
    from ._common import missing

    raise missing("crewai", "crewai", _e) from _e

import numpy as np

from ..memory import Memory
from ..types import MemoryItem
from ._common import WRITE_ROLES, put_memory, require_user
from .tools import VeraMemTools, description

__all__ = ["VeraMemStorage", "veramem_tools"]


def _in_scope(scope: str, prefix: str | None) -> bool:
    if prefix is None or not prefix.strip("/"):
        return True
    return scope.startswith(prefix.rstrip("/"))


class VeraMemStorage:
    """CrewAI ``StorageBackend`` that persists records as VeraMem memories."""

    def __init__(
        self,
        memory: Memory,
        *,
        user_id: str,
        agent_id: str | None = None,
        role: str = "assistant",
        hard_delete: bool = False,
    ) -> None:
        if role not in WRITE_ROLES:
            raise ValueError(f"role must be one of {WRITE_ROLES}")
        self.memory = memory
        self.user_id = require_user(user_id)
        self.agent_id = agent_id
        self.role = role
        self.hard_delete = hard_delete

    # ------------------------------------------------------------ helpers
    def _items(self, scope_prefix: str | None = None) -> list[MemoryItem]:
        rows = self.memory.store.memories(
            "m.user_id IS ? AND m.agent_id IS ? AND m.status='active' "
            "AND json_extract(m.metadata,'$.crewai.id') IS NOT NULL",
            (self.user_id, self.agent_id),
            order="created_at DESC",
        )
        return [m for m in rows if _in_scope(m.metadata["crewai"].get("scope", "/"), scope_prefix)]

    @staticmethod
    def _record(m: MemoryItem) -> MemoryRecord:
        c = m.metadata["crewai"]
        emb = m.metadata.get("crewai_embedding")
        return MemoryRecord(
            id=c["id"],
            content=m.text,
            scope=c.get("scope", "/"),
            categories=c.get("categories", []),
            metadata=c.get("metadata", {}),
            importance=c.get("importance", 0.5),
            created_at=datetime.fromisoformat(c["created_at"]),
            last_accessed=datetime.fromisoformat(c.get("last_accessed", c["created_at"])),
            source=c.get("source"),
            private=bool(c.get("private", False)),
            embedding=emb,
        )

    @staticmethod
    def _matches(
        m: MemoryItem,
        categories: list[str] | None,
        metadata_filter: dict[str, Any] | None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
    ) -> bool:
        c = m.metadata["crewai"]
        if record_ids is not None and c["id"] not in record_ids:
            return False
        if categories and not any(x in c.get("categories", []) for x in categories):
            return False
        if metadata_filter and any(
            c.get("metadata", {}).get(k) != v for k, v in metadata_filter.items()
        ):
            return False
        return not (older_than and datetime.fromisoformat(c["created_at"]) >= older_than)

    def _remove(self, m: MemoryItem) -> None:
        if self.hard_delete:
            self.memory.forget(m.id, user_id=self.user_id, agent_id=self.agent_id)
        else:
            self.memory.delete(m.id, user_id=self.user_id, agent_id=self.agent_id)

    def _find(self, record_id: str) -> MemoryItem | None:
        rows = self.memory.store.memories(
            "m.user_id IS ? AND m.agent_id IS ? AND m.status='active' "
            "AND json_extract(m.metadata,'$.crewai.id')=?",
            (self.user_id, self.agent_id, record_id),
            limit=1,
        )
        return rows[0] if rows else None

    # ------------------------------------------------------------ protocol
    def save(self, records: list[MemoryRecord]) -> None:
        for rec in records:
            data = rec.model_dump(mode="json", exclude={"content", "embedding"})
            meta: dict[str, Any] = {"crewai": data}
            if rec.embedding is not None:
                meta["crewai_embedding"] = [float(x) for x in rec.embedding]
            put_memory(
                self.memory,
                rec.content,
                role=self.role,
                user_id=self.user_id,
                agent_id=self.agent_id,
                metadata=meta,
                episode_metadata={"crewai": data},
            )

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        q = np.asarray(query_embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q)) or 1.0
        scored: list[tuple[MemoryRecord, float]] = []
        for m in self._items(scope_prefix):
            emb = m.metadata.get("crewai_embedding")
            if not emb or len(emb) != len(q) or not self._matches(m, categories, metadata_filter):
                continue
            v = np.asarray(emb, dtype=np.float32)
            score = float(q @ v) / (qn * (float(np.linalg.norm(v)) or 1.0))
            if score >= min_score:
                scored.append((self._record(m), score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:limit]

    def delete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        n = 0
        for m in self._items(scope_prefix):
            if self._matches(m, categories, metadata_filter, record_ids, older_than):
                self._remove(m)
                n += 1
        return n

    def update(self, record: MemoryRecord) -> None:
        old = self._find(record.id)
        self.save([record])
        if old is not None:
            self._remove(old)

    def get_record(self, record_id: str) -> MemoryRecord | None:
        m = self._find(record_id)
        return self._record(m) if m else None

    def list_records(
        self, scope_prefix: str | None = None, limit: int = 200, offset: int = 0
    ) -> list[MemoryRecord]:
        return [self._record(m) for m in self._items(scope_prefix)[offset : offset + limit]]

    def get_scope_info(self, scope: str) -> ScopeInfo:
        recs = [self._record(m) for m in self._items(scope)]
        return ScopeInfo(
            path=scope,
            record_count=len(recs),
            categories=sorted({c for r in recs for c in r.categories}),
            oldest_record=min((r.created_at for r in recs), default=None),
            newest_record=max((r.created_at for r in recs), default=None),
            child_scopes=self.list_scopes(scope),
        )

    def list_scopes(self, parent: str = "/") -> list[str]:
        base = parent.rstrip("/")
        prefix = base + "/"
        children: set[str] = set()
        for m in self._items(base or None):
            sc = str(m.metadata["crewai"].get("scope", "/"))
            if sc.startswith(prefix) and sc.rstrip("/") != base:
                first = sc[len(prefix) :].split("/", 1)[0]
                if first:
                    children.add(prefix + first)
        return sorted(children)

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self._items(scope_prefix):
            for c in m.metadata["crewai"].get("categories", []):
                counts[c] = counts.get(c, 0) + 1
        return counts

    def count(self, scope_prefix: str | None = None) -> int:
        return len(self._items(scope_prefix))

    def reset(self, scope_prefix: str | None = None) -> None:
        for m in self._items(scope_prefix):
            self._remove(m)

    def close(self) -> None:
        """No-op: the VeraMem ``Memory`` is owned (and closed) by the caller."""

    async def asave(self, records: list[MemoryRecord]) -> None:
        await asyncio.to_thread(self.save, records)

    async def asearch(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        return await asyncio.to_thread(
            self.search,
            query_embedding,
            scope_prefix,
            categories,
            metadata_filter,
            limit,
            min_score,
        )

    async def adelete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        return await asyncio.to_thread(
            self.delete, scope_prefix, categories, record_ids, older_than, metadata_filter
        )


# ---------------------------------------------------------------- tools
class _RememberArgs(BaseModel):
    content: str = Field(..., description="What to remember.")
    role: str = Field(
        "assistant", description="Who the content comes from: assistant, tool or document."
    )


class _RecallArgs(BaseModel):
    query: str = Field(..., description="What to look for.")
    limit: int = Field(8, description="Maximum number of memories.")


class _FeedbackArgs(BaseModel):
    trace_id: str = Field(..., description="trace_id returned by recall.")
    reward: float = Field(..., description="-1 (wrong) to 1 (helpful).")
    cited: list[str] | None = Field(None, description="Ids of the memories you used.")


class _CheckArgs(BaseModel):
    memory_ids: list[str] = Field(..., description="Memory ids you want to act on.")


class _VeraMemTool(BaseTool):
    core: Any = Field(exclude=True)

    def _call(self, name: str, **kwargs: Any) -> str:
        import json

        return json.dumps(self.core.call(name, kwargs), default=str)


class VeraMemRememberTool(_VeraMemTool):
    name: str = "veramem_remember"
    description: str = description("remember")
    args_schema: type[BaseModel] = _RememberArgs

    def _run(self, content: str, role: str = "assistant", **_: Any) -> str:
        return self._call("remember", content=content, role=role)


class VeraMemRecallTool(_VeraMemTool):
    name: str = "veramem_recall"
    description: str = description("recall")
    args_schema: type[BaseModel] = _RecallArgs

    def _run(self, query: str, limit: int = 8, **_: Any) -> str:
        return self._call("recall", query=query, limit=limit)


class VeraMemFeedbackTool(_VeraMemTool):
    name: str = "veramem_feedback"
    description: str = description("feedback")
    args_schema: type[BaseModel] = _FeedbackArgs

    def _run(self, trace_id: str, reward: float, cited: list[str] | None = None, **_: Any) -> str:
        return self._call("feedback", trace_id=trace_id, reward=reward, cited=cited)


class VeraMemCheckActionTool(_VeraMemTool):
    name: str = "veramem_check_action"
    description: str = description("check_action")
    args_schema: type[BaseModel] = _CheckArgs

    def _run(self, memory_ids: list[str], **_: Any) -> str:
        return self._call("check_action", memory_ids=memory_ids)


def veramem_tools(
    memory: Memory, *, user_id: str, agent_id: str | None = None, **options: Any
) -> list[BaseTool]:
    """CrewAI tools (remember, recall, feedback, check_action) bound to one user.

    ``options`` are passed to ``veramem.integrations.tools.VeraMemTools``.
    """
    core = VeraMemTools(memory, user_id=user_id, agent_id=agent_id, **options)
    return [
        VeraMemRememberTool(core=core),
        VeraMemRecallTool(core=core),
        VeraMemFeedbackTool(core=core),
        VeraMemCheckActionTool(core=core),
    ]
