"""Uniform adapters so every memory system is driven through the same interface.

Each adapter exposes the operations a benchmark needs. When a system has no
equivalent for a capability (for example revocable agent grants or tamper detection),
the adapter says so via `supports`, and the benchmark scores it as unsupported
instead of guessing.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

OLLAMA = os.environ.get("BENCH_OLLAMA", "http://localhost:11434")
LLM_MODEL = os.environ.get("BENCH_LLM", "qwen3.6:35b")
EMBED_MODEL = os.environ.get("BENCH_EMBED", "nomic-embed-text:latest")


class System:
    name = "base"
    supports: set[str] = set()

    def add(
        self,
        user_id: str,
        messages: list[dict[str, Any]],
        *,
        session_id: str | None = None,
        agent_id: str | None = None,
        resources: list[str] | None = None,
    ) -> None:
        raise NotImplementedError

    def search(
        self, user_id: str, query: str, k: int = 10, agent_id: str | None = None
    ) -> list[dict[str, Any]]:
        """-> [{"id", "text", "date"?, "authority"?}]"""
        raise NotImplementedError

    def memories(self, user_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def forget_user(self, user_id: str) -> None:
        raise NotImplementedError

    def storage_files(self) -> list[Path]:
        return []

    def close(self) -> None:
        pass

    # optional capabilities ------------------------------------------------
    def feedback(self, trace: Any, reward: float) -> None:  # "feedback"
        raise NotImplementedError

    def grant_agent(self, user_id: str, agent_id: str) -> None:  # "grants"
        raise NotImplementedError

    def revoke_agent(self, user_id: str, agent_id: str) -> None:
        raise NotImplementedError

    def tamper_one(self, user_id: str) -> None:  # "verify"
        raise NotImplementedError

    def verify(self) -> bool:
        raise NotImplementedError

    def authority_of(self, item: dict[str, Any]) -> str | None:
        return item.get("authority")


# ------------------------------------------------------------------ VeraMem
class VeraMemSystem(System):
    name = "veramem"
    supports = {"feedback", "grants", "verify", "authority", "evidence", "forget", "share"}

    def __init__(
        self,
        workdir: str | None = None,
        llm: Any = None,
        embedder: Any = None,
        cache: str | None = None,
    ):
        from veramem import Memory
        from veramem.providers import OllamaEmbedder, OllamaLLM, ResponseCache

        self.dir = Path(workdir or tempfile.mkdtemp(prefix="vm_bench_"))
        c = ResponseCache(cache) if cache else None
        self.mem = Memory(
            str(self.dir / "memory.db"),
            llm=llm or OllamaLLM(LLM_MODEL, host=OLLAMA, cache=c),
            embedder=embedder or OllamaEmbedder(EMBED_MODEL, host=OLLAMA, cache=c),
        )
        self.last_trace: str | None = None

    def add(self, user_id, messages, *, session_id=None, agent_id=None, resources=None):
        self.mem.add(
            messages, user_id=user_id, session_id=session_id, agent_id=agent_id, resources=resources
        )

    def search(self, user_id, query, k=10, agent_id=None):
        r = self.mem.search(query, user_id=user_id, agent_id=agent_id, limit=k)
        self.last_trace = r.trace_id
        return [
            {
                "id": h.memory.id,
                "text": h.memory.text,
                "date": time.strftime(
                    "%Y-%m-%d", time.gmtime(h.memory.valid_from or h.memory.created_at)
                ),
                "authority": h.memory.authority.value,
                "granularity": h.granularity,
            }
            for h in r.hits
        ]

    def memories(self, user_id):
        return [
            {
                "id": m.id,
                "text": m.text,
                "authority": m.authority.value,
                "evidence": len(m.evidence),
                "kind": m.kind.value,
            }
            for m in self.mem.get_all(user_id=user_id)
        ]

    def forget_user(self, user_id):
        self.mem.forget(user_id=user_id, everything=True)

    def storage_files(self):
        self.mem.store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return [p for p in self.dir.rglob("*") if p.is_file()]

    def feedback(self, trace, reward):
        self.mem.feedback(trace, reward)

    def grant_agent(self, user_id, agent_id):
        self.mem.grant_agent(user_id, agent_id)

    def revoke_agent(self, user_id, agent_id):
        self.mem.revoke_agent(user_id, agent_id)

    def tamper_one(self, user_id):
        mid = self.mem.get_all(user_id=user_id)[0].id
        self.mem.store.db.execute("UPDATE memories SET text=text||' (edited)' WHERE id=?", (mid,))

    def verify(self):
        return not self.mem.verify()["ok"]  # True == tampering detected

    def close(self):
        self.mem.close()


# --------------------------------------------------------------------- mem0
class Mem0System(System):
    """mem0 OSS (pip `mem0ai`), configured with the same local models as VeraMem."""

    name = "mem0"
    supports = {"forget"}

    def __init__(self, workdir: str | None = None):
        os.environ["MEM0_TELEMETRY"] = "False"
        from mem0 import Memory

        self.dir = Path(workdir or tempfile.mkdtemp(prefix="m0_bench_"))
        cfg = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": LLM_MODEL,
                    "temperature": 0,
                    "max_tokens": 4000,
                    "ollama_base_url": OLLAMA,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {"model": EMBED_MODEL, "ollama_base_url": OLLAMA},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": str(self.dir / "qdrant"),
                    "collection_name": "bench",
                    "embedding_model_dims": 768,
                    "on_disk": True,
                },
            },
            "history_db_path": str(self.dir / "history.db"),
        }
        self.mem = Memory.from_config(cfg)
        _disable_thinking(self.mem)

    def add(self, user_id, messages, *, session_id=None, agent_id=None, resources=None):
        msgs = []
        for m in messages:
            role = m.get("role", "user")
            content = m["content"]
            if m.get("name"):
                content = f"{m['name']}: {content}"
            if role not in ("user", "assistant", "system"):  # mem0 accepts only these roles
                content = f"[{role} output] {content}"
                role = "user"
            if m.get("timestamp"):  # OSS mem0 rejects `timestamp=`; give it the date inline
                content = f"[{m['timestamp']}] {content}"
            msgs.append({"role": role, "content": content})
        kw: dict[str, Any] = {"user_id": user_id}
        if agent_id:
            kw["agent_id"] = agent_id
        ts = next((m.get("timestamp") for m in messages if m.get("timestamp")), None)
        if ts:
            kw["metadata"] = {"timestamp": str(ts)}
        self.mem.add(msgs, **kw)

    def search(self, user_id, query, k=10, agent_id=None):
        filters: dict[str, Any] = {"user_id": user_id}
        if agent_id:
            filters["agent_id"] = agent_id
        res = self.mem.search(query=query, filters=filters, top_k=k)
        out = []
        for r in res.get("results", []):
            md = r.get("metadata") or {}
            out.append(
                {
                    "id": r["id"],
                    "text": r["memory"],
                    "date": str(md.get("timestamp") or r.get("created_at", ""))[:10],
                    "authority": None,
                    "attributed_to": r.get("attributed_to"),
                }
            )
        return out

    def memories(self, user_id):
        res = self.mem.get_all(filters={"user_id": user_id}, top_k=1000)
        items = res.get("results", res) if isinstance(res, dict) else res
        return [
            {
                "id": r["id"],
                "text": r["memory"],
                "authority": None,
                "attributed_to": r.get("attributed_to"),
                "evidence": 0,
            }
            for r in items
        ]

    def forget_user(self, user_id):
        self.mem.delete_all(user_id=user_id)

    def storage_files(self):
        return [p for p in self.dir.rglob("*") if p.is_file()]

    def authority_of(self, item):
        return None  # mem0 has no source-authority model

    def close(self):
        with contextlib.suppress(Exception):
            self.mem.vector_store.client.close()


def _disable_thinking(mem0_memory: Any) -> None:
    """Parity with VeraMem: run qwen3 without hidden reasoning tokens in both systems."""
    client = getattr(mem0_memory.llm, "client", None)
    if client is None or not hasattr(client, "chat"):
        return
    orig = client.chat

    def chat(**kw):
        kw.setdefault("think", False)
        return orig(**kw)

    client.chat = chat


def files_containing(paths: list[Path], needle: str) -> list[str]:
    """Raw byte scan of storage files, plus a decoded scan of SQLite databases."""
    hits = []
    nb = needle.encode()
    for p in paths:
        try:
            data = p.read_bytes()
        except OSError:
            continue
        if nb in data:
            hits.append(str(p))
            continue
        if data[:16] == b"SQLite format 3\x00":
            tmp = Path(tempfile.mkdtemp()) / "copy.db"
            shutil.copy(p, tmp)
            try:
                con = sqlite3.connect(tmp)
                if needle in "\n".join(con.iterdump()):
                    hits.append(str(p))
                con.close()
            except sqlite3.DatabaseError:
                pass
    return hits
