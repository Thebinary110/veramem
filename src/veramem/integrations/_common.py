"""Shared helpers for the framework adapters (no framework imports here)."""

from __future__ import annotations

import time
from typing import Any

from ..memory import Memory
from ..types import Authority, Evidence, Hit, MemoryItem, SourceType

# Roles VeraMem understands, by the authority they confer.
WRITE_ROLES = ("user", "system", "assistant", "tool", "document")


def missing(framework: str, extra: str, err: ImportError) -> ImportError:
    """A clear ImportError telling the user which extra to install."""
    return ImportError(
        f"veramem.integrations.{extra} needs {framework}. "
        f'Install it with: pip install "veramem[{extra}]"  (import error: {err})'
    )


def require_user(user_id: str | None) -> str:
    if not user_id or not isinstance(user_id, str):
        raise ValueError("user_id is required and is bound when the adapter is constructed")
    return user_id


def iso_date(ts: float | None) -> str | None:
    return None if ts is None else time.strftime("%Y-%m-%d", time.gmtime(ts))


def hit_metadata(hit: Hit, trace_id: str) -> dict[str, Any]:
    """Flat, JSON-safe metadata describing one retrieval hit."""
    m = hit.memory
    return {
        "id": m.id,
        "authority": m.authority.value,
        "kind": m.kind.value,
        "date": iso_date(m.valid_from or m.created_at),
        "trace_id": trace_id,
        "granularity": hit.granularity,
        "score": hit.score,
    }


def put_memory(
    mem: Memory,
    text: str,
    *,
    role: str,
    user_id: str,
    agent_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    episode_metadata: dict[str, Any] | None = None,
) -> MemoryItem:
    """Store `text` verbatim as exactly one memory, with authority taken from `role`.

    The text is first written as a raw episode through ``Memory.add(infer=False)``, so
    it is hashed and ledgered like any other evidence. VeraMem's raw mode only turns
    user/system turns into memories; for assistant/tool/document content the memory
    is created explicitly here, pointing at that episode, with the lower authority its
    source implies (Attested or Unendorsed). Authority is never raised.

    `metadata` is attached to the memory; `episode_metadata` (default: the same) to
    the raw episode.
    """
    if role not in WRITE_ROLES:
        raise ValueError(f"role must be one of {WRITE_ROLES}, got {role!r}")
    text = str(text).strip()
    if not text:
        raise ValueError("cannot store empty content")
    meta = dict(metadata or {})
    r = mem.add(
        {"role": role, "content": text},
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        infer=False,
        metadata=meta if episode_metadata is None else episode_metadata,
    )
    created = [c.memory_id for c in r.changes if c.memory_id]
    if created:
        mid = created[0]
        if meta:
            mem.store.update_memory(mid, metadata=meta)
    else:
        source = SourceType(role)
        ep = mem.store.get_episode(r.episodes[0])
        item = MemoryItem(
            text=text,
            user_id=user_id,
            agent_id=agent_id,
            authority=Authority.for_source(source),
            evidence=[
                Evidence(
                    episode_id=r.episodes[0], quote=text, start=0, end=len(text), source=source
                )
            ],
            valid_from=ep.occurred_at if ep else None,
            metadata=meta,
        )
        mem.import_({"format": "veramem.export.v1", "memories": [item.model_dump(mode="json")]})
        mid = item.id
    out = mem.store.get_memory(mid)
    assert out is not None
    return out
