"""Optional approximate nearest-neighbour index (HNSW via `usearch`).

    pip install "veramem[ann]"
    Memory(vector_index="hnsw")      # or "auto": HNSW only for scopes above ANN_MIN rows

Security model: one index per *owner scope* (a user's private memory, a team, the public
pool), never one shared index. A search only probes the scopes the caller may read, and
every candidate still passes the SQL access predicate before it is scored, so the
guarantee "unreadable rows never reach ranking" is unchanged.

Freshness: indexes are keyed by SQLite rowid and catch up on rows with a larger rowid
before each search, so writes from this or another process become searchable without a
rebuild. Erased rows are simply never matched again (their rowid no longer resolves).
Only changes to a row's owner or visibility force a rebuild of the affected indexes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .store.sqlite import SQLiteStore

ANN_MIN = 50_000  # "auto" uses HNSW only for scopes at least this large


def available() -> bool:
    try:
        import usearch.index  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class _ScopeIndex:
    index: Any  # None until the scope has its first vector
    max_rowid: int = 0
    owner_generation: int = 0
    size: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class AnnManager:
    def __init__(self, store: SQLiteStore, connectivity: int = 16, expansion_search: int = 128):
        if not available():
            raise ImportError('approximate search needs: pip install "veramem[ann]"')
        self.store = store
        self.connectivity = connectivity
        self.expansion_search = expansion_search
        self.indexes: dict[tuple[str, str], _ScopeIndex] = {}
        self.lock = threading.Lock()

    # --------------------------------------------------------------- building
    def _source(self, kind: str) -> tuple[str, str]:
        if kind == "m":
            return "memories", "vm_owner(user_id, agent_id, team_id, visibility)"
        return "episodes", "vm_owner(user_id, agent_id, NULL, 'private')"

    def _new_index(self, dim: int) -> Any:
        from usearch.index import Index

        return Index(
            ndim=dim,
            metric="cos",
            dtype="f16",
            connectivity=self.connectivity,
            expansion_add=128,
            expansion_search=self.expansion_search,
        )

    def _catch_up(self, kind: str, token: str, si: _ScopeIndex) -> None:
        table, owner = self._source(kind)
        top = self.store.total_rows(table)
        if top <= si.max_rowid:
            return  # nothing written since the last look (the common case)
        rows = self.store.db.execute(
            f"SELECT rowid, embedding FROM {table} WHERE rowid > ? AND embedding IS NOT NULL"
            f" AND {owner} = ?",
            (si.max_rowid, token),
        ).fetchall()
        if rows:
            keys = np.array([r[0] for r in rows], dtype=np.uint64)
            vecs = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(
                len(rows), -1
            )
            if si.index is None:
                si.index = self._new_index(vecs.shape[1])
            if vecs.shape[1] == si.index.ndim:
                si.index.add(keys, vecs, threads=0)
                si.size += len(rows)
        si.max_rowid = top

    def _get(self, kind: str, token: str) -> _ScopeIndex:
        key = (kind, token)
        with self.lock:
            si = self.indexes.get(key)
            if si is None or si.owner_generation != self.store.owner_generation:
                # new scope, or rows moved between scopes: (re)build. Empty scopes are
                # cached too, so they are never rescanned.
                si = _ScopeIndex(index=None, owner_generation=self.store.owner_generation)
                self.indexes[key] = si
        with si.lock:
            self._catch_up(kind, token, si)
        return si

    def scope_size(self, kind: str, tokens: list[str]) -> int:
        return sum(self._get(kind, t).size for t in tokens)

    # ---------------------------------------------------------------- search
    def search(self, kind: str, tokens: list[str], qv: np.ndarray, k: int) -> list[int]:
        """Candidate rowids from the caller's scope indexes (unfiltered; the caller must
        still apply the SQL access predicate)."""
        out: list[tuple[float, int]] = []
        for t in tokens:
            si = self._get(kind, t)
            if si.index is None or si.size == 0:
                continue
            res = si.index.search(qv.astype(np.float32), min(k, si.size))
            out += [(float(d), int(key)) for key, d in zip(res.keys, res.distances)]
        out.sort()
        return [key for _, key in out[: k * max(1, len(tokens))]]
