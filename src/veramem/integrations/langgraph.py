"""LangGraph long-term memory store (``BaseStore``) backed by VeraMem.

    pip install "veramem[langgraph]"

    store = VeraMemStore(mem, user_namespace_index=1)   # namespace ("memories", user_id)
    graph = builder.compile(store=store)
    # inside a node: store.put(("memories", user_id), "food", {"content": "likes ramen"})
    #                store.search(("memories", user_id), query="what food?")

Namespace -> user mapping (pick exactly one):

* ``user_id="priya"``: the store is bound to one VeraMem user; every namespace lives
  inside that user's scope. Use this when the store is built per request/user.
* ``user_namespace_index=i``: the VeraMem user is ``namespace[i]``. Use this for one
  shared store compiled into a multi-user graph, where your code builds namespaces
  from trusted config (e.g. ``config["configurable"]["user_id"]``). Never build a
  namespace from model output. Searches must name the user (prefix length > i):
  VeraMem never searches across users.

Each item is one VeraMem memory. Its text (what gets embedded and searched) is
``value["content"]`` / ``value["text"]`` / ``value["memory"]`` when that is a string,
the configured ``index_fields`` / ``PutOp.index`` fields when given, and otherwise the
JSON of the value. The full namespace, key and value are kept in the memory's
metadata, so ``get`` returns exactly what was put.

Semantics that differ from LangGraph's in-memory/Postgres stores:

* Overwriting a key retracts the old memory (kept for audit, hidden from retrieval)
  and writes a new one; ``created_at`` is carried over.
* Deleting retracts by default; pass ``hard_delete=True`` to erase with
  ``Memory.forget``.
* ``index=False`` is accepted but the item is still embedded: VeraMem indexes every
  memory.
* Search with a ``query`` ranks a bounded candidate pool (VeraMem hybrid search,
  over-fetched), then applies namespace/filter/offset; very selective filters may
  return fewer than ``limit`` items. Search scores are VeraMem's, not cosine.
* TTL is not supported (``supports_ttl = False``).
* ``list_namespaces`` covers only the bound user with ``user_id=...``; with
  ``user_namespace_index`` it lists every user's namespaces (an operator-side call,
  like LangGraph's own stores), so don't expose it to a model.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

try:
    from langgraph.store.base import (
        BaseStore,
        GetOp,
        Item,
        ListNamespacesOp,
        Op,
        PutOp,
        Result,
        SearchItem,
        SearchOp,
    )
    from langgraph.store.base.embed import get_text_at_path
except ImportError as _e:  # pragma: no cover - exercised only without the extra
    from ._common import missing

    raise missing("langgraph", "langgraph", _e) from _e

from ..memory import Memory
from ..types import MemoryItem
from ._common import WRITE_ROLES, put_memory

__all__ = ["VeraMemStore"]

_TEXT_KEYS = ("content", "text", "memory")


class VeraMemStore(BaseStore):
    """LangGraph ``BaseStore`` over a VeraMem ``Memory`` (see module docs)."""

    supports_ttl = False

    def __init__(
        self,
        memory: Memory,
        *,
        user_id: str | None = None,
        user_namespace_index: int | None = None,
        role: str = "assistant",
        index_fields: list[str] | None = None,
        hard_delete: bool = False,
        search_overfetch: int = 4,
    ) -> None:
        """
        role: VeraMem role of stored values, which sets their authority. Defaults to
            ``assistant`` (Attested) because graph code usually writes model output;
            use ``user`` only for values that are the user's own words.
        """
        if (user_id is None) == (user_namespace_index is None):
            raise ValueError("pass exactly one of user_id or user_namespace_index")
        if role not in WRITE_ROLES:
            raise ValueError(f"role must be one of {WRITE_ROLES}")
        self.memory = memory
        self.user_id = user_id
        self.user_namespace_index = user_namespace_index
        self.role = role
        self.index_fields = index_fields
        self.hard_delete = hard_delete
        self.search_overfetch = max(1, search_overfetch)
        self.last_trace_id: str | None = None

    # ------------------------------------------------------------- mapping
    def _user(self, namespace: tuple[str, ...]) -> str:
        if self.user_id is not None:
            return self.user_id
        i = self.user_namespace_index
        assert i is not None
        if len(namespace) <= i:
            raise ValueError(
                f"namespace {namespace!r} has no user component at index {i}; "
                "VeraMem does not read or search across users"
            )
        return namespace[i]

    @staticmethod
    def _ns_key(namespace: tuple[str, ...]) -> str:
        return json.dumps(list(namespace))

    def _text(self, value: dict[str, Any], index: list[str] | None | bool) -> str:
        fields = index if isinstance(index, list) else self.index_fields
        if fields:
            parts = [t for f in fields for t in get_text_at_path(value, f)]
            if parts:
                return "\n".join(parts)
        for k in _TEXT_KEYS:
            if isinstance(value.get(k), str) and value[k].strip():
                return str(value[k])
        return json.dumps(value, sort_keys=True, default=str)

    def _find(self, namespace: tuple[str, ...], key: str) -> MemoryItem | None:
        found = self.memory.store.memories(
            "m.user_id IS ? AND m.status='active' "
            "AND json_extract(m.metadata,'$.lg_ns')=? AND json_extract(m.metadata,'$.lg_key')=?",
            (self._user(namespace), self._ns_key(namespace), key),
            order="created_at DESC",
            limit=1,
        )
        return found[0] if found else None

    def _scan(self, user_id: str | None) -> list[MemoryItem]:
        where = "m.status='active' AND json_extract(m.metadata,'$.lg_ns') IS NOT NULL"
        params: tuple[Any, ...] = ()
        if user_id is not None:
            where = "m.user_id IS ? AND " + where
            params = (user_id,)
        return self.memory.store.memories(where, params, order="updated_at DESC")

    @staticmethod
    def _item_parts(m: MemoryItem) -> dict[str, Any]:
        md = m.metadata
        created = md.get("lg_created_at", m.created_at)
        return {
            "namespace": tuple(json.loads(md["lg_ns"])),
            "key": md["lg_key"],
            "value": md.get("lg_value", {}),
            "created_at": datetime.fromtimestamp(created, tz=timezone.utc),
            "updated_at": datetime.fromtimestamp(m.created_at, tz=timezone.utc),
        }

    def _remove(self, m: MemoryItem, user_id: str) -> None:
        if self.hard_delete:
            self.memory.forget(m.id, user_id=user_id)
        else:
            self.memory.delete(m.id, user_id=user_id)

    # ----------------------------------------------------------------- ops
    def _get(self, op: GetOp) -> Item | None:
        m = self._find(op.namespace, op.key)
        return Item(**self._item_parts(m)) if m else None

    def _put(self, op: PutOp) -> None:
        user = self._user(op.namespace)
        old = self._find(op.namespace, op.key)
        if op.value is None:
            if old is not None:
                self._remove(old, user)
            return
        value = dict(op.value)
        meta = {
            "lg_ns": self._ns_key(op.namespace),
            "lg_key": op.key,
            "lg_value": value,
            "lg_created_at": old.metadata.get("lg_created_at", old.created_at) if old else None,
            "lg_replaces": old.id if old else None,
        }
        meta = {k: v for k, v in meta.items() if v is not None}
        put_memory(
            self.memory,
            self._text(value, op.index),
            role=self.role,
            user_id=user,
            metadata=meta,
        )
        if old is not None:
            self._remove(old, user)

    @staticmethod
    def _in_prefix(ns: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
        return ns[: len(prefix)] == prefix

    def _search(self, op: SearchOp) -> list[SearchItem]:
        prefix = tuple(op.namespace_prefix)
        user = self._user(prefix)
        wanted = op.offset + op.limit
        rows: list[tuple[MemoryItem, float | None]] = []
        if op.query:
            r = self.memory.search(
                op.query,
                user_id=user,
                limit=max(wanted * self.search_overfetch, 20),
                granularity="fact",
                include_shared=False,
                include_raw=False,
            )
            self.last_trace_id = r.trace_id
            rows = [(h.memory, h.score) for h in r.hits if "lg_ns" in h.memory.metadata]
        else:
            rows = [(m, None) for m in self._scan(user)]
        out: list[SearchItem] = []
        for m, score in rows:
            parts = self._item_parts(m)
            if not self._in_prefix(parts["namespace"], prefix):
                continue
            if op.filter and not all(
                _compare(parts["value"].get(k), v) for k, v in op.filter.items()
            ):
                continue
            out.append(SearchItem(score=score, **parts))
        return out[op.offset : wanted]

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        seen: set[tuple[str, ...]] = set()
        for m in self._scan(self.user_id):
            ns = tuple(json.loads(m.metadata["lg_ns"]))
            if op.match_conditions and not all(
                _matches(c.match_type, tuple(c.path), ns) for c in op.match_conditions
            ):
                continue
            seen.add(ns[: op.max_depth] if op.max_depth is not None else ns)
        return sorted(seen)[op.offset : op.offset + op.limit]

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get(op))
            elif isinstance(op, PutOp):
                self._put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            else:
                raise ValueError(f"unknown store operation: {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, list(ops))


def _matches(match_type: str, path: tuple[str, ...], ns: tuple[str, ...]) -> bool:
    if len(ns) < len(path):
        return False
    pairs = zip(ns, path) if match_type == "prefix" else zip(reversed(ns), reversed(path))
    return all(p == "*" or k == p for k, p in pairs)


def _compare(value: Any, cond: Any) -> bool:
    """LangGraph filter semantics: equality, nested dicts, and $eq/$ne/$gt/$gte/$lt/$lte."""
    if isinstance(cond, dict):
        if any(k.startswith("$") for k in cond):
            return all(_op(value, k, v) for k, v in cond.items())
        return isinstance(value, dict) and all(_compare(value.get(k), v) for k, v in cond.items())
    return bool(value == cond)


def _op(value: Any, op: str, arg: Any) -> bool:
    if op == "$eq":
        return bool(value == arg)
    if op == "$ne":
        return bool(value != arg)
    if value is None:
        return False
    if op == "$gt":
        return float(value) > float(arg)
    if op == "$gte":
        return float(value) >= float(arg)
    if op == "$lt":
        return float(value) < float(arg)
    if op == "$lte":
        return float(value) <= float(arg)
    raise ValueError(f"unsupported filter operator {op!r}")
