"""Append-only, hash-chained audit log of every memory write.

entry_hash = sha256(prev_hash || canonical_json(ts, actor, op, target, payload))

Rewriting, deleting or reordering any past row breaks every hash after it, which
`verify()` detects. Erasure (`forget`) writes a tombstone that holds only hashes,
so the chain survives GDPR-style deletion without retaining content.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .store.sqlite import SQLiteStore
from .types import now


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _digest(
    prev: str, ts: float, actor: str | None, op: str, target: str | None, payload: str
) -> str:
    body = _canon([ts, actor, op, target, payload])
    return hashlib.sha256((prev + body).encode()).hexdigest()


@dataclass
class VerifyReport:
    ok: bool
    entries: int
    head: str
    first_bad_seq: int | None = None
    reason: str = ""


class Ledger:
    def __init__(self, store: SQLiteStore):
        self.store = store

    def append(
        self, op: str, target: str | None, payload: dict[str, Any], actor: str | None = None
    ) -> str:
        ts = now()
        body = _canon(payload)
        with self.store._lock:
            _, prev = self.store.ledger_tail()
            h = _digest(prev, ts, actor, op, target, body)
            self.store.ledger_append(ts, actor, op, target, body, prev, h)
        return h

    def history(self, target: str) -> list[dict[str, Any]]:
        return [
            {
                "seq": r["seq"],
                "ts": r["ts"],
                "actor": r["actor"],
                "op": r["op"],
                "payload": json.loads(r["payload"]),
                "hash": r["hash"],
            }
            for r in self.store.ledger_rows(target)
        ]

    def verify(self) -> VerifyReport:
        prev = "0" * 64
        rows = self.store.ledger_rows()
        for r in rows:
            if r["prev_hash"] != prev:
                return VerifyReport(False, len(rows), prev, r["seq"], "broken link")
            h = _digest(prev, r["ts"], r["actor"], r["op"], r["target"], r["payload"])
            if h != r["hash"]:
                return VerifyReport(False, len(rows), prev, r["seq"], "content altered")
            prev = h
        return VerifyReport(True, len(rows), prev)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
