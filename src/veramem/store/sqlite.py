"""Single-file storage: SQLite (WAL) + FTS5 for lexical search + float32 vector blobs.

No server, no extension modules. Vector search is exact (brute-force cosine) over
the *already access-filtered* candidate set, which keeps it correct by construction:
a memory the caller may not read never reaches the scorer.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..types import (
    Authority,
    Episode,
    Evidence,
    Kind,
    MemoryItem,
    SourceType,
    Status,
    Visibility,
)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS episodes (
  id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  source TEXT NOT NULL,
  actor TEXT,
  user_id TEXT,
  agent_id TEXT,
  session_id TEXT,
  occurred_at REAL NOT NULL,
  ingested_at REAL NOT NULL,
  sha256 TEXT NOT NULL,
  metadata TEXT NOT NULL DEFAULT '{}',
  embedding BLOB
);
CREATE INDEX IF NOT EXISTS ep_user ON episodes(user_id, session_id, occurred_at);

CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  kind TEXT NOT NULL,
  user_id TEXT,
  agent_id TEXT,
  team_id TEXT,
  visibility TEXT NOT NULL,
  authority TEXT NOT NULL,
  status TEXT NOT NULL,
  supersedes TEXT,
  superseded_by TEXT,
  valid_from REAL,
  valid_to REAL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  confidence REAL NOT NULL DEFAULT 1.0,
  uses INTEGER NOT NULL DEFAULT 0,
  successes REAL NOT NULL DEFAULT 0,
  failures REAL NOT NULL DEFAULT 0,
  last_used_at REAL,
  shareable INTEGER NOT NULL DEFAULT 0,
  entities TEXT NOT NULL DEFAULT '[]',
  metadata TEXT NOT NULL DEFAULT '{}',
  embedding BLOB
);
CREATE INDEX IF NOT EXISTS mem_scope ON memories(status, user_id, visibility, team_id);

CREATE TABLE IF NOT EXISTS evidence (
  memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  episode_id TEXT NOT NULL REFERENCES episodes(id),
  quote TEXT NOT NULL,
  start INTEGER NOT NULL DEFAULT -1,
  end INTEGER NOT NULL DEFAULT -1,
  source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ev_mem ON evidence(memory_id);
CREATE INDEX IF NOT EXISTS ev_ep ON evidence(episode_id);

CREATE TABLE IF NOT EXISTS entity_links (
  entity TEXT NOT NULL,
  memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  PRIMARY KEY (entity, memory_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(id UNINDEXED, text, tokenize='porter');
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(id UNINDEXED, content, tokenize='porter');

CREATE TABLE IF NOT EXISTS traces (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  query TEXT NOT NULL,
  user_id TEXT,
  agent_id TEXT,
  created_at REAL NOT NULL,
  memory_ids TEXT NOT NULL,
  payload TEXT NOT NULL DEFAULT '{}',
  outcome REAL,
  outcome_at REAL
);

CREATE TABLE IF NOT EXISTS grants (
  kind TEXT NOT NULL,        -- user_agent | agent_resource | team_member
  subject TEXT NOT NULL,
  object TEXT NOT NULL,
  granted_at REAL NOT NULL,
  revoked_at REAL
);
CREATE INDEX IF NOT EXISTS grants_idx ON grants(kind, subject, object);

CREATE TABLE IF NOT EXISTS ledger (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  actor TEXT,
  op TEXT NOT NULL,
  target TEXT,
  payload TEXT NOT NULL,
  prev_hash TEXT NOT NULL,
  hash TEXT NOT NULL
);
"""

_FTS_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 OR-query (no operator injection)."""
    toks = [t.lower() for t in _FTS_TOKEN.findall(text) if len(t) > 1]
    toks = [t for t in toks if t not in _STOP][:32]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(toks))


_STOP = set(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "this",
        "that",
        "with",
        "as",
        "by",
        "from",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "do",
        "does",
        "did",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "my",
        "your",
        "me",
        "his",
        "her",
        "our",
        "their",
        "its",
        "not",
        "no",
        "yes",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "may",
        "might",
        "have",
        "has",
        "had",
        "about",
        "into",
        "than",
        "then",
        "there",
        "here",
        "also",
        "just",
        "so",
        "if",
        "but",
    ]
)


def _vec(blob: bytes | None) -> np.ndarray | None:
    return None if blob is None else np.frombuffer(blob, dtype=np.float32)


def _blob(v: Sequence[float] | np.ndarray | None) -> bytes | None:
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return (a / n if n > 0 else a).astype(np.float32).tobytes()


class SQLiteStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser())
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # ------------------------------------------------------------------ utils
    def tx(self):
        return _Tx(self)

    def close(self) -> None:
        self.db.close()

    def meta_get(self, key: str) -> str | None:
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def meta_set(self, key: str, value: str) -> None:
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

    # --------------------------------------------------------------- episodes
    def put_episode(self, ep: Episode, embedding: Sequence[float] | None = None) -> None:
        with self._lock:
            self.db.execute(
                "INSERT OR IGNORE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ep.id,
                    ep.content,
                    ep.source.value,
                    ep.actor,
                    ep.user_id,
                    ep.agent_id,
                    ep.session_id,
                    ep.occurred_at,
                    ep.ingested_at,
                    ep.sha256,
                    json.dumps(ep.metadata),
                    _blob(embedding),
                ),
            )
            self.db.execute("INSERT INTO episodes_fts VALUES (?,?)", (ep.id, ep.content))

    def get_episode(self, episode_id: str) -> Episode | None:
        r = self.db.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()
        return self._row_episode(r) if r else None

    def episodes(self, where: str = "1=1", params: Iterable[Any] = ()) -> list[Episode]:
        rows = self.db.execute(
            f"SELECT * FROM episodes e WHERE {where} ORDER BY occurred_at", tuple(params)
        ).fetchall()
        return [self._row_episode(r) for r in rows]

    @staticmethod
    def _row_episode(r: sqlite3.Row) -> Episode:
        return Episode(
            id=r["id"],
            content=r["content"],
            source=SourceType(r["source"]),
            actor=r["actor"],
            user_id=r["user_id"],
            agent_id=r["agent_id"],
            session_id=r["session_id"],
            occurred_at=r["occurred_at"],
            ingested_at=r["ingested_at"],
            sha256=r["sha256"],
            metadata=json.loads(r["metadata"]),
        )

    def episode_vectors(self, where: str, params: Iterable[Any]) -> tuple[list[str], np.ndarray]:
        rows = self.db.execute(
            f"SELECT e.id, e.embedding FROM episodes e WHERE e.embedding IS NOT NULL AND {where}",
            tuple(params),
        ).fetchall()
        return _stack(rows)

    def episode_fts(self, query: str, ids_where: str, params: Iterable[Any], limit: int):
        q = fts_query(query)
        if not q:
            return []
        return self.db.execute(
            f"SELECT f.id, bm25(episodes_fts) AS s FROM episodes_fts f JOIN episodes e "
            f"ON e.id=f.id WHERE episodes_fts MATCH ? AND {ids_where} ORDER BY s LIMIT ?",
            (q, *params, limit),
        ).fetchall()

    # --------------------------------------------------------------- memories
    def put_memory(self, m: MemoryItem, embedding: Sequence[float] | None = None) -> None:
        with self._lock:
            self.db.execute(
                "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    m.id,
                    m.text,
                    m.kind.value,
                    m.user_id,
                    m.agent_id,
                    m.team_id,
                    m.visibility.value,
                    m.authority.value,
                    m.status.value,
                    m.supersedes,
                    m.superseded_by,
                    m.valid_from,
                    m.valid_to,
                    m.created_at,
                    m.updated_at,
                    m.confidence,
                    m.uses,
                    m.successes,
                    m.failures,
                    m.last_used_at,
                    int(m.shareable),
                    json.dumps(m.entities),
                    json.dumps(m.metadata),
                    _blob(embedding),
                ),
            )
            self.db.execute("INSERT INTO memories_fts VALUES (?,?)", (m.id, m.text))
            for ev in m.evidence:
                self.db.execute(
                    "INSERT INTO evidence VALUES (?,?,?,?,?,?)",
                    (m.id, ev.episode_id, ev.quote, ev.start, ev.end, ev.source.value),
                )
            for ent in set(m.entities):
                self.db.execute(
                    "INSERT OR IGNORE INTO entity_links VALUES (?,?)", (ent.lower(), m.id)
                )

    def update_memory(self, memory_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols, vals = [], []
        for k, v in fields.items():
            if k == "embedding":
                v = _blob(v)
            elif k in ("entities", "metadata"):
                v = json.dumps(v)
            elif hasattr(v, "value"):
                v = v.value
            elif isinstance(v, bool):
                v = int(v)
            cols.append(f"{k}=?")
            vals.append(v)
        with self._lock:
            self.db.execute(f"UPDATE memories SET {', '.join(cols)} WHERE id=?", (*vals, memory_id))
            if "text" in fields:
                self.db.execute("DELETE FROM memories_fts WHERE id=?", (memory_id,))
                self.db.execute(
                    "INSERT INTO memories_fts VALUES (?,?)", (memory_id, fields["text"])
                )

    def add_evidence(self, memory_id: str, evs: list[Evidence]) -> None:
        with self._lock:
            for ev in evs:
                self.db.execute(
                    "INSERT INTO evidence VALUES (?,?,?,?,?,?)",
                    (memory_id, ev.episode_id, ev.quote, ev.start, ev.end, ev.source.value),
                )

    def get_memory(self, memory_id: str) -> MemoryItem | None:
        r = self.db.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return self._row_memory(r) if r else None

    def memories(
        self,
        where: str = "1=1",
        params: Iterable[Any] = (),
        order: str = "created_at",
        limit: int | None = None,
    ) -> list[MemoryItem]:
        sql = f"SELECT * FROM memories m WHERE {where} ORDER BY {order}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [self._row_memory(r) for r in self.db.execute(sql, tuple(params)).fetchall()]

    def memory_vector(self, memory_id: str) -> np.ndarray | None:
        r = self.db.execute("SELECT embedding FROM memories WHERE id=?", (memory_id,)).fetchone()
        return _vec(r[0]) if r else None

    def memory_vectors(self, where: str, params: Iterable[Any]) -> tuple[list[str], np.ndarray]:
        rows = self.db.execute(
            f"SELECT m.id, m.embedding FROM memories m WHERE m.embedding IS NOT NULL AND {where}",
            tuple(params),
        ).fetchall()
        return _stack(rows)

    def memory_fts(self, query: str, where: str, params: Iterable[Any], limit: int):
        q = fts_query(query)
        if not q:
            return []
        return self.db.execute(
            f"SELECT f.id, bm25(memories_fts) AS s FROM memories_fts f JOIN memories m "
            f"ON m.id=f.id WHERE memories_fts MATCH ? AND {where} ORDER BY s LIMIT ?",
            (q, *params, limit),
        ).fetchall()

    def memories_for_entities(self, entities: list[str], where: str, params: Iterable[Any]):
        if not entities:
            return []
        marks = ",".join("?" * len(entities))
        return self.db.execute(
            f"SELECT l.memory_id AS id, COUNT(*) AS n FROM entity_links l JOIN memories m "
            f"ON m.id=l.memory_id WHERE l.entity IN ({marks}) AND {where} "
            f"GROUP BY l.memory_id ORDER BY n DESC LIMIT 200",
            (*[e.lower() for e in entities], *params),
        ).fetchall()

    def evidence_for(self, memory_id: str) -> list[Evidence]:
        rows = self.db.execute("SELECT * FROM evidence WHERE memory_id=?", (memory_id,)).fetchall()
        return [
            Evidence(
                episode_id=r["episode_id"],
                quote=r["quote"],
                start=r["start"],
                end=r["end"],
                source=SourceType(r["source"]),
            )
            for r in rows
        ]

    def _row_memory(self, r: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=r["id"],
            text=r["text"],
            kind=Kind(r["kind"]),
            user_id=r["user_id"],
            agent_id=r["agent_id"],
            team_id=r["team_id"],
            visibility=Visibility(r["visibility"]),
            authority=Authority(r["authority"]),
            status=Status(r["status"]),
            supersedes=r["supersedes"],
            superseded_by=r["superseded_by"],
            valid_from=r["valid_from"],
            valid_to=r["valid_to"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            confidence=r["confidence"],
            uses=r["uses"],
            successes=r["successes"],
            failures=r["failures"],
            last_used_at=r["last_used_at"],
            shareable=bool(r["shareable"]),
            entities=json.loads(r["entities"]),
            metadata=json.loads(r["metadata"]),
            evidence=self.evidence_for(r["id"]),
        )

    # ----------------------------------------------------------------- traces
    def put_trace(
        self,
        trace_id: str,
        kind: str,
        query: str,
        user_id: str | None,
        agent_id: str | None,
        created_at: float,
        memory_ids: list[str],
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self.db.execute(
                "INSERT INTO traces VALUES (?,?,?,?,?,?,?,?,NULL,NULL)",
                (
                    trace_id,
                    kind,
                    query,
                    user_id,
                    agent_id,
                    created_at,
                    json.dumps(memory_ids),
                    json.dumps(payload or {}),
                ),
            )

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        r = self.db.execute("SELECT * FROM traces WHERE id=?", (trace_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["memory_ids"] = json.loads(d["memory_ids"])
        d["payload"] = json.loads(d["payload"])
        return d

    def set_trace_outcome(self, trace_id: str, outcome: float, at: float) -> None:
        with self._lock:
            self.db.execute(
                "UPDATE traces SET outcome=?, outcome_at=? WHERE id=?", (outcome, at, trace_id)
            )

    def traces(self, where: str = "1=1", params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        out = []
        for r in self.db.execute(
            f"SELECT * FROM traces WHERE {where} ORDER BY created_at", tuple(params)
        ):
            d = dict(r)
            d["memory_ids"] = json.loads(d["memory_ids"])
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out

    # ----------------------------------------------------------------- grants
    def grant(self, kind: str, subject: str, obj: str, at: float) -> None:
        with self._lock:
            if not self.active_grant(kind, subject, obj, at):
                self.db.execute(
                    "INSERT INTO grants VALUES (?,?,?,?,NULL)", (kind, subject, obj, at)
                )

    def revoke(self, kind: str, subject: str, obj: str, at: float) -> None:
        with self._lock:
            self.db.execute(
                "UPDATE grants SET revoked_at=? WHERE kind=? AND subject=? AND object=? "
                "AND revoked_at IS NULL",
                (at, kind, subject, obj),
            )

    def active_grant(self, kind: str, subject: str, obj: str, at: float) -> bool:
        r = self.db.execute(
            "SELECT 1 FROM grants WHERE kind=? AND subject=? AND object=? AND granted_at<=? "
            "AND (revoked_at IS NULL OR revoked_at>?) LIMIT 1",
            (kind, subject, obj, at, at),
        ).fetchone()
        return r is not None

    def grant_objects(self, kind: str, subject: str, at: float) -> set[str]:
        rows = self.db.execute(
            "SELECT object FROM grants WHERE kind=? AND subject=? AND granted_at<=? "
            "AND (revoked_at IS NULL OR revoked_at>?)",
            (kind, subject, at, at),
        ).fetchall()
        return {r[0] for r in rows}

    def has_any_grants(self, kind: str) -> bool:
        return (
            self.db.execute("SELECT 1 FROM grants WHERE kind=? LIMIT 1", (kind,)).fetchone()
            is not None
        )

    # ----------------------------------------------------------------- ledger
    def ledger_tail(self) -> tuple[int, str]:
        r = self.db.execute("SELECT seq, hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
        return (r[0], r[1]) if r else (0, "0" * 64)

    def ledger_append(
        self,
        ts: float,
        actor: str | None,
        op: str,
        target: str | None,
        payload: str,
        prev_hash: str,
        h: str,
    ) -> None:
        self.db.execute(
            "INSERT INTO ledger (ts, actor, op, target, payload, prev_hash, hash) "
            "VALUES (?,?,?,?,?,?,?)",
            (ts, actor, op, target, payload, prev_hash, h),
        )

    def ledger_rows(self, target: str | None = None) -> list[sqlite3.Row]:
        if target:
            return self.db.execute(
                "SELECT * FROM ledger WHERE target=? ORDER BY seq", (target,)
            ).fetchall()
        return self.db.execute("SELECT * FROM ledger ORDER BY seq").fetchall()

    def stats(self) -> dict[str, int]:
        q = self.db.execute
        return {
            "episodes": q("SELECT COUNT(*) FROM episodes").fetchone()[0],
            "memories_active": q("SELECT COUNT(*) FROM memories WHERE status='active'").fetchone()[
                0
            ],
            "memories_total": q("SELECT COUNT(*) FROM memories").fetchone()[0],
            "traces": q("SELECT COUNT(*) FROM traces").fetchone()[0],
            "ledger_entries": q("SELECT COUNT(*) FROM ledger").fetchone()[0],
        }


class _Tx:
    def __init__(self, store: SQLiteStore):
        self.s = store

    def __enter__(self):
        self.s._lock.acquire()
        self.s.db.execute("BEGIN")
        return self.s

    def __exit__(self, et, ev, tb):
        try:
            self.s.db.execute("ROLLBACK" if et else "COMMIT")
        finally:
            self.s._lock.release()


def _stack(rows: list[sqlite3.Row]) -> tuple[list[str], np.ndarray]:
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    ids = [r[0] for r in rows]
    mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    return ids, mat
