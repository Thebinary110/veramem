"""Single-file storage: SQLite (WAL) + FTS5 for lexical search + float32 vector blobs.

No server, no extension modules. Vector search is exact (brute-force cosine) over
the *already access-filtered* candidate set, which keeps it correct by construction:
a memory the caller may not read never reaches the scorer.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from collections import OrderedDict
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

# Bump SCHEMA_VERSION and add an entry to MIGRATIONS for every schema change. SCHEMA
# always describes the newest layout; MIGRATIONS[n] upgrades a v(n-1) database to vn.
SCHEMA_VERSION = 3
MIGRATIONS: dict[int, list[str]] = {
    # v2: scope tokens inside the full-text indexes, so lexical search only touches the
    # caller's rows instead of ranking every tenant's documents.
    2: [
        "DROP TABLE IF EXISTS memories_fts",
        "DROP TABLE IF EXISTS episodes_fts",
        "CREATE VIRTUAL TABLE memories_fts USING fts5(id UNINDEXED, owner, text,"
        " tokenize='porter')",
        "CREATE VIRTUAL TABLE episodes_fts USING fts5(id UNINDEXED, owner, content,"
        " tokenize='porter')",
        "INSERT INTO memories_fts SELECT id, vm_owner(user_id, agent_id, team_id, visibility),"
        " text FROM memories",
        "INSERT INTO episodes_fts SELECT id, vm_owner(user_id, agent_id, NULL, 'private'),"
        " content FROM episodes",
    ],
    # v3: query vectors on traces and per-memory outcome rows (contextual utility)
    3: [
        "ALTER TABLE traces ADD COLUMN qvec BLOB",
        "CREATE TABLE IF NOT EXISTS outcomes (memory_id TEXT NOT NULL, trace_id TEXT NOT NULL,"
        " qvec BLOB, pos REAL NOT NULL, neg REAL NOT NULL, ts REAL NOT NULL)",
        "CREATE INDEX IF NOT EXISTS outcomes_mem ON outcomes(memory_id)",
    ],
}


def owner_token(
    user_id: str | None, agent_id: str | None, team_id: str | None, visibility: str | None
) -> str:
    """Opaque FTS token naming the scope a row lives in (never a raw id)."""
    if visibility == "public":
        return "pub"
    if visibility == "team":
        return "t" + hashlib.sha256(f"team:{team_id}".encode()).hexdigest()[:20]
    if user_id is not None:
        return "o" + hashlib.sha256(f"user:{user_id}".encode()).hexdigest()[:20]
    return "a" + hashlib.sha256(f"agent:{agent_id}".encode()).hexdigest()[:20]


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA secure_delete=ON;
PRAGMA synchronous=NORMAL;

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

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
  id UNINDEXED, owner, text, tokenize='porter');
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
  id UNINDEXED, owner, content, tokenize='porter');

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
  outcome_at REAL,
  qvec BLOB
);

CREATE TABLE IF NOT EXISTS outcomes (
  memory_id TEXT NOT NULL,
  trace_id TEXT NOT NULL,
  qvec BLOB,
  pos REAL NOT NULL,
  neg REAL NOT NULL,
  ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS outcomes_mem ON outcomes(memory_id);

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

# Identifiers that may be interpolated into SQL. Values are always bound parameters;
# WHERE fragments come only from internal code (policy.read_filter etc.), never users.
_ORDERS = {"created_at", "created_at DESC", "updated_at DESC", "valid_from"}
_MEMORY_COLUMNS = {
    "text",
    "kind",
    "user_id",
    "agent_id",
    "team_id",
    "visibility",
    "authority",
    "status",
    "supersedes",
    "superseded_by",
    "valid_from",
    "valid_to",
    "created_at",
    "updated_at",
    "confidence",
    "uses",
    "successes",
    "failures",
    "last_used_at",
    "shareable",
    "entities",
    "metadata",
    "embedding",
}

Vector = Sequence[float] | np.ndarray
VECTOR_CACHE_ENTRIES = 32

_FTS_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def fts_query(text: str, scopes: list[str] | None = None, column: str = "text") -> str:
    """Turn free text into a safe FTS5 query (no operator injection).

    `scopes` restricts matches to rows whose owner token is one of the caller's scopes,
    so the index never ranks other tenants' rows. The SQL access predicate is still
    applied afterwards; this is an optimisation, not the security boundary.
    """
    toks = [t.lower() for t in _FTS_TOKEN.findall(text) if len(t) > 1]
    toks = [t for t in toks if t not in _STOP][:32]
    terms = " OR ".join(f'"{t}"' for t in dict.fromkeys(toks))
    if not terms:
        return ""
    if not scopes:
        return terms
    owners = " OR ".join(f'"{s}"' for s in scopes)
    return f"owner : ({owners}) AND {column} : ({terms})"


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


def _blob(v: Vector | None) -> bytes | None:
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
        self._tx_depth = 0
        # bumped on every write that can change which vectors a query sees
        self.generation = 0
        self._vec_cache: OrderedDict[tuple[Any, ...], tuple[int, list[str], np.ndarray]] = (
            OrderedDict()
        )
        self.raw = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.raw.row_factory = sqlite3.Row
        # every statement goes through the store lock: a transaction opened by one thread
        # can never absorb another thread's statements on this shared connection
        self.db = _LockedConnection(self.raw, self._lock)
        self.raw.create_function("vm_owner", 4, owner_token, deterministic=True)
        existing = self.db.execute("SELECT 1 FROM sqlite_master WHERE name='meta'").fetchone()
        if (
            existing
            and not self.db.execute("SELECT 1 FROM meta WHERE key='schema_version'").fetchone()
        ):
            # pre-versioning database (v1)
            self.db.execute("INSERT INTO meta VALUES ('schema_version', '1')")
        self.db.executescript(SCHEMA)
        self._migrate()

    # -------------------------------------------------------------- migrations
    def _migrate(self) -> None:
        """Bring an existing database up to SCHEMA_VERSION, one step at a time.

        Each step in MIGRATIONS runs in its own transaction. A database created by a
        newer VeraMem is refused rather than risk silently corrupting it.
        """
        row = self.db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        current = int(row[0]) if row else SCHEMA_VERSION
        if current > SCHEMA_VERSION:
            raise RuntimeError(
                f"database {self.path} uses schema v{current}, but this VeraMem supports up to "
                f"v{SCHEMA_VERSION}. Upgrade with: pip install -U veramem"
            )
        while current < SCHEMA_VERSION:
            with self.tx():
                for stmt in MIGRATIONS[current + 1]:
                    try:
                        self.db.execute(stmt)
                    except sqlite3.OperationalError as e:
                        # idempotent re-runs: a column added before an interrupted upgrade
                        if "duplicate column name" not in str(e):
                            raise
                current += 1
                self.db.execute(
                    "INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(current),)
                )
        if row is None:
            self.db.execute(
                "INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )

    # ------------------------------------------------------------------ utils
    def tx(self):
        return _Tx(self)

    def scrub(self) -> None:
        """Make erasure physical: rebuild FTS indexes, truncate the WAL, rewrite the file."""
        with self._lock:
            self.db.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            self.db.execute("INSERT INTO episodes_fts(episodes_fts) VALUES('rebuild')")
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            if self.path != ":memory:":
                self.db.execute("VACUUM")
                self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        self.raw.close()

    def meta_get(self, key: str) -> str | None:
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def meta_set(self, key: str, value: str) -> None:
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

    # --------------------------------------------------------------- episodes
    def put_episode(self, ep: Episode, embedding: Vector | None = None) -> None:
        self.generation += 1
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
            self.db.execute(
                "INSERT INTO episodes_fts VALUES (?,?,?)",
                (ep.id, owner_token(ep.user_id, ep.agent_id, None, "private"), ep.content),
            )

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
        return self._cached_vectors(
            "e",
            f"SELECT e.id, e.embedding FROM episodes e WHERE e.embedding IS NOT NULL AND {where}",
            tuple(params),
        )

    def episode_fts(
        self,
        query: str,
        ids_where: str,
        params: Iterable[Any],
        limit: int,
        scopes: list[str] | None = None,
    ):
        q = fts_query(query, scopes, "content")
        if not q:
            return []
        return self.db.execute(
            f"SELECT f.id, bm25(episodes_fts, 0, 0, 1) AS s FROM episodes_fts f JOIN episodes e "
            f"ON e.id=f.id WHERE episodes_fts MATCH ? AND {ids_where} ORDER BY s LIMIT ?",
            (q, *params, limit),
        ).fetchall()

    # --------------------------------------------------------------- memories
    def put_memory(self, m: MemoryItem, embedding: Vector | None = None) -> None:
        self.generation += 1
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
            self.db.execute(
                "INSERT INTO memories_fts VALUES (?,?,?)",
                (m.id, owner_token(m.user_id, m.agent_id, m.team_id, m.visibility.value), m.text),
            )
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
        bad = set(fields) - _MEMORY_COLUMNS
        if bad:
            raise ValueError(f"unknown memory columns: {sorted(bad)}")
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
        self.generation += 1
        with self._lock:
            self.db.execute(f"UPDATE memories SET {', '.join(cols)} WHERE id=?", (*vals, memory_id))
            if fields.keys() & {"text", "user_id", "agent_id", "team_id", "visibility"}:
                r = self.db.execute(
                    "SELECT text, user_id, agent_id, team_id, visibility FROM memories WHERE id=?",
                    (memory_id,),
                ).fetchone()
                self.db.execute("DELETE FROM memories_fts WHERE id=?", (memory_id,))
                if r is not None:
                    self.db.execute(
                        "INSERT INTO memories_fts VALUES (?,?,?)",
                        (memory_id, owner_token(r[1], r[2], r[3], r[4]), r[0]),
                    )

    def add_evidence(self, memory_id: str, evs: list[Evidence]) -> None:
        self.generation += 1
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
        if order not in _ORDERS:
            raise ValueError(f"unsupported order {order!r}")
        sql = f"SELECT * FROM memories m WHERE {where} ORDER BY {order}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [self._row_memory(r) for r in self.db.execute(sql, tuple(params)).fetchall()]

    def memory_vector(self, memory_id: str) -> np.ndarray | None:
        r = self.db.execute("SELECT embedding FROM memories WHERE id=?", (memory_id,)).fetchone()
        return _vec(r[0]) if r else None

    def bump(self) -> None:
        self.generation += 1

    def _cached_vectors(self, kind: str, sql: str, params: tuple[Any, ...]):
        key = (kind, sql, params)
        hit = self._vec_cache.get(key)
        if hit is not None and hit[0] == self.generation:
            self._vec_cache.move_to_end(key)
            return hit[1], hit[2]
        ids, mat = _stack(self.db.execute(sql, params).fetchall())
        self._vec_cache[key] = (self.generation, ids, mat)
        while len(self._vec_cache) > VECTOR_CACHE_ENTRIES:
            self._vec_cache.popitem(last=False)
        return ids, mat

    def memory_vectors(self, where: str, params: Iterable[Any]) -> tuple[list[str], np.ndarray]:
        return self._cached_vectors(
            "m",
            f"SELECT m.id, m.embedding FROM memories m WHERE m.embedding IS NOT NULL AND {where}",
            tuple(params),
        )

    def get_memories(self, ids: list[str]) -> dict[str, MemoryItem]:
        """Batch fetch (2 queries instead of 2 per id)."""
        if not ids:
            return {}
        out: dict[str, MemoryItem] = {}
        evs: dict[str, list[Evidence]] = {}
        for chunk in _chunks(ids, 500):
            marks = ",".join("?" * len(chunk))
            for r in self.db.execute(
                f"SELECT * FROM evidence WHERE memory_id IN ({marks})", chunk
            ).fetchall():
                evs.setdefault(r["memory_id"], []).append(_row_evidence(r))
            for r in self.db.execute(
                f"SELECT * FROM memories WHERE id IN ({marks})", chunk
            ).fetchall():
                out[r["id"]] = self._row_memory(r, evs.get(r["id"], []))
        return out

    def get_episodes(self, ids: list[str]) -> dict[str, Episode]:
        out: dict[str, Episode] = {}
        for chunk in _chunks(ids, 500):
            marks = ",".join("?" * len(chunk))
            for r in self.db.execute(
                f"SELECT * FROM episodes WHERE id IN ({marks})", chunk
            ).fetchall():
                out[r["id"]] = self._row_episode(r)
        return out

    def memory_fts(
        self,
        query: str,
        where: str,
        params: Iterable[Any],
        limit: int,
        scopes: list[str] | None = None,
    ):
        q = fts_query(query, scopes, "text")
        if not q:
            return []
        return self.db.execute(
            f"SELECT f.id, bm25(memories_fts, 0, 0, 1) AS s FROM memories_fts f JOIN memories m "
            f"ON m.id=f.id WHERE memories_fts MATCH ? AND {where} ORDER BY s LIMIT ?",
            (q, *params, limit),
        ).fetchall()

    def memories_for_entities(self, entities: list[str], where: str, params: Iterable[Any]):
        if not entities:
            return []
        marks = ",".join("?" * len(entities))
        return self.db.execute(
            f"SELECT l.memory_id AS id, COUNT(*) AS n FROM entity_links l CROSS JOIN memories m "
            f"ON m.id=l.memory_id WHERE l.entity IN ({marks}) AND {where} "
            f"GROUP BY l.memory_id ORDER BY n DESC LIMIT 200",
            (*[e.lower() for e in entities], *params),
        ).fetchall()

    def evidence_for(self, memory_id: str) -> list[Evidence]:
        rows = self.db.execute("SELECT * FROM evidence WHERE memory_id=?", (memory_id,)).fetchall()
        return [_row_evidence(r) for r in rows]

    def _row_memory(self, r: sqlite3.Row, evidence: list[Evidence] | None = None) -> MemoryItem:
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
            evidence=evidence if evidence is not None else self.evidence_for(r["id"]),
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
        qvec: Vector | None = None,
    ) -> None:
        with self._lock:
            self.db.execute(
                "INSERT INTO traces (id, kind, query, user_id, agent_id, created_at, memory_ids,"
                " payload, qvec) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    trace_id,
                    kind,
                    query,
                    user_id,
                    agent_id,
                    created_at,
                    json.dumps(memory_ids),
                    json.dumps(payload or {}),
                    _blob(qvec),
                ),
            )

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        r = self.db.execute("SELECT * FROM traces WHERE id=?", (trace_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["memory_ids"] = json.loads(d["memory_ids"])
        d["payload"] = json.loads(d["payload"])
        d["qvec"] = _vec(d.get("qvec"))
        return d

    def put_outcomes(self, rows: list[tuple[str, str, Vector | None, float, float, float]]) -> None:
        """rows: (memory_id, trace_id, query_vector, positive_mass, negative_mass, ts)"""
        with self._lock:
            for mid, tid, qv, pos, neg, ts in rows:
                self.db.execute(
                    "INSERT INTO outcomes VALUES (?,?,?,?,?,?)", (mid, tid, _blob(qv), pos, neg, ts)
                )

    def outcomes_for(
        self, memory_ids: list[str]
    ) -> dict[str, list[tuple[np.ndarray, float, float]]]:
        out: dict[str, list[tuple[np.ndarray, float, float]]] = {}
        for chunk in _chunks(memory_ids, 500):
            marks = ",".join("?" * len(chunk))
            for r in self.db.execute(
                f"SELECT memory_id, qvec, pos, neg FROM outcomes WHERE qvec IS NOT NULL"
                f" AND memory_id IN ({marks})",
                chunk,
            ).fetchall():
                v = _vec(r[1])
                if v is not None:
                    out.setdefault(r[0], []).append((v, float(r[2]), float(r[3])))
        return out

    def update_trace_payload(self, trace_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self.db.execute(
                "UPDATE traces SET payload=? WHERE id=?", (json.dumps(payload), trace_id)
            )

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
            d["qvec"] = _vec(d.get("qvec"))
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

    def count(self, sql: str) -> int:
        row = self.db.execute(sql).fetchone()
        return int(row[0]) if row else 0

    def stats(self) -> dict[str, int]:
        return {
            "episodes": self.count("SELECT COUNT(*) FROM episodes"),
            "memories_active": self.count("SELECT COUNT(*) FROM memories WHERE status='active'"),
            "memories_total": self.count("SELECT COUNT(*) FROM memories"),
            "traces": self.count("SELECT COUNT(*) FROM traces"),
            "ledger_entries": self.count("SELECT COUNT(*) FROM ledger"),
        }


class _Result:
    """Fully materialised cursor result, safe to read after the lock is released."""

    __slots__ = ("_rows", "lastrowid", "rowcount")

    def __init__(self, rows: list[sqlite3.Row], lastrowid: int | None, rowcount: int):
        self._rows, self.lastrowid, self.rowcount = rows, lastrowid, rowcount

    def fetchone(self) -> sqlite3.Row | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[sqlite3.Row]:
        return self._rows

    def __iter__(self):
        return iter(self._rows)


class _LockedConnection:
    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock):
        self._conn, self._lock = conn, lock

    def execute(self, sql: str, params: Iterable[Any] = ()) -> _Result:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            return _Result(cur.fetchall(), cur.lastrowid, cur.rowcount)

    def executescript(self, script: str) -> None:
        with self._lock:
            self._conn.executescript(script)

    def iterdump(self):
        with self._lock:
            return list(self._conn.iterdump())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class _Tx:
    """Re-entrant transaction: only the outermost block issues BEGIN / COMMIT."""

    def __init__(self, store: SQLiteStore):
        self.s = store

    def __enter__(self):
        self.s._lock.acquire()
        if self.s._tx_depth == 0:
            self.s.db.execute("BEGIN")
        self.s._tx_depth += 1
        return self.s

    def __exit__(self, et, ev, tb):
        try:
            self.s._tx_depth -= 1
            if self.s._tx_depth == 0:
                self.s.db.execute("ROLLBACK" if et else "COMMIT")
        finally:
            self.s._lock.release()


def _chunks(items: list[str], n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _row_evidence(r: sqlite3.Row) -> Evidence:
    return Evidence(
        episode_id=r["episode_id"],
        quote=r["quote"],
        start=r["start"],
        end=r["end"],
        source=SourceType(r["source"]),
    )


def _stack(rows: list[sqlite3.Row]) -> tuple[list[str], np.ndarray]:
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    ids = [r[0] for r in rows]
    mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    return ids, mat
