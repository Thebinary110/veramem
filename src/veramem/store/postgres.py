"""PostgreSQL backend for multi-process / multi-replica deployments.

    pip install "veramem[postgres]"
    Memory("postgresql://veramem@db.example.com:5432/veramem")   # or VERAMEM_DB=postgresql://...

Keep the password out of the URL: use libpq's PGPASSWORD environment variable or a
~/.pgpass file, so it never appears in logs, process lists or `Memory.info()`.

It reuses the SQLite store's logic and SQL; a thin dialect layer translates the few
SQLite-specific constructs. Differences that matter:

* full-text search uses generated `tsvector` columns with GIN indexes (`ts_rank_cd`),
  scoped by a generated `owner` column, instead of FTS5;
* `rowid` is a BIGSERIAL column, so vector caches catch up exactly as on SQLite;
* ledger appends take a transaction-scoped advisory lock, so the hash chain stays
  linear across any number of server replicas;
* cache generation counters live in the `meta` table, shared by all replicas;
* vectors are stored as BYTEA and scored with numpy (no pgvector extension needed).
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from .sqlite import (
    SCHEMA_VERSION,
    SQLiteStore,
    _Result,
    fts_query,
)

PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE OR REPLACE FUNCTION vm_owner(u TEXT, a TEXT, t TEXT, v TEXT) RETURNS TEXT
LANGUAGE SQL IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE
    WHEN v = 'public' THEN 'pub'
    WHEN v = 'team' THEN 't' || left(encode(sha256(convert_to('team:' || coalesce(t, 'None'), 'UTF8')), 'hex'), 20)
    WHEN u IS NOT NULL THEN 'o' || left(encode(sha256(convert_to('user:' || u, 'UTF8')), 'hex'), 20)
    ELSE 'a' || left(encode(sha256(convert_to('agent:' || coalesce(a, 'None'), 'UTF8')), 'hex'), 20)
  END $$;

CREATE TABLE IF NOT EXISTS episodes (
  id TEXT PRIMARY KEY, content TEXT NOT NULL, source TEXT NOT NULL, actor TEXT, user_id TEXT,
  agent_id TEXT, session_id TEXT, occurred_at DOUBLE PRECISION NOT NULL,
  ingested_at DOUBLE PRECISION NOT NULL, sha256 TEXT NOT NULL,
  metadata TEXT NOT NULL DEFAULT '{}', embedding BYTEA,
  rowid BIGSERIAL UNIQUE,
  owner TEXT GENERATED ALWAYS AS (vm_owner(user_id, agent_id, NULL, 'private')) STORED,
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, content)) STORED
);
CREATE INDEX IF NOT EXISTS ep_user ON episodes(user_id, session_id, occurred_at);
CREATE INDEX IF NOT EXISTS ep_tsv ON episodes USING GIN (tsv);
CREATE INDEX IF NOT EXISTS ep_owner ON episodes(owner);

CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY, text TEXT NOT NULL, kind TEXT NOT NULL, user_id TEXT, agent_id TEXT,
  team_id TEXT, visibility TEXT NOT NULL, authority TEXT NOT NULL, status TEXT NOT NULL,
  supersedes TEXT, superseded_by TEXT, valid_from DOUBLE PRECISION, valid_to DOUBLE PRECISION,
  created_at DOUBLE PRECISION NOT NULL, updated_at DOUBLE PRECISION NOT NULL,
  confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0, uses INTEGER NOT NULL DEFAULT 0,
  successes DOUBLE PRECISION NOT NULL DEFAULT 0, failures DOUBLE PRECISION NOT NULL DEFAULT 0,
  last_used_at DOUBLE PRECISION, shareable INTEGER NOT NULL DEFAULT 0,
  entities TEXT NOT NULL DEFAULT '[]', metadata TEXT NOT NULL DEFAULT '{}', embedding BYTEA,
  rowid BIGSERIAL UNIQUE,
  owner TEXT GENERATED ALWAYS AS (vm_owner(user_id, agent_id, team_id, visibility)) STORED,
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, text)) STORED
);
CREATE INDEX IF NOT EXISTS mem_scope ON memories(status, user_id, visibility, team_id);
CREATE INDEX IF NOT EXISTS mem_tsv ON memories USING GIN (tsv);
CREATE INDEX IF NOT EXISTS mem_owner ON memories(owner);

CREATE TABLE IF NOT EXISTS evidence (
  memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  episode_id TEXT NOT NULL REFERENCES episodes(id),
  quote TEXT NOT NULL, start INTEGER NOT NULL DEFAULT -1, "end" INTEGER NOT NULL DEFAULT -1,
  source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ev_mem ON evidence(memory_id);
CREATE INDEX IF NOT EXISTS ev_ep ON evidence(episode_id);

CREATE TABLE IF NOT EXISTS entity_links (
  entity TEXT NOT NULL, memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  PRIMARY KEY (entity, memory_id)
);

CREATE TABLE IF NOT EXISTS traces (
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, query TEXT NOT NULL, user_id TEXT, agent_id TEXT,
  created_at DOUBLE PRECISION NOT NULL, memory_ids TEXT NOT NULL,
  payload TEXT NOT NULL DEFAULT '{}', outcome DOUBLE PRECISION, outcome_at DOUBLE PRECISION,
  qvec BYTEA
);

CREATE TABLE IF NOT EXISTS outcomes (
  memory_id TEXT NOT NULL, trace_id TEXT NOT NULL, qvec BYTEA, pos DOUBLE PRECISION NOT NULL,
  neg DOUBLE PRECISION NOT NULL, ts DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS outcomes_mem ON outcomes(memory_id);

CREATE TABLE IF NOT EXISTS grants (
  kind TEXT NOT NULL, subject TEXT NOT NULL, object TEXT NOT NULL,
  granted_at DOUBLE PRECISION NOT NULL, revoked_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS grants_idx ON grants(kind, subject, object);

CREATE TABLE IF NOT EXISTS ledger (
  seq BIGSERIAL PRIMARY KEY, ts DOUBLE PRECISION NOT NULL, actor TEXT, op TEXT NOT NULL,
  target TEXT, payload TEXT NOT NULL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
  key_hash TEXT PRIMARY KEY, prefix TEXT NOT NULL, role TEXT NOT NULL, user_id TEXT,
  agent_id TEXT, label TEXT, created_at DOUBLE PRECISION NOT NULL,
  revoked_at DOUBLE PRECISION, last_used_at DOUBLE PRECISION
)
"""

LEDGER_LOCK = 7_316_225  # arbitrary advisory-lock key for ledger appends

_JSON_EXTRACT = re.compile(r"json_extract\(\s*([\w.]+)\s*,\s*'\$\.([\w.]+)'\s*\)")


@lru_cache(maxsize=1024)
def translate(sql: str) -> str | None:
    """SQLite dialect -> PostgreSQL. Returns None for statements that are no-ops on PG."""
    s = sql.strip()
    head = s[:40].upper()
    if head.startswith("PRAGMA") or head.startswith("VACUUM"):
        return None
    if re.search(r"\b(memories_fts|episodes_fts)\b", s) and not s.upper().startswith("SELECT"):
        return None  # FTS5 shadow tables: PG keeps generated tsvector columns instead
    if head.startswith("BEGIN IMMEDIATE"):
        s = "BEGIN"
    s = _JSON_EXTRACT.sub(
        lambda m: f"(({m.group(1)})::jsonb #>> '{{{m.group(2).replace('.', ',')}}}')", s
    )
    s = s.replace("CROSS JOIN", "JOIN")
    if s.upper().startswith("INSERT OR REPLACE INTO META"):
        s = re.sub(r"(?i)^INSERT OR REPLACE INTO", "INSERT INTO", s)
        s += " ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
    elif s.upper().startswith("INSERT OR IGNORE INTO"):
        s = re.sub(r"(?i)^INSERT OR IGNORE INTO", "INSERT INTO", s) + " ON CONFLICT DO NOTHING"
    s = s.replace("%", "%%")  # literal percent signs (LIKE patterns) under psycopg
    s = re.sub(r"\bIS \?", "IS NOT DISTINCT FROM ?", s)
    return s.replace("?", "%s")


class PGRow(tuple):
    """Row supporting r[0], r["col"] and dict(r), like sqlite3.Row."""

    _index: dict[str, int]

    def __new__(cls, values: Iterable[Any], index: dict[str, int]) -> PGRow:
        row = super().__new__(cls, values)
        row._index = index
        return row

    def __getitem__(self, key: Any) -> Any:  # type: ignore[override]
        if isinstance(key, str):
            return tuple.__getitem__(self, self._index[key])
        return tuple.__getitem__(self, key)

    def keys(self) -> list[str]:
        return list(self._index)


def _row_factory(cursor: Any) -> Any:
    names = [c.name for c in cursor.description] if cursor.description else []
    index = {n: i for i, n in enumerate(names)}
    return lambda values: PGRow(values, index)


class _PGConnection:
    def __init__(self, conn: Any, lock: threading.RLock):
        self._conn, self._lock = conn, lock

    def execute(self, sql: str, params: Iterable[Any] = ()) -> _Result:
        tsql = translate(sql)
        if tsql is None:
            return _Result([], None, 0)
        with self._lock:
            cur = self._conn.execute(tsql, tuple(params))
            rows = cur.fetchall() if cur.description else []
            return _Result(rows, None, cur.rowcount)

    def executescript(self, script: str) -> None:
        with self._lock:
            self._conn.execute(script)

    def iterdump(self) -> list[str]:
        raise NotImplementedError("iterdump is SQLite-only")

    def close(self) -> None:
        self._conn.close()


class PostgresStore(SQLiteStore):
    def __init__(self, dsn: str):  # noqa: D107 - deliberately not calling SQLiteStore.__init__
        try:
            import psycopg
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError('PostgreSQL support needs: pip install "veramem[postgres]"') from e
        self.path = dsn
        self._lock = threading.RLock()
        self._tx_depth = 0
        self._vec_cache = OrderedDict()
        self.raw = psycopg.connect(  # type: ignore[assignment]
            dsn, autocommit=True, row_factory=_row_factory
        )
        self.db = _PGConnection(self.raw, self._lock)  # type: ignore[assignment]
        with self._lock:
            self.raw.execute("SELECT pg_advisory_lock(%s)", (LEDGER_LOCK + 1,))  # one migrator
            try:
                self.raw.execute(PG_SCHEMA)
                self._migrate()
                for counter in ("generation", "owner_generation"):
                    self.db.execute("INSERT OR IGNORE INTO meta VALUES (?, ?)", (counter, "0"))
            finally:
                self.raw.execute("SELECT pg_advisory_unlock(%s)", (LEDGER_LOCK + 1,))

    def _migrate(self) -> None:
        row = self.db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        current = int(row[0]) if row else SCHEMA_VERSION
        if current > SCHEMA_VERSION:
            raise RuntimeError(
                f"database uses schema v{current}, but this VeraMem supports up to "
                f"v{SCHEMA_VERSION}. Upgrade with: pip install -U veramem"
            )
        # PG databases are created at the current layout; future layout changes add
        # PG-specific steps here alongside the SQLite MIGRATIONS entries.
        self.db.execute(
            "INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
        )

    # -------------------------------------------------------------- dialect
    @staticmethod
    def resources_subset_sql(alias: str) -> str:
        return (
            f"NOT EXISTS (SELECT 1 FROM jsonb_array_elements_text(COALESCE(({alias}.metadata)"
            f"::jsonb -> 'resources', '[]'::jsonb)) AS r(value) WHERE r.value NOT IN "
            f"(SELECT j.value FROM jsonb_array_elements_text(?::jsonb) AS j(value)))"
        )

    def lock_ledger(self) -> None:
        self.db.execute("SELECT pg_advisory_xact_lock(?)", (LEDGER_LOCK,))

    def _fts(
        self,
        table: str,
        alias: str,
        query: str,
        where: str,
        params: Iterable[Any],
        limit: int,
        scopes: list[str] | None,
    ):
        terms = [t.strip('"') for t in fts_query(query).split(" OR ") if t.strip('"')]
        if not terms:
            return []
        tsq = " | ".join(
            re.sub(r"[^a-z0-9_]", "", t) for t in terms if re.sub(r"[^a-z0-9_]", "", t)
        )
        if not tsq:
            return []
        owner = f" AND {alias}.owner = ANY(?)" if scopes else ""
        extra = [list(scopes)] if scopes else []
        return self.db.execute(
            f"SELECT {alias}.id AS id, -ts_rank_cd({alias}.tsv, q) AS s FROM {table} {alias},"
            f" to_tsquery('english', ?) q WHERE {alias}.tsv @@ q{owner} AND {where}"
            f" ORDER BY s LIMIT ?",
            (tsq, *extra, *params, limit),
        ).fetchall()

    def memory_fts(self, query, where, params, limit, scopes=None):  # type: ignore[override]
        return self._fts("memories", "m", query, where, params, limit, scopes)

    def episode_fts(self, query, ids_where, params, limit, scopes=None):  # type: ignore[override]
        return self._fts("episodes", "e", query, ids_where, params, limit, scopes)

    def scrub(self) -> None:
        """Physically reclaim erased rows (VACUUM; run VACUUM FULL for page-level rewrite)."""
        with self._lock:
            for table in ("memories", "episodes", "evidence", "outcomes", "traces"):
                self.raw.execute(f"VACUUM {table}")

    def close(self) -> None:
        self.raw.close()
