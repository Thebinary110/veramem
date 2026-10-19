from .sqlite import SQLiteStore, fts_query


def make_store(path: str) -> SQLiteStore:
    """SQLite for a file path (or ":memory:"), PostgreSQL for a postgres:// URL."""
    if str(path).startswith(("postgresql://", "postgres://")):
        from .postgres import PostgresStore

        return PostgresStore(str(path))
    return SQLiteStore(path)


__all__ = ["SQLiteStore", "fts_query", "make_store"]
