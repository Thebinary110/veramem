"""Optional HNSW index: same answers as exact search, same isolation guarantees."""

import pytest

pytest.importorskip("usearch")

from veramem import Memory  # noqa: E402
from veramem.providers import HashEmbedder, NoLLM  # noqa: E402


def _mem(path, index):
    return Memory(str(path), llm=NoLLM(), embedder=HashEmbedder(dim=256), vector_index=index)


def _seed(m):
    for u in ("ana", "ben"):
        m.add([f"{u} note {i} about topic {i % 7} and item {i}" for i in range(300)], user_id=u)


def test_hnsw_matches_exact_on_small_store(tmp_path):
    exact = _mem(tmp_path / "x.db", "exact")
    _seed(exact)
    exact.close()
    ann = _mem(tmp_path / "x.db", "hnsw")
    ex = _mem(tmp_path / "x.db", "exact")
    agree = 0
    for q in ("topic 3 item", "note about topic 5", "item 42"):
        a = [
            h.memory.id for h in ann.search(q, user_id="ana", include_raw=False, record=False).hits
        ]
        e = [h.memory.id for h in ex.search(q, user_id="ana", include_raw=False, record=False).hits]
        agree += len(set(a[:5]) & set(e[:5]))
    assert agree >= 12  # >= 80% top-5 overlap


def test_hnsw_never_returns_other_tenants(tmp_path):
    m = _mem(tmp_path / "i.db", "hnsw")
    _seed(m)
    for q in ("ana note", "topic 3", "item 7"):
        hits = m.search(q, user_id="ben", include_raw=False, limit=50).hits
        assert hits and all(h.memory.user_id == "ben" for h in hits)


def test_hnsw_sees_new_writes_and_drops_erased(tmp_path):
    m = _mem(tmp_path / "w.db", "hnsw")
    _seed(m)
    m.search("warm the index", user_id="ana")
    mid = m.add("ana keeps the spare key under the zebra statue", user_id="ana").changes[0]
    hits = m.search("spare key zebra statue", user_id="ana", include_raw=False).hits
    assert hits[0].memory.id == mid.memory_id
    m.forget(mid.memory_id, user_id="ana")
    hits = m.search("spare key zebra statue", user_id="ana", include_raw=False).hits
    assert all(h.memory.id != mid.memory_id for h in hits)


def test_hnsw_sees_rows_written_by_another_process(tmp_path):
    path = tmp_path / "p.db"
    reader = _mem(path, "hnsw")
    _seed(reader)
    reader.search("warm", user_id="ana")
    writer = _mem(path, "exact")
    new = writer.add("ana bought a violet bicycle", user_id="ana").changes[0].memory_id
    hits = reader.search("violet bicycle", user_id="ana", include_raw=False).hits
    assert hits[0].memory.id == new


def test_hnsw_rebuilds_when_rows_change_scope(tmp_path):
    m = _mem(tmp_path / "s.db", "hnsw")
    _seed(m)
    m.join_team("ben", "crew")
    m.search("warm", user_id="ben")
    mid = m.add("crew rendezvous is pier nine", user_id="ana").changes[0].memory_id
    m.store.update_memory(mid, visibility="team", team_id="crew", user_id=None)
    hits = m.search("crew rendezvous pier", user_id="ben", include_raw=False).hits
    assert any(h.memory.id == mid for h in hits)
