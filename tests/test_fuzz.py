"""Property-based fuzzing of the query compiler, tenant isolation and LLM-output parsing.

Needs `hypothesis` (dev dependency); skipped when it is not installed.
"""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3

import pytest

hypothesis = pytest.importorskip("hypothesis")

from conftest import fake_llm  # noqa: E402
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from veramem import Memory  # noqa: E402
from veramem.providers import HashEmbedder, NoLLM, ScriptedLLM  # noqa: E402
from veramem.providers.llm import LLMError, parse_json  # noqa: E402
from veramem.store.sqlite import fts_query, owner_token  # noqa: E402

FUZZ = settings(
    max_examples=150,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

# text biased towards FTS5 syntax, SQL fragments and awkward unicode
_ATOMS = st.sampled_from(
    [
        '"', "'", "*", "^", ":", "(", ")", "{", "}", "-", "+", ",", ";", "\\", "\x00",
        "NEAR", "NEAR/2", "AND", "OR", "NOT", "MATCH", "owner", "text", "content", "id",
        "owner:", "text:", "pub", "--", "/*", "*/", "DROP TABLE memories", "1=1", "%", "_",
        "​", "‮", "﻿", "ｖａｕｌｔ", "🦓", "vault", "code", "zebra", "9917",
    ]
)  # fmt: skip
QUERY = st.lists(st.one_of(_ATOMS, st.text(max_size=12)), max_size=24).map(" ".join)
VICTIM_TEXT = "Victim vault code zebra quartz 9917"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    path = tmp_path_factory.mktemp("fuzz") / "f.db"
    mem = Memory(str(path), llm=NoLLM(), embedder=HashEmbedder())
    mem.add(VICTIM_TEXT, user_id="victim")
    mem.add({"role": "tool", "content": VICTIM_TEXT + " tool copy"}, user_id="victim")
    mem.add("Mallory keeps a kayak and a vault", user_id="mallory")
    victim_ids = {m.id for m in mem.store.memories("m.user_id='victim'")}
    victim_ids |= {e.id for e in mem.store.episodes("e.user_id='victim'")}
    yield mem, victim_ids
    mem.close()


@FUZZ
@given(q=QUERY)
def test_fts_query_is_always_valid_fts5(q):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE t USING fts5(id UNINDEXED, owner, text, tokenize='porter')")
    db.execute("INSERT INTO t VALUES ('1', 'oabc', 'vault code zebra')")
    db.execute("INSERT INTO t VALUES ('2', 'oxyz', 'vault code zebra')")
    compiled = fts_query(q, ["oabc"])
    if not compiled:
        return
    rows = db.execute("SELECT id FROM t WHERE t MATCH ?", (compiled,)).fetchall()
    assert all(r[0] == "1" for r in rows)  # the owner pin can never be escaped
    head, body = compiled.split(" AND ", 1)
    assert head == 'owner : ("oabc")'
    terms = body[len("text : (") : -1].split(" OR ")
    assert all(re.fullmatch(r'"[a-z0-9_]+"', t) for t in terms)


@FUZZ
@given(q=QUERY, k=st.integers(min_value=1, max_value=100))
def test_search_never_crosses_tenants(world, q, k):
    mem, victim_ids = world
    for g in ("auto", "raw", "fact", "dual"):
        res = mem.search(q, user_id="mallory", limit=k, granularity=g, record=False)
        for h in res.hits:
            assert h.memory.id not in victim_ids and h.memory.user_id != "victim"


@FUZZ
@given(q=QUERY)
def test_victim_owner_token_in_query_does_not_widen(world, q):
    mem, victim_ids = world
    tok = owner_token("victim", None, None, "private")
    for query in (f"{tok} {q}", f"owner:{tok} {q}", f'owner : "{tok}" OR {q}'):
        rows = mem.store.memory_fts(
            query, "1=1", [], 100, [owner_token("mallory", None, None, "private")]
        )
        assert not {r["id"] for r in rows} & victim_ids


@FUZZ
@given(content=st.text(max_size=400), uid=st.text(min_size=1, max_size=20))
def test_add_then_search_roundtrip_is_isolated(world, content, uid):
    mem, victim_ids = world
    if uid == "victim":
        return
    if any(ord(c) < 32 or c == "\x7f" for c in uid):
        with pytest.raises(ValueError):  # rejected outright, never cleaned into another id
            mem.add(content, user_id=uid)
        return
    r = mem.add(content, user_id=uid)
    assert len(r.episodes) == (1 if content.replace("\x00", "").strip() else 0)
    for h in mem.search(content or "x", user_id=uid, limit=50, record=False).hits:
        assert h.memory.id not in victim_ids


@FUZZ
@given(text=st.text(max_size=300))
def test_parse_json_only_raises_llm_error(text):
    with contextlib.suppress(LLMError):
        parse_json(text)


JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=20),
    lambda c: st.lists(c, max_size=4) | st.dictionaries(st.text(max_size=8), c, max_size=4),
    max_leaves=12,
)
EXTRACTION = st.fixed_dictionaries(
    {
        "memories": st.lists(
            st.fixed_dictionaries(
                {
                    "text": st.text(max_size=40),
                    "kind": st.sampled_from(["fact", "preference", "event", "x", ""]),
                    "source": st.sampled_from(["m0", "m1", "m9", "", "[m0]"]),
                    "quote": st.sampled_from(["Mallory likes tea", "tea", "", "never said"]),
                    "when": st.none() | st.sampled_from(["2026-01-01", "null", "soon"]),
                    "entities": st.lists(st.text(max_size=10), max_size=10),
                }
            ),
            max_size=6,
        )
    }
)


@FUZZ
@given(out=EXTRACTION)
def test_well_typed_hostile_extraction_keeps_guarantees(tmp_path_factory, out):
    path = tmp_path_factory.mktemp("ex") / "e.db"
    llm = ScriptedLLM(
        lambda m: out if m[0]["content"].startswith("You are the memory") else fake_llm(m),
        retries=0,
    )
    mem = Memory(str(path), llm=llm, embedder=HashEmbedder())
    try:
        mem.add(VICTIM_TEXT, user_id="victim", infer=False)
        mem.add({"role": "tool", "content": "Mallory likes tea"}, user_id="mallory")
        for m in mem.get_all(user_id="mallory"):
            assert m.authority.value == "unendorsed" and m.kind.value != "preference"
            assert m.evidence and "tea" in m.evidence[0].quote.lower()
        assert all("9917" not in h.memory.text for h in mem.search("vault", user_id="mallory").hits)
        assert mem.verify()["ok"]
    finally:
        mem.close()


@FUZZ
@given(out=JSON)
def test_arbitrary_extraction_json_never_crashes(tmp_path_factory, out):
    path = tmp_path_factory.mktemp("aj") / "a.db"
    payload = json.dumps(out)
    llm = ScriptedLLM(
        lambda m: payload if m[0]["content"].startswith("You are the memory") else fake_llm(m),
        retries=0,
    )
    mem = Memory(str(path), llm=llm, embedder=HashEmbedder())
    try:
        mem.add("Mallory likes tea", user_id="mallory")
        mem.add("Mallory now likes coffee", user_id="mallory")
    finally:
        mem.close()
