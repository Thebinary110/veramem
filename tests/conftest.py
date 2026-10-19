"""Deterministic fixtures: a rule-based fake LLM + hash embeddings. No network needed."""

from __future__ import annotations

import json
import os
import re

import pytest

from veramem import Memory
from veramem.providers import HashEmbedder, ScriptedLLM

_LINE = re.compile(r"^\[(m\d+)\] \((\w+)[^)]*\) (.*)$")


def fake_llm(messages):
    system, user = messages[0]["content"], messages[-1]["content"]
    if system.startswith("You are the memory constructor"):
        new = user.split("NEW messages:\n", 1)[1]
        mems = []
        for line in new.splitlines():
            m = _LINE.match(line)
            if not m:
                continue
            mid, role, content = m.groups()
            if content.startswith("hello"):
                continue
            quote = content if "INVENT" not in content else "something never said"
            text = content
            if role == "tool":
                text = f"According to the tool, {content}"
            kind = "preference" if re.search(r"\b(love|like|prefer|hate)\b", content) else "fact"
            ents = re.findall(r"\b[A-Z][a-z]{2,}\b", content)
            mems.append(
                {
                    "text": text,
                    "kind": kind,
                    "source": mid,
                    "quote": quote,
                    "when": None,
                    "entities": ents,
                }
            )
        return {"memories": mems}
    if system.startswith("You are the memory manager"):
        new = json.loads(user.split("NEW facts:\n", 1)[1])
        ops = []
        for f in new:
            t = f["text"].lower()
            if "no longer" in t:
                ops.append({"fact": f["index"], "event": "DELETE", "id": "0", "text": f["text"]})
            elif " now " in f" {t} " or "moved" in t:
                ops.append({"fact": f["index"], "event": "UPDATE", "id": "0", "text": f["text"]})
            else:
                ops.append({"fact": f["index"], "event": "ADD", "id": None, "text": f["text"]})
        return {"operations": ops}
    if system.startswith("You answer questions"):
        lines = [ln for ln in user.split("MEMORIES:\n", 1)[1].splitlines() if ln.startswith("[")]
        if not lines:
            return {"selected": [], "answer": "I don't know", "sufficient": False}
        first = lines[0]
        return {
            "selected": [first[1 : first.index("]")], "999"],
            "answer": first.split(") ", 1)[1],
            "sufficient": True,
        }
    if system.startswith("You turn a completed task"):
        return {
            "experience": "## book_table\nApplies when: booking a {venue}\nInputs: {date}\n"
            "Procedure:\n[step 1] ask for {party_size}\n[step 2] confirm\n"
            "Caution: never book without confirmation",
            "scope": "booking",
        }
    if system.startswith("You maintain a pool"):
        cand = user.split("CANDIDATE:\n", 1)[1]
        return {"action": "add", "id": None, "content": cand}
    if system.startswith("Rewrite this experience"):
        return {"content": user}
    if system.startswith("Summarise"):
        return {"title": "Session", "summary": "The user talked about things."}
    raise AssertionError(f"unexpected prompt: {system[:60]}")


@pytest.fixture
def mem(tmp_path):
    m = Memory(str(tmp_path / "test.db"), llm=ScriptedLLM(fake_llm), embedder=HashEmbedder())
    yield m
    m.close()


@pytest.fixture
def raw_mem(tmp_path):
    from veramem.providers import NoLLM

    m = Memory(str(tmp_path / "raw.db"), llm=NoLLM(), embedder=HashEmbedder())
    yield m
    m.close()


# --------------------------------------------------------------------- PostgreSQL
# Set VERAMEM_TEST_PG=postgresql://veramem@localhost:5432/veramem to run the whole suite against
# PostgreSQL: every Memory(...) a test creates gets its own fresh schema (stable per path,
# so tests that reopen a path see the same data).
_PG = os.environ.get("VERAMEM_TEST_PG")


@pytest.fixture(autouse=True)
def _route_to_postgres(monkeypatch):
    if not _PG:
        yield
        return
    import uuid

    import psycopg

    import veramem.memory as vm
    from veramem.store.postgres import PostgresStore

    schemas: dict[str, str] = {}
    admin = psycopg.connect(_PG, autocommit=True)

    def make_store(path):
        key = str(path) if str(path) != ":memory:" else uuid.uuid4().hex
        if key not in schemas:
            schemas[key] = "t_" + uuid.uuid4().hex[:12]
            admin.execute(f"CREATE SCHEMA {schemas[key]}")
        sep = "&" if "?" in _PG else "?"
        return PostgresStore(f"{_PG}{sep}options=-csearch_path%3D{schemas[key]}")

    monkeypatch.setattr(vm, "make_store", make_store)
    yield
    for schema in schemas.values():
        admin.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    admin.close()


def pytest_collection_modifyitems(config, items):
    if not _PG:
        return
    skip = pytest.mark.skip(reason="SQLite-specific (file layout / iterdump / v1 migration)")
    for item in items:
        if "sqlite_only" in item.keywords:
            item.add_marker(skip)
