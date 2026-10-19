"""Framework adapters. Each framework section is skipped when its package is missing."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest

from veramem import Memory
from veramem.providers import HashEmbedder, NoLLM


@pytest.fixture
def mem(tmp_path):
    m = Memory(str(tmp_path / "int.db"), llm=NoLLM(), embedder=HashEmbedder())
    yield m
    m.close()


def test_import_veramem_needs_no_framework():
    code = (
        "import sys, veramem, veramem.integrations, veramem.integrations.tools\n"
        "bad = [m for m in ('langchain_core','langgraph','llama_index','crewai') "
        "if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_framework_error_names_extra(monkeypatch):
    for name in [m for m in sys.modules if m.split(".")[0] == "langchain_core"] + [
        "langchain_core"
    ]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, "veramem.integrations.langchain", raising=False)
    with pytest.raises(ImportError, match=r'pip install "veramem\[langchain\]"'):
        import veramem.integrations.langchain  # noqa: F401
    monkeypatch.delitem(sys.modules, "veramem.integrations.langchain", raising=False)


# --------------------------------------------------------------------- tools
def test_tools_roundtrip_and_user_binding(mem):
    from veramem.integrations.tools import VeraMemTools

    mem.add("My passport number is X123", user_id="bob", infer=False)
    t = VeraMemTools(mem, user_id="alice")
    assert [x["function"]["name"] for x in t.openai_tools()] == [
        "remember",
        "recall",
        "feedback",
        "check_action",
    ]
    assert all("input_schema" in x for x in t.anthropic_tools())
    assert "user" not in t.json_schemas()["remember"]["properties"]["role"]["enum"]

    r = t.call("remember", json.dumps({"content": "I prefer aisle seats"}))
    mid = r["stored"][0]
    assert r["authority"] == "attested"
    assert t.call("remember", {"content": "x", "role": "user"})["error"]
    assert "unexpected" in t.call("recall", {"query": "passport", "user_id": "bob"})["error"]

    got = t.call("recall", {"query": "aisle seats"})
    ids = [m["id"] for m in got["memories"]]
    assert mid in ids
    assert all("X123" not in m["text"] for m in got["memories"])
    assert t.last_trace_id == got["trace_id"]

    fb = t.call("feedback", {"trace_id": got["trace_id"], "reward": 1, "cited": [mid]})
    assert "error" not in fb
    # bob's trace can't be rated through alice's tools
    bob_trace = mem.search("passport", user_id="bob").trace_id
    assert "error" in t.call("feedback", {"trace_id": bob_trace, "reward": -1})

    gate = t.call("check_action", {"memory_ids": [mid]})
    assert gate["allowed"] is False  # attested: confirm with the user first
    assert t.call("nope")["error"]


def test_tools_user_role_when_allowed(mem):
    from veramem.integrations.tools import VeraMemTools

    t = VeraMemTools(mem, user_id="alice", allow_user_role=True)
    r = t.remember("My email is a@x.org", role="user")
    assert r["authority"] == "authorized"
    assert t.check_action(r["stored"])["allowed"] is True


# ----------------------------------------------------------------- langchain
def test_langchain_retriever_and_history(mem):
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    from veramem.integrations.langchain import VeraMemChatMessageHistory, VeraMemRetriever

    h = VeraMemChatMessageHistory(mem, user_id="alice", session_id="s1")
    h.add_messages(
        [
            SystemMessage(content="be brief"),
            HumanMessage(content="I live in Lisbon"),
            AIMessage(content="Noted, Lisbon."),
            ToolMessage(content="weather: sunny", tool_call_id="call_1"),
        ]
    )
    msgs = h.messages
    assert [m.type for m in msgs] == ["system", "human", "ai"]
    assert msgs[1].content == "I live in Lisbon"
    h2 = VeraMemChatMessageHistory(
        mem, user_id="alice", session_id="s1", include_tool_messages=True
    )
    assert h2.messages[-1].type == "tool"
    assert VeraMemChatMessageHistory(mem, user_id="bob", session_id="s1").messages == []

    r = VeraMemRetriever(memory=mem, user_id="alice", limit=5)
    docs = r.invoke("where do I live Lisbon")
    assert docs and "Lisbon" in docs[0].page_content
    md = docs[0].metadata
    assert {"id", "authority", "kind", "date", "trace_id", "granularity"} <= set(md)
    assert md["trace_id"] == r.last_trace_id
    authorities = {d.page_content: d.metadata["authority"] for d in docs}
    assert authorities.get("I live in Lisbon") == "authorized"
    assert "error" not in r.feedback(1.0)
    assert VeraMemRetriever(memory=mem, user_id="bob").invoke("Lisbon") == []

    h.clear()
    assert h.messages == []
    assert r.invoke("Lisbon")  # clear() hides the transcript, not long-term memory


# ----------------------------------------------------------------- langgraph
def test_langgraph_store(mem):
    pytest.importorskip("langgraph")
    from veramem.integrations.langgraph import VeraMemStore

    store = VeraMemStore(mem, user_namespace_index=1)
    ns = ("memories", "alice")
    store.put(ns, "food", {"content": "likes ramen", "stars": 5})
    store.put(ns, "city", {"content": "lives in Porto", "stars": 2})
    store.put(("memories", "bob"), "food", {"content": "likes ramen too"})
    store.put(("prefs", "alice", "ui"), "theme", {"mode": "dark"})

    it = store.get(ns, "food")
    assert it is not None and it.value == {"content": "likes ramen", "stars": 5}
    assert it.key == "food" and it.namespace == ns
    created = it.created_at

    store.put(ns, "food", {"content": "likes sushi", "stars": 4})
    it2 = store.get(ns, "food")
    assert it2 is not None and it2.value["content"] == "likes sushi"
    assert it2.created_at == created

    res = store.search(ns, query="what food does alice like sushi")
    assert res and res[0].key == "food" and res[0].score is not None
    assert all(r.namespace == ns for r in res)
    assert store.last_trace_id
    assert [r.key for r in store.search(ns, filter={"stars": {"$gte": 3}})] == ["food"]
    assert len(store.search(ns, limit=1)) == 1

    assert set(store.list_namespaces(prefix=("memories",))) == {ns, ("memories", "bob")}
    assert store.list_namespaces(prefix=("prefs",), max_depth=2) == [("prefs", "alice")]

    with pytest.raises(ValueError, match="across users"):
        store.search(("memories",), query="ramen")

    store.delete(ns, "food")
    assert store.get(ns, "food") is None
    assert store.get(("memories", "bob"), "food") is not None

    # authority: values are the agent's claims by default
    city = mem.store.memories("json_extract(m.metadata,'$.lg_key')='city'")[0]
    assert city.authority.value == "attested" and city.user_id == "alice"

    async def go():
        await store.aput(ns, "k", {"text": "async value"})
        return await store.aget(ns, "k")

    got = asyncio.run(go())
    assert got is not None and got.value == {"text": "async value"}


def test_langgraph_store_bound_user(mem):
    pytest.importorskip("langgraph")
    from veramem.integrations.langgraph import VeraMemStore

    with pytest.raises(ValueError):
        VeraMemStore(mem)
    store = VeraMemStore(mem, user_id="carol", role="user")
    store.put(("notes",), "n1", {"content": "allergic to peanuts"})
    m = mem.get_all(user_id="carol")[0]
    assert m.authority.value == "authorized" and m.metadata["lg_key"] == "n1"
    assert store.list_namespaces() == [("notes",)]
    assert store.search(("notes",), query="peanuts")[0].key == "n1"


# ---------------------------------------------------------------- llamaindex
def test_llamaindex_retriever_and_block(mem):
    pytest.importorskip("llama_index.core")
    from llama_index.core.base.llms.types import ChatMessage

    from veramem.integrations.llamaindex import VeraMemMemoryBlock, VeraMemRetriever

    block = VeraMemMemoryBlock(memory=mem, user_id="alice", infer=False)
    asyncio.run(
        block.aput(
            [
                ChatMessage(role="user", content="My dog is called Miso"),
                ChatMessage(role="assistant", content="Cute name!"),
            ],
            session_id="s9",
        )
    )
    ctx = asyncio.run(block.aget([ChatMessage(role="user", content="what is my dog called Miso")]))
    assert "Miso" in ctx and block.last_trace_id
    assert asyncio.run(block.aget([])) == ""
    eps = mem.store.episodes("e.session_id='s9'")
    assert [e.source.value for e in eps] == ["user", "assistant"]

    r = VeraMemRetriever(mem, user_id="alice", include_raw=False)
    nodes = r.retrieve("dog Miso")
    assert nodes and nodes[0].node.get_content() == "My dog is called Miso"
    assert nodes[0].node.node_id == nodes[0].node.metadata["id"]
    assert nodes[0].node.metadata["authority"] == "authorized"
    assert "error" not in r.feedback(0.5)
    assert VeraMemRetriever(mem, user_id="bob").retrieve("Miso") == []
    assert asyncio.run(r.aretrieve("Miso"))


# -------------------------------------------------------------------- crewai
def test_crewai_storage_backend(mem):
    pytest.importorskip("crewai")
    from crewai.memory.storage.backend import StorageBackend
    from crewai.memory.types import MemoryRecord

    from veramem.integrations.crewai import VeraMemStorage

    st = VeraMemStorage(mem, user_id="alice")
    assert isinstance(st, StorageBackend)
    recs = [
        MemoryRecord(
            content="Deploys happen on Fridays",
            scope="/team/eng",
            categories=["process"],
            metadata={"src": "retro"},
            embedding=[1.0, 0.0, 0.0],
        ),
        MemoryRecord(
            content="Budget is 10k",
            scope="/team/finance",
            categories=["money"],
            embedding=[0.0, 1.0, 0.0],
        ),
    ]
    st.save(recs)
    assert st.count() == 2 and st.count("/team/eng") == 1
    got = st.get_record(recs[0].id)
    assert got is not None and got.content == recs[0].content and got.scope == "/team/eng"
    assert got.metadata == {"src": "retro"}

    hits = st.search([0.9, 0.1, 0.0], limit=5)
    assert hits[0][0].id == recs[0].id and hits[0][1] > hits[1][1]
    assert [r.id for r, _ in st.search([0.9, 0.1, 0.0], categories=["money"])] == [recs[1].id]
    assert st.search([0.9, 0.1, 0.0], scope_prefix="/team/finance")[0][0].id == recs[1].id
    assert st.list_scopes("/") == ["/team"]
    assert st.list_scopes("/team") == ["/team/eng", "/team/finance"]
    assert st.list_categories() == {"process": 1, "money": 1}
    info = st.get_scope_info("/team")
    assert info.record_count == 2 and info.child_scopes == ["/team/eng", "/team/finance"]

    # stored as the agent's claims, owned by alice, visible to VeraMem search
    items = mem.get_all(user_id="alice")
    assert {m.authority.value for m in items} == {"attested"}
    assert VeraMemStorage(mem, user_id="bob").count() == 0

    upd = recs[0].model_copy(update={"content": "Deploys happen on Thursdays"})
    st.update(upd)
    rec = st.get_record(recs[0].id)
    assert rec is not None and rec.content == "Deploys happen on Thursdays"
    assert st.count() == 2

    assert st.delete(record_ids=[recs[1].id]) == 1
    assert st.get_record(recs[1].id) is None
    assert asyncio.run(st.asearch([1.0, 0.0, 0.0]))
    st.reset()
    assert st.count() == 0


def test_crewai_tools(mem):
    pytest.importorskip("crewai")
    from veramem.integrations.crewai import veramem_tools

    tools = {t.name: t for t in veramem_tools(mem, user_id="alice")}
    assert set(tools) == {
        "veramem_remember",
        "veramem_recall",
        "veramem_feedback",
        "veramem_check_action",
    }
    out = json.loads(tools["veramem_remember"].run(content="The launch code word is Falcon"))
    mid = out["stored"][0]
    rec = json.loads(tools["veramem_recall"].run(query="launch code word Falcon"))
    assert mid in [m["id"] for m in rec["memories"]]
    fb = json.loads(tools["veramem_feedback"].run(trace_id=rec["trace_id"], reward=1.0))
    assert "error" not in fb
    gate = json.loads(tools["veramem_check_action"].run(memory_ids=[mid]))
    assert gate["allowed"] is False
