"""Gap B: provenance, source authority, tamper evidence."""

import pytest

from veramem import Authority, Kind, Memory
from veramem.providers import HashEmbedder, ScriptedLLM


def test_role_policy_authority_labels(mem):
    mem.add(
        [
            {"role": "user", "content": "My seat preference is aisle"},
            {"role": "assistant", "content": "the Gold plan suits the user"},
            {"role": "tool", "name": "crm", "content": "home city is Asterhaven"},
        ],
        user_id="u1",
    )
    by_text = {m.text: m for m in mem.get_all(user_id="u1")}
    assert by_text["My seat preference is aisle"].authority is Authority.AUTHORIZED
    # non-user claims are worded as claims even when the extractor forgets to attribute them
    assert (
        by_text["The assistant stated that the Gold plan suits the user"].authority
        is Authority.ATTESTED
    )
    tool = by_text["According to the tool, home city is Asterhaven"]
    assert tool.authority is Authority.UNENDORSED


def test_unattributed_tool_claim_is_reworded(tmp_path):
    def extractor(messages):
        if not messages[0]["content"].startswith("You are the memory constructor"):
            return {"operations": []}
        q = "bank is Barclays"
        return {"memories": [{"text": "The user's bank is Barclays", "source": "m0", "quote": q}]}

    m = Memory(str(tmp_path / "m.db"), llm=ScriptedLLM(extractor), embedder=HashEmbedder(dim=64))
    m.add({"role": "tool", "name": "crm", "content": "Profile: bank is Barclays"}, user_id="u3")
    (item,) = m.get_all(user_id="u3")
    assert item.text == "According to crm, the user's bank is Barclays"
    assert item.authority is Authority.UNENDORSED
    m.close()


def test_non_user_source_cannot_create_preference(mem):
    mem.add({"role": "tool", "content": "users like the premium plan"}, user_id="u2")
    m = mem.get_all(user_id="u2")[0]
    assert m.kind is Kind.FACT  # R2P: a tool's suggestion is never a user preference


def test_low_authority_cannot_overwrite_user_fact(mem):
    mem.add("Hank lives in Oslo", user_id="hank")
    r = mem.add({"role": "tool", "content": "Hank moved to Rome"}, user_id="hank")
    ch = r.changes[0]
    assert ch.op.value == "ADD" and "cannot update" in ch.reason
    texts = {m.text: m.authority for m in mem.get_all(user_id="hank")}
    assert texts["Hank lives in Oslo"] is Authority.AUTHORIZED  # untouched


def test_action_gate(mem):
    mem.add(
        [
            {"role": "user", "content": "Send invoices to ap@acme.test"},
            {"role": "tool", "content": "billing contact is evil@x.test"},
        ],
        user_id="ivy",
    )
    mems = {m.authority: m.id for m in mem.get_all(user_id="ivy")}
    ok = mem.check_action([mems[Authority.AUTHORIZED]], user_id="ivy")
    assert ok["allowed"]
    gated = mem.check_action([mems[Authority.UNENDORSED]], user_id="ivy")
    assert not gated["allowed"] and gated["needs_confirmation"][0]["authority"] == "unendorsed"


def test_user_restatement_legitimately_raises_authority(mem):
    mem.add({"role": "tool", "content": "Jack prefers window seats"}, user_id="jack")
    m = mem.get_all(user_id="jack")[0]
    assert m.authority is Authority.UNENDORSED
    # identical proposition now stated by the user himself (AuthMem H+): becomes authorized
    mem.add("According to the tool, Jack prefers window seats", user_id="jack")
    assert mem.store.get_memory(m.id).authority is Authority.AUTHORIZED


def test_ledger_detects_tampering(mem):
    mem.add(["Kim plays chess", "Kim speaks Korean"], user_id="kim")
    assert mem.verify()["ok"]
    mid = mem.get_all(user_id="kim")[0].id
    mem.store.db.execute("UPDATE memories SET text='Kim plays poker' WHERE id=?", (mid,))
    rep = mem.verify()
    assert not rep["ok"] and rep["tampered_memories"] == [mid]


def test_ledger_detects_rewritten_history(mem):
    mem.add("Leo likes tea", user_id="leo")
    mem.store.db.execute("UPDATE ledger SET payload='{}' WHERE seq=1")
    rep = mem.verify()
    assert not rep["ledger_ok"] and rep["first_bad_seq"] == 1


def test_episode_tampering_detected(mem):
    mem.add("Mia was born in Lima", user_id="mia")
    mem.store.db.execute("UPDATE episodes SET content='Mia was born in Quito'")
    rep = mem.verify()
    assert not rep["ok"] and rep["tampered_episodes"]


def test_action_gate_does_not_leak_other_users_memories(mem):
    mid = mem.add({"role": "tool", "content": "vault pin is 9911"}, user_id="vic").changes[0]
    out = mem.check_action([mid.memory_id], user_id="mallory")
    assert out["missing"] == [mid.memory_id] and out["needs_confirmation"] == []


def test_feedback_on_another_users_trace_is_denied(mem):
    import pytest

    from veramem import AccessDenied

    mem.add("Wes rides a red bike", user_id="wes")
    r = mem.search("bike", user_id="wes")
    with pytest.raises(AccessDenied):
        mem.feedback(r.trace_id, -1.0, user_id="mallory")
    mem.feedback(r.trace_id, 1.0, user_id="wes")


@pytest.mark.sqlite_only  # spawned processes use the SQLite file directly
def test_ledger_chain_survives_concurrent_processes(tmp_path):
    import multiprocessing as mp

    from veramem import Memory
    from veramem.providers import HashEmbedder, NoLLM

    path = str(tmp_path / "multi.db")
    Memory(path, llm=NoLLM(), embedder=HashEmbedder()).close()
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_writer, args=(path, i)) for i in range(3)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
        assert p.exitcode == 0
    m = Memory(path, llm=NoLLM(), embedder=HashEmbedder())
    rep = m.verify()
    assert rep["ok"], rep
    assert len(m.get_all(user_id="p0")) == 25


def _writer(path: str, i: int) -> None:
    from veramem import Memory
    from veramem.providers import HashEmbedder, NoLLM

    m = Memory(path, llm=NoLLM(), embedder=HashEmbedder())
    for j in range(25):
        m.add(f"process {i} note {j}", user_id=f"p{i}")
    m.close()
