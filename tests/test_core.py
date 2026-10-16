from veramem import Authority, Status


def test_add_search_with_evidence(mem):
    r = mem.add("Alice adopted a beagle named Rex", user_id="alice")
    assert [c.op.value for c in r.changes] == ["ADD"]
    res = mem.search("what dog does Alice have", user_id="alice")
    top = res.hits[0].memory
    assert "Rex" in top.text
    assert top.authority is Authority.AUTHORIZED
    ev = mem.evidence(top.id, user_id="alice")
    assert ev[0]["intact"] and ev[0]["quote"] in ev[0]["episode"]
    assert res.trace_id.startswith("tr_")


def test_ungrounded_candidate_is_rejected(mem):
    r = mem.add("INVENT a fact about Bob", user_id="alice")
    assert r.changes == []
    assert r.rejected and "ungrounded" in r.rejected[0]["reason"]


def test_greeting_skipped_and_batch_dedup(mem):
    r = mem.add(
        ["hello there", "Carol works at Initech", "Carol works at Initech"], user_id="carol"
    )
    ops = [c.op.value for c in r.changes]
    assert ops.count("ADD") == 1


def test_update_is_versioned_and_time_travel(mem):
    mem.add("Dana lives in Paris", user_id="dana")
    before = mem.store.db.execute("SELECT MAX(created_at) FROM memories").fetchone()[0]
    r = mem.add("Dana moved to Berlin", user_id="dana")
    assert r.changes[0].op.value == "UPDATE"
    new_id, old_id = r.changes[0].memory_id, r.changes[0].previous
    assert mem.store.get_memory(old_id).status is Status.SUPERSEDED
    hist = mem.history(new_id, user_id="dana")
    assert [h["id"] for h in hist] == [old_id, new_id]
    now_hits = [h.memory.text for h in mem.search("where does Dana live", user_id="dana").hits]
    assert any("Berlin" in t for t in now_hits) and not any("Paris" in t for t in now_hits)
    past = mem.search("where does Dana live", user_id="dana", as_of=before + 1e-6)
    assert any("Paris" in h.memory.text for h in past.hits)


def test_delete_retracts_but_keeps_successor(mem):
    mem.add("Erin has a red car", user_id="erin")
    r = mem.add("Erin no longer has a car", user_id="erin")
    ops = [c.op.value for c in r.changes]
    assert ops == ["DELETE", "ADD"]
    texts = [h.memory.text for h in mem.search("Erin car", user_id="erin").hits]
    assert any("no longer" in t for t in texts) and not any("red car" in t for t in texts)


def test_answer_citation_lock_and_abstention(mem):
    mem.add("Frank's favourite band is Radiohead", user_id="frank")
    a = mem.answer("What is Frank's favourite band?", user_id="frank")
    assert "Radiohead" in a.answer
    assert len(a.citations) == 1  # "999" was not shown to the model, so it is stripped
    a2 = mem.answer("What is my passport number?", user_id="nobody")
    assert a2.abstained and a2.answer == "I don't know"


def test_raw_mode_without_llm(raw_mem):
    raw_mem.add(
        [
            {"role": "user", "content": "I keep my bike in the garage"},
            {"role": "assistant", "content": "Noted!"},
        ],
        user_id="u",
    )
    hits = raw_mem.search("where is the bike", user_id="u").hits
    assert hits and "garage" in hits[0].memory.text
    assert len(raw_mem.get_all(user_id="u")) == 1  # assistant turns are not memories


def test_manual_update_and_delete_by_owner_only(mem):
    import pytest

    from veramem import AccessDenied

    mid = mem.add("Gina studies law", user_id="gina").changes[0].memory_id
    with pytest.raises(AccessDenied):
        mem.update(mid, "Gina studies medicine", user_id="mallory")
    new = mem.update(mid, "Gina studies medicine", user_id="gina")
    assert new.supersedes == mid and new.authority is Authority.AUTHORIZED
    with pytest.raises(AccessDenied):
        mem.delete(new.id, user_id="mallory")
    mem.delete(new.id, user_id="gina")
    assert mem.store.get_memory(new.id).status is Status.RETRACTED
