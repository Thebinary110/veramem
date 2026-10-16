"""Gap C: isolation, grants, safe sharing, erasure."""

import pytest

from veramem import AccessDenied, Authority


def test_private_isolation_across_users(mem):
    mem.add("Nora's PIN is 4821", user_id="nora")
    for q in ["Nora PIN", "what is the pin", "4821", "Nora"]:
        hits = mem.search(q, user_id="oscar").hits
        assert all(h.memory.user_id != "nora" for h in hits)
    mid = mem.get_all(user_id="nora")[0].id
    with pytest.raises(AccessDenied):
        mem.get(mid, user_id="oscar")
    with pytest.raises(AccessDenied):
        mem.evidence(mid, user_id="oscar")


def test_dedup_does_not_cross_users(mem):
    mem.add("Paul owns a cat named Tom", user_id="paul")
    r = mem.add("Paul owns a cat named Tom", user_id="eve")  # same text, other user
    assert r.changes[0].op.value == "ADD"  # not NOOP pointing at Paul's memory
    assert r.changes[0].memory_id != mem.get_all(user_id="paul")[0].id


def test_agent_grant_revocation_is_retroactive(mem):
    mem.add("Quinn's project codename is Falcon", user_id="quinn", agent_id="research-bot")
    mem.grant_agent("quinn", "research-bot")
    assert mem.search("project codename", user_id="quinn").hits
    mem.revoke_agent("quinn", "research-bot")
    assert not [h for h in mem.search("project codename", user_id="quinn").hits
                if h.memory.agent_id == "research-bot"]
    mem.grant_agent("quinn", "research-bot")
    assert mem.search("project codename", user_id="quinn").hits


def test_resource_grants_hide_derived_memories(mem):
    mem.add("Rita's salary is confidential", user_id="rita", agent_id="hr-bot",
            resources=["hr_db"])
    mem.grant_resource("hr-bot", "hr_db")
    assert mem.search("salary", user_id="rita", agent_id="hr-bot").hits
    mem.revoke_resource("hr-bot", "hr_db")
    assert not mem.search("salary", user_id="rita", agent_id="hr-bot").hits


def test_preferences_cannot_be_shared(mem):
    mem.add("I love spicy food", user_id="sam")
    pref = mem.get_all(user_id="sam")[0]
    with pytest.raises(AccessDenied):
        mem.set_shareable(pref.id, True, user_id="sam")


def _learn(mem, user, reward=1.0):
    return mem.learn("book a table", ["asked party size", "confirmed booking"], reward,
                     user_id=user, shareable=True)


def test_experience_sharing_to_team(mem):
    mem.join_team("tara", "ops")
    mem.join_team("uma", "ops")
    out = _learn(mem, "tara")
    assert out["action"] == "add"
    rep = mem.share(user_id="tara", team_id="ops")
    assert len(rep["shared"]) == 1
    shared_id = rep["shared"][0]["shared_id"]
    s = mem.store.get_memory(shared_id)
    assert s.user_id is None and s.authority is Authority.UNENDORSED and s.evidence == []
    hits = mem.search("how to book a venue", user_id="uma").hits
    assert any(h.memory.id == shared_id and h.granularity == "shared" for h in hits)
    assert not any(h.memory.id == shared_id for h in mem.search("book venue",
                                                                 user_id="outsider").hits)
    assert not mem.check_action([shared_id])["allowed"]


def test_privacy_gate_blocks_pii_and_canaries(mem):
    mem.add("Victor's email is victor@corp.test and he lives at 12 Baker Street",
            user_id="victor")
    item = mem.learner._manage_experience(
        "## x\nApplies when: email victor@corp.test about Baker Street", None, "victor",
        None, [], 1.0, True, __import__("veramem").Visibility.PRIVATE, None)[1]
    rep = mem.share(user_id="victor", memory_ids=[item.id])
    assert not rep["shared"] and rep["rejected"][0]["reason"] == "privacy gate"


def test_share_requires_consent_and_membership(mem):
    mem.learn("book a table", ["done"], 1.0, user_id="wes", shareable=False)
    assert mem.share(user_id="wes")["shared"] == []
    with pytest.raises(PermissionError):
        mem.share(user_id="wes", team_id="not-my-team")


def test_forget_user_erases_and_withdraws(mem):
    mem.join_team("xena", "t")
    mem.add("Xena's secret phrase is purple-otter-42", user_id="xena")
    _learn(mem, "xena")
    shared = mem.share(user_id="xena", team_id="t")["shared"][0]["shared_id"]
    rep = mem.forget(user_id="xena", everything=True)
    assert rep["memories"] >= 2 and rep["shared"] == 1
    dump = "\n".join(mem.store.db.iterdump())
    assert "purple-otter-42" not in dump
    assert mem.store.get_memory(shared).status.value == "retracted"
    assert mem.verify()["ok"]
