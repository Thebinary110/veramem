"""Gap A: outcome feedback drives utility, ranking, archival and training data."""

import json


def test_feedback_credits_cited_memories(mem):
    mem.add("Yara's gate code is 1234", user_id="yara")
    a = mem.answer("What is Yara's gate code?", user_id="yara")
    mid = a.citations[0]
    mem.feedback(a.trace_id, 1.0)
    assert mem.store.get_memory(mid).utility > 0.6


def test_negative_feedback_demotes_and_archives(mem):
    mem.add(
        ["Zoe's locker is number 7", "Zoe's locker combination is written on a note"], user_id="zoe"
    )
    bad = mem.search("Zoe locker number", user_id="zoe", limit=1).hits[0].memory.id
    for _ in range(4):
        res = mem.search("Zoe locker number", user_id="zoe", limit=1)
        mem.store.db.execute(
            "UPDATE traces SET memory_ids=? WHERE id=?", (json.dumps([bad]), res.trace_id)
        )
        mem.store.db.execute("UPDATE memories SET uses=uses+1 WHERE id=?", (bad,))
        mem.feedback(res.trace_id, -1.0)
    assert mem.store.get_memory(bad).utility < 0.3
    rep = mem.consolidate(user_id="zoe", summarize=False)
    assert bad in rep["archived"]
    assert all(h.memory.id != bad for h in mem.search("Zoe locker", user_id="zoe").hits)


def test_utility_reorders_ranking(mem):
    mem.add(["Ava drinks green tea", "Ava drinks black tea"], user_id="ava")
    res = mem.search("what tea does Ava drink", user_id="ava", limit=2)
    second = res.hits[1].memory.id
    for _ in range(5):
        r = mem.search("what tea does Ava drink", user_id="ava", limit=2)
        mem.store.db.execute(
            "UPDATE traces SET memory_ids=? WHERE id=?", (json.dumps([second]), r.trace_id)
        )
        mem.feedback(r.trace_id, 1.0)
    assert mem.search("what tea does Ava drink", user_id="ava").hits[0].memory.id == second


def test_learn_requires_decisive_outcome(mem):
    assert mem.learn("t", ["x"], 0.3, user_id="b")["action"] == "none"
    assert mem.learn("t", ["x"], 1.0, user_id="b")["action"] == "add"


def test_export_trajectories(mem, tmp_path):
    mem.add("Ben's dentist is Dr. Ortiz", user_id="ben")
    a = mem.answer("Who is Ben's dentist?", user_id="ben")
    mem.feedback(a.trace_id, 1.0)
    path = tmp_path / "traj.jsonl"
    n = mem.export_trajectories(str(path))
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert n == len(rows) >= 1
    assert rows[0]["schema"] == "veramem.answer.v1" and rows[0]["reward"] == 1.0


def test_consolidate_dedupes_and_summarizes(mem):
    msgs = [
        "Cleo plays the violin",
        "Cleo is learning Spanish",
        "Cleo visited Rome",
        "Cleo adopted a parrot",
    ]
    mem.add(msgs, user_id="cleo", session_id="s1")
    rep = mem.consolidate(user_id="cleo")
    assert len(rep["summaries"]) == 1
    hits = mem.search("summarize Cleo's life overall", user_id="cleo").hits
    assert any(h.granularity == "summary" for h in hits)
