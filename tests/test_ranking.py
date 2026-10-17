"""Outcome-informed ranking: citation lock, contextual utility, exploration, learned ranker."""

import numpy as np
import pytest

from veramem import Memory
from veramem.providers import HashEmbedder, NoLLM
from veramem.ranking import FEATURES, LogisticRanker


def _mem(tmp_path, **kw):
    return Memory(str(tmp_path / "r.db"), llm=NoLLM(), embedder=HashEmbedder(), **kw)


def test_feedback_citation_lock_and_single_rating(tmp_path):
    m = _mem(tmp_path)
    m.add(["Ana's dentist is Dr. Lee", "Ana's gym opens at 6"], user_id="ana")
    other = m.add("Bo's dentist is Dr. Kim", user_id="bo").changes[0].memory_id
    r = m.search("Ana dentist", user_id="ana", include_raw=False)
    top = r.hits[0].memory.id
    out = m.feedback(r.trace_id, 1.0, cited=[top, other])  # other is not in this trace
    assert out["credited"] == [top]
    with pytest.raises(ValueError):
        m.feedback(r.trace_id, 1.0)


def test_contextual_utility_is_query_specific(tmp_path):
    """A memory praised for dentist questions is boosted for dentist questions only."""
    for mode, boosts_unrelated in (("global", True), ("contextual", False)):
        m = Memory(str(tmp_path / f"{mode}.db"), llm=NoLLM(), embedder=HashEmbedder(), ranking=mode)
        mid = m.add("Note: Lee clinic, Oak street, room four", user_id="u").changes[0].memory_id
        for _ in range(6):
            r = m.search("dentist appointment clinic Lee", user_id="u", include_raw=False)
            m.feedback(r.trace_id, 1.0, cited=[mid])
        # an unrelated query that still retrieves the memory through "street"
        r = m.search("street parking rules downtown", user_id="u", include_raw=False)
        s = {h.memory.id: h.score for h in r.hits}[mid]
        base = {h.memory.id: h.signals for h in r.hits}[mid]
        assert (s > 1.5) is boosts_unrelated, (mode, s, base)


def test_thompson_exploration_surfaces_unrated_memories(tmp_path):
    m = _mem(tmp_path, exploration="thompson", seed=7)
    m.add([f"widget fact number {i}" for i in range(6)], user_id="u")
    tops = {
        m.search("widget fact", user_id="u", limit=1, include_raw=False).hits[0].memory.id
        for _ in range(30)
    }
    assert len(tops) > 1  # without exploration the same memory would win every time


def test_ranker_trains_and_is_used(tmp_path):
    m = _mem(tmp_path, exploration="thompson", seed=3)
    good = [f"Gate code for block {c} is {i}{i}{i}" for i, c in enumerate("ABCDEFGH")]
    bad = [f"Someone asked about the gate code for block {c} again" for c in "ABCDEFGH"]
    m.add(good + bad, user_id="u")
    for c in "ABCDEFGH" * 5:
        r = m.search(f"gate code for block {c}", user_id="u", include_raw=False, limit=4)
        top = r.hits[0].memory
        m.feedback(r.trace_id, 1.0 if top.text.startswith("Gate code") else -1.0, cited=[top.id])
    rep = m.train_ranker(min_examples=10)
    assert rep["trained"], rep
    assert set(rep["coefficients"]) == set(FEATURES)
    m2 = Memory(str(tmp_path / "r.db"), llm=NoLLM(), embedder=HashEmbedder(), ranking="ranker")
    assert m2.ranker is not None  # persisted
    hits = m2.search("gate code for block C", user_id="u", include_raw=False).hits
    assert all(0.0 <= h.score <= 1.0 for h in hits)


def test_logistic_ranker_recovers_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, len(FEATURES)))
    y = (x[:, 1] - 0.5 * x[:, 4] + rng.normal(scale=0.3, size=400) > 0).astype(float)
    model = LogisticRanker.fit(x, y, l2=0.1)
    top = list(model.explain())[:2]
    assert set(top) == {FEATURES[1], FEATURES[4]}
    assert ((model.predict(x) > 0.5) == (y > 0.5)).mean() > 0.9


def test_forget_erases_outcome_rows(tmp_path):
    m = _mem(tmp_path)
    mid = m.add("Zed's locker is 12", user_id="zed").changes[0].memory_id
    r = m.search("locker", user_id="zed", include_raw=False)
    m.feedback(r.trace_id, 1.0, cited=[mid])
    assert m.store.count("SELECT COUNT(*) FROM outcomes") == 1
    m.forget(user_id="zed", everything=True)
    assert m.store.count("SELECT COUNT(*) FROM outcomes") == 0
