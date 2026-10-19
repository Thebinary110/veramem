"""Can the outcome ranker learn source authority, and transfer it to unseen attributes?

    python -m benchmarks.authority_ranking --seeds 5

LoCoMo never varies authority or kind (every memory is a user-stated fact), so the
transfer experiment there could not test those ranker features. This controlled benchmark
does, and only that:

* 8 attribute families (airline, home city, diet, ...). Each seed splits them 4/4 into
  feedback families and held-out families (no attribute is shared).
* Per user: the user's own statement of the value (authorized; a preference for
  preference-like attributes), plus two tool claims and one assistant suggestion with
  *wrong* values, worded to overlap the question more than the user's statement does,
  so plain retrieval prefers them.
* Memories go through VeraMem's real write path (a scripted extractor stands in for the
  LLM), so authority and kind are assigned by production code.
* The agent cites its top hit; reward +1 if it is the user's statement, else -1.
* Conditions: none, genuine, shuffled (permutation of the genuine rewards).

This is synthetic by design: it shows whether the ranker *can* learn and transfer an
authority preference when authority matters, not how often that happens in practice.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from veramem import Memory
from veramem.providers import HashEmbedder, ScriptedLLM

FAMILIES = {
    # name: (user statement, tool claim, assistant suggestion, question, values, is preference)
    "airline": (
        "I always fly with {v}.",
        "Booking history shows the user usually flies with {v}.",
        "You might like flying with {v}.",
        "Which airline does the user usually fly with?",
        ["KLM", "Iberia", "Finnair", "Qantas", "Delta", "Emirates"],
        True,
    ),
    "home_city": (
        "I live in {v}.",
        "Address records say the user's home city is {v}.",
        "Maybe you live near {v}?",
        "What is the user's home city?",
        ["Lyon", "Porto", "Graz", "Leeds", "Turin", "Ghent"],
        False,
    ),
    "diet": (
        "I'm {v}.",
        "The meal profile marks the user's diet as {v}.",
        "A {v} diet could suit you.",
        "What is the user's diet?",
        ["vegan", "pescatarian", "keto", "vegetarian", "halal", "paleo"],
        True,
    ),
    "seat": (
        "I prefer the {v} seat.",
        "Seat history: the user's preferred seat is {v}.",
        "The {v} seat is popular.",
        "What is the user's preferred seat?",
        ["window", "aisle", "middle", "exit-row", "bulkhead", "front"],
        True,
    ),
    "bank": (
        "My bank is {v}.",
        "Payment data shows the user's bank is {v}.",
        "Many people bank with {v}.",
        "Which bank does the user use?",
        ["Monzo", "Revolut", "Barclays", "Santander", "Nordea", "Rabobank"],
        False,
    ),
    "gym_time": (
        "I go to the gym at {v}.",
        "Check-in logs show the user's gym time is {v}.",
        "{v} is a good gym time.",
        "What is the user's gym time?",
        ["6am", "noon", "7pm", "9pm", "5am", "8am"],
        True,
    ),
    "phone_plan": (
        "I'm on the {v} plan.",
        "Billing shows the user's phone plan is {v}.",
        "The {v} plan is cheaper.",
        "What is the user's phone plan?",
        ["Lite", "Max", "Family", "Unlimited", "Basic", "Pro"],
        False,
    ),
    "shipping": (
        "Ship my orders to {v}.",
        "Order records list the user's shipping address as {v}.",
        "Shipping to {v} is faster.",
        "What is the user's shipping address?",
        ["Dock 4", "Gate 9", "Unit 12", "Bay 7", "Pier 3", "Lot 21"],
        False,
    ),
}
USERS_PER_FAMILY = 25
_LINE = re.compile(r"^\[(m\d+)\] \((\w+)[^)]*\) (.*)$")


def _extractor(prefs: set[str]):
    """Scripted stand-in for the LLM: one memory per message, quoting it verbatim."""

    def run(messages):
        system, user = messages[0]["content"], messages[-1]["content"]
        if not system.startswith("You are the memory constructor"):
            return {"operations": []} if "memory manager" in system else {}
        mems = []
        for line in user.split("NEW messages:\n", 1)[1].splitlines():
            m = _LINE.match(line)
            if m:
                label, role, content = m.groups()
                kind = "preference" if role == "user" and content in prefs else "fact"
                mems.append({"text": content, "kind": kind, "source": label, "quote": content})
        return {"memories": mems}

    return run


def build(path: Path, rng: random.Random) -> dict[str, list[dict[str, Any]]]:
    """Create all users; returns family -> list of {user, question, gold memory id}."""
    prefs = {tpl[0].format(v=v) for tpl in FAMILIES.values() if tpl[5] for v in tpl[4]}
    mem = Memory(str(path), llm=ScriptedLLM(_extractor(prefs)), embedder=HashEmbedder(dim=512))
    tasks: dict[str, list[dict[str, Any]]] = {}
    for fam, (u_t, t_t, a_t, question, values, _) in FAMILIES.items():
        for i in range(USERS_PER_FAMILY):
            uid = f"{fam}-{i}"
            true, *wrong = rng.sample(values, 4)
            msgs = [
                {"role": "user", "content": u_t.format(v=true)},
                {"role": "tool", "name": "records", "content": t_t.format(v=wrong[0])},
                {"role": "tool", "name": "crm", "content": t_t.format(v=wrong[1])},
                {"role": "assistant", "content": a_t.format(v=wrong[2])},
            ]
            rng.shuffle(msgs)
            res = mem.add(msgs, user_id=uid)
            gold = next(c.memory_id for c in res.changes if c.text == u_t.format(v=true))
            tasks.setdefault(fam, []).append({"user": uid, "q": question, "gold": gold})
    mem.close()
    return tasks


def run_condition(
    db: Path, train: list[dict], test: list[dict], cond: str, seed: int, genuine: list[float] | None
) -> tuple[float, list[float], dict]:
    mem = Memory(
        str(db),
        llm=ScriptedLLM(lambda m: {}),
        embedder=HashEmbedder(dim=512),
        ranking="contextual",
        exploration="thompson",
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(genuine)) if cond == "shuffled" and genuine else None
    rewards = []
    for _ in range(2):
        for j in rng.permutation(len(train)):
            t = train[j]
            r = mem.search(t["q"], user_id=t["user"], limit=5, include_raw=False)
            top = r.hits[0].memory.id
            reward = 1.0 if top == t["gold"] else -1.0
            rewards.append(reward)
            if perm is not None:
                reward = float(perm[len(rewards) - 1])
            mem.feedback(r.trace_id, reward, cited=[top])
    gate = mem.train_ranker(min_examples=20)
    mem.close()
    ev = Memory(
        str(db), llm=ScriptedLLM(lambda m: {}), embedder=HashEmbedder(dim=512), ranking="ranker"
    )
    hits = [
        float(
            ev.search(t["q"], user_id=t["user"], limit=5, include_raw=False, record=False)
            .hits[0]
            .memory.id
            == t["gold"]
        )
        for t in test
    ]
    ev.close()
    info = {k: gate.get(k) for k in ("active", "cv_gain_lower_95")}
    if gate.get("coefficients"):
        c = gate["coefficients"]
        info["coef"] = {
            k: round(c[k], 3)
            for k in (
                "auth_authorized",
                "auth_unendorsed",
                "auth_attested",
                "kind_preference",
                "bm25_rr",
                "dense",
            )
            if k in c
        }
    return float(np.mean(hits)), rewards, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="benchmarks/runs/authority_ranking.json")
    a = ap.parse_args()
    results = []
    for seed in range(a.seeds):
        rng = random.Random(seed)
        d = Path(tempfile.mkdtemp())
        base = d / "base.db"
        tasks = build(base, rng)
        fams = sorted(tasks)
        rng.shuffle(fams)
        train_f, test_f = fams[:4], fams[4:]
        train = [t for f in train_f for t in tasks[f]]
        test = [t for f in test_f for t in tasks[f]]
        none_mem = Memory(
            str(base), llm=ScriptedLLM(lambda m: {}), embedder=HashEmbedder(dim=512), ranking="none"
        )
        none_hit = float(
            np.mean(
                [
                    none_mem.search(
                        t["q"], user_id=t["user"], limit=5, include_raw=False, record=False
                    )
                    .hits[0]
                    .memory.id
                    == t["gold"]
                    for t in test
                ]
            )
        )
        none_mem.close()
        row = {"seed": seed, "train": train_f, "test": test_f, "none": none_hit}
        genuine = None
        for cond in ("genuine", "shuffled"):
            db = d / f"{cond}.db"
            db.write_bytes(base.read_bytes())
            hit, rewards, info = run_condition(db, train, test, cond, seed, genuine)
            if cond == "genuine":
                genuine = rewards
            row[cond] = hit
            row[f"{cond}_gate"] = info
        results.append(row)
        print(json.dumps(row), flush=True)
    for k in ("none", "genuine", "shuffled"):
        xs = [r[k] for r in results]
        m = statistics.mean(xs)
        h = 2.78 * statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0
        print(f"held-out Hit@1 {k:9}: {m:.3f} [{m - h:.3f}, {m + h:.3f}]")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
