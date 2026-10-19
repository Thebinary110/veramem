"""Does outcome feedback teach VeraMem which memories help on *new* tasks?

    python -m benchmarks.outcome_transfer                # 5 seeds, all methods/conditions
    python -m benchmarks.outcome_transfer --seeds 1 --quick

Placebo-controlled transfer experiment on LoCoMo (no LLM in the loop):

* Memories are the raw conversation turns of all 10 LoCoMo conversations
  (`infer=False`), so every memory maps to exactly one gold turn id.
* Task families are conversations (different people, topics and vocabulary). Each seed
  splits them into 5 feedback families and 5 held-out families; every memory in a
  held-out family is unseen during feedback.
* Within the feedback families, questions are split 50/50 into feedback questions and
  within-family held-out questions; a held-out question is dropped if it is a
  near-duplicate (cosine > 0.9) of any feedback question in the same family.
* A simulated agent retrieves the top 10, cites the top 3, and gets reward +1 if a
  cited memory is a gold evidence turn, else -1 (feedback goes to the cited memories).
* Conditions: none, genuine, shuffled (an exact permutation of the same seed's genuine
  reward sequence: same distribution, association broken), noisy (20% flipped).
* Evaluation never records traces or gives feedback (no leakage into held-out tasks).

Metrics: Hit@3 (the agent's task success), MRR@10, Recall@10 and nDCG@10 over gold
turns. Reported as mean over seeds with a 95% t-interval, plus paired bootstrap CIs of
the difference against the no-feedback baseline and against the shuffled control.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from veramem import Memory
from veramem.providers import OllamaEmbedder, ResponseCache

from .adapters import EMBED_MODEL, OLLAMA
from .locomo import CATEGORY, load, sessions

K_RETRIEVE, K_CITE = 10, 3
METHODS = {
    # name: (ranking during feedback, exploration during feedback, ranking at evaluation)
    "global": ("global", "none", "global"),
    "contextual": ("contextual", "none", "contextual"),
    "contextual+ts": ("contextual", "thompson", "contextual"),
    "ranker": ("contextual", "thompson", "ranker"),
}
CONDITIONS = ["genuine", "shuffled", "noisy"]
_DIA = re.compile(r"D\d+:\d+")


# ------------------------------------------------------------------ setup
def build_base(out: Path, data: list[dict[str, Any]], cache: ResponseCache) -> Path:
    """Ingest every conversation once as raw turns; map memory id -> gold turn id."""
    db = out / "base.db"
    mapping = out / "memory_turns.json"
    if db.exists() and mapping.exists():
        return db
    mem = Memory(str(db), llm=_no_llm(), embedder=OllamaEmbedder(EMBED_MODEL, OLLAMA, cache=cache))
    turns: dict[str, str] = {}
    for conv in data:
        cid = conv["sample_id"]
        for key, _when, msgs in sessions(conv):
            # align with the messages VeraMem keeps (it skips empty content)
            dia = [
                t["dia_id"]
                for t, m in zip(conv["conversation"][key], msgs)
                if str(m["content"]).strip()
            ]
            res = mem.add(msgs, user_id=cid, infer=False)
            for ch in res.changes:  # one raw memory per non-empty turn, in order
                ep = mem.store.get_memory(ch.memory_id).evidence[0].episode_id
                idx = res.episodes.index(ep)
                turns[ch.memory_id] = dia[idx]
    mapping.write_text(json.dumps(turns), encoding="utf-8")
    mem.close()
    return db


def _no_llm():
    from veramem.providers import NoLLM

    return NoLLM()


def questions(data: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for conv in data:
        qs = []
        for i, q in enumerate(conv["qa"]):
            ev = set(_DIA.findall(" ".join(map(str, q.get("evidence", [])))))
            if q.get("category") in CATEGORY and ev:
                qs.append(
                    {
                        "qid": f"{conv['sample_id']}#{i}",
                        "q": q["question"],
                        "ev": ev,
                        "cat": CATEGORY[q["category"]],
                    }
                )
        out[conv["sample_id"]] = qs
    return out


def split(
    convs: list[str], qs: dict[str, list[dict[str, Any]]], seed: int, emb: OllamaEmbedder
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(convs))
    train_f, test_f = order[:5], order[5:]
    feedback, within, dropped = [], [], 0
    for c in train_f:
        idx = rng.permutation(len(qs[c]))
        half = len(idx) // 2
        fb = [qs[c][i] for i in idx[:half]]
        held = [qs[c][i] for i in idx[half:]]
        fv = emb.embed([q["q"] for q in fb])
        hv = emb.embed([q["q"] for q in held])
        keep = (hv @ fv.T).max(axis=1) <= 0.9 if len(fb) else np.ones(len(held), bool)
        dropped += int((~keep).sum())
        feedback += [(c, q) for q in fb]
        within += [(c, q) for q, k in zip(held, keep) if k]
    test = [(c, q) for c in test_f for q in qs[c]]
    return {
        "train_families": train_f,
        "test_families": test_f,
        "feedback": feedback,
        "within": within,
        "test": test,
        "near_duplicates_dropped": dropped,
    }


# -------------------------------------------------------------- one run
def run_feedback(
    db: Path,
    turns: dict[str, str],
    sp: dict[str, Any],
    method: str,
    cond: str,
    seed: int,
    emb: OllamaEmbedder,
    epochs: int,
    genuine_rewards: list[float] | None,
) -> tuple[list[float], float, dict[str, Any]]:
    ranking, exploration, _ = METHODS[method]
    mem = Memory(
        str(db), llm=_no_llm(), embedder=emb, ranking=ranking, exploration=exploration, seed=seed
    )
    rng = np.random.default_rng(1000 + seed)
    perm = (
        list(rng.permutation(genuine_rewards)) if cond == "shuffled" and genuine_rewards else None
    )
    rewards: list[float] = []
    t0 = time.time()
    for _ in range(epochs):
        for j in rng.permutation(len(sp["feedback"])):
            cid, q = sp["feedback"][j]
            r = mem.search(
                q["q"], user_id=cid, limit=K_RETRIEVE, granularity="fact", include_raw=False
            )
            cited = [h.memory.id for h in r.hits[:K_CITE]]
            reward = 1.0 if any(turns.get(m) in q["ev"] for m in cited) else -1.0
            rewards.append(reward)
            if cond == "shuffled":
                reward = float(perm[len(rewards) - 1])
            elif cond == "noisy" and rng.random() < 0.2:
                reward = -reward
            if cited:
                mem.feedback(r.trace_id, reward, cited=cited)
    gate: dict[str, Any] = {}
    if METHODS[method][2] == "ranker":
        rep = mem.train_ranker(min_examples=20)
        gate = {k: rep.get(k) for k in ("active", "cv_gain_per_example", "cv_gain_lower_95", "l2")}
    mem.close()
    return rewards, time.time() - t0, gate


def evaluate(
    db: Path,
    turns: dict[str, str],
    items: list[tuple[str, dict[str, Any]]],
    ranking: str,
    emb: OllamaEmbedder,
) -> tuple[dict[str, float], list[dict], float]:
    mem = Memory(str(db), llm=_no_llm(), embedder=emb, ranking=ranking)
    per_q, lat = [], []
    for cid, q in items:
        t = time.perf_counter()
        r = mem.search(
            q["q"],
            user_id=cid,
            limit=K_RETRIEVE,
            granularity="fact",
            include_raw=False,
            record=False,
        )  # no traces, no feedback: no leakage
        lat.append(time.perf_counter() - t)
        got = [turns.get(h.memory.id) for h in r.hits]
        rel = [g in q["ev"] for g in got]
        first = next((i for i, x in enumerate(rel) if x), None)
        dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(rel) if x)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(q["ev"]), K_RETRIEVE)))
        per_q.append(
            {
                "qid": q["qid"],
                "cat": q["cat"],
                "hit3": float(any(rel[:K_CITE])),
                "mrr": 0.0 if first is None else 1.0 / (first + 1),
                "recall10": len({g for g in got if g in q["ev"]}) / len(q["ev"]),
                "ndcg10": dcg / idcg if idcg else 0.0,
            }
        )
    mem.close()
    agg = {k: float(np.mean([p[k] for p in per_q])) for k in ("hit3", "mrr", "recall10", "ndcg10")}
    return agg, per_q, float(np.median(lat) * 1000)


# --------------------------------------------------- static baseline
STATIC_GRID = [(b, n) for b in (0.0, 1.0, 2.0, 4.0) for n in (0.0, 0.25, 0.5, 1.0)]


def _static_ranker(w_bm25: float, w_len: float) -> str:
    from veramem.ranking import FEATURES, LogisticRanker

    w = [0.0] * len(FEATURES)
    w[FEATURES.index("bm25_rr")] = w_bm25
    w[FEATURES.index("log_len")] = w_len
    model = LogisticRanker(
        weights=w, bias=0.0, mean=[0.0] * len(FEATURES), scale=[1.0] * len(FEATURES), active=True
    )
    model.metrics = {"static": True, "w_bm25": w_bm25, "w_len": w_len}
    return model.to_json()


def tune_static(
    base: Path, out: Path, turns: dict[str, str], sp: dict[str, Any], emb: OllamaEmbedder, seed: int
) -> tuple[dict[str, float], Path]:
    """No-feedback baseline: a fixed re-weighting of the same retrieval features the learned
    ranker uses (BM25 rank, memory length), grid-tuned on the feedback families' questions
    with *gold* labels (strictly more information than feedback provides)."""
    db = out / f"static_s{seed}.db"
    shutil.copy(base, db)
    best, best_hit = (0.0, 0.0), -1.0
    for wb, wl in STATIC_GRID:
        m = Memory(str(db), llm=_no_llm(), embedder=emb)
        m.store.meta_set("ranker", _static_ranker(wb, wl))
        m.close()
        agg, _, _ = evaluate(db, turns, sp["feedback"], "ranker", emb)
        if agg["hit3"] > best_hit + 1e-9:
            best, best_hit = (wb, wl), agg["hit3"]
    m = Memory(str(db), llm=_no_llm(), embedder=emb)
    m.store.meta_set("ranker", _static_ranker(*best))
    m.close()
    return {"w_bm25": best[0], "w_len": best[1], "tuned_hit3": round(best_hit, 4)}, db


# ---------------------------------------------------------------- stats
def t_ci(xs: list[float]) -> tuple[float, float, float]:
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, m, m
    tcrit = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}
    h = tcrit.get(len(xs), 1.96) * statistics.stdev(xs) / math.sqrt(len(xs))
    return m, m - h, m + h


def paired_bootstrap(a: list[float], b: list[float], n: int = 5000, seed: int = 0):
    """95% CI of mean(a - b) over paired questions."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ----------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--out", default="benchmarks/runs/outcome_transfer")
    ap.add_argument("--quick", action="store_true", help="1 epoch, global+ranker only")
    ap.add_argument(
        "--static-baseline",
        action="store_true",
        help="also run the gold-tuned static re-weighting baseline",
    )
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    methods = ["global", "ranker"] if a.quick else a.methods.split(",")
    epochs = 1 if a.quick else a.epochs
    cache = ResponseCache(out / "embed_cache.db")
    emb = OllamaEmbedder(EMBED_MODEL, OLLAMA, cache=cache)
    data = load(Path("benchmarks/data/locomo10.json"))
    base = build_base(out, data, cache)
    turns = json.loads((out / "memory_turns.json").read_text(encoding="utf-8"))
    qs = questions(data)
    convs = sorted(qs)
    results: list[dict[str, Any]] = []
    rows_path = out / "per_question.jsonl"
    rows_path.unlink(missing_ok=True)

    def record(seed, method, cond, split_name, agg, per_q, extra):
        results.append(
            {"seed": seed, "method": method, "condition": cond, "split": split_name, **agg, **extra}
        )
        with rows_path.open("a", encoding="utf-8") as f:
            for p in per_q:
                f.write(
                    json.dumps(
                        {
                            "seed": seed,
                            "method": method,
                            "condition": cond,
                            "split": split_name,
                            **p,
                        }
                    )
                    + "\n"
                )

    for seed in range(a.seeds):
        sp = split(convs, qs, seed, emb)
        print(
            f"seed {seed}: train {sp['train_families']} test {sp['test_families']} | "
            f"feedback {len(sp['feedback'])} within {len(sp['within'])} "
            f"test {len(sp['test'])} (dropped {sp['near_duplicates_dropped']} near-dups)",
            flush=True,
        )
        for name in ("within", "test"):
            agg, per_q, ms = evaluate(base, turns, sp[name], "none", emb)
            record(seed, "none", "none", name, agg, per_q, {"search_ms": ms})
        if a.static_baseline:
            weights, tuned = tune_static(base, out, turns, sp, emb, seed)
            for name in ("within", "test"):
                agg, per_q, ms = evaluate(tuned, turns, sp[name], "ranker", emb)
                record(
                    seed,
                    "static-tuned",
                    "none",
                    name,
                    agg,
                    per_q,
                    {"search_ms": ms, "static_weights": weights},
                )
                print(
                    f"  s{seed} static-tuned   {name:6} hit@3 {agg['hit3']:.3f} {weights}",
                    flush=True,
                )
            tuned.unlink(missing_ok=True)
        for method in methods:
            genuine = None
            for cond in CONDITIONS:
                db = out / f"run_s{seed}_{method}_{cond}.db"
                shutil.copy(base, db)
                for suffix in ("-wal", "-shm"):
                    Path(str(db) + suffix).unlink(missing_ok=True)
                rewards, fb_secs, gate = run_feedback(
                    db, turns, sp, method, cond, seed, emb, epochs, genuine
                )
                if cond == "genuine":
                    genuine = rewards
                for name in ("within", "test"):
                    agg, per_q, ms = evaluate(db, turns, sp[name], METHODS[method][2], emb)
                    record(
                        seed,
                        method,
                        cond,
                        name,
                        agg,
                        per_q,
                        {
                            "feedback_seconds": round(fb_secs, 1),
                            "search_ms": ms,
                            "feedback_events": len(rewards),
                            "ranker_gate": gate,
                            "feedback_success_rate": float(np.mean(np.asarray(rewards) > 0)),
                        },
                    )
                    print(
                        f"  s{seed} {method:14} {cond:8} {name:6} hit@3 {agg['hit3']:.3f} "
                        f"mrr {agg['mrr']:.3f} ndcg {agg['ndcg10']:.3f}",
                        flush=True,
                    )
                db.unlink(missing_ok=True)
                for suffix in ("-wal", "-shm"):
                    Path(str(db) + suffix).unlink(missing_ok=True)
        (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    report(out, results)


def report(out: Path, results: list[dict[str, Any]]) -> None:
    rows = [json.loads(x) for x in (out / "per_question.jsonl").read_text().splitlines()]

    def series(method, cond, split_name, metric):
        return [
            r[metric]
            for r in rows
            if r["method"] == method and r["condition"] == cond and r["split"] == split_name
        ]

    def seed_means(method, cond, split_name, metric):
        return [
            r[metric]
            for r in results
            if r["method"] == method and r["condition"] == cond and r["split"] == split_name
        ]

    lines = [
        "| split | method | condition | Hit@3 (95% CI over seeds) | MRR@10 | nDCG@10 | "
        "Δ Hit@3 vs none [95% CI] | Δ Hit@3 vs shuffled [95% CI] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    combos = sorted(
        {(r["method"], r["condition"]) for r in results},
        key=lambda mc: (
            mc[0] != "none",
            mc[0],
            CONDITIONS.index(mc[1]) if mc[1] in CONDITIONS else -1,
        ),
    )
    for split_name in ("within", "test"):
        base = series("none", "none", split_name, "hit3")
        for method, cond in combos:
            h = seed_means(method, cond, split_name, "hit3")
            if not h:
                continue
            m, lo, hi = t_ci(h)
            mrr = statistics.mean(seed_means(method, cond, split_name, "mrr"))
            nd = statistics.mean(seed_means(method, cond, split_name, "ndcg10"))
            cur = series(method, cond, split_name, "hit3")
            d_none = paired_bootstrap(cur, base) if method != "none" else None
            sh = series(method, "shuffled", split_name, "hit3")
            if method == "static-tuned":
                sh = []
            d_sh = (
                paired_bootstrap(cur, sh)
                if cond != "shuffled" and method != "none" and len(sh) == len(cur)
                else None
            )
            fmt = lambda d: "" if d is None else f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"  # noqa: E731
            lines.append(
                f"| {split_name} | {method} | {cond} | {m:.3f} [{lo:.3f}, {hi:.3f}] | "
                f"{mrr:.3f} | {nd:.3f} | {fmt(d_none)} | {fmt(d_sh)} |"
            )
    cost = {}
    for r in results:
        if "feedback_seconds" in r:
            cost.setdefault(r["method"], []).append((r["feedback_seconds"], r["search_ms"]))
    lines += ["", "| method | feedback phase (s, mean) | eval search p50 (ms) |", "|---|---|---|"]
    lines.append(
        f"| none | 0 | {statistics.mean(r['search_ms'] for r in results if r['method'] == 'none'):.1f} |"
    )
    for mth, v in cost.items():
        lines.append(
            f"| {mth} | {statistics.mean(x[0] for x in v):.1f} | "
            f"{statistics.mean(x[1] for x in v):.1f} |"
        )
    text = "\n".join(lines)
    (out / "report.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
