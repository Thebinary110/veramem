"""Gap A: outcome-weighted memory ranking and retention (no RL training).

What this is, and is not: VeraMem re-ranks and retains memories according to their
observed usefulness. It does not learn which memories to *write*, update or delete;
those decisions are prompted. See docs/OUTCOME_LEARNING.md for what transfers to new
tasks and what does not.

The papers learn a memory policy with RL (Memory-R1: EM reward through a frozen
answer agent; AgeMem: task + context + memory rewards with step-wise GRPO). VeraMem
keeps the same *signals* at inference time:

* every retrieval / answer is a trace; `feedback()` turns a task outcome into
  fractional Beta updates on the memories that were actually cited (credit
  assignment by citation, falling back to the retrieved set);
* utility re-weights ranking, and `consolidate()` archives memories that keep
  failing, merges duplicates and writes session summaries (AgeMem STM -> LTM);
* `learn()` distils a successful (or instructive failed) trajectory into a reusable
  experience and reconciles it with existing ones (ShareMem stage 1);
* `export_trajectories()` emits GRPO-ready records so the manager / answer models
  can be fine-tuned later on a user's own data.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from . import prompts, ranking
from .grounding import same_proposition
from .ledger import content_hash
from .providers.llm import LLMError
from .types import (
    Authority,
    Evidence,
    Kind,
    MemoryItem,
    SourceType,
    Status,
    Visibility,
    now,
)
from .write import normalise_messages

if TYPE_CHECKING:
    from .memory import Memory

ARCHIVE_UTILITY = 0.3
ARCHIVE_MIN_USES = 3
ADMIT_REWARD = 0.8
GATE_MARGIN = 0.01  # ranker must beat retrieval-relevance AUC by this much to deploy  # ShareMem: only well-rewarded trajectories become experiences
DUP_COS = 0.95


class Learner:
    def __init__(self, mem: Memory):
        self.m = mem

    # ------------------------------------------------------------ feedback
    def feedback(
        self,
        trace_id: str,
        reward: float,
        actor: str | None = None,
        cited: list[str] | None = None,
    ) -> dict[str, Any]:
        reward = max(-1.0, min(1.0, float(reward)))
        tr = self.m.store.get_trace(trace_id)
        if tr is None:
            raise KeyError(f"unknown trace {trace_id}")
        if tr["outcome"] is not None:
            raise ValueError(f"trace {trace_id} already has feedback")
        if cited is not None:
            # citation lock: only memories this trace actually returned can be credited
            cited = [i for i in cited if i in set(tr["memory_ids"]) and i.startswith("mem_")]
        else:
            cited = [i for i in tr["payload"].get("cited", []) if i.startswith("mem_")]
        pool = cited or [i for i in tr["memory_ids"] if i.startswith("mem_")][:5]
        weight = 1.0 if cited else 1.0 / max(1, len(pool)) ** 0.5
        pos, neg = weight * (1 + reward) / 2, weight * (1 - reward) / 2
        with self.m.store._lock:
            for mid in pool:
                self.m.store.db.execute(
                    "UPDATE memories SET successes=successes+?, failures=failures+? WHERE id=?",
                    (pos, neg, mid),
                )
        t = now()
        self.m.store.put_outcomes([(mid, trace_id, tr["qvec"], pos, neg, t) for mid in pool])
        tr["payload"]["credited"] = pool
        self.m.store.update_trace_payload(trace_id, tr["payload"])
        self.m.store.set_trace_outcome(trace_id, reward, t)
        self.m.ledger.append(
            "FEEDBACK", trace_id, {"reward": reward, "credited": pool}, actor=actor
        )
        return {"trace_id": trace_id, "reward": reward, "credited": pool}

    # --------------------------------------------------------- consolidate
    def consolidate(
        self,
        user_id: str | None = None,
        summarize: bool = True,
        archive: bool = True,
        dedupe: bool = True,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {"archived": [], "merged": [], "summaries": []}
        scope_sql, scope_params = ("user_id IS ?", [user_id]) if user_id else ("1=1", [])
        if archive:
            for m in self.m.store.memories(
                f"status='active' AND uses>=? AND {scope_sql}", [ARCHIVE_MIN_USES, *scope_params]
            ):
                if m.successes + m.failures >= 1 and m.utility < ARCHIVE_UTILITY:
                    self.m.store.update_memory(m.id, status=Status.ARCHIVED, updated_at=now())
                    self.m.ledger.append("ARCHIVE", m.id, {"utility": round(m.utility, 3)})
                    report["archived"].append(m.id)
        if dedupe:
            report["merged"] = self._dedupe(scope_sql, scope_params)
        if summarize and self.m.has_llm:
            report["summaries"] = self._summarize_sessions(user_id)
        return report

    def _dedupe(self, scope_sql: str, scope_params: list[Any]) -> list[str]:
        merged: list[str] = []
        owners = self.m.store.db.execute(
            f"SELECT DISTINCT user_id, agent_id, visibility, team_id FROM memories "
            f"WHERE status='active' AND {scope_sql}",
            scope_params,
        ).fetchall()
        for o in owners:
            where = (
                "m.status='active' AND m.user_id IS ? AND m.agent_id IS ? "
                "AND m.visibility=? AND m.team_id IS ?"
            )
            ids, mat = self.m.store.memory_vectors(where, list(o))
            if len(ids) < 2:
                continue
            sims = mat @ mat.T
            np.fill_diagonal(sims, 0)
            gone: set[str] = set()
            for i, j in zip(*np.where(np.triu(sims) > DUP_COS)):
                a, b = ids[int(i)], ids[int(j)]
                if a in gone or b in gone:
                    continue
                ma, mb = self.m.store.get_memory(a), self.m.store.get_memory(b)
                if not ma or not mb or ma.kind != mb.kind:
                    continue
                if not same_proposition(ma.text, mb.text):
                    continue  # similar wording, different value: not a duplicate
                keep, drop = (ma, mb) if ma.authority.rank >= mb.authority.rank else (mb, ma)
                self.m.store.add_evidence(keep.id, drop.evidence)
                self.m.store.update_memory(
                    drop.id, status=Status.SUPERSEDED, superseded_by=keep.id, updated_at=now()
                )
                self.m.ledger.append("MERGE", keep.id, {"absorbed": drop.id})
                gone.add(drop.id)
                merged.append(drop.id)
        return merged

    def _summarize_sessions(self, user_id: str | None) -> list[str]:
        where = "e.session_id IS NOT NULL" + (" AND e.user_id IS ?" if user_id else "")
        rows = self.m.store.db.execute(
            f"SELECT session_id, user_id, agent_id, COUNT(*) n FROM episodes e WHERE {where} "
            f"GROUP BY session_id, user_id, agent_id HAVING n>=4",
            [user_id] if user_id else [],
        ).fetchall()
        out = []
        for r in rows:
            key = f"summary:{r['user_id']}:{r['session_id']}"
            if self.m.store.meta_get(key):
                continue
            eps = self.m.store.episodes(
                "e.session_id=? AND e.user_id IS ?", [r["session_id"], r["user_id"]]
            )
            convo = "\n".join(
                f"({time.strftime('%Y-%m-%d', time.gmtime(e.occurred_at))}) "
                f"{e.actor or e.source.value}: {e.content[:600]}"
                for e in eps
            )[:24000]
            try:
                d = self.m.llm.json(
                    [
                        {"role": "system", "content": prompts.SUMMARY_SYSTEM},
                        {"role": "user", "content": convo},
                    ],
                    max_tokens=700,
                )
            except LLMError:
                continue
            text = str(d.get("summary", "")).strip() if isinstance(d, dict) else ""
            if not text:
                continue
            title = str(d.get("title", "")).strip()
            item = MemoryItem(
                text=(f"{title}: " if title else "") + text,
                kind=Kind.SUMMARY,
                user_id=r["user_id"],
                agent_id=r["agent_id"],
                authority=Authority.weakest([Authority.for_source(e.source) for e in eps]),
                evidence=[
                    Evidence(
                        episode_id=e.id,
                        quote=e.content[:120],
                        start=0,
                        end=min(120, len(e.content)),
                        source=e.source,
                    )
                    for e in eps
                ],
                valid_from=eps[0].occurred_at,
                metadata={"session": r["session_id"]},
            )
            self.m.store.put_memory(item, self.m.embedder.embed_one(item.text))
            self.m.store.meta_set(key, item.id)
            self.m.ledger.append(
                "SUMMARY",
                item.id,
                {
                    "text_sha256": content_hash(item.text),
                    "session": r["session_id"],
                    "episodes": len(eps),
                },
            )
            out.append(item.id)
        return out

    # --------------------------------------------------------------- learn
    def learn(
        self,
        task: str,
        trajectory: Any,
        reward: float,
        *,
        user_id: str | None,
        agent_id: str | None,
        scope: str | None = None,
        shareable: bool = False,
    ) -> dict[str, Any]:
        """Turn a finished task into (at most) one reusable experience."""
        from .types import new_id

        session = new_id("task")
        msgs = normalise_messages(trajectory, default_role="assistant")
        res = self.m.writer.add(
            [{"role": "user", "content": f"TASK: {task}"}, *msgs],
            user_id=user_id,
            agent_id=agent_id,
            session_id=session,
            team_id=None,
            infer=False,
            timestamp=None,
            metadata={"trajectory": True},
            resources=None,
            actor=user_id or agent_id,
            episodes_only=True,
        )
        episodes = [ep for e in res.episodes if (ep := self.m.store.get_episode(e)) is not None]
        out: dict[str, Any] = {"episodes": res.episodes, "action": "none", "experience": None}
        if not self.m.has_llm or (0 <= reward < ADMIT_REWARD) or (-0.5 < reward < 0):
            out["reason"] = "outcome not decisive enough to learn from"
            return out
        transcript = "\n".join(f"{e.source.value}: {e.content[:800]}" for e in episodes)[:20000]
        verdict = "SUCCESS" if reward >= ADMIT_REWARD else "FAILURE (describe what to avoid)"
        try:
            d = self.m.llm.json(
                [
                    {"role": "system", "content": prompts.EXPERIENCE_SYSTEM},
                    {
                        "role": "user",
                        "content": f"OUTCOME: {verdict}, reward={reward}\n\n{transcript}",
                    },
                ],
                max_tokens=900,
            )
        except LLMError as e:
            out["reason"] = f"distillation failed: {e}"
            return out
        content = str(d.get("experience", "")).strip() if isinstance(d, dict) else ""
        if not content:
            out["reason"] = "no transferable lesson"
            return out
        scope = scope or (str(d.get("scope", "")).strip() if isinstance(d, dict) else "") or None
        action, item = self._manage_experience(
            content, scope, user_id, agent_id, episodes, reward, shareable, Visibility.PRIVATE, None
        )
        out.update(action=action, experience=item.id if item else None)
        return out

    def _manage_experience(
        self,
        content: str,
        scope: str | None,
        user_id: str | None,
        agent_id: str | None,
        episodes: list,
        reward: float,
        shareable: bool,
        visibility: Visibility,
        team_id: str | None,
        extra_meta: dict[str, Any] | None = None,
    ):
        if visibility is Visibility.PRIVATE:
            where = "m.status='active' AND m.kind='experience' AND m.user_id IS ? AND m.visibility='private'"
            params: list[Any] = [user_id]
        else:
            where = (
                "m.status='active' AND m.kind='experience' AND m.visibility=? AND m.team_id IS ?"
            )
            params = [visibility.value, team_id]
        v = self.m.embedder.embed_one(content)
        ids, mat = self.m.store.memory_vectors(where, params)
        neigh: list[MemoryItem] = []
        if len(ids):
            order = np.argsort(-(mat @ v))[:5]
            neigh = [x for x in (self.m.store.get_memory(ids[i]) for i in order) if x]
        action, target_id, final = "add", None, content
        if neigh:
            listing = "\n\n".join(f"<<id {i}>>\n{n.text}" for i, n in enumerate(neigh))
            try:
                d = self.m.llm.json(
                    [
                        {"role": "system", "content": prompts.EXPERIENCE_MANAGER_SYSTEM},
                        {
                            "role": "user",
                            "content": f"EXISTING:\n{listing}\n\nCANDIDATE:\n{content}",
                        },
                    ],
                    max_tokens=900,
                )
                action = str(d.get("action", "add")).lower()
                if action == "replace":
                    try:
                        target_id = neigh[int(str(d.get("id")).strip())].id
                    except (ValueError, IndexError, TypeError):
                        action = "add"
                final = str(d.get("content") or content).strip()
            except LLMError:
                action = "add"
        if action == "none":
            return "none", None
        src = [e.source for e in episodes] or [SourceType.SHARED]
        item = MemoryItem(
            text=final,
            kind=Kind.EXPERIENCE,
            user_id=user_id,
            agent_id=agent_id,
            team_id=team_id,
            visibility=visibility,
            authority=Authority.UNENDORSED
            if visibility is not Visibility.PRIVATE
            else Authority.weakest([Authority.for_source(s) for s in src]),
            evidence=[
                Evidence(
                    episode_id=e.id,
                    quote=e.content[:160],
                    start=0,
                    end=min(160, len(e.content)),
                    source=e.source,
                )
                for e in episodes
            ],
            successes=max(reward, 0.0),
            failures=max(-reward, 0.0),
            uses=1,
            shareable=shareable,
            supersedes=target_id,
            metadata={"scope": scope, "reward": reward, **(extra_meta or {})},
        )
        self.m.store.put_memory(item, self.m.embedder.embed_one(final))
        if target_id:
            self.m.store.update_memory(
                target_id,
                status=Status.SUPERSEDED,
                superseded_by=item.id,
                valid_to=now(),
                updated_at=now(),
            )
        self.m.ledger.append(
            "EXPERIENCE",
            item.id,
            {
                "text_sha256": content_hash(final),
                "action": action,
                "replaces": target_id,
                "visibility": visibility.value,
                "reward": reward,
            },
        )
        return ("replace" if target_id else "add"), item

    # --------------------------------------------------------------- ranker
    def training_examples(
        self, user_ids: list[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """(features, labels, trace ids) from rated traces, using the features logged at
        retrieval time, so later outcomes never leak into an example."""
        where, params = "outcome IS NOT NULL AND outcome != 0", []
        if user_ids:
            where += f" AND user_id IN ({','.join('?' * len(user_ids))})"
            params = list(user_ids)
        xs, ys, tids = [], [], []
        for t in self.m.store.traces(where, params):
            feats = t["payload"].get("features") or {}
            for mid in t["payload"].get("credited", []):
                fv = feats.get(mid)
                if fv is None or len(fv) != len(ranking.FEATURES):
                    continue
                xs.append(fv)
                ys.append(1.0 if t["outcome"] > 0 else 0.0)
                tids.append(t["id"])
        if not xs:
            return np.zeros((0, len(ranking.FEATURES))), np.zeros(0), []
        return np.asarray(xs, dtype=float), np.asarray(ys), tids

    def train_ranker(
        self,
        user_ids: list[str] | None = None,
        min_examples: int = 30,
        l2: float = 1.0,
        save: bool = True,
    ) -> dict[str, Any]:
        x, y, tids = self.training_examples(user_ids)
        if len(y) < min_examples or len(set(y.tolist())) < 2:
            return {
                "trained": False,
                "examples": int(len(y)),
                "reason": f"need >= {min_examples} examples with both outcomes",
            }
        # Grouped cross-validation picks the regularisation; the model is deployed only if
        # its out-of-fold log-loss beats the relevance-only model with 95% confidence
        # (bootstrap over traces). Uninformative feedback therefore cannot take over ranking.
        cv = ranking.cross_validate(x, y, tids)
        gain = cv["null_loss"] - cv["model_loss"]  # > 0 means the model predicts better
        uniq = sorted(set(tids))
        idx = {t: i for i, t in enumerate(uniq)}
        per_trace = np.zeros(len(uniq))
        np.add.at(per_trace, [idx[t] for t in tids], gain)
        rng = np.random.default_rng(0)
        boots = per_trace[rng.integers(0, len(uniq), size=(2000, len(uniq)))].sum(axis=1)
        lower = float(np.percentile(boots, 5)) / len(y)
        active = bool(lower > 0)
        model = ranking.LogisticRanker.fit(x, y, l2=cv["best_l2"])  # final model on all data
        model.active = active
        model.metrics = {
            "active": active,
            "reason": (
                "beats the relevance-only model on held-out feedback (95% bound)"
                if active
                else "no reliable gain over retrieval relevance; keeping baseline ranking"
            ),
            "cv_gain_per_example": float(gain.mean()),
            "cv_gain_lower_95": lower,
            "l2": cv["best_l2"],
            "examples": int(len(y)),
            "positive_rate": float(y.mean()),
        }
        if save:
            self.m.store.meta_set("ranker", model.to_json())
            self.m.ledger.append("TRAIN_RANKER", None, model.metrics)
        self.m.ranker = model
        return {"trained": True, **model.metrics, "coefficients": model.explain()}

    # ---------------------------------------------------------- action gate
    def check_action(self, memory_ids: list[str]) -> dict[str, Any]:
        """AuthMem-Bench Module B: may these memories bind a protected tool argument?

        Only Authorized memories may. Attested / Unendorsed ones require the agent to
        confirm with the user first (blanket distrust would destroy task success).
        """
        needs, missing = [], []
        for mid in memory_ids:
            m = self.m.store.get_memory(mid)
            if m is None:
                missing.append(mid)
            elif m.authority is not Authority.AUTHORIZED or m.status is not Status.ACTIVE:
                needs.append(
                    {
                        "id": mid,
                        "authority": m.authority.value,
                        "status": m.status.value,
                        "text": m.text,
                    }
                )
        return {
            "allowed": not needs and not missing,
            "needs_confirmation": needs,
            "missing": missing,
        }

    # ------------------------------------------------------ trajectories
    def export_trajectories(self, path: str, user_id: str | None = None) -> int:
        where = "outcome IS NOT NULL" + (" AND user_id IS ?" if user_id else "")
        answered = self.m.store.traces(where, [user_id] if user_id else [])
        credit: dict[str, list[float]] = {}
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for t in answered:
                for mid in t["payload"].get("cited", []) or t["memory_ids"]:
                    credit.setdefault(mid, []).append(t["outcome"])
                if t["kind"] != "answer":
                    continue
                mems = [self.m.store.get_memory(i) for i in t["memory_ids"] if i.startswith("mem_")]
                f.write(
                    json.dumps(
                        {
                            "schema": "veramem.answer.v1",
                            "trace_id": t["id"],
                            "question": t["query"],
                            "retrieved": [
                                {
                                    "id": m.id,
                                    "text": m.text,
                                    "authority": m.authority.value,
                                    "date": m.valid_from,
                                }
                                for m in mems
                                if m
                            ],
                            "selected": t["payload"].get("cited", []),
                            "answer": t["payload"].get("answer"),
                            "reward": t["outcome"],
                        }
                    )
                    + "\n"
                )
                n += 1
            for t in self.m.store.traces(
                "kind='manage'" + (" AND user_id IS ?" if user_id else ""),
                [user_id] if user_id else [],
            ):
                rewards = [r for mid in t["memory_ids"] for r in credit.get(mid, [])]
                if not rewards:
                    continue
                f.write(
                    json.dumps(
                        {
                            "schema": "veramem.manager.v1",
                            "trace_id": t["id"],
                            "existing": t["payload"]["existing"],
                            "new": t["payload"]["new"],
                            "ops": t["payload"]["ops"],
                            "reward": float(np.mean(rewards)),
                        }
                    )
                    + "\n"
                )
                n += 1
        return n


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank-based ROC AUC (probability a positive outranks a negative)."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
