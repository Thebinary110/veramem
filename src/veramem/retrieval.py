"""Read path: access filter -> routed, hybrid candidate generation -> fused ranking.

* Access is enforced in SQL before anything is scored (Collaborative Memory / AIM).
* Granularity routing follows AMA: fine-grained questions go to raw episodes,
  abstract ones to summaries, multi-entity ones to both facts and raw ("dual").
* Dense + BM25 + entity channels are fused with reciprocal rank fusion (k=60,
  Agent Zero Memory), then re-weighted by learned utility (Memory-R1 / AgeMem outcome
  signal) and, for temporal questions, re-ordered chronologically.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from .policy import Principal
from .types import (
    Authority,
    Episode,
    Evidence,
    Hit,
    Kind,
    MemoryItem,
    SearchResult,
    Visibility,
    new_id,
    now,
)

if TYPE_CHECKING:
    from .memory import Memory

RRF_K = 60
POOL = 60
UTILITY_WEIGHT = 0.6

_TEMPORAL = re.compile(
    r"\b(when|what (date|time|day|year|month)|how long|since|until|before|after|latest|"
    r"most recent(ly)?|currently|current|now|nowadays|still|first|last time|ago|recently|"
    r"which year|what year|anymore|used to)\b",
    re.I,
)
_FINE = re.compile(
    r"\b(exact(ly)?|word for word|verbatim|quote|what did .{1,30} say|"
    r"precisely|specific(ally)?|name of|title of|how many|how much)\b",
    re.I,
)
_ABSTRACT = re.compile(
    r"\b(summari[sz]e|overall|in general|overview|tell me about|"
    r"describe|what kind of|personality|relationship between)\b",
    re.I,
)
_CAP = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_QWORDS = {
    "What",
    "When",
    "Where",
    "Who",
    "Why",
    "How",
    "Which",
    "Did",
    "Does",
    "Do",
    "Is",
    "Are",
    "Was",
    "Were",
    "Has",
    "Have",
    "Can",
    "Could",
    "Would",
    "Should",
    "Will",
    "The",
}


def route(query: str) -> str:
    """Rule-based AMA router: no LLM call on the hot path."""
    ents = [e for e in _CAP.findall(query) if e not in _QWORDS]
    multi = len(set(ents)) >= 2 or bool(re.search(r"\b(both|compare|and)\b", query, re.I) and ents)
    if _ABSTRACT.search(query):
        return "summary"
    if _FINE.search(query) and multi:
        return "dual"
    if _FINE.search(query):
        return "raw"
    if multi:
        return "dual"
    return "fact"


def is_temporal(query: str) -> bool:
    return bool(_TEMPORAL.search(query))


def query_entities(query: str) -> list[str]:
    ents = [e for e in _CAP.findall(query) if e not in _QWORDS]
    words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", query)]
    return list(dict.fromkeys([e.lower() for e in ents] + words))[:12]


def _rrf(rank_lists: list[list[str]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for r, i in enumerate(ranks):
            scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
    return scores


def episode_as_memory(ep: Episode) -> MemoryItem:
    return MemoryItem(
        id=ep.id,
        text=(f"{ep.actor}: " if ep.actor else "") + ep.content,
        kind=Kind.EVENT,
        user_id=ep.user_id,
        agent_id=ep.agent_id,
        authority=Authority.for_source(ep.source),
        evidence=[
            Evidence(
                episode_id=ep.id,
                quote=ep.content[:300],
                start=0,
                end=len(ep.content),
                source=ep.source,
            )
        ],
        valid_from=ep.occurred_at,
        created_at=ep.ingested_at,
        metadata={"raw": True, "session": ep.session_id},
    )


class Retriever:
    def __init__(self, mem: Memory):
        self.m = mem

    # ------------------------------------------------------------ memories
    def _memory_where(
        self, p: Principal, kinds: list[Kind] | None, include_shared: bool, as_of: float | None
    ) -> tuple[str, list]:
        where, params = self.m.policy.read_filter(p, include_shared=include_shared)
        if as_of is None:
            where += " AND m.status='active'"
        else:
            where += (
                " AND m.created_at<=? AND m.status IN ('active','superseded') "
                "AND (m.valid_to IS NULL OR m.valid_to>?)"
            )
            params += [as_of, as_of]
        if kinds:
            where += f" AND m.kind IN ({','.join('?' * len(kinds))})"
            params += [k.value for k in kinds]
        return where, params

    def _rank_memories(
        self, query: str, qv: np.ndarray, where: str, params: list, pool: int = POOL
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        signals: dict[str, dict[str, float]] = {}
        ids, mat = self.m.store.memory_vectors(where, params)
        dense: list[str] = []
        if len(ids):
            sims = mat @ qv
            order = np.argsort(-sims)[:pool]
            dense = [ids[i] for i in order]
            for i in order:
                signals.setdefault(ids[i], {})["dense"] = round(float(sims[i]), 4)
        lexical = [r["id"] for r in self.m.store.memory_fts(query, where, params, pool)]
        for r, i in enumerate(lexical):
            signals.setdefault(i, {})["bm25_rank"] = r + 1
        ent = [
            r["id"]
            for r in self.m.store.memories_for_entities(query_entities(query), where, params)
        ][:pool]
        for i in ent:
            signals.setdefault(i, {})["entity"] = 1.0
        return _rrf([dense, lexical, ent]), signals

    # ------------------------------------------------------------- episodes
    def _episode_where(self, p: Principal) -> tuple[str, list]:
        if p.user_id is not None:
            return "e.user_id IS ?", [p.user_id]
        return "e.user_id IS NULL AND e.agent_id IS ?", [p.agent_id]

    def _rank_episodes(
        self, query: str, qv: np.ndarray, p: Principal, pool: int = POOL
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        where, params = self._episode_where(p)
        signals: dict[str, dict[str, float]] = {}
        ids, mat = self.m.store.episode_vectors(where, params)
        dense: list[str] = []
        if len(ids):
            sims = mat @ qv
            order = np.argsort(-sims)[:pool]
            dense = [ids[i] for i in order]
            for i in order:
                signals.setdefault(ids[i], {})["dense"] = round(float(sims[i]), 4)
        lexical = [r["id"] for r in self.m.store.episode_fts(query, where, params, pool)]
        for r, i in enumerate(lexical):
            signals.setdefault(i, {})["bm25_rank"] = r + 1
        return _rrf([dense, lexical]), signals

    # --------------------------------------------------------------- search
    def search(
        self,
        query: str,
        p: Principal,
        limit: int = 10,
        granularity: str = "auto",
        kinds: list[Kind] | None = None,
        include_shared: bool = True,
        as_of: float | None = None,
        record: bool = True,
    ) -> SearchResult:
        g = route(query) if granularity == "auto" else granularity
        temporal = is_temporal(query)
        qv = self.m.embedder.embed_one(query)
        hits: list[Hit] = []

        if g in ("fact", "dual", "summary"):
            k = [Kind.SUMMARY] + (kinds or []) if g == "summary" and kinds else kinds
            where, params = self._memory_where(p, k, include_shared, as_of)
            fused, sig = self._rank_memories(query, qv, where, params)
            if g == "summary":  # coarse level first, facts fill the rest
                where2, params2 = self._memory_where(p, [Kind.SUMMARY], include_shared, as_of)
                f2, s2 = self._rank_memories(query, qv, where2, params2, pool=10)
                for i, v in f2.items():
                    fused[i] = fused.get(i, 0) + v * 2
                    sig.setdefault(i, {}).update(s2.get(i, {}))
            top = max(fused.values(), default=1.0)
            for mid, s in sorted(fused.items(), key=lambda x: -x[1])[: max(limit * 3, 30)]:
                item = self.m.store.get_memory(mid)
                if item is None:
                    continue
                score = s / top
                if item.uses > 0:
                    score *= 1 + UTILITY_WEIGHT * (item.utility - 0.5)
                sig[mid]["utility"] = round(item.utility, 3)
                hits.append(
                    Hit(
                        memory=item,
                        score=round(score, 5),
                        signals=sig[mid],
                        granularity="summary" if item.kind is Kind.SUMMARY else "fact",
                    )
                )

        if g in ("raw", "dual") and as_of is None:
            fused, sig = self._rank_episodes(query, qv, p)
            top = max(fused.values(), default=1.0)
            weight = 1.0 if g == "raw" else 0.85
            for eid, s in sorted(fused.items(), key=lambda x: -x[1])[: max(limit * 2, 20)]:
                ep = self.m.store.get_episode(eid)
                if ep is None:
                    continue
                hits.append(
                    Hit(
                        memory=episode_as_memory(ep),
                        score=round(weight * s / top, 5),
                        signals=sig[eid],
                        granularity="raw",
                    )
                )

        own, shared = [], []
        for h in sorted(hits, key=lambda h: -h.score):
            is_shared = (
                h.memory.visibility is not Visibility.PRIVATE and h.memory.user_id != p.user_id
            )
            if is_shared:
                h.granularity = "shared"
                shared.append(h)
            else:
                own.append(h)
        # ShareMem precedence: shared experience never displaces the caller's own memory
        selected = own[:limit] + shared[: max(2, limit // 4)]
        if temporal:
            selected.sort(
                key=lambda h: (
                    h.granularity == "shared",
                    h.memory.valid_from or h.memory.created_at,
                )
            )

        result = SearchResult(trace_id=new_id("tr"), query=query, hits=selected, granularity=g)
        if record:
            self.m.store.put_trace(
                result.trace_id,
                "search",
                query,
                p.user_id,
                p.agent_id,
                now(),
                [h.memory.id for h in selected],
                {"granularity": g, "temporal": temporal},
            )
            self._touch([h.memory.id for h in selected if h.granularity != "raw"])
        return result

    def _touch(self, ids: list[str]) -> None:
        t = now()
        with self.m.store._lock:
            for i in ids:
                self.m.store.db.execute(
                    "UPDATE memories SET uses=uses+1, last_used_at=? WHERE id=?", (t, i)
                )

    # --------------------------------------------------------- expansions
    def source_episodes(self, hits: list[Hit], p: Principal, neighbours: int = 1) -> list[Hit]:
        """AMA retry ladder: facts -> their source turns -> adjacent turns."""
        seen = {h.memory.id for h in hits}
        out: list[Hit] = []
        own_where, own_params = self._episode_where(p)
        for h in hits:
            for ev in h.memory.evidence:
                ep = self.m.store.get_episode(ev.episode_id)
                if ep is None or ep.id in seen:
                    continue
                if not self.m.store.episodes(f"e.id=? AND {own_where}", [ep.id, *own_params]):
                    continue  # never expand into evidence the caller cannot read
                around = [ep]
                if ep.session_id and neighbours:
                    sess = self.m.store.episodes(
                        f"e.session_id=? AND {own_where}", [ep.session_id, *own_params]
                    )
                    idx = next((i for i, e in enumerate(sess) if e.id == ep.id), None)
                    if idx is not None:
                        around = sess[max(0, idx - neighbours) : idx + neighbours + 1]
                for e in around:
                    if e.id not in seen:
                        seen.add(e.id)
                        out.append(
                            Hit(memory=episode_as_memory(e), score=h.score * 0.9, granularity="raw")
                        )
        return out
