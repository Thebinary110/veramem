"""Write path: episodes -> grounded candidates -> reconciled, versioned memories."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from . import prompts
from .grounding import locate
from .ledger import content_hash
from .providers.llm import LLMError
from .types import (
    AddResult,
    Authority,
    Change,
    Episode,
    Evidence,
    Kind,
    MemoryItem,
    Operation,
    SourceType,
    Status,
    Visibility,
    new_id,
    now,
)

if TYPE_CHECKING:
    from .memory import Memory

log = logging.getLogger("veramem")

DUP_COS = 0.95  # near-identical -> NOOP + extra evidence
NEIGHBOURS = 5
EXTRACT_BATCH = 12
CONTEXT_TURNS = 6


@dataclass
class Candidate:
    text: str
    kind: Kind
    evidence: Evidence
    authority: Authority
    entities: list[str]
    valid_from: float | None
    vec: np.ndarray | None = None
    neighbours: list[tuple[MemoryItem, float]] = field(default_factory=list)


def normalise_messages(messages: Any, default_role: str = "user") -> list[dict[str, Any]]:
    if isinstance(messages, str):
        return [{"role": default_role, "content": messages}]
    if isinstance(messages, dict):
        return [messages]
    out = []
    for m in messages:
        out.append({"role": default_role, "content": m} if isinstance(m, str) else dict(m))
    return out


def _role(role: str) -> SourceType:
    r = (role or "user").lower()
    if r in ("user", "human"):
        return SourceType.USER
    if r in ("assistant", "ai", "agent", "model"):
        return SourceType.ASSISTANT
    if r in ("tool", "function", "observation"):
        return SourceType.TOOL
    if r in ("document", "doc", "file", "web"):
        return SourceType.DOCUMENT
    if r == "system":
        return SourceType.SYSTEM
    return SourceType.USER


def _ts(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    s = str(value).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%I:%M %p on %d %B, %Y",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return fallback


def _fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


class Writer:
    def __init__(self, mem: Memory):
        self.m = mem

    # ------------------------------------------------------------------ add
    def add(
        self,
        messages: Any,
        *,
        user_id: str | None,
        agent_id: str | None,
        session_id: str | None,
        team_id: str | None,
        infer: bool,
        timestamp: Any,
        metadata: dict[str, Any] | None,
        resources: list[str] | None,
        actor: str | None,
        episodes_only: bool = False,
    ) -> AddResult:
        msgs = normalise_messages(messages)
        base_ts = _ts(timestamp, now())
        episodes: list[Episode] = []
        for i, msg in enumerate(msgs):
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            occurred = _ts(msg.get("timestamp"), base_ts + i * 1e-3)
            ep = Episode(
                content=content,
                source=_role(msg.get("role", "user")),
                actor=msg.get("name"),
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                occurred_at=occurred,
                sha256=hashlib.sha256(content.encode()).hexdigest(),
                metadata={**(metadata or {}), **({"resources": resources} if resources else {})},
            )
            episodes.append(ep)
        if not episodes:
            return AddResult(episodes=[], changes=[])

        vecs = self.m.embedder.embed([e.content for e in episodes])
        for ep, v in zip(episodes, vecs):
            self.m.store.put_episode(ep, v)
            self.m.ledger.append(
                "EPISODE",
                ep.id,
                {
                    "sha256": ep.sha256,
                    "source": ep.source.value,
                    "user": user_id,
                    "session": session_id,
                },
                actor=actor or user_id,
            )

        if episodes_only:
            return AddResult(episodes=[e.id for e in episodes], changes=[])
        scope = dict(user_id=user_id, agent_id=agent_id, team_id=team_id)
        if not infer or not self.m.has_llm:
            changes = [
                self._raw_memory(ep, scope, resources)
                for ep in episodes
                if ep.source in (SourceType.USER, SourceType.SYSTEM)
            ]
            return AddResult(episodes=[e.id for e in episodes], changes=changes)

        changes: list[Change] = []
        rejected: list[dict[str, Any]] = []
        for start in range(0, len(episodes), EXTRACT_BATCH):
            batch = episodes[start : start + EXTRACT_BATCH]
            cands, rej = self._extract(batch, session_id, user_id)
            rejected += rej
            if cands:
                changes += self._reconcile(cands, scope, resources, actor or user_id)
        return AddResult(episodes=[e.id for e in episodes], changes=changes, rejected=rejected)

    # -------------------------------------------------------------- extract
    def _extract(
        self, batch: list[Episode], session_id: str | None, user_id: str | None
    ) -> tuple[list[Candidate], list[dict[str, Any]]]:
        ctx = []
        if session_id:
            prev = self.m.store.episodes(
                "session_id=? AND user_id IS ? AND occurred_at<? ",
                (session_id, user_id, batch[0].occurred_at),
            )[-CONTEXT_TURNS:]
            ctx = [
                f"({p.source.value}{'/' + p.actor if p.actor else ''}, {_fmt(p.occurred_at)}) "
                f"{p.content[:500]}"
                for p in prev
            ]
        labels = {f"m{i}": ep for i, ep in enumerate(batch)}
        new_lines = [
            f"[{k}] ({ep.source.value}{'/' + ep.actor if ep.actor else ''}, "
            f"{_fmt(ep.occurred_at)}) {ep.content}"
            for k, ep in labels.items()
        ]
        user_msg = (
            "CONTEXT (reference only):\n"
            + ("\n".join(ctx) or "(none)")
            + "\n\nNEW messages:\n"
            + "\n".join(new_lines)
        )
        try:
            data = self.m.llm.json(
                [
                    {"role": "system", "content": prompts.EXTRACT_SYSTEM},
                    {"role": "user", "content": user_msg},
                ]
            )
        except LLMError as e:
            log.warning("extraction failed, storing raw turns: %s", e)
            return [], [{"reason": f"extraction failed: {e}"}]
        items = data.get("memories", []) if isinstance(data, dict) else data
        cands: list[Candidate] = []
        rejected: list[dict[str, Any]] = []
        for it in items or []:
            if not isinstance(it, dict) or not str(it.get("text", "")).strip():
                continue
            ep = labels.get(str(it.get("source", "")).strip("[] "))
            quote = str(it.get("quote", "")).strip()
            if ep is None:  # model forgot the source: search all new messages for the quote
                ep = next((e for e in batch if locate(quote, e.content)[0] >= 0), None)
            if ep is None:
                rejected.append({"text": it.get("text"), "reason": "unknown source message"})
                continue
            start, end, score = locate(quote, ep.content)
            if start < 0:
                # the quote must exist in the source; second chance: any new message
                alt = next((e for e in batch if locate(quote, e.content)[0] >= 0), None)
                if alt is None:
                    rejected.append(
                        {
                            "text": it.get("text"),
                            "quote": quote,
                            "reason": "quote not found in source (ungrounded)",
                        }
                    )
                    continue
                ep = alt
                start, end, score = locate(quote, ep.content)
            try:
                kind = Kind(str(it.get("kind", "fact")).lower())
            except ValueError:
                kind = Kind.FACT
            if kind not in (Kind.FACT, Kind.PREFERENCE, Kind.EVENT):
                kind = Kind.FACT
            when = it.get("when")
            valid_from = _ts(when, ep.occurred_at) if when and when != "null" else ep.occurred_at
            authority = Authority.for_source(ep.source)
            if authority is not Authority.AUTHORIZED and kind is Kind.PREFERENCE:
                # a non-user source cannot establish a user preference (AuthMem R2P / S2D)
                kind = Kind.FACT
            cands.append(
                Candidate(
                    text=str(it["text"]).strip(),
                    kind=kind,
                    evidence=Evidence(
                        episode_id=ep.id,
                        quote=ep.content[start:end] if score < 1 else quote,
                        start=start,
                        end=end,
                        source=ep.source,
                    ),
                    authority=authority,
                    entities=[str(e) for e in (it.get("entities") or []) if str(e).strip()][:8],
                    valid_from=valid_from,
                )
            )
        return cands, rejected

    # ------------------------------------------------------------ reconcile
    def _scope_where(self, scope: dict[str, Any]) -> tuple[str, list[Any]]:
        # scope-restricted comparison set (AIM §3.5): dedup never looks across owners
        if scope["user_id"] is None:
            return (
                "m.status='active' AND m.visibility='private' AND m.user_id IS NULL AND m.agent_id IS ?",
                [scope["agent_id"]],
            )
        return "m.status='active' AND m.visibility='private' AND m.user_id IS ?", [scope["user_id"]]

    def _reconcile(
        self,
        cands: list[Candidate],
        scope: dict[str, Any],
        resources: list[str] | None,
        actor: str | None,
    ) -> list[Change]:
        vecs = self.m.embedder.embed([c.text for c in cands])
        where, params = self._scope_where(scope)
        ids, mat = self.m.store.memory_vectors(where, params)
        changes: list[Change] = []
        pending: list[Candidate] = []
        batch_vecs: list[np.ndarray] = []
        for c, v in zip(cands, vecs):
            c.vec = v
            # dedup within this batch
            if batch_vecs and max(float(v @ b) for b in batch_vecs) > DUP_COS:
                changes.append(Change(op=Operation.NOOP, text=c.text, reason="duplicate in batch"))
                continue
            batch_vecs.append(v)
            if len(ids):
                sims = mat @ v
                order = np.argsort(-sims)[:NEIGHBOURS]
                c.neighbours = [
                    (self.m.store.get_memory(ids[i]), float(sims[i]))
                    for i in order
                    if sims[i] > 0.3
                ]
                c.neighbours = [(n, s) for n, s in c.neighbours if n is not None]
                # same-entity memories are update candidates even when phrased differently
                seen = {n.id for n, _ in c.neighbours}
                if c.entities and len(c.neighbours) < NEIGHBOURS:
                    id_pos = {mid: i for i, mid in enumerate(ids)}
                    for row in self.m.store.memories_for_entities(c.entities, where, params):
                        if row["id"] in seen or row["id"] not in id_pos:
                            continue
                        n = self.m.store.get_memory(row["id"])
                        if n is not None:
                            c.neighbours.append((n, float(sims[id_pos[row["id"]]])))
                            seen.add(n.id)
                        if len(c.neighbours) >= NEIGHBOURS:
                            break
            top = c.neighbours[0] if c.neighbours else None
            if top and top[1] > DUP_COS:
                changes.append(self._confirm(top[0], c, actor))
                continue
            if not c.neighbours:
                changes.append(self._add(c, scope, resources, actor))
                continue
            pending.append(c)
        if pending:
            changes += self._decide(pending, scope, resources, actor)
        return changes

    def _decide(
        self,
        cands: list[Candidate],
        scope: dict[str, Any],
        resources: list[str] | None,
        actor: str | None,
    ) -> list[Change]:
        # short integer ids for the model (avoids UUID hallucination, as in mem0)
        existing: dict[str, MemoryItem] = {}
        for c in cands:
            for n, _ in c.neighbours:
                if n.id not in {e.id for e in existing.values()}:
                    existing[str(len(existing))] = n
        ex_lines = [
            {
                "id": k,
                "text": m.text,
                "date": _fmt(m.valid_from or m.created_at)[:10],
                "authority": m.authority.value,
            }
            for k, m in existing.items()
        ]
        new_lines = [
            {
                "index": i,
                "text": c.text,
                "authority": c.authority.value,
                "date": _fmt(c.valid_from or now())[:10],
            }
            for i, c in enumerate(cands)
        ]
        try:
            data = self.m.llm.json(
                [
                    {"role": "system", "content": prompts.RECONCILE_SYSTEM},
                    {
                        "role": "user",
                        "content": "EXISTING memories:\n"
                        + json.dumps(ex_lines, indent=0)
                        + "\n\nNEW facts:\n"
                        + json.dumps(new_lines, indent=0),
                    },
                ]
            )
            ops = data.get("operations", []) if isinstance(data, dict) else []
        except LLMError as e:
            log.warning("reconcile failed, defaulting to ADD: %s", e)
            ops = []
        by_fact: dict[int, dict[str, Any]] = {}
        for op in ops:
            try:
                by_fact.setdefault(int(op.get("fact")), op)
            except (TypeError, ValueError):
                continue
        changes: list[Change] = []
        for i, c in enumerate(cands):
            op = by_fact.get(i, {"event": "ADD"})
            event = str(op.get("event", "ADD")).upper()
            target = existing.get(str(op.get("id"))) if op.get("id") is not None else None
            text = str(op.get("text") or c.text).strip()
            if event in ("UPDATE", "DELETE") and target is None:
                event = "ADD"
            if event == "UPDATE" and target is not None:
                changes.append(self._update(target, c, text, scope, resources, actor))
            elif event == "DELETE" and target is not None:
                # non-destructive: retract the old memory, keep the new statement as its
                # successor so "no longer X" stays queryable (Graphiti-style invalidation)
                ch = self._retract(target, c, actor)
                changes.append(ch)
                if ch.op is Operation.DELETE:
                    changes.append(self._add(c, scope, resources, actor, supersedes=target.id))
            elif event in ("NONE", "NOOP") and target is not None:
                changes.append(self._confirm(target, c, actor))
            elif event in ("NONE", "NOOP"):
                changes.append(Change(op=Operation.NOOP, text=c.text, reason="judged irrelevant"))
            else:
                changes.append(self._add(c, scope, resources, actor))
        # replayable manager state for later RL fine-tuning (Memory-R1 manager records)
        self.m.store.put_trace(
            new_id("tr"),
            "manage",
            json.dumps(new_lines),
            scope["user_id"],
            scope["agent_id"],
            now(),
            [ch.memory_id for ch in changes if ch.memory_id],
            {
                "existing": ex_lines,
                "new": new_lines,
                "ops": ops,
                "applied": [ch.model_dump(mode="json") for ch in changes],
            },
        )
        return changes

    # ------------------------------------------------------------ appliers
    def _add(
        self,
        c: Candidate,
        scope: dict[str, Any],
        resources: list[str] | None,
        actor: str | None,
        supersedes: str | None = None,
    ) -> Change:
        item = MemoryItem(
            text=c.text,
            kind=c.kind,
            user_id=scope["user_id"],
            agent_id=scope["agent_id"],
            team_id=scope.get("team_id"),
            visibility=Visibility.PRIVATE,
            authority=c.authority,
            evidence=[c.evidence],
            entities=c.entities,
            valid_from=c.valid_from,
            supersedes=supersedes,
            metadata={"resources": resources} if resources else {},
        )
        self.m.store.put_memory(item, c.vec)
        self.m.ledger.append(
            "ADD",
            item.id,
            {
                "text_sha256": content_hash(item.text),
                "authority": item.authority.value,
                "evidence": [c.evidence.episode_id],
                "supersedes": supersedes,
            },
            actor=actor,
        )
        return Change(op=Operation.ADD, memory_id=item.id, text=item.text)

    def _update(
        self,
        old: MemoryItem,
        c: Candidate,
        text: str,
        scope: dict[str, Any],
        resources: list[str] | None,
        actor: str | None,
    ) -> Change:
        # Authority guard: lower-authority evidence never rewrites a higher-authority memory.
        if c.authority.rank < old.authority.rank:
            ch = self._add(c, scope, resources, actor)
            self.m.store.update_memory(ch.memory_id, metadata={"conflicts_with": [old.id]})
            ch.reason = f"kept separate: {c.authority.value} source cannot update {old.authority.value} memory"
            return ch
        # No authority upgrade: if the merged text still carries the old content, the
        # weakest authority among the retained evidence wins.
        merged_keeps_old = text.strip().lower() != c.text.strip().lower()
        auth = Authority.weakest([c.authority, old.authority]) if merged_keeps_old else c.authority
        evidence = [c.evidence] + (old.evidence if merged_keeps_old else [])
        item = MemoryItem(
            text=text,
            kind=c.kind if c.kind is not Kind.FACT else old.kind,
            user_id=old.user_id,
            agent_id=old.agent_id,
            team_id=old.team_id,
            visibility=old.visibility,
            authority=auth,
            evidence=evidence,
            entities=sorted(set(old.entities) | set(c.entities)),
            valid_from=c.valid_from,
            supersedes=old.id,
            uses=old.uses,
            successes=old.successes,
            failures=old.failures,
            metadata={**old.metadata, **({"resources": resources} if resources else {})},
        )
        vec = self.m.embedder.embed_one(text) if text != c.text else c.vec
        self.m.store.put_memory(item, vec)
        t = now()
        self.m.store.update_memory(
            old.id,
            status=Status.SUPERSEDED,
            superseded_by=item.id,
            valid_to=c.valid_from or t,
            updated_at=t,
        )
        self.m.ledger.append(
            "UPDATE",
            item.id,
            {
                "text_sha256": content_hash(text),
                "previous": old.id,
                "authority": auth.value,
                "evidence": [e.episode_id for e in evidence],
            },
            actor=actor,
        )
        return Change(op=Operation.UPDATE, memory_id=item.id, text=text, previous=old.id)

    def _retract(self, old: MemoryItem, c: Candidate, actor: str | None) -> Change:
        if c.authority.rank < old.authority.rank:
            return Change(
                op=Operation.NOOP,
                memory_id=old.id,
                text=c.text,
                reason=f"{c.authority.value} source cannot retract {old.authority.value} memory",
            )
        t = now()
        self.m.store.update_memory(
            old.id, status=Status.RETRACTED, valid_to=c.valid_from or t, updated_at=t
        )
        self.m.store.add_evidence(old.id, [c.evidence])
        self.m.ledger.append(
            "DELETE",
            old.id,
            {"reason_sha256": content_hash(c.text), "evidence": [c.evidence.episode_id]},
            actor=actor,
        )
        return Change(op=Operation.DELETE, memory_id=old.id, text=old.text, reason=c.text)

    def _confirm(self, old: MemoryItem, c: Candidate, actor: str | None) -> Change:
        """Same proposition seen again: attach evidence; a direct user restatement
        legitimately establishes it with user authority (AuthMem H+ case)."""
        self.m.store.add_evidence(old.id, [c.evidence])
        fields: dict[str, Any] = {"updated_at": now()}
        if c.authority.rank > old.authority.rank:
            fields["authority"] = c.authority
        self.m.store.update_memory(old.id, **fields)
        self.m.ledger.append(
            "CONFIRM",
            old.id,
            {
                "evidence": [c.evidence.episode_id],
                "authority": (fields.get("authority") or old.authority).value,
            },
            actor=actor,
        )
        return Change(op=Operation.NOOP, memory_id=old.id, text=old.text, reason="already known")

    def _raw_memory(
        self, ep: Episode, scope: dict[str, Any], resources: list[str] | None
    ) -> Change:
        c = Candidate(
            text=ep.content,
            kind=Kind.FACT,
            evidence=Evidence(
                episode_id=ep.id, quote=ep.content, start=0, end=len(ep.content), source=ep.source
            ),
            authority=Authority.for_source(ep.source),
            entities=[],
            valid_from=ep.occurred_at,
            vec=self.m.embedder.embed_one(ep.content),
        )
        return self._add(c, scope, resources, scope["user_id"])


def iso(ts: float | None) -> str | None:
    return None if ts is None else time.strftime("%Y-%m-%d", time.gmtime(ts))
