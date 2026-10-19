"""Write path: episodes -> grounded candidates -> reconciled, versioned memories."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from . import prompts
from .grounding import (
    contains_token,
    locate,
    missing_values,
    name_tokens,
    same_proposition,
    value_tokens,
)
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
    check_id,
    clean_text,
    day,
    new_id,
    now,
)

if TYPE_CHECKING:
    from .memory import Memory

log = logging.getLogger("veramem")

DUP_COS = 0.95
MAX_CANDIDATES = 50  # memories one extraction call may create
MAX_MEMORY_CHARS = 2000  # near-identical -> NOOP + extra evidence
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
    extra_evidence: list[Evidence] = field(default_factory=list)
    neighbours: list[tuple[MemoryItem, float]] = field(default_factory=list)


_ATTRIBUTED = ("according to", "the assistant", "an assistant", "the agent")
_LOWER_OK = {"the", "a", "an", "this", "that", "these", "those", "it", "its", "there", "their"}


def _attribute(text: str, ep: Episode) -> str:
    """Prefix a non-user claim with its source unless the model already attributed it."""
    if any(a in text.lower() for a in _ATTRIBUTED):
        return text
    first = text.split(" ", 1)[0]
    body = text[0].lower() + text[1:] if first.lower() in _LOWER_OK else text
    if ep.source is SourceType.ASSISTANT:
        return f"The assistant stated that {body}"
    name = ep.actor or {SourceType.TOOL: "a tool", SourceType.DOCUMENT: "a document"}.get(
        ep.source, "another source"
    )
    return f"According to {name}, {body}"


def _entities(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [e.strip()[:80] for e in value if isinstance(e, str) and e.strip()][:8]


def normalise_messages(messages: Any, default_role: str = "user") -> list[dict[str, Any]]:
    """Validate and normalise `add()` input. Raises ValueError on anything malformed."""
    if isinstance(messages, str):
        return [{"role": default_role, "content": clean_text(messages)}]
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, (list, tuple)):
        raise ValueError("messages must be a string, a message dict, or a list of them")
    out = []
    for m in messages:
        if isinstance(m, str):
            out.append({"role": default_role, "content": clean_text(m)})
            continue
        if not isinstance(m, dict):
            raise ValueError("each message must be a string or a dict")
        role = m.get("role", default_role)
        if not isinstance(role, str):
            raise ValueError("message role must be a string")
        _role(role)  # unknown roles are rejected, never trusted as the user
        content = m.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        name = m.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError("message name must be a string")
        out.append(
            {
                **m,
                "role": role,
                "content": clean_text(content),
                **({"name": clean_text(name)} if name is not None else {}),
            }
        )
    return out


_ROLES = {
    **dict.fromkeys(("user", "human"), SourceType.USER),
    **dict.fromkeys(("assistant", "ai", "agent", "model"), SourceType.ASSISTANT),
    **dict.fromkeys(("tool", "function", "observation"), SourceType.TOOL),
    **dict.fromkeys(("document", "doc", "file", "web"), SourceType.DOCUMENT),
    "system": SourceType.SYSTEM,
}


def _role(role: str) -> SourceType:
    """Map a message role to its source. Unknown roles raise: authority is never guessed."""
    r = (role or "user").strip().lower()
    if r not in _ROLES:
        raise ValueError(
            f"unknown message role {role!r}; use user, assistant, tool, document or system"
        )
    return _ROLES[r]


# plausible event times: 1900-01-01 .. 2200-01-01
MIN_TS, MAX_TS = -2208988800.0, 7258118400.0


def _valid_ts(t: float) -> bool:
    return math.isfinite(t) and MIN_TS <= t <= MAX_TS


def _ts(value: Any, fallback: float, strict: bool = False) -> float:
    """Parse a timestamp. Invalid/out-of-range values give `fallback`, or raise if strict."""
    if value is None:
        return fallback
    t: float | None = None
    if isinstance(value, bool):
        t = None
    elif isinstance(value, (int, float)):
        t = float(value)
    elif isinstance(value, datetime):
        t = value.timestamp()
    else:
        s = str(value).strip()[:64]
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
                t = dt.timestamp()
                break
            except ValueError:
                continue
        if t is None:
            try:
                t = datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
            except ValueError:
                t = None
    if t is not None and _valid_ts(t):
        return t
    if strict:
        raise ValueError(f"invalid or out-of-range timestamp: {str(value)[:40]!r}")
    return fallback


def _fmt(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return "unknown date"


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
        for value, what in (
            (user_id, "user_id"),
            (agent_id, "agent_id"),
            (session_id, "session_id"),
            (team_id, "team_id"),
        ):
            check_id(value, what)
        msgs = normalise_messages(messages)
        base_ts = _ts(timestamp, now(), strict=True)
        episodes: list[Episode] = []
        for i, msg in enumerate(msgs):
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            occurred = _ts(msg.get("timestamp"), base_ts + i * 1e-3, strict=True)
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
        with self.m.store.tx():
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
            with self.m.store.tx():
                raw = [
                    self._raw_memory(ep, v, scope, resources)
                    for ep, v in zip(episodes, vecs)
                    if ep.source in (SourceType.USER, SourceType.SYSTEM)
                ]
            return AddResult(episodes=[e.id for e in episodes], changes=raw)

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
        prev: list[Episode] = []
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
        if not isinstance(items, list):
            items = []  # model output is untrusted: wrong shapes are ignored, never raised
        cands: list[Candidate] = []
        rejected: list[dict[str, Any]] = []
        for it in items[:MAX_CANDIDATES]:
            if not isinstance(it, dict) or not isinstance(it.get("text"), str):
                continue
            if not it["text"].strip() or len(it["text"]) > MAX_MEMORY_CHARS:
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
            text = it["text"].strip()
            # Value provenance (AuthMem "MIX"): the label follows the source that supplies
            # the claim's values, not the message whose words happen to be quoted.
            cited = f"{ep.content}\n{ep.actor or ''}"
            pool = [e for e in [*batch, *prev] if e.id != ep.id]
            needed = {
                t for t in (value_tokens(text) | name_tokens(text)) if not contains_token(t, cited)
            }
            suppliers: dict[str, Episode] = {}
            unsupported = set()
            for t in needed:
                found = [e for e in pool if contains_token(t, f"{e.content}\n{e.actor or ''}")]
                if found:
                    weakest = min(found, key=lambda e: Authority.for_source(e.source).rank)
                    suppliers[weakest.id] = weakest
                elif t in value_tokens(text):
                    unsupported.add(t)
            if unsupported:
                rejected.append(
                    {
                        "text": text,
                        "reason": "values not found in any source: "
                        + ", ".join(sorted(unsupported)),
                    }
                )
                continue
            authority = Authority.weakest(
                [Authority.for_source(ep.source)]
                + [Authority.for_source(e.source) for e in suppliers.values()]
            )
            extra = [
                Evidence(
                    episode_id=e.id,
                    quote=e.content[:200],
                    start=0,
                    end=min(200, len(e.content)),
                    source=e.source,
                )
                for e in suppliers.values()
                if Authority.for_source(e.source).rank < Authority.for_source(ep.source).rank
            ]
            if authority is not Authority.AUTHORIZED and kind is Kind.PREFERENCE:
                # a non-user source cannot establish a user preference (AuthMem R2P / S2D)
                kind = Kind.FACT
            if authority is not Authority.AUTHORIZED:
                # the label is enforced above; the wording must not read as the user's own
                weakest_ep = min(
                    [ep, *suppliers.values()], key=lambda e: Authority.for_source(e.source).rank
                )
                text = _attribute(text, weakest_ep)
            cands.append(
                Candidate(
                    text=text,
                    kind=kind,
                    evidence=Evidence(
                        episode_id=ep.id,
                        quote=ep.content[start:end] if score < 1 else quote,
                        start=start,
                        end=end,
                        source=ep.source,
                    ),
                    authority=authority,
                    entities=_entities(it.get("entities")),
                    valid_from=valid_from,
                    extra_evidence=extra,
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
        batch: list[tuple[np.ndarray, str]] = []
        with self.m.store.tx():
            for c, v in zip(cands, vecs):
                c.vec = v
                # dedup within this batch
                if any(float(v @ b) > DUP_COS and same_proposition(c.text, t) for b, t in batch):
                    changes.append(
                        Change(op=Operation.NOOP, text=c.text, reason="duplicate in batch")
                    )
                    continue
                batch.append((v, c.text))
                if len(ids):
                    sims = mat @ v
                    order = np.argsort(-sims)[:NEIGHBOURS]
                    c.neighbours = [
                        (n, float(sims[i]))
                        for i in order
                        if sims[i] > 0.3 and (n := self.m.store.get_memory(ids[i])) is not None
                    ]
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
                if top and top[1] > DUP_COS and same_proposition(top[0].text, c.text):
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
        if not isinstance(ops, list):
            ops = []
        by_fact: dict[int, dict[str, Any]] = {}
        for op in ops:
            if not isinstance(op, dict):
                continue
            try:
                idx = int(op.get("fact"))  # type: ignore[arg-type]
            except (TypeError, ValueError, OverflowError):
                continue
            if 0 <= idx < len(cands):
                by_fact.setdefault(idx, op)
        changes: list[Change] = []
        with self.m.store.tx():
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
                    changes.append(
                        Change(op=Operation.NOOP, text=c.text, reason="judged irrelevant")
                    )
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
            evidence=[c.evidence, *c.extra_evidence],
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
            assert ch.memory_id is not None
            self.m.store.update_memory(ch.memory_id, metadata={"conflicts_with": [old.id]})
            ch.reason = f"kept separate: {c.authority.value} source cannot update {old.authority.value} memory"
            return ch
        # The merged text is model output: every value in it must come from the new
        # evidence, the old memory or their sources. Otherwise keep the grounded statement.
        if text.strip().lower() != c.text.strip().lower():
            ep_ids = [e.episode_id for e in [c.evidence, *c.extra_evidence, *old.evidence]]
            sources = [c.text, old.text] + [
                e.content for e in self.m.store.get_episodes(ep_ids).values()
            ]
            if missing_values(text, sources):
                text = c.text
        # No authority upgrade: if the merged text still carries the old content, the
        # weakest authority among the retained evidence wins.
        merged_keeps_old = text.strip().lower() != c.text.strip().lower()
        auth = Authority.weakest([c.authority, old.authority]) if merged_keeps_old else c.authority
        evidence = [c.evidence, *c.extra_evidence] + (old.evidence if merged_keeps_old else [])
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
        self, ep: Episode, vec: np.ndarray, scope: dict[str, Any], resources: list[str] | None
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
            vec=vec,  # same text as the episode: reuse its embedding
        )
        return self._add(c, scope, resources, scope["user_id"])


def iso(ts: float | None) -> str | None:
    return None if ts is None else day(ts)
