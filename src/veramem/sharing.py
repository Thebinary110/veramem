"""Gap C: safe cross-user experience transfer.

Stage 1 happens in `Learner.learn` (local consolidation of a user's experiences).
Stage 2 here (ShareMem §3.2), hardened beyond the paper:

1. eligibility: owner consent (`shareable`), kind == experience (never preferences,
   never personal facts), utility above a floor (outcome-admitted, Gap A);
2. LLM abstraction into {slots};
3. deterministic gate: PII patterns + canary n-grams from the owner's private memory;
4. independent shared-pool manager (a local op is never replayed verbatim);
5. the shared item is Unendorsed, carries no evidence pointer into the owner's private
   episodes, and records only an HMAC pseudonym of the contributor, so it can be
   withdrawn on erasure without being re-identifiable.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

from . import prompts
from .privacy import check_shareable, pseudonym
from .providers.llm import LLMError
from .types import Kind, Status, Visibility, now

if TYPE_CHECKING:
    from .memory import Memory


class Sharer:
    def __init__(self, mem: Memory):
        self.m = mem

    @property
    def secret(self) -> bytes:
        s = self.m.store.meta_get("pseudonym_secret")
        if not s:
            s = secrets.token_hex(32)
            self.m.store.meta_set("pseudonym_secret", s)
        return bytes.fromhex(s)

    def share(
        self,
        user_id: str,
        team_id: str | None = None,
        min_utility: float = 0.6,
        memory_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        visibility = Visibility.TEAM if team_id else Visibility.PUBLIC
        if team_id and team_id not in self.m.policy.teams(user_id):
            raise PermissionError(f"user {user_id} is not a member of team {team_id}")
        where = (
            "m.status='active' AND m.kind='experience' AND m.visibility='private' "
            "AND m.user_id IS ? AND m.shareable=1"
        )
        params: list[Any] = [user_id]
        if memory_ids:
            where += f" AND m.id IN ({','.join('?' * len(memory_ids))})"
            params += memory_ids
        cands = self.m.store.memories(where, params)
        # canaries = the owner's personal values: facts/preferences/events and ordinary
        # messages. Task statements behind experiences are what the experience generalises.
        private_texts = [
            m.text
            for m in self.m.store.memories(
                "m.user_id IS ? AND m.kind IN ('fact','preference','event')", [user_id]
            )
        ]
        private_texts += [
            e.content
            for e in self.m.store.episodes(
                "e.user_id IS ? AND e.source='user' AND "
                "(e.session_id IS NULL OR e.session_id NOT LIKE 'task_%')",
                [user_id],
            )
        ][-500:]
        report: dict[str, Any] = {"shared": [], "rejected": [], "skipped": []}
        for c in cands:
            if c.metadata.get("shared_as"):
                report["skipped"].append({"id": c.id, "reason": "already shared"})
                continue
            if c.utility < min_utility:
                report["skipped"].append(
                    {"id": c.id, "reason": f"utility {c.utility:.2f} < {min_utility}"}
                )
                continue
            content = c.text
            if self.m.has_llm:
                try:
                    d = self.m.llm.json(
                        [
                            {"role": "system", "content": prompts.SHARE_ABSTRACT_SYSTEM},
                            {"role": "user", "content": c.text},
                        ],
                        max_tokens=900,
                    )
                    content = str(d.get("content", "")).strip() if isinstance(d, dict) else ""
                except LLMError as e:
                    report["rejected"].append({"id": c.id, "reason": f"abstraction failed: {e}"})
                    continue
            if not content:
                report["rejected"].append({"id": c.id, "reason": "not generalisable"})
                continue
            check = check_shareable(content, private_texts)
            if not check.ok:
                report["rejected"].append(
                    {"id": c.id, "reason": "privacy gate", "findings": check.findings[:10]}
                )
                self.m.ledger.append(
                    "SHARE_REJECTED", c.id, {"findings": len(check.findings)}, actor=user_id
                )
                continue
            contributor = pseudonym(user_id, self.secret)
            action, item = self.m.learner._manage_experience(
                content,
                c.metadata.get("scope"),
                None,
                None,
                [],
                c.utility * 2 - 1,
                False,
                visibility,
                team_id,
                extra_meta={
                    "contributor": contributor,
                    "derived_from_hash": pseudonym(c.id, self.secret),
                },
            )
            if item is None:
                report["skipped"].append({"id": c.id, "reason": "already covered by shared pool"})
                continue
            self.m.store.update_memory(c.id, metadata={**c.metadata, "shared_as": item.id})
            self.m.ledger.append(
                "SHARE",
                item.id,
                {"visibility": visibility.value, "team": team_id, "action": action},
                actor=contributor,
            )
            report["shared"].append({"from": c.id, "shared_id": item.id, "action": action})
        return report

    # ------------------------------------------------------------ erasure
    def forget(
        self, memory_id: str | None = None, user_id: str | None = None, actor: str | None = None
    ) -> dict[str, int]:
        """Hard erase. Leaves only hash tombstones in the ledger."""
        st = self.m.store
        n_mem = n_ep = n_shared = 0
        with st._lock:
            if memory_id:
                m = st.get_memory(memory_id)
                if m is None:
                    return {"memories": 0, "episodes": 0, "shared": 0}
                self._erase_memory(memory_id)
                n_mem = 1
                self.m.ledger.append("FORGET", memory_id, {"kind": m.kind.value}, actor=actor)
            if user_id:
                ids = [
                    r[0]
                    for r in st.db.execute("SELECT id FROM memories WHERE user_id IS ?", (user_id,))
                ]
                for i in ids:
                    self._erase_memory(i)
                n_mem += len(ids)
                eps = [
                    r[0]
                    for r in st.db.execute("SELECT id FROM episodes WHERE user_id IS ?", (user_id,))
                ]
                for e in eps:
                    st.db.execute("DELETE FROM evidence WHERE episode_id=?", (e,))
                    st.db.execute("DELETE FROM episodes_fts WHERE id=?", (e,))
                    st.db.execute("DELETE FROM episodes WHERE id=?", (e,))
                n_ep = len(eps)
                contributor = pseudonym(user_id, self.secret)
                for m in st.memories(
                    "json_extract(m.metadata,'$.contributor')=? AND m.status='active'",
                    [contributor],
                ):
                    st.update_memory(m.id, status=Status.RETRACTED, valid_to=now())
                    n_shared += 1
                st.db.execute("DELETE FROM traces WHERE user_id IS ?", (user_id,))
                self.m.ledger.append(
                    "FORGET_USER",
                    pseudonym(user_id, b"tombstone"),
                    {"memories": n_mem, "episodes": n_ep, "shared_withdrawn": n_shared},
                    actor=actor,
                )
        return {"memories": n_mem, "episodes": n_ep, "shared": n_shared}

    def _erase_memory(self, memory_id: str) -> None:
        st = self.m.store
        st.db.execute("DELETE FROM evidence WHERE memory_id=?", (memory_id,))
        st.db.execute("DELETE FROM entity_links WHERE memory_id=?", (memory_id,))
        st.db.execute("DELETE FROM memories_fts WHERE id=?", (memory_id,))
        st.db.execute("UPDATE memories SET supersedes=NULL WHERE supersedes=?", (memory_id,))
        st.db.execute("DELETE FROM memories WHERE id=?", (memory_id,))


def ensure_not_preference(kind: Kind, visibility: Visibility) -> None:
    if kind is Kind.PREFERENCE and visibility is not Visibility.PRIVATE:
        raise PermissionError("preferences are user-bound and cannot be shared")
