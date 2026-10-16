"""The public API. Everything a user needs is a method on `Memory`."""

from __future__ import annotations

import os
from typing import Any

from .answer import Answerer
from .config import Config, ollama_host, resolve
from .learning import Learner
from .ledger import Ledger, content_hash
from .policy import AccessDenied, Policy, Principal
from .providers import (
    LLM,
    AnthropicLLM,
    Embedder,
    HashEmbedder,
    NoLLM,
    OllamaEmbedder,
    OllamaLLM,
    OpenAICompatibleEmbedder,
    OpenAICompatibleLLM,
    ResponseCache,
)
from .retrieval import Retriever
from .sharing import Sharer
from .store import SQLiteStore
from .types import (
    AddResult,
    Answer,
    Authority,
    Episode,
    Evidence,
    Kind,
    MemoryItem,
    SearchResult,
    SourceType,
    Status,
    Visibility,
    now,
)
from .write import Candidate, Writer


class Memory:
    """Verifiable, outcome-aware, share-safe memory for AI agents.

    >>> mem = Memory()                         # auto-detects Ollama / OpenAI / Anthropic
    >>> mem.add("I moved to Berlin in March", user_id="alice")
    >>> mem.search("where does alice live?", user_id="alice").hits[0].memory.text
    """

    def __init__(
        self,
        path: str | None = None,
        *,
        llm: LLM | None = None,
        embedder: Embedder | None = None,
        config: Config | None = None,
        **config_overrides: Any,
    ):
        self.config = config or Config.load(**config_overrides, db_path=path)
        choice = resolve(self.config)
        cache = ResponseCache(self.config.cache_path) if self.config.cache_path else None
        self.llm = llm or _make_llm(*choice["llm"], cache=cache)
        self.embedder = embedder or _make_embedder(*choice["embed"], cache=cache)
        self.store = SQLiteStore(self.config.db_path)
        self._check_embedder()
        self.ledger = Ledger(self.store)
        self.policy = Policy(self.store)
        self.writer = Writer(self)
        self.retriever = Retriever(self)
        self.answerer = Answerer(self)
        self.learner = Learner(self)
        self.sharer = Sharer(self)

    # ---------------------------------------------------------------- info
    @property
    def has_llm(self) -> bool:
        return not isinstance(self.llm, NoLLM)

    def info(self) -> dict[str, Any]:
        return {
            "db": self.store.path,
            "llm": f"{self.llm.name}:{self.llm.model}" if self.has_llm else "none (raw-turn mode)",
            "embedder": f"{self.embedder.name}:{self.embedder.model}",
            **self.store.stats(),
        }

    def _check_embedder(self) -> None:
        ident = f"{self.embedder.name}:{self.embedder.model}"
        stored = self.store.meta_get("embedder")
        if stored is None:
            self.store.meta_set("embedder", ident)
        elif stored != ident:
            self._embedder_mismatch = stored
            # vectors from a different model are not comparable: re-embed lazily on demand
            if os.environ.get("VERAMEM_AUTO_REINDEX", "1") == "1":
                self.reindex()

    def reindex(self) -> int:
        """Re-embed every memory and episode with the current embedder."""
        n = 0
        rows = self.store.db.execute("SELECT id, text FROM memories").fetchall()
        for start in range(0, len(rows), 128):
            chunk = rows[start : start + 128]
            vecs = self.embedder.embed([r[1] for r in chunk])
            for r, v in zip(chunk, vecs):
                self.store.update_memory(r[0], embedding=v)
            n += len(chunk)
        rows = self.store.db.execute("SELECT id, content FROM episodes").fetchall()
        for start in range(0, len(rows), 128):
            chunk = rows[start : start + 128]
            vecs = self.embedder.embed([r[1] for r in chunk])
            from .store.sqlite import _blob

            with self.store._lock:
                for r, v in zip(chunk, vecs):
                    self.store.db.execute(
                        "UPDATE episodes SET embedding=? WHERE id=?", (_blob(v), r[0])
                    )
            n += len(chunk)
        self.store.meta_set("embedder", f"{self.embedder.name}:{self.embedder.model}")
        return n

    # --------------------------------------------------------------- write
    def add(
        self,
        messages: Any,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        team_id: str | None = None,
        infer: bool = True,
        timestamp: Any = None,
        metadata: dict[str, Any] | None = None,
        resources: list[str] | None = None,
    ) -> AddResult:
        """Remember a message, a list of chat messages, a tool result or a document.

        messages: str | {"role","content"[,"name","timestamp"]} | list of those.
        Roles: user, assistant, tool, document, system. The role decides authority.
        """
        if user_id is None and agent_id is None:
            user_id = "default"
        return self.writer.add(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            team_id=team_id,
            infer=infer,
            timestamp=timestamp,
            metadata=metadata,
            resources=resources,
            actor=user_id or agent_id,
        )

    # ---------------------------------------------------------------- read
    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 10,
        granularity: str = "auto",
        kinds: list[str] | None = None,
        include_shared: bool = True,
        as_of: float | None = None,
    ) -> SearchResult:
        """Hybrid, access-controlled retrieval. Returns hits with evidence and a trace_id."""
        p = self._principal(user_id, agent_id)
        k = [Kind(x) for x in kinds] if kinds else None
        return self.retriever.search(
            query,
            p,
            limit=limit,
            granularity=granularity,
            kinds=k,
            include_shared=include_shared,
            as_of=as_of,
        )

    def answer(
        self,
        question: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 20,
        style: str = "concise",
        max_rounds: int = 2,
        today: str | None = None,
    ) -> Answer:
        """Answer from memory only, with citations; abstains when evidence is missing."""
        if not self.has_llm:
            raise RuntimeError("answer() needs an LLM; configure one with `veramem init`")
        return self.answerer.answer(
            question,
            self._principal(user_id, agent_id),
            limit=limit,
            style=style,
            max_rounds=max_rounds,
            now_hint=today,
        )

    def get(
        self, memory_id: str, *, user_id: str | None = None, agent_id: str | None = None
    ) -> MemoryItem | None:
        m = self.store.get_memory(memory_id)
        if m is None:
            return None
        if (
            not self.policy.can_read(self._principal(user_id, agent_id), m)
            and m.status is Status.ACTIVE
        ):
            raise AccessDenied(f"not allowed to read {memory_id}")
        if (
            m.status is not Status.ACTIVE
            and m.user_id != (user_id or "default")
            and m.visibility is Visibility.PRIVATE
        ):
            raise AccessDenied(f"not allowed to read {memory_id}")
        return m

    def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        include_inactive: bool = False,
        limit: int = 1000,
    ) -> list[MemoryItem]:
        p = self._principal(user_id, agent_id)
        where, params = self.policy.read_filter(p, include_shared=False)
        if not include_inactive:
            where += " AND m.status='active'"
        return self.store.memories(where, params, order="created_at DESC", limit=limit)

    def evidence(
        self, memory_id: str, *, user_id: str | None = None, agent_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Resolve a memory's evidence pointers back to the raw episodes."""
        m = self.get(memory_id, user_id=user_id, agent_id=agent_id)
        out = []
        for ev in m.evidence if m else []:
            ep = self.store.get_episode(ev.episode_id)
            out.append(
                {
                    "episode_id": ev.episode_id,
                    "quote": ev.quote,
                    "source": ev.source.value,
                    "span": [ev.start, ev.end],
                    "episode": ep.content if ep and ep.user_id == m.user_id else None,
                    "occurred_at": ep.occurred_at if ep else None,
                    "intact": bool(ep and content_hash(ep.content) == ep.sha256),
                }
            )
        return out

    def history(
        self, memory_id: str, *, user_id: str | None = None, agent_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Version chain (oldest first) plus the ledger entries for each version."""
        self.get(memory_id, user_id=user_id, agent_id=agent_id)
        chain: list[MemoryItem] = []
        cur = self.store.get_memory(memory_id)
        while cur and cur.superseded_by:  # walk forward to the newest version
            nxt = self.store.get_memory(cur.superseded_by)
            if not nxt:
                break
            cur = nxt
        while cur:
            chain.append(cur)
            cur = self.store.get_memory(cur.supersedes) if cur.supersedes else None
        chain.reverse()
        return [
            {
                "id": m.id,
                "text": m.text,
                "status": m.status.value,
                "authority": m.authority.value,
                "valid_from": m.valid_from,
                "valid_to": m.valid_to,
                "ledger": self.ledger.history(m.id),
            }
            for m in chain
        ]

    # -------------------------------------------------------------- update
    def update(
        self, memory_id: str, text: str, *, user_id: str | None = None, agent_id: str | None = None
    ) -> MemoryItem:
        """Manual correction by the owner. Creates a new version and keeps history.

        The correction itself is stored as a user episode, so the new version is
        grounded in evidence like any other memory.
        """
        p = self._principal(user_id, agent_id)
        old = self.store.get_memory(memory_id)
        if old is None:
            raise KeyError(memory_id)
        self.policy.check_write(p, old)
        ep = Episode(
            content=text,
            source=SourceType.USER,
            user_id=old.user_id,
            agent_id=old.agent_id,
            sha256=content_hash(text),
            metadata={"manual_edit": True},
        )
        vec = self.embedder.embed_one(text)
        self.store.put_episode(ep, vec)
        self.ledger.append(
            "EPISODE",
            ep.id,
            {"sha256": ep.sha256, "source": "user", "manual_edit": True},
            actor=p.user_id,
        )
        cand = Candidate(
            text=text,
            kind=old.kind,
            evidence=Evidence(
                episode_id=ep.id, quote=text, start=0, end=len(text), source=SourceType.USER
            ),
            authority=Authority.AUTHORIZED,
            entities=old.entities,
            valid_from=ep.occurred_at,
            vec=vec,
        )
        scope = {"user_id": old.user_id, "agent_id": old.agent_id, "team_id": old.team_id}
        ch = self.writer._update(old, cand, text, scope, None, p.user_id or p.agent_id)
        return self.store.get_memory(ch.memory_id)  # type: ignore[return-value]

    def delete(
        self, memory_id: str, *, user_id: str | None = None, agent_id: str | None = None
    ) -> None:
        """Retract a memory (kept for audit, hidden from retrieval). See forget()."""
        p = self._principal(user_id, agent_id)
        m = self.store.get_memory(memory_id)
        if m is None:
            raise KeyError(memory_id)
        self.policy.check_write(p, m)
        self.store.update_memory(
            memory_id, status=Status.RETRACTED, valid_to=now(), updated_at=now()
        )
        self.ledger.append("DELETE", memory_id, {"manual": True}, actor=p.user_id or p.agent_id)

    def forget(
        self,
        memory_id: str | None = None,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        everything: bool = False,
    ) -> dict[str, int]:
        """Hard-erase one memory, or everything about a user (right to be forgotten)."""
        p = self._principal(user_id, agent_id)
        if memory_id:
            m = self.store.get_memory(memory_id)
            if m is None:
                return {"memories": 0, "episodes": 0, "shared": 0}
            self.policy.check_write(p, m)
            return self.sharer.forget(memory_id=memory_id, actor=p.user_id)
        if everything and p.user_id:
            return self.sharer.forget(user_id=p.user_id, actor=p.user_id)
        raise ValueError("pass memory_id, or everything=True with user_id")

    # ------------------------------------------------------------ outcomes
    def feedback(self, trace_id: str, reward: float) -> dict[str, Any]:
        """Report how well a search/answer worked: reward in [-1, 1]."""
        return self.learner.feedback(trace_id, reward)

    def learn(
        self,
        task: str,
        trajectory: Any,
        reward: float,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        scope: str | None = None,
        shareable: bool = False,
    ) -> dict[str, Any]:
        """Distil a finished task into a reusable experience (if the outcome is decisive)."""
        return self.learner.learn(
            task,
            trajectory,
            reward,
            user_id=user_id or "default",
            agent_id=agent_id,
            scope=scope,
            shareable=shareable,
        )

    def consolidate(self, *, user_id: str | None = None, summarize: bool = True) -> dict[str, Any]:
        return self.learner.consolidate(user_id=user_id, summarize=summarize)

    def check_action(self, memory_ids: list[str]) -> dict[str, Any]:
        """May these memories bind a protected tool call? (authority gate)"""
        return self.learner.check_action(memory_ids)

    def export_trajectories(self, path: str, *, user_id: str | None = None) -> int:
        return self.learner.export_trajectories(path, user_id=user_id)

    # ------------------------------------------------------------- sharing
    def share(
        self,
        *,
        user_id: str,
        team_id: str | None = None,
        memory_ids: list[str] | None = None,
        min_utility: float = 0.6,
    ) -> dict[str, Any]:
        """Share the user's consented experiences (de-identified) with a team or everyone."""
        return self.sharer.share(
            user_id, team_id=team_id, memory_ids=memory_ids, min_utility=min_utility
        )

    def set_shareable(
        self, memory_id: str, shareable: bool = True, *, user_id: str | None = None
    ) -> None:
        m = self.store.get_memory(memory_id)
        if m is None:
            raise KeyError(memory_id)
        self.policy.check_write(self._principal(user_id, None), m)
        if m.kind is Kind.PREFERENCE and shareable:
            raise AccessDenied("preferences are user-bound and cannot be shared")
        self.store.update_memory(memory_id, shareable=shareable)
        self.ledger.append("CONSENT", memory_id, {"shareable": shareable}, actor=user_id)

    def grant_agent(self, user_id: str, agent_id: str) -> None:
        self.policy.grant_agent(user_id, agent_id)
        self.ledger.append("GRANT", f"{user_id}->{agent_id}", {"kind": "user_agent"})

    def revoke_agent(self, user_id: str, agent_id: str) -> None:
        self.policy.revoke_agent(user_id, agent_id)
        self.ledger.append("REVOKE", f"{user_id}->{agent_id}", {"kind": "user_agent"})

    def grant_resource(self, agent_id: str, resource: str) -> None:
        self.policy.grant_resource(agent_id, resource)
        self.ledger.append("GRANT", f"{agent_id}->{resource}", {"kind": "agent_resource"})

    def revoke_resource(self, agent_id: str, resource: str) -> None:
        self.policy.revoke_resource(agent_id, resource)
        self.ledger.append("REVOKE", f"{agent_id}->{resource}", {"kind": "agent_resource"})

    def join_team(self, user_id: str, team_id: str) -> None:
        self.policy.add_to_team(user_id, team_id)
        self.ledger.append("GRANT", f"{user_id}->{team_id}", {"kind": "team_member"})

    def leave_team(self, user_id: str, team_id: str) -> None:
        self.policy.remove_from_team(user_id, team_id)
        self.ledger.append("REVOKE", f"{user_id}->{team_id}", {"kind": "team_member"})

    # -------------------------------------------------------- portability
    def export(self, *, user_id: str) -> dict[str, Any]:
        """Portable JSON of everything stored about a user (data-subject access)."""
        mems = self.store.memories("m.user_id IS ?", [user_id])
        eps = self.store.episodes("e.user_id IS ?", [user_id])
        return {
            "format": "veramem.export.v1",
            "user_id": user_id,
            "exported_at": now(),
            "episodes": [e.model_dump(mode="json") for e in eps],
            "memories": [m.model_dump(mode="json") for m in mems],
        }

    def import_(self, data: dict[str, Any]) -> dict[str, int]:
        """Load a `veramem.export.v1` document (ids are preserved, existing ids skipped)."""
        if data.get("format") != "veramem.export.v1":
            raise ValueError("unsupported export format")
        n_ep = n_mem = 0
        eps = [Episode(**e) for e in data.get("episodes", [])]
        new_eps = [e for e in eps if self.store.get_episode(e.id) is None]
        for e, v in zip(new_eps, self.embedder.embed([e.content for e in new_eps])):
            if content_hash(e.content) != e.sha256:
                raise ValueError(f"episode {e.id} fails its hash check")
            self.store.put_episode(e, v)
            self.ledger.append("IMPORT", e.id, {"sha256": e.sha256})
            n_ep += 1
        mems = [MemoryItem(**m) for m in data.get("memories", [])]
        new_mems = [m for m in mems if self.store.get_memory(m.id) is None]
        for m, v in zip(new_mems, self.embedder.embed([m.text for m in new_mems])):
            self.store.put_memory(m, v)
            self.ledger.append("IMPORT", m.id, {"text_sha256": content_hash(m.text)})
            n_mem += 1
        return {"episodes": n_ep, "memories": n_mem}

    # ------------------------------------------------------------- verify
    def verify(self) -> dict[str, Any]:
        """Check the ledger chain, episode hashes and memory text against the ledger."""
        rep = self.ledger.verify()
        bad_eps = [
            r[0]
            for r in self.store.db.execute("SELECT id, content, sha256 FROM episodes")
            if content_hash(r[1]) != r[2]
        ]
        latest: dict[str, str] = {}
        for r in self.store.ledger_rows():
            import json as _json

            payload = _json.loads(r["payload"])
            if "text_sha256" in payload and r["target"]:
                latest[r["target"]] = payload["text_sha256"]
        bad_mem = [
            r[0]
            for r in self.store.db.execute("SELECT id, text FROM memories")
            if r[0] in latest and content_hash(r[1]) != latest[r[0]]
        ]
        return {
            "ok": rep.ok and not bad_eps and not bad_mem,
            "ledger_ok": rep.ok,
            "ledger_entries": rep.entries,
            "head": rep.head,
            "first_bad_seq": rep.first_bad_seq,
            "tampered_episodes": bad_eps,
            "tampered_memories": bad_mem,
        }

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _principal(user_id: str | None, agent_id: str | None) -> Principal:
        if user_id is None and agent_id is None:
            user_id = "default"
        return Principal(user_id=user_id, agent_id=agent_id)


def _make_llm(provider: str, model: str, base_url: str, cache: ResponseCache | None) -> LLM:
    if provider == "ollama":
        return OllamaLLM(model=model or "qwen3.6:35b", host=base_url or ollama_host(), cache=cache)
    if provider == "openai":
        return OpenAICompatibleLLM(
            model=model or "gpt-4o-mini",
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            cache=cache,
        )
    if provider == "anthropic":
        return AnthropicLLM(
            model=model or "claude-haiku-5-5",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            base_url=base_url or "https://api.anthropic.com",
            cache=cache,
        )
    return NoLLM()


def _make_embedder(
    provider: str, model: str, base_url: str, cache: ResponseCache | None
) -> Embedder:
    if provider == "ollama":
        return OllamaEmbedder(
            model=model or "nomic-embed-text", host=base_url or ollama_host(), cache=cache
        )
    if provider == "openai":
        return OpenAICompatibleEmbedder(
            model=model or "text-embedding-3-small",
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            cache=cache,
        )
    return HashEmbedder(cache=cache)
