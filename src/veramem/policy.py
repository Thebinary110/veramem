"""Access control, evaluated deterministically and *before* ranking.

Model (Collaborative Memory, Rezazadeh et al. 2025; AIM 2026):
  * time-varying bipartite graphs  G_UA(t) users->agents, G_AR(t) agents->resources,
    plus team membership;
  * a memory is readable by (user u, agent a) at time t iff
        visibility rule  (AIM Eq.1, extended with teams)
      AND provenance rule (C6 Eq.3: the agent that produced it is one u may use now,
                           and every resource it touched is one a may access now)
  * revocation is retroactive because the predicate is evaluated at read time;
  * only the owner may update / delete (AIM write ownership).

The predicate is compiled to SQL so that unreadable rows never leave the database.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .store.sqlite import SQLiteStore, owner_token
from .types import Kind, MemoryItem, Visibility, now


class AccessDenied(PermissionError):
    pass


@dataclass(frozen=True)
class Principal:
    user_id: str | None = None
    agent_id: str | None = None

    def __str__(self) -> str:
        return f"user={self.user_id} agent={self.agent_id}"


class Policy:
    def __init__(self, store: SQLiteStore):
        self.store = store

    # ------------------------------------------------------------ graph edits
    def grant_agent(self, user_id: str, agent_id: str) -> None:
        self.store.grant("user_agent", user_id, agent_id, now())

    def revoke_agent(self, user_id: str, agent_id: str) -> None:
        self.store.revoke("user_agent", user_id, agent_id, now())

    def grant_resource(self, agent_id: str, resource: str) -> None:
        self.store.grant("agent_resource", agent_id, resource, now())

    def revoke_resource(self, agent_id: str, resource: str) -> None:
        self.store.revoke("agent_resource", agent_id, resource, now())

    def add_to_team(self, user_id: str, team_id: str) -> None:
        self.store.grant("team_member", user_id, team_id, now())

    def remove_from_team(self, user_id: str, team_id: str) -> None:
        self.store.revoke("team_member", user_id, team_id, now())

    def teams(self, user_id: str | None, at: float | None = None) -> set[str]:
        if not user_id:
            return set()
        return self.store.grant_objects("team_member", user_id, at or now())

    # --------------------------------------------------------------- reading
    def read_filter(
        self, p: Principal, alias: str = "m", include_shared: bool = True, at: float | None = None
    ) -> tuple[str, list]:
        """SQL WHERE fragment selecting exactly the memories `p` may read."""
        t = at or now()
        clauses, params = [], []
        own = f"({alias}.visibility='private' AND {alias}.user_id IS ?)"
        if p.user_id is None:
            # anonymous / agent-only callers see only agent-scoped private memory
            own = f"({alias}.visibility='private' AND {alias}.user_id IS NULL AND {alias}.agent_id IS ?)"
            params.append(p.agent_id)
        else:
            params.append(p.user_id)
        clauses.append(own)
        if include_shared:
            teams = sorted(self.teams(p.user_id, t))
            if teams:
                marks = ",".join("?" * len(teams))
                clauses.append(f"({alias}.visibility='team' AND {alias}.team_id IN ({marks}))")
                params.extend(teams)
            clauses.append(f"({alias}.visibility='public')")
        where = "(" + " OR ".join(clauses) + ")"

        # provenance rule: agent edges only constrain once the operator has configured them
        if p.user_id and self.store.has_any_grants("user_agent"):
            agents = sorted(self.store.grant_objects("user_agent", p.user_id, t))
            if agents:
                marks = ",".join("?" * len(agents))
                where += f" AND ({alias}.agent_id IS NULL OR {alias}.agent_id IN ({marks}))"
                params.extend(agents)
            else:
                where += f" AND {alias}.agent_id IS NULL"
        if p.agent_id and self.store.has_any_grants("agent_resource"):
            allowed = self.store.grant_objects("agent_resource", p.agent_id, t)
            where += f" AND {_resources_subset_sql(alias)}"
            params.append(json.dumps(sorted(allowed)))
        return where, params

    def episode_filter(
        self, p: Principal, alias: str = "e", at: float | None = None
    ) -> tuple[str, list]:
        """Raw episodes are always private to their owner and obey the same provenance
        rules as memories (agent grants, resource grants)."""
        t = at or now()
        params: list[Any] = []
        if p.user_id is not None:
            where = f"{alias}.user_id IS ?"
            params.append(p.user_id)
        else:
            where = f"{alias}.user_id IS NULL AND {alias}.agent_id IS ?"
            params.append(p.agent_id)
        if p.user_id and self.store.has_any_grants("user_agent"):
            agents = sorted(self.store.grant_objects("user_agent", p.user_id, t))
            if agents:
                marks = ",".join("?" * len(agents))
                where += f" AND ({alias}.agent_id IS NULL OR {alias}.agent_id IN ({marks}))"
                params.extend(agents)
            else:
                where += f" AND {alias}.agent_id IS NULL"
        if p.agent_id and self.store.has_any_grants("agent_resource"):
            allowed = self.store.grant_objects("agent_resource", p.agent_id, t)
            where += f" AND {_resources_subset_sql(alias)}"
            params.append(json.dumps(sorted(allowed)))
        return where, params

    def fts_scopes(
        self, p: Principal, include_shared: bool = True, at: float | None = None
    ) -> list[str]:
        """Owner tokens of every scope `p` may read (narrows full-text search)."""
        own = owner_token(p.user_id, p.agent_id, None, "private")
        if not include_shared:
            return [own]
        teams = [owner_token(None, None, t, "team") for t in sorted(self.teams(p.user_id, at))]
        return [own, *teams, "pub"]

    def can_read(self, p: Principal, m: MemoryItem) -> bool:
        where, params = self.read_filter(p)
        r = self.store.db.execute(
            f"SELECT 1 FROM memories m WHERE m.id=? AND {where}", (m.id, *params)
        ).fetchone()
        return r is not None

    # --------------------------------------------------------------- writing
    def check_write(self, p: Principal, m: MemoryItem) -> None:
        """Only the owner may modify a memory; agent-owned memories by that agent."""
        if m.user_id is not None:
            if p.user_id != m.user_id:
                raise AccessDenied(f"{p} may not modify memory {m.id} owned by another user")
        elif m.agent_id is not None and p.agent_id != m.agent_id and p.user_id is not None:
            raise AccessDenied(f"{p} may not modify agent memory {m.id}")

    @staticmethod
    def check_visibility(m: MemoryItem) -> None:
        """Invariant from ShareMem §4.4: preferences are user-bound and never shared."""
        if m.kind is Kind.PREFERENCE and m.visibility is not Visibility.PRIVATE:
            raise AccessDenied("preferences are user-bound and cannot be shared")


def _resources_subset_sql(alias: str) -> str:
    # every resource in the memory's provenance must be in the allowed set
    return (
        f"NOT EXISTS (SELECT 1 FROM json_each(COALESCE(json_extract({alias}.metadata,"
        f"'$.resources'),'[]')) r WHERE r.value NOT IN (SELECT value FROM json_each(?)))"
    )
