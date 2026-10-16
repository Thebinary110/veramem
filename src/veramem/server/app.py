"""REST API + built-in web UI.

Security defaults: binds to 127.0.0.1; when VERAMEM_API_KEY is set every /v1 request
must send `Authorization: Bearer <key>` (constant-time compare). The CLI refuses to
listen on a non-loopback address without a key.
"""

from __future__ import annotations

import hmac
import os
from importlib import resources
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..__about__ import __version__
from ..memory import Memory
from ..policy import AccessDenied


class AddBody(BaseModel):
    messages: Any = Field(..., description="string, message dict or list of message dicts")
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    team_id: str | None = None
    infer: bool = True
    timestamp: Any = None
    metadata: dict[str, Any] | None = None
    resources: list[str] | None = None


class SearchBody(BaseModel):
    query: str
    user_id: str | None = None
    agent_id: str | None = None
    limit: int = Field(10, ge=1, le=100)
    granularity: str = "auto"
    kinds: list[str] | None = None
    include_shared: bool = True
    as_of: float | None = None


class AnswerBody(BaseModel):
    question: str
    user_id: str | None = None
    agent_id: str | None = None
    limit: int = Field(20, ge=1, le=100)
    style: str = "concise"


class FeedbackBody(BaseModel):
    trace_id: str
    reward: float = Field(..., ge=-1, le=1)


class LearnBody(BaseModel):
    task: str
    trajectory: Any
    reward: float = Field(..., ge=-1, le=1)
    user_id: str | None = None
    agent_id: str | None = None
    scope: str | None = None
    shareable: bool = False


class UpdateBody(BaseModel):
    text: str
    user_id: str | None = None
    agent_id: str | None = None


class ShareBody(BaseModel):
    user_id: str
    team_id: str | None = None
    memory_ids: list[str] | None = None
    min_utility: float = 0.6


class GrantBody(BaseModel):
    kind: str = Field(..., pattern="^(user_agent|agent_resource|team_member)$")
    subject: str
    object: str
    revoke: bool = False


class ActionBody(BaseModel):
    memory_ids: list[str]


def create_app(mem: Memory | None = None) -> FastAPI:
    mem = mem or Memory()
    api_key = os.environ.get("VERAMEM_API_KEY", "")
    app = FastAPI(
        title="VeraMem",
        version=__version__,
        description="Verifiable, outcome-aware, share-safe memory for AI agents.",
    )

    def auth(authorization: str | None = Header(default=None)) -> None:
        if not api_key:
            return
        token = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(token.encode(), api_key.encode()):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    guard = [Depends(auth)]

    def _call(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except AccessDenied as e:
            raise HTTPException(status_code=403, detail=str(e)) from e
        except KeyError as e:
            raise HTTPException(status_code=404, detail=f"not found: {e}") from e
        except (ValueError, PermissionError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui() -> str:
        return resources.files("veramem.server").joinpath("static/index.html").read_text()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "auth": bool(api_key)}

    @app.get("/v1/info", dependencies=guard)
    def info() -> dict[str, Any]:
        return mem.info()

    @app.post("/v1/memories", dependencies=guard)
    def add(b: AddBody) -> dict[str, Any]:
        return _call(
            mem.add,
            b.messages,
            user_id=b.user_id,
            agent_id=b.agent_id,
            session_id=b.session_id,
            team_id=b.team_id,
            infer=b.infer,
            timestamp=b.timestamp,
            metadata=b.metadata,
            resources=b.resources,
        ).model_dump(mode="json")

    @app.get("/v1/memories", dependencies=guard)
    def list_memories(
        user_id: str | None = None,
        agent_id: str | None = None,
        include_inactive: bool = False,
        limit: int = Query(200, ge=1, le=5000),
    ) -> list[dict[str, Any]]:
        items = _call(
            mem.get_all,
            user_id=user_id,
            agent_id=agent_id,
            include_inactive=include_inactive,
            limit=limit,
        )
        return [m.model_dump(mode="json") for m in items]

    @app.get("/v1/memories/{memory_id}", dependencies=guard)
    def get(
        memory_id: str, user_id: str | None = None, agent_id: str | None = None
    ) -> dict[str, Any]:
        m = _call(mem.get, memory_id, user_id=user_id, agent_id=agent_id)
        if m is None:
            raise HTTPException(404, "not found")
        return m.model_dump(mode="json")

    @app.put("/v1/memories/{memory_id}", dependencies=guard)
    def update(memory_id: str, b: UpdateBody) -> dict[str, Any]:
        return _call(
            mem.update, memory_id, b.text, user_id=b.user_id, agent_id=b.agent_id
        ).model_dump(mode="json")

    @app.delete("/v1/memories/{memory_id}", dependencies=guard)
    def delete(
        memory_id: str, user_id: str | None = None, agent_id: str | None = None, hard: bool = False
    ) -> dict[str, Any]:
        if hard:
            return _call(mem.forget, memory_id, user_id=user_id, agent_id=agent_id)
        _call(mem.delete, memory_id, user_id=user_id, agent_id=agent_id)
        return {"retracted": memory_id}

    @app.get("/v1/memories/{memory_id}/evidence", dependencies=guard)
    def evidence(
        memory_id: str, user_id: str | None = None, agent_id: str | None = None
    ) -> list[dict[str, Any]]:
        return _call(mem.evidence, memory_id, user_id=user_id, agent_id=agent_id)

    @app.get("/v1/memories/{memory_id}/history", dependencies=guard)
    def history(
        memory_id: str, user_id: str | None = None, agent_id: str | None = None
    ) -> list[dict[str, Any]]:
        return _call(mem.history, memory_id, user_id=user_id, agent_id=agent_id)

    @app.post("/v1/search", dependencies=guard)
    def search(b: SearchBody) -> dict[str, Any]:
        r = _call(
            mem.search,
            b.query,
            user_id=b.user_id,
            agent_id=b.agent_id,
            limit=b.limit,
            granularity=b.granularity,
            kinds=b.kinds,
            include_shared=b.include_shared,
            as_of=b.as_of,
        )
        d = r.model_dump(mode="json")
        d["context"] = r.context()
        return d

    @app.post("/v1/answer", dependencies=guard)
    def answer(b: AnswerBody) -> dict[str, Any]:
        return _call(
            mem.answer,
            b.question,
            user_id=b.user_id,
            agent_id=b.agent_id,
            limit=b.limit,
            style=b.style,
        ).model_dump(mode="json")

    @app.post("/v1/feedback", dependencies=guard)
    def feedback(b: FeedbackBody) -> dict[str, Any]:
        return _call(mem.feedback, b.trace_id, b.reward)

    @app.post("/v1/learn", dependencies=guard)
    def learn(b: LearnBody) -> dict[str, Any]:
        return _call(
            mem.learn,
            b.task,
            b.trajectory,
            b.reward,
            user_id=b.user_id,
            agent_id=b.agent_id,
            scope=b.scope,
            shareable=b.shareable,
        )

    @app.post("/v1/share", dependencies=guard)
    def share(b: ShareBody) -> dict[str, Any]:
        return _call(
            mem.share,
            user_id=b.user_id,
            team_id=b.team_id,
            memory_ids=b.memory_ids,
            min_utility=b.min_utility,
        )

    @app.post("/v1/grants", dependencies=guard)
    def grants(b: GrantBody) -> dict[str, str]:
        fn = {
            ("user_agent", False): mem.grant_agent,
            ("user_agent", True): mem.revoke_agent,
            ("agent_resource", False): mem.grant_resource,
            ("agent_resource", True): mem.revoke_resource,
            ("team_member", False): mem.join_team,
            ("team_member", True): mem.leave_team,
        }[(b.kind, b.revoke)]
        _call(fn, b.subject, b.object)
        return {"status": "revoked" if b.revoke else "granted"}

    @app.post("/v1/check-action", dependencies=guard)
    def check_action(b: ActionBody) -> dict[str, Any]:
        return mem.check_action(b.memory_ids)

    @app.post("/v1/consolidate", dependencies=guard)
    def consolidate(user_id: str | None = None) -> dict[str, Any]:
        return mem.consolidate(user_id=user_id)

    @app.get("/v1/verify", dependencies=guard)
    def verify() -> dict[str, Any]:
        return mem.verify()

    @app.get("/v1/export", dependencies=guard)
    def export(user_id: str) -> dict[str, Any]:
        return mem.export(user_id=user_id)

    @app.delete("/v1/users/{user_id}", dependencies=guard)
    def forget_user(user_id: str) -> dict[str, int]:
        return _call(mem.forget, user_id=user_id, everything=True)

    return app
