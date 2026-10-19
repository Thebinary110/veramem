"""REST API + built-in web UI.

Authentication (any combination; with none configured the server is "open" and the CLI
only allows it on localhost):
  * per-user API keys created with `veramem keys create` (stored as SHA-256 hashes): a
    user key is pinned to its user, so `user_id` in a request cannot widen its scope;
  * `VERAMEM_API_KEY`: an admin key from the environment (constant-time compare);
  * `VERAMEM_JWT_JWKS_URL` or `VERAMEM_JWT_SECRET`: OIDC / JWT bearer tokens, `sub` = user.
Hardening: per-caller rate limit (`VERAMEM_RATE_LIMIT`/min), body size limit, security
headers + CSP, CORS off unless `VERAMEM_CORS_ORIGINS` is set.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from importlib import resources
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .. import auth
from ..__about__ import __version__
from ..auth import Caller
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
    cited: list[str] | None = None
    user_id: str | None = None
    agent_id: str | None = None


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
    user_id: str | None = None
    agent_id: str | None = None


class ShareableBody(BaseModel):
    shareable: bool = True
    user_id: str | None = None


class KeyBody(BaseModel):
    role: str = Field("user", pattern="^(admin|user)$")
    user_id: str | None = None
    agent_id: str | None = None
    label: str | None = Field(None, max_length=120)


def create_app(mem: Memory | None = None) -> FastAPI:
    mem = mem or Memory()
    env_key = os.environ.get("VERAMEM_API_KEY", "")
    jwt_verifier = _jwt_from_env()
    limiter = _RateLimiter(int(os.environ.get("VERAMEM_RATE_LIMIT", "600")))
    max_body = int(os.environ.get("VERAMEM_MAX_BODY_BYTES", str(2 * 1024 * 1024)))
    app = FastAPI(
        title="VeraMem",
        version=__version__,
        description="Verifiable, outcome-aware, share-safe memory for AI agents.",
    )

    origins = [o.strip() for o in os.environ.get("VERAMEM_CORS_ORIGINS", "").split(",") if o]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.middleware("http")
    async def harden(request: Request, call_next):
        size = request.headers.get("content-length")
        if size and size.isdigit() and int(size) > max_body:
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response

    def auth_configured() -> bool:
        return bool(env_key) or jwt_verifier is not None or auth.has_keys(mem.store)

    def caller(request: Request, authorization: str | None = Header(default=None)) -> Caller:
        token = (authorization or "").removeprefix("Bearer ").strip()
        who: Caller | None = None
        if not auth_configured():
            who = Caller(role="anonymous")  # open mode: the CLI only allows it on localhost
        elif token:
            if env_key and hmac.compare_digest(token.encode(), env_key.encode()):
                who = Caller(role="admin", key_prefix="env")
            elif token.startswith(auth.KEY_PREFIX):
                who = auth.lookup(mem.store, token)
            elif jwt_verifier is not None:
                who = jwt_verifier.verify(token)
        if who is None:
            raise HTTPException(status_code=401, detail="invalid or missing credentials")
        bucket = who.key_prefix or who.user_id or (request.client.host if request.client else "?")
        if not limiter.allow(bucket):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        return who

    def scope(
        c: Caller, user_id: str | None, agent_id: str | None
    ) -> tuple[str | None, str | None]:
        """Admins act for anyone; a user key is pinned to its own user (and agent)."""
        if c.is_admin:
            return user_id, agent_id
        if user_id is not None and user_id != c.user_id:
            raise HTTPException(status_code=403, detail="this key is bound to another user")
        if c.agent_id is not None and agent_id not in (None, c.agent_id):
            raise HTTPException(status_code=403, detail="this key is bound to another agent")
        return c.user_id, c.agent_id or agent_id

    def admin_only(c: Caller) -> None:
        if not c.is_admin:
            raise HTTPException(status_code=403, detail="admin key required")

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
        return (
            resources.files("veramem.server")
            .joinpath("static/index.html")
            .read_text(encoding="utf-8")
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "auth": auth_configured()}

    @app.get("/v1/me")
    def me(c: Caller = Depends(caller)) -> dict[str, Any]:
        return {"role": c.role, "user_id": c.user_id, "agent_id": c.agent_id}

    @app.get("/v1/info")
    def info(c: Caller = Depends(caller)) -> dict[str, Any]:
        data = mem.info()
        if not c.is_admin:  # users see configuration, not global counts
            data = {k: v for k, v in data.items() if k in ("llm", "embedder")}
        return data

    @app.post("/v1/memories")
    def add(b: AddBody, c: Caller = Depends(caller)) -> dict[str, Any]:
        u, a = scope(c, b.user_id, b.agent_id)
        return _call(
            mem.add,
            b.messages,
            user_id=u,
            agent_id=a,
            session_id=b.session_id,
            team_id=b.team_id,
            infer=b.infer,
            timestamp=b.timestamp,
            metadata=b.metadata,
            resources=b.resources,
        ).model_dump(mode="json")

    @app.get("/v1/memories")
    def list_memories(
        c: Caller = Depends(caller),
        user_id: str | None = None,
        agent_id: str | None = None,
        include_inactive: bool = False,
        limit: int = Query(200, ge=1, le=5000),
    ) -> list[dict[str, Any]]:
        u, a = scope(c, user_id, agent_id)
        items = _call(
            mem.get_all, user_id=u, agent_id=a, include_inactive=include_inactive, limit=limit
        )
        return [m.model_dump(mode="json") for m in items]

    @app.get("/v1/memories/{memory_id}")
    def get(
        memory_id: str,
        c: Caller = Depends(caller),
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        u, a = scope(c, user_id, agent_id)
        m = _call(mem.get, memory_id, user_id=u, agent_id=a)
        if m is None:
            raise HTTPException(404, "not found")
        return m.model_dump(mode="json")

    @app.put("/v1/memories/{memory_id}")
    def update(memory_id: str, b: UpdateBody, c: Caller = Depends(caller)) -> dict[str, Any]:
        u, a = scope(c, b.user_id, b.agent_id)
        return _call(mem.update, memory_id, b.text, user_id=u, agent_id=a).model_dump(mode="json")

    @app.put("/v1/memories/{memory_id}/shareable")
    def set_shareable(
        memory_id: str, b: ShareableBody, c: Caller = Depends(caller)
    ) -> dict[str, Any]:
        u, _ = scope(c, b.user_id, None)
        _call(mem.set_shareable, memory_id, b.shareable, user_id=u)
        return {"id": memory_id, "shareable": b.shareable}

    @app.delete("/v1/memories/{memory_id}")
    def delete(
        memory_id: str,
        c: Caller = Depends(caller),
        user_id: str | None = None,
        agent_id: str | None = None,
        hard: bool = False,
    ) -> dict[str, Any]:
        u, a = scope(c, user_id, agent_id)
        if hard:
            return _call(mem.forget, memory_id, user_id=u, agent_id=a)
        _call(mem.delete, memory_id, user_id=u, agent_id=a)
        return {"retracted": memory_id}

    @app.get("/v1/memories/{memory_id}/evidence")
    def evidence(
        memory_id: str,
        c: Caller = Depends(caller),
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        u, a = scope(c, user_id, agent_id)
        return _call(mem.evidence, memory_id, user_id=u, agent_id=a)

    @app.get("/v1/memories/{memory_id}/history")
    def history(
        memory_id: str,
        c: Caller = Depends(caller),
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        u, a = scope(c, user_id, agent_id)
        return _call(mem.history, memory_id, user_id=u, agent_id=a)

    @app.post("/v1/search")
    def search(b: SearchBody, c: Caller = Depends(caller)) -> dict[str, Any]:
        u, a = scope(c, b.user_id, b.agent_id)
        r = _call(
            mem.search,
            b.query,
            user_id=u,
            agent_id=a,
            limit=b.limit,
            granularity=b.granularity,
            kinds=b.kinds,
            include_shared=b.include_shared,
            as_of=b.as_of,
        )
        d = r.model_dump(mode="json")
        d["context"] = r.context()
        return d

    @app.post("/v1/answer")
    def answer(b: AnswerBody, c: Caller = Depends(caller)) -> dict[str, Any]:
        u, a = scope(c, b.user_id, b.agent_id)
        return _call(
            mem.answer, b.question, user_id=u, agent_id=a, limit=b.limit, style=b.style
        ).model_dump(mode="json")

    @app.post("/v1/feedback")
    def feedback(b: FeedbackBody, c: Caller = Depends(caller)) -> dict[str, Any]:
        if c.is_admin and b.user_id is None:
            return _call(mem.feedback, b.trace_id, b.reward, cited=b.cited)
        u, a = scope(c, b.user_id, b.agent_id)
        return _call(mem.feedback, b.trace_id, b.reward, cited=b.cited, user_id=u, agent_id=a)

    @app.post("/v1/learn")
    def learn(b: LearnBody, c: Caller = Depends(caller)) -> dict[str, Any]:
        u, a = scope(c, b.user_id, b.agent_id)
        return _call(
            mem.learn,
            b.task,
            b.trajectory,
            b.reward,
            user_id=u,
            agent_id=a,
            scope=b.scope,
            shareable=b.shareable,
        )

    @app.post("/v1/share")
    def share(b: ShareBody, c: Caller = Depends(caller)) -> dict[str, Any]:
        u, _ = scope(c, b.user_id, None)
        if u is None:
            raise HTTPException(400, "user_id required")
        return _call(
            mem.share,
            user_id=u,
            team_id=b.team_id,
            memory_ids=b.memory_ids,
            min_utility=b.min_utility,
        )

    @app.get("/v1/shared")
    def shared(
        c: Caller = Depends(caller),
        user_id: str | None = None,
        limit: int = Query(200, ge=1, le=2000),
    ):
        u, _ = scope(c, user_id, None)
        return [m.model_dump(mode="json") for m in mem.shared_with(user_id=u, limit=limit)]

    @app.get("/v1/teams")
    def teams(c: Caller = Depends(caller), user_id: str | None = None) -> dict[str, Any]:
        u, _ = scope(c, user_id, None)
        return {"user_id": u, "teams": sorted(mem.policy.teams(u)), "agents": mem.agents_of(u)}

    @app.post("/v1/grants")
    def grants(b: GrantBody, c: Caller = Depends(caller)) -> dict[str, str]:
        # users may manage which agents can use *their own* memory; teams and agent
        # resources are organisation-level decisions reserved for admins
        if not c.is_admin and not (b.kind == "user_agent" and b.subject == c.user_id):
            raise HTTPException(status_code=403, detail="admin key required for this grant")
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

    @app.post("/v1/check-action")
    def check_action(b: ActionBody, c: Caller = Depends(caller)) -> dict[str, Any]:
        u, a = scope(c, b.user_id, b.agent_id)
        return mem.check_action(b.memory_ids, user_id=u, agent_id=a)

    @app.post("/v1/consolidate")
    def consolidate(c: Caller = Depends(caller), user_id: str | None = None) -> dict[str, Any]:
        u, _ = scope(c, user_id, None)
        return mem.consolidate(user_id=u)

    @app.get("/v1/verify")
    def verify(c: Caller = Depends(caller)) -> dict[str, Any]:
        admin_only(c)
        return mem.verify()

    @app.get("/v1/export")
    def export(c: Caller = Depends(caller), user_id: str | None = None) -> dict[str, Any]:
        u, _ = scope(c, user_id, None)
        if u is None:
            raise HTTPException(400, "user_id required")
        return mem.export(user_id=u)

    @app.delete("/v1/users/{user_id}")
    def forget_user(user_id: str, c: Caller = Depends(caller)) -> dict[str, int]:
        u, _ = scope(c, user_id, None)
        return _call(mem.forget, user_id=u, everything=True)

    # ------------------------------------------------------------ admin: keys
    @app.get("/v1/keys")
    def list_keys(c: Caller = Depends(caller)) -> list[dict[str, Any]]:
        admin_only(c)
        return auth.list_keys(mem.store)

    @app.post("/v1/keys")
    def create_key(b: KeyBody, c: Caller = Depends(caller)) -> dict[str, Any]:
        admin_only(c)
        try:
            key = auth.create_key(
                mem.store, role=b.role, user_id=b.user_id, agent_id=b.agent_id, label=b.label
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        mem.ledger.append("KEY_CREATE", key[:10], {"role": b.role, "user": b.user_id})
        return {"key": key, "prefix": key[:10], "note": "store it now; it is not shown again"}

    @app.delete("/v1/keys/{prefix}")
    def revoke_key(prefix: str, c: Caller = Depends(caller)) -> dict[str, int]:
        admin_only(c)
        n = _call(auth.revoke_key, mem.store, prefix)
        mem.ledger.append("KEY_REVOKE", prefix, {"revoked": n})
        return {"revoked": n}

    return app


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' "
        "'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'self'"
    ),
}


class _RateLimiter:
    """Token bucket per caller: `per_minute` requests, bursting up to the same amount."""

    def __init__(self, per_minute: int):
        self.rate = per_minute / 60.0
        self.cap = float(per_minute)
        self.state: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.rate <= 0:
            return True
        t = time.monotonic()
        with self.lock:
            tokens, last = self.state.get(key, (self.cap, t))
            tokens = min(self.cap, tokens + (t - last) * self.rate)
            if tokens < 1:
                self.state[key] = (tokens, t)
                return False
            self.state[key] = (tokens - 1, t)
            return True


def _jwt_from_env() -> auth.JWTVerifier | None:
    jwks = os.environ.get("VERAMEM_JWT_JWKS_URL")
    secret = os.environ.get("VERAMEM_JWT_SECRET")
    if not jwks and not secret:
        return None
    return auth.JWTVerifier(
        jwks_url=jwks,
        secret=secret,
        audience=os.environ.get("VERAMEM_JWT_AUDIENCE"),
        issuer=os.environ.get("VERAMEM_JWT_ISSUER"),
    )
