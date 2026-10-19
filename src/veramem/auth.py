"""API keys and token verification for the REST server.

Model:
  * an **admin** key may act for any user (operators, backends that do their own auth);
  * a **user** key is bound to one user (and optionally one agent): the server ignores
    any other `user_id` in the request and refuses mismatches;
  * keys are random 160-bit secrets shown once; only their SHA-256 hash is stored;
  * optional JWT/OIDC: a verified token's `sub` claim is the user, and the claim
    `veramem_role: admin` makes it an admin token (needs `pip install "veramem[auth]"`).
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Any

from .store.sqlite import SQLiteStore
from .types import now

KEY_PREFIX = "vm_"


@dataclass(frozen=True)
class Caller:
    role: str  # admin | user | anonymous
    user_id: str | None = None
    agent_id: str | None = None
    key_prefix: str | None = None

    @property
    def is_admin(self) -> bool:
        return self.role in ("admin", "anonymous")


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_key(
    store: SQLiteStore,
    *,
    role: str = "user",
    user_id: str | None = None,
    agent_id: str | None = None,
    label: str | None = None,
) -> str:
    """Create a key and return it. It cannot be shown again."""
    if role not in ("admin", "user"):
        raise ValueError("role must be admin or user")
    if role == "user" and not user_id:
        raise ValueError("a user key must be bound to a user_id")
    key = KEY_PREFIX + secrets.token_urlsafe(30)
    with store._lock:
        store.db.execute(
            "INSERT INTO api_keys (key_hash, prefix, role, user_id, agent_id, label, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (_hash(key), key[:10], role, user_id, agent_id, label, now()),
        )
    return key


def lookup(store: SQLiteStore, key: str) -> Caller | None:
    if not key.startswith(KEY_PREFIX):
        return None
    r = store.db.execute(
        "SELECT prefix, role, user_id, agent_id FROM api_keys WHERE key_hash=? AND revoked_at IS NULL",
        (_hash(key),),
    ).fetchone()
    if r is None:
        return None
    store.db.execute("UPDATE api_keys SET last_used_at=? WHERE key_hash=?", (now(), _hash(key)))
    return Caller(
        role=r["role"], user_id=r["user_id"], agent_id=r["agent_id"], key_prefix=r["prefix"]
    )


def list_keys(store: SQLiteStore) -> list[dict[str, Any]]:
    rows = store.db.execute(
        "SELECT prefix, role, user_id, agent_id, label, created_at, revoked_at, last_used_at"
        " FROM api_keys ORDER BY created_at"
    ).fetchall()
    return [dict(r) for r in rows]


def revoke_key(store: SQLiteStore, prefix: str) -> int:
    with store._lock:
        rows = store.db.execute(
            "SELECT key_hash FROM api_keys WHERE prefix=? AND revoked_at IS NULL", (prefix,)
        ).fetchall()
        if len(rows) > 1:
            raise ValueError("prefix matches several keys; use a longer prefix")
        store.db.execute(
            "UPDATE api_keys SET revoked_at=? WHERE prefix=? AND revoked_at IS NULL",
            (now(), prefix),
        )
    return len(rows)


def has_keys(store: SQLiteStore) -> bool:
    return (
        store.db.execute("SELECT 1 FROM api_keys WHERE revoked_at IS NULL LIMIT 1").fetchone()
        is not None
    )


class JWTVerifier:
    """Verify bearer JWTs from an OIDC provider (JWKS URL) or a shared HS256 secret."""

    def __init__(
        self,
        *,
        jwks_url: str | None = None,
        secret: str | None = None,
        audience: str | None = None,
        issuer: str | None = None,
    ):
        try:
            import jwt  # PyJWT
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError('JWT auth needs: pip install "veramem[auth]"') from e
        self.jwt = jwt
        self.audience, self.issuer, self.secret = audience, issuer, secret
        self.jwks = jwt.PyJWKClient(jwks_url) if jwks_url else None

    def verify(self, token: str) -> Caller | None:
        jwt = self.jwt
        try:
            if self.jwks is not None:
                key = self.jwks.get_signing_key_from_jwt(token).key
                algs = ["RS256", "ES256", "PS256", "EdDSA"]
            else:
                key, algs = self.secret, ["HS256"]
            claims = jwt.decode(
                token,
                key,
                algorithms=algs,
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "sub"], "verify_aud": self.audience is not None},
            )
        except Exception:  # noqa: BLE001 - any verification failure is a rejection
            return None
        role = "admin" if claims.get("veramem_role") == "admin" else "user"
        return Caller(role=role, user_id=str(claims["sub"]), agent_id=claims.get("veramem_agent"))
