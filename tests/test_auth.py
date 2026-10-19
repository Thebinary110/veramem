"""REST authentication and authorisation."""

import time

import pytest
from fastapi.testclient import TestClient

from veramem import auth
from veramem.server import create_app


def _client(mem, monkeypatch, **env):
    for k in (
        "VERAMEM_API_KEY",
        "VERAMEM_JWT_SECRET",
        "VERAMEM_JWT_JWKS_URL",
        "VERAMEM_RATE_LIMIT",
    ):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return TestClient(create_app(mem))


def _h(key):
    return {"Authorization": f"Bearer {key}"}


def test_open_mode_without_any_auth(mem, monkeypatch):
    c = _client(mem, monkeypatch)
    assert c.get("/v1/me").json()["role"] == "anonymous"
    assert c.get("/health").json()["auth"] is False


def test_user_key_is_pinned_to_its_user(mem, monkeypatch):
    alice = auth.create_key(mem.store, user_id="alice")
    c = _client(mem, monkeypatch)
    assert c.get("/v1/info").status_code == 401  # auth now configured: anonymous refused
    r = c.post("/v1/memories", json={"messages": "Alice likes tea"}, headers=_h(alice))
    assert r.status_code == 200
    assert mem.get_all(user_id="alice")  # user_id taken from the key
    bad = c.post("/v1/memories", json={"messages": "x", "user_id": "bob"}, headers=_h(alice))
    assert bad.status_code == 403
    assert c.get("/v1/memories", params={"user_id": "bob"}, headers=_h(alice)).status_code == 403
    s = c.post("/v1/search", json={"query": "tea"}, headers=_h(alice)).json()
    assert s["hits"] and all(h["memory"]["user_id"] == "alice" for h in s["hits"])


def test_user_key_cannot_touch_other_users_data(mem, monkeypatch):
    mid = mem.add("Bob's PIN is 4321", user_id="bob").changes[0].memory_id
    r = mem.search("PIN", user_id="bob")
    alice = auth.create_key(mem.store, user_id="alice")
    c = _client(mem, monkeypatch)
    h = _h(alice)
    assert c.get(f"/v1/memories/{mid}", headers=h).status_code == 403
    assert c.get(f"/v1/memories/{mid}/evidence", headers=h).status_code == 403
    assert c.delete(f"/v1/memories/{mid}", headers=h).status_code == 403
    gate = c.post("/v1/check-action", json={"memory_ids": [mid]}, headers=h).json()
    assert gate["missing"] == [mid] and "4321" not in str(gate)
    fb = c.post("/v1/feedback", json={"trace_id": r.trace_id, "reward": -1}, headers=h)
    assert fb.status_code == 403
    assert c.delete("/v1/users/bob", headers=h).status_code == 403
    assert c.get("/v1/export", params={"user_id": "bob"}, headers=h).status_code == 403


def test_admin_only_endpoints_and_grants(mem, monkeypatch):
    admin = auth.create_key(mem.store, role="admin")
    alice = auth.create_key(mem.store, user_id="alice")
    c = _client(mem, monkeypatch)
    for path in ("/v1/verify", "/v1/keys"):
        assert c.get(path, headers=_h(alice)).status_code == 403
        assert c.get(path, headers=_h(admin)).status_code == 200
    own = {"kind": "user_agent", "subject": "alice", "object": "bot"}
    assert c.post("/v1/grants", json=own, headers=_h(alice)).status_code == 200
    team = {"kind": "team_member", "subject": "alice", "object": "finance"}
    assert c.post("/v1/grants", json=team, headers=_h(alice)).status_code == 403
    assert c.post("/v1/grants", json=team, headers=_h(admin)).status_code == 200
    t = c.get("/v1/teams", headers=_h(alice)).json()
    assert t["teams"] == ["finance"] and t["agents"] == ["bot"]
    assert c.get("/v1/info", headers=_h(alice)).json().keys() <= {"llm", "embedder"}


def test_key_lifecycle_and_env_admin(mem, monkeypatch):
    c = _client(mem, monkeypatch, VERAMEM_API_KEY="env-secret")
    created = c.post("/v1/keys", json={"user_id": "carol"}, headers=_h("env-secret")).json()
    key = created["key"]
    assert c.get("/v1/me", headers=_h(key)).json() == {
        "role": "user",
        "user_id": "carol",
        "agent_id": None,
    }
    assert c.delete(f"/v1/keys/{created['prefix']}", headers=_h("env-secret")).json() == {
        "revoked": 1
    }
    assert c.get("/v1/me", headers=_h(key)).status_code == 401
    listed = c.get("/v1/keys", headers=_h("env-secret")).json()
    assert key not in str(listed)  # only hashes and prefixes are stored
    assert (
        mem.store.db.execute("SELECT COUNT(*) FROM api_keys WHERE key_hash=?", (key,)).fetchone()[0]
        == 0
    )


def test_jwt_bearer_tokens(mem, monkeypatch):
    jwt = pytest.importorskip("jwt")
    c = _client(mem, monkeypatch, VERAMEM_JWT_SECRET="s" * 32)
    tok = jwt.encode({"sub": "dana", "exp": int(time.time()) + 60}, "s" * 32, algorithm="HS256")
    assert c.get("/v1/me", headers=_h(tok)).json()["user_id"] == "dana"
    expired = jwt.encode({"sub": "dana", "exp": int(time.time()) - 5}, "s" * 32, "HS256")
    assert c.get("/v1/me", headers=_h(expired)).status_code == 401
    forged = jwt.encode({"sub": "dana", "exp": int(time.time()) + 60}, "x" * 32, "HS256")
    assert c.get("/v1/me", headers=_h(forged)).status_code == 401


def test_rate_limit_body_limit_and_headers(mem, monkeypatch):
    key = auth.create_key(mem.store, user_id="erin")
    c = _client(mem, monkeypatch, VERAMEM_RATE_LIMIT="5")
    codes = [c.get("/v1/me", headers=_h(key)).status_code for _ in range(8)]
    assert codes[:5] == [200] * 5 and 429 in codes[5:]
    r = c.get("/health")
    assert r.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    monkeypatch.setenv("VERAMEM_MAX_BODY_BYTES", "100")
    big = TestClient(create_app(mem)).post(
        "/v1/memories", content="x" * 500, headers={**_h(key), "content-type": "application/json"}
    )
    assert big.status_code == 413


def test_cli_keys(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from veramem.cli import app

    monkeypatch.setenv("VERAMEM_HOME", str(tmp_path))
    monkeypatch.setenv("VERAMEM_LLM", "none")
    monkeypatch.setenv("VERAMEM_EMBEDDER", "hash")
    run = CliRunner()
    out = run.invoke(app, ["keys", "create", "-u", "fay", "--label", "laptop"])
    assert out.exit_code == 0 and out.output.startswith("vm_")
    prefix = out.output.strip().splitlines()[0][:10]
    assert "fay" in run.invoke(app, ["keys", "list"]).output
    assert "revoked 1" in run.invoke(app, ["keys", "revoke", prefix]).output
    assert run.invoke(app, ["keys", "create"]).exit_code == 2  # user key needs a user
