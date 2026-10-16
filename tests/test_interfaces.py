"""REST API, MCP server and CLI wiring."""

import json

from fastapi.testclient import TestClient

from veramem.server import create_app


def test_rest_roundtrip(mem):
    c = TestClient(create_app(mem))
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/v1/memories", json={"messages": "Olga collects stamps", "user_id": "olga"})
    assert r.status_code == 200 and r.json()["changes"][0]["op"] == "ADD"
    s = c.post("/v1/search", json={"query": "what does Olga collect", "user_id": "olga"}).json()
    mid = s["hits"][0]["memory"]["id"]
    assert "stamps" in s["context"]
    assert c.get(f"/v1/memories/{mid}/evidence", params={"user_id": "olga"}).status_code == 200
    assert c.get(f"/v1/memories/{mid}", params={"user_id": "intruder"}).status_code == 403
    a = c.post("/v1/answer", json={"question": "What does Olga collect?", "user_id": "olga"})
    assert "stamps" in a.json()["answer"]
    fb = c.post("/v1/feedback", json={"trace_id": a.json()["trace_id"], "reward": 1})
    assert fb.status_code == 200
    assert c.get("/v1/verify").json()["ok"]
    assert "VeraMem" in c.get("/").text


def test_rest_api_key(mem, monkeypatch):
    monkeypatch.setenv("VERAMEM_API_KEY", "s3cret")
    c = TestClient(create_app(mem))
    assert c.get("/v1/info").status_code == 401
    assert c.get("/v1/info", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/v1/info", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_mcp_tools_registered(mem, monkeypatch):
    import asyncio

    from veramem.mcp_server import build

    monkeypatch.setenv("VERAMEM_USER", "mcp-user")
    server = build(mem)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"remember", "recall", "ask", "feedback", "check_action", "forget"} <= names
    res = asyncio.run(server.call_tool("remember", {"content": "Pia speaks Finnish"}))
    assert res is not None
    assert mem.get_all(user_id="mcp-user")[0].text == "Pia speaks Finnish"


def test_export_import_roundtrip(mem, tmp_path):
    from veramem import Memory
    from veramem.providers import HashEmbedder, NoLLM

    mem.add(["Raj plays cricket", "Raj lives in Pune"], user_id="raj")
    data = json.loads(json.dumps(mem.export(user_id="raj"), default=str))
    other = Memory(str(tmp_path / "other.db"), llm=NoLLM(), embedder=HashEmbedder())
    assert other.import_(data) == {"episodes": 2, "memories": 2}
    assert other.search("where does Raj live", user_id="raj").hits
    assert other.verify()["ok"]


def test_cli_smoke(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from veramem.cli import app

    monkeypatch.setenv("VERAMEM_HOME", str(tmp_path))
    monkeypatch.setenv("VERAMEM_LLM", "none")
    monkeypatch.setenv("VERAMEM_EMBEDDER", "hash")
    run = CliRunner()
    assert run.invoke(app, ["add", "I keep spare keys under the mat", "-u", "t"]).exit_code == 0
    out = run.invoke(app, ["search", "where are the keys", "-u", "t"]).output
    assert "mat" in out
    assert run.invoke(app, ["verify"]).exit_code == 0
