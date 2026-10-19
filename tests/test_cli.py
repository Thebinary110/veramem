"""CLI commands end to end (model-free)."""

import json

import pytest
from typer.testing import CliRunner

from veramem.cli import app


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("VERAMEM_HOME", str(tmp_path))
    monkeypatch.setenv("VERAMEM_LLM", "none")
    monkeypatch.setenv("VERAMEM_EMBEDDER", "hash")
    runner = CliRunner()
    return lambda *args: runner.invoke(app, list(args))


def test_status_version_list_history(run):
    assert run("version").exit_code == 0
    assert run("add", "My bike is green", "-u", "zoe").exit_code == 0
    status = json.loads(run("status").output)
    assert status["memories_active"] == 1
    listing = run("list", "-u", "zoe").output
    mid = listing.split()[0]
    assert mid.startswith("mem_") and "bike is green" in listing
    hist = json.loads(run("history", mid, "-u", "zoe").output)
    assert hist[0]["id"] == mid
    found = json.loads(run("search", "bike", "-u", "zoe", "--json").output)
    assert found["hits"] and found["hits"][0]["memory"]["id"] == mid


def test_export_import_forget(run, tmp_path):
    run("add", "Zoe's locker is 12", "-u", "zoe")
    out = tmp_path / "zoe.json"
    assert run("export", "-u", "zoe", "--out", str(out)).exit_code == 0
    data = json.loads(out.read_text())
    assert data["format"] == "veramem.export.v1" and data["memories"]
    assert json.loads(run("forget", "-u", "zoe", "--yes").output)["memories"] == 1
    assert json.loads(run("import", str(out)).output)["memories"] == 1


def test_consolidate_feedback_and_trajectories(run, tmp_path):
    run("add", "Zoe drinks oolong", "-u", "zoe")
    trace = run("search", "oolong", "-u", "zoe").output.split("trace=")[1].split()[0]
    assert run("feedback", trace, "1").exit_code == 0
    assert "archived" in run("consolidate", "-u", "zoe").output
    assert "wrote" in run("export-trajectories", str(tmp_path / "t.jsonl")).output


def test_mcp_config_doctor_and_verify(run):
    cfg = json.loads(run("mcp-config", "-u", "zoe").stdout.split("\n\n")[0])
    assert cfg["mcpServers"]["veramem"]["env"]["VERAMEM_USER"] == "zoe"
    assert run("doctor").exit_code == 0
    assert run("verify").exit_code == 0


def test_serve_refuses_public_bind_without_auth(run, monkeypatch):
    monkeypatch.delenv("VERAMEM_API_KEY", raising=False)
    r = run("serve", "--host", "0.0.0.0")
    assert r.exit_code == 2 and "Refusing" in r.output
