"""`veramem` command line."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import typer

from .__about__ import __version__
from .config import Config, home, ollama_models, pick_ollama

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="VeraMem: verifiable long-term memory for AI agents.",
)


def _load_keys() -> None:
    cfg = Config.load()
    for var, key in (
        ("OPENAI_API_KEY", "openai_api_key"),
        ("ANTHROPIC_API_KEY", "anthropic_api_key"),
    ):
        if not os.environ.get(var) and cfg.extra.get(key):
            os.environ[var] = cfg.extra[key]


def _mem(db: str | None = None):
    _load_keys()
    from .memory import Memory

    return Memory(db)


def _out(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command()
def init(yes: bool = typer.Option(False, "--yes", "-y", help="accept detected defaults")) -> None:
    """Guided setup: pick a model provider and create ~/.veramem."""
    cfg = Config.load()
    typer.secho("VeraMem setup", bold=True)
    models = ollama_models()
    chat, emb = pick_ollama(models)
    if chat:
        typer.echo(f"Found Ollama with model '{chat}'" + (f" and embedder '{emb}'" if emb else ""))
        if yes or typer.confirm("Use it (fully local, no API key)?", default=True):
            cfg.llm_provider, cfg.llm_model = "ollama", chat
            cfg.embed_provider, cfg.embed_model = ("ollama", emb) if emb else ("hash", "")
            if not emb:
                typer.echo("Tip: `ollama pull nomic-embed-text` gives better search.")
    if cfg.llm_provider in ("auto", ""):
        choice = (
            "none"
            if yes
            else typer.prompt("Provider? [openai/anthropic/ollama/none]", default="openai")
        )
        choice = choice.strip().lower()
        if choice in ("openai", "anthropic"):
            var = f"{choice.upper()}_API_KEY"
            key = os.environ.get(var) or typer.prompt(f"{var}", hide_input=True)
            cfg.extra[f"{choice}_api_key"] = key
            cfg.llm_provider = choice
            cfg.embed_provider = "openai" if choice == "openai" else "hash"
        elif choice == "ollama":
            typer.echo(
                "Install Ollama from https://ollama.com, then run:\n"
                "  ollama pull qwen3:8b && ollama pull nomic-embed-text\n"
                "and re-run `veramem init`."
            )
            cfg.llm_provider = "ollama"
        else:
            cfg.llm_provider = "none"
            typer.echo("No model: VeraMem will store and search raw messages only.")
    p = cfg.save()
    if cfg.extra:
        os.chmod(p, 0o600)
    typer.secho(f"Saved {p}", fg="green")
    mem = _mem()
    typer.echo(f"Database: {mem.store.path}")
    typer.echo(f"LLM: {mem.info()['llm']} | embedder: {mem.info()['embedder']}")
    typer.echo(
        '\nNext:\n  veramem add "I prefer window seats"\n  veramem ask "what seat do I like?"\n'
        "  veramem serve        # web UI at http://127.0.0.1:8765\n"
        "  veramem mcp-config   # connect Claude Desktop / Cursor"
    )


@app.command()
def status() -> None:
    """Show configuration and counts."""
    _out(_mem().info())


@app.command()
def add(
    text: str,
    user: str = typer.Option("default", "--user", "-u"),
    role: str = typer.Option("user", help="user | assistant | tool | document"),
    session: str = typer.Option(None, "--session", "-s"),
    raw: bool = typer.Option(False, help="store verbatim, no LLM extraction"),
) -> None:
    """Remember something."""
    r = _mem().add({"role": role, "content": text}, user_id=user, session_id=session, infer=not raw)
    for c in r.changes:
        typer.echo(f"{c.op.value:7} {c.text or ''}" + (f"  ({c.reason})" if c.reason else ""))
    for x in r.rejected:
        typer.secho(f"REJECTED {x.get('text')}: {x['reason']}", fg="yellow")
    if not r.changes and not r.rejected:
        typer.echo("nothing new to remember")


@app.command()
def search(
    query: str,
    user: str = typer.Option("default", "--user", "-u"),
    limit: int = 8,
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Search memory."""
    r = _mem().search(query, user_id=user, limit=limit)
    if as_json:
        _out(r.model_dump(mode="json"))
        return
    typer.echo(f"route={r.granularity} trace={r.trace_id}")
    for h in r.hits:
        typer.echo(f"{h.score:.3f} [{h.memory.authority.value[:4]}] {h.memory.text}")


@app.command()
def ask(question: str, user: str = typer.Option("default", "--user", "-u")) -> None:
    """Answer from memory, with citations."""
    mem = _mem()
    a = mem.answer(question, user_id=user)
    typer.secho(a.answer, bold=True)
    for cid in a.citations:
        m = mem.store.get_memory(cid)
        if m:
            typer.echo(f"  [{m.authority.value}] {m.text}")
    typer.echo(f"trace={a.trace_id}  (rate it: veramem feedback {a.trace_id} 1)")


@app.command()
def feedback(trace_id: str, reward: float) -> None:
    """Tell VeraMem whether a recall helped (-1..1)."""
    _out(_mem().feedback(trace_id, reward))


@app.command("list")
def list_(
    user: str = typer.Option("default", "--user", "-u"),
    all_: bool = typer.Option(False, "--all", help="include superseded/retracted"),
) -> None:
    """List memories."""
    for m in _mem().get_all(user_id=user, include_inactive=all_):
        flag = "" if m.status.value == "active" else f" ({m.status.value})"
        typer.echo(f"{m.id}  [{m.authority.value[:4]}] {m.kind.value:10} {m.text}{flag}")


@app.command()
def history(memory_id: str, user: str = typer.Option("default", "--user", "-u")) -> None:
    """Show the version history of a memory."""
    _out(_mem().history(memory_id, user_id=user))


@app.command()
def verify() -> None:
    """Verify the audit ledger and content hashes."""
    rep = _mem().verify()
    _out(rep)
    raise typer.Exit(0 if rep["ok"] else 1)


@app.command()
def consolidate(user: str = typer.Option(None, "--user", "-u")) -> None:
    """Archive failing memories, merge duplicates, summarise sessions."""
    _out(_mem().consolidate(user_id=user))


@app.command()
def forget(
    user: str = typer.Option(..., "--user", "-u"),
    memory_id: str = typer.Option(None),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    """Permanently erase one memory, or everything about a user."""
    what = f"memory {memory_id}" if memory_id else f"EVERYTHING about user '{user}'"
    if not yes and not typer.confirm(f"Permanently erase {what}?"):
        raise typer.Abort()
    mem = _mem()
    _out(
        mem.forget(memory_id, user_id=user)
        if memory_id
        else mem.forget(user_id=user, everything=True)
    )


@app.command()
def export(user: str = typer.Option(..., "--user", "-u"), out: Path = typer.Option(None)) -> None:
    """Export everything about a user as portable JSON."""
    data = json.dumps(_mem().export(user_id=user), indent=2, default=str)
    if out:
        out.write_text(data)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(data)


@app.command("import")
def import_(path: Path) -> None:
    """Import a VeraMem export file."""
    _out(_mem().import_(json.loads(path.read_text())))


@app.command("export-trajectories")
def export_trajectories(out: Path, user: str = typer.Option(None, "--user", "-u")) -> None:
    """Write GRPO-ready training records from rated traces."""
    n = _mem().export_trajectories(str(out), user_id=user)
    typer.echo(f"wrote {n} records to {out}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the REST API and web UI."""
    if host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("VERAMEM_API_KEY"):
        typer.secho("Refusing to listen on a public interface without VERAMEM_API_KEY.", fg="red")
        raise typer.Exit(2)
    import uvicorn

    from .server import create_app

    typer.echo(f"VeraMem UI: http://{host}:{port}   API docs: http://{host}:{port}/docs")
    uvicorn.run(create_app(_mem()), host=host, port=port, log_level="warning")


@app.command()
def mcp() -> None:
    """Run as an MCP server over stdio (for Claude Desktop, Cursor, ...)."""
    _load_keys()
    try:
        from .mcp_server import main
    except ImportError:
        typer.secho("MCP support needs: pip install 'veramem[mcp]'", fg="red", err=True)
        raise typer.Exit(1) from None
    main()


@app.command("mcp-config")
def mcp_config(user: str = typer.Option("default", "--user", "-u")) -> None:
    """Print the JSON to paste into your MCP client's config."""
    exe = Path(sys.argv[0]).resolve()
    cfg = {
        "mcpServers": {
            "veramem": {"command": str(exe), "args": ["mcp"], "env": {"VERAMEM_USER": user}}
        }
    }
    typer.echo(json.dumps(cfg, indent=2))
    typer.echo(
        "\nClaude Desktop: Settings > Developer > Edit Config, paste under mcpServers.\n"
        "Cursor: ~/.cursor/mcp.json   |   Claude Code: claude mcp add veramem -- "
        f"{exe} mcp",
        err=True,
    )


@app.command()
def doctor() -> None:
    """Check that everything works end to end."""
    ok = True
    typer.echo(f"veramem {__version__}, python {sys.version.split()[0]}, home {home()}")
    t = time.time()
    try:
        mem = _mem()
        info = mem.info()
        typer.echo(f"[ok] database {info['db']}")
        typer.echo(f"[ok] llm {info['llm']}")
        typer.echo(f"[ok] embedder {info['embedder']} (dim {len(mem.embedder.embed_one('x'))})")
        if mem.has_llm:
            mem.llm.complete([{"role": "user", "content": "Reply with OK"}], max_tokens=5)
            typer.echo(f"[ok] llm responds ({time.time() - t:.1f}s)")
        rep = mem.verify()
        typer.echo(f"[{'ok' if rep['ok'] else 'FAIL'}] ledger ({rep['ledger_entries']} entries)")
        ok = rep["ok"]
    except Exception as e:  # noqa: BLE001 - doctor reports everything
        typer.secho(f"[FAIL] {e}", fg="red")
        ok = False
    raise typer.Exit(0 if ok else 1)


if __name__ == "__main__":
    app()
