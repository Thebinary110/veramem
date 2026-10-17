# VeraMem

**Long-term memory for AI agents that you can verify, that learns from outcomes, and that
is safe to share.**

- One SQLite file, no servers to run.
- Works fully offline with Ollama, or with OpenAI / Anthropic.
- Python SDK, REST API with a web UI, and an MCP server for Claude Desktop, Cursor and Claude Code.

```bash
pip install "veramem[mcp]"
veramem init          # finds Ollama / OpenAI / Anthropic and sets everything up
veramem add "I'm vegetarian and I moved to Lisbon last week"
veramem ask "where do I live?"
veramem serve         # web UI on http://127.0.0.1:8765
```

```python
from veramem import Memory

mem = Memory()                                    # auto-detects your model provider
mem.add("I moved to Lisbon last week", user_id="priya")
mem.answer("Where does Priya live?", user_id="priya")
# Answer(answer='Lisbon', citations=['mem_…'], abstained=False, trace_id='tr_…')
```

---

## Why another memory system?

Today's memory layers (Mem0, Letta, Graphiti/Zep, MemOS, TencentDB-Agent-Memory) have three
gaps. VeraMem fills each one with techniques from recent research:

| | The gap | What VeraMem does |
|---|---|---|
| **A** | Memories are kept or dropped by heuristics. Nothing learns whether a memory actually *helped*. | `feedback(trace_id, reward)` credits the memories an answer cited. Utility re-ranks retrieval and archives memories that keep misleading, and successful tasks become reusable experiences. Rated traces export as GRPO training data. *(Memory-R1, AgeMem)* |
| **B** | Extracted "facts" lose the link to what was said. A tool's claim quietly becomes "the user's preference". | Every memory points to a **verbatim, verified quote** in an immutable, hashed source message. Each carries a **source-authority** label (user / assistant / tool) that consolidation can never raise. Updates are versioned and time-travel queryable. Every write goes into a **hash-chained ledger**. Answers cite or abstain, and `check_action()` blocks low-authority memories from driving payments, bookings or sends. *(Agent Zero Memory, AMA, AuthMem-Bench)* |
| **C** | Sharing is all-or-nothing. Private data leaks through shared pools, dedup, or other users' preferences. | Access rules are compiled into SQL **before** ranking. Revocable user↔agent and agent↔resource grants apply retroactively. Preferences can never be shared. Experiences are shared through a two-stage pipeline with de-identification plus a **deterministic PII/canary gate**. Erasure is physical. *(ShareMem, Collaborative Memory, AIM)* |

The paper-by-paper mapping is in [docs/RESEARCH.md](docs/RESEARCH.md), and the design is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Benchmarks

Measured head-to-head on the same machine, with the same local model (`qwen3.6:35b`,
thinking off, temperature 0) and the same embedder (`nomic-embed-text`) for every system.
The full method and raw outputs are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

<!-- BENCHMARK_TABLE -->

Reproduce:
```bash
python -m benchmarks.guarantees --systems veramem,mem0
python -m benchmarks.locomo --systems veramem,mem0 --convs all
```

## What you get

| Feature | API | CLI |
|---|---|---|
| Remember chats, tool results, documents (role sets authority) | `add(messages, user_id=…)` | `veramem add` |
| Hybrid search (dense + BM25 + entity, routed by granularity) | `search(q, user_id=…)` | `veramem search` |
| Grounded answers with citations and abstention | `answer(q, user_id=…)` | `veramem ask` |
| Learn from outcomes | `feedback(trace_id, reward)`, `learn(task, trajectory, reward)` | `veramem feedback` |
| Evidence and version history | `evidence(id)`, `history(id)`, `search(…, as_of=ts)` | `veramem history` |
| Authority gate for actions | `check_action([ids])` | |
| Teams, grants, revocation | `join_team`, `grant_agent`, `revoke_agent`, `grant_resource` | |
| Safe sharing | `set_shareable`, `share(user_id, team_id)` | |
| Tamper check | `verify()` | `veramem verify` |
| Right to be forgotten / data export | `forget(user_id=…, everything=True)`, `export(user_id)` | `veramem forget`, `veramem export` |
| Training data for your own memory model | `export_trajectories(path)` | `veramem export-trajectories` |
| Maintenance (archive, dedupe, summarise) | `consolidate()` | `veramem consolidate` |

### Use it from any app

- **MCP** (Claude Desktop, Claude Code, Cursor, VS Code): `veramem mcp-config` prints the
  config to paste. See [examples/mcp_clients.md](examples/mcp_clients.md).
- **REST**: `veramem serve` serves interactive docs at `/docs`. Set `VERAMEM_API_KEY` to
  require a bearer token; the CLI refuses to listen publicly without one.
- **Docker**: `VERAMEM_API_KEY=… docker compose up` starts VeraMem plus Ollama, fully local.

### Configuration

`veramem init` writes `~/.veramem/config.json`. You can override any setting with an
environment variable:

| Variable | Meaning |
|---|---|
| `VERAMEM_LLM` | `ollama` · `openai` · `anthropic` · `none` |
| `VERAMEM_LLM_MODEL`, `VERAMEM_LLM_BASE_URL` | e.g. `qwen3:8b`, or any OpenAI-compatible URL (vLLM, LM Studio, Groq, Together) |
| `VERAMEM_EMBEDDER`, `VERAMEM_EMBED_MODEL` | `ollama` · `openai` · `hash` (no model at all) |
| `VERAMEM_DB` | Path of the SQLite file |
| `VERAMEM_API_KEY` | Bearer token for the REST API |

With no model at all, VeraMem still works. It stores and searches raw messages, with
the same access control and ledger.

## Security and trust

- No telemetry, ever. The only network calls go to the model provider you configure.
- Releases are built in CI with SLSA provenance, signed with Sigstore, and published to
  PyPI with OIDC trusted publishing.
- Contributions follow a vouched-contributor model: open an issue first. See
  [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and [GOVERNANCE.md](GOVERNANCE.md).

## License

Apache-2.0. Some prompts are adapted from mem0 and AMA (both Apache-2.0); see [NOTICE](NOTICE).
