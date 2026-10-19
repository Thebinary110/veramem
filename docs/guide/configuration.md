# Configuration

Settings are resolved in this order: explicit arguments, then environment variables,
then `~/.veramem/config.json` (written by `veramem init`), then auto-detection.

| Variable | Meaning | Default |
|---|---|---|
| `VERAMEM_HOME` | directory for config and database | `~/.veramem` |
| `VERAMEM_DB` | SQLite file | `$VERAMEM_HOME/memory.db` |
| `VERAMEM_LLM` | `ollama` · `openai` · `anthropic` · `none` | auto |
| `VERAMEM_LLM_MODEL` | model name | provider default |
| `VERAMEM_LLM_BASE_URL` | server URL (OpenAI-compatible or Ollama) | provider default |
| `VERAMEM_EMBEDDER` | `ollama` · `openai` · `hash` | auto |
| `VERAMEM_EMBED_MODEL` | embedding model | `nomic-embed-text` / `text-embedding-3-small` |
| `VERAMEM_CACHE` | file for an LLM response cache (replayable runs) | off |
| `VERAMEM_DB` set to `postgresql://veramem@db.example.com/veramem` | use PostgreSQL instead of a SQLite file (`pip install "veramem[postgres]"`); give the password via `PGPASSWORD` or `~/.pgpass`, not in the URL | |
| `VERAMEM_API_KEY` | an admin bearer token for the REST API | unset |
| `VERAMEM_JWT_JWKS_URL` / `VERAMEM_JWT_SECRET` | accept OIDC/JWT bearer tokens (`sub` = user) | unset |
| `VERAMEM_JWT_AUDIENCE` / `VERAMEM_JWT_ISSUER` | required JWT audience / issuer | unset |
| `VERAMEM_RATE_LIMIT` | requests per minute per caller (0 = off) | `600` |
| `VERAMEM_MAX_BODY_BYTES` | largest accepted request body | `2097152` |
| `VERAMEM_CORS_ORIGINS` | comma-separated origins allowed to call the API from a browser | unset (CORS off) |
| `VERAMEM_VECTOR_INDEX` | `exact` · `auto` · `hnsw` (`pip install "veramem[ann]"`) | `exact` |
| `VERAMEM_RANKING` | `contextual` (default) · `ranker` · `global` · `none` | `contextual` |
| `VERAMEM_EXPLORATION` | `none` · `thompson` · `ucb` | `none` |
| `VERAMEM_USER` / `VERAMEM_AGENT` | identity used by the MCP server | `default` |
| `OLLAMA_HOST`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ANTHROPIC_API_KEY` | provider settings | |

Auto-detection order:
1. A running Ollama server with a chat model.
2. `OPENAI_API_KEY`.
3. `ANTHROPIC_API_KEY`.
4. No model: raw-message mode.

If you change the embedding model, the next start re-embeds stored data automatically
(`VERAMEM_AUTO_REINDEX=0` turns this off; `Memory.reindex()` runs it by hand).

## Limits

Measured on one machine (`python -m benchmarks.scale`, model-free, 768-d vectors):

| Memories in one user's scope | Search p50, exact index |
|---:|---:|
| 100,000 | 11 ms |
| 300,000 | 34 ms |
| 1,000,000 | 171 ms |

- **Writes are cheap:** about 0.5 ms per memory in raw mode. The first search after a write stays fast (30 ms at 100k), because caches append new rows instead of reloading.
- **Many users share a database without slowing each other down:** one user's search in a 100,000-memory, 50-user store takes 12 ms p50.
- **`vector_index="hnsw"` is experimental.** In the same tests it was *slower* than exact search up to 300k memories (19 ms vs 11 ms at 100k), with 69–82% top-10 agreement. Use it only when RAM, not latency, is the constraint, at millions of memories per scope. Exact search is the default because it is both faster and exact at every size measured.
