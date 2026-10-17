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
| `VERAMEM_API_KEY` | bearer token for the REST API | unset (localhost only) |
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

Vector search is exact, with no approximate index. Measured on one machine
(`python -m benchmarks.scale`):

| Memories in one user's scope | Search p50 | Search p95 |
|---:|---:|---:|
| 10,000 | 13 ms | 32 ms |
| 50,000 | 33 ms | 89 ms |
| 100,000 | 69 ms | 117 ms |

Many users in one database don't slow each other down: one user's search in a
100,000-memory, 50-user store takes 12 ms p50.
