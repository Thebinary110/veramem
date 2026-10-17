# Integrations

## MCP (Claude Desktop, Claude Code, Cursor, VS Code)

```bash
pip install "veramem[mcp]"
veramem mcp-config -u your-name     # prints the JSON to paste into your client
```

Tools exposed: `remember`, `recall`, `ask`, `feedback`, `check_action`, `correct`,
`forget`, `history`. The acting user comes from `VERAMEM_USER`; the model can't pick
another user's id.

## REST API

```bash
VERAMEM_API_KEY=$(openssl rand -hex 24) veramem serve --host 0.0.0.0
```

The interactive OpenAPI docs are at `/docs`. Main endpoints:

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/memories` | add messages |
| POST | `/v1/search` | search, returns `context` ready for a prompt |
| POST | `/v1/answer` | grounded answer with citations |
| POST | `/v1/feedback` | outcome reward for a trace |
| GET | `/v1/memories/{id}/evidence` | resolve evidence pointers |
| GET | `/v1/memories/{id}/history` | version chain and ledger hashes |
| POST | `/v1/grants` | grant or revoke user↔agent, agent↔resource, team membership |
| POST | `/v1/check-action` | authority gate |
| GET | `/v1/verify` | ledger verification |
| DELETE | `/v1/users/{id}` | right to be forgotten |

!!! warning
    The REST API trusts the `user_id` you send. Put it behind your application's own
    authentication; don't expose it directly to end users.

## Docker

```bash
VERAMEM_API_KEY=... docker compose up -d     # VeraMem + Ollama
docker compose exec ollama ollama pull qwen3:8b
docker compose exec ollama ollama pull nomic-embed-text
```

## Any OpenAI-compatible model server

```bash
export VERAMEM_LLM=openai
export VERAMEM_LLM_BASE_URL=http://localhost:30000/v1   # vLLM, SGLang, LM Studio, llama.cpp
export VERAMEM_LLM_MODEL=Qwen/Qwen3-32B
export OPENAI_API_KEY=unused
```
