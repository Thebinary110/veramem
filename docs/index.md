# VeraMem

**Long-term memory for AI agents that you can verify, that learns from outcomes, and that
is safe to share.**

- One SQLite file, no servers to run.
- Works offline with Ollama, or with OpenAI / Anthropic / any OpenAI-compatible server
  (vLLM, SGLang, LM Studio).

```bash
pip install "veramem[mcp]"
veramem init
veramem add "I'm vegetarian and I moved to Lisbon last week"
veramem ask "where do I live?"
```

| Gap in today's memory systems | What VeraMem does |
|---|---|
| Nothing learns whether a memory actually helped | `feedback()` credits the memories an answer cited. Utility re-ranks retrieval and archives memories that keep misleading. |
| Facts lose the link to what was said; a tool's claim becomes "the user's preference" | Verified quotes and source-authority labels on every memory, a hash-chained ledger, and an action gate for low-authority memories |
| Sharing leaks private data | Access is enforced in SQL before ranking, grants are revocable, preferences are never shared, and a PII/canary gate runs before sharing |

Start with the [Quickstart](guide/quickstart.md), or read how it works in
[Concepts](guide/concepts.md).
