# VeraMem

![VeraMem: agent memory with receipts](assets/banner-light.png#only-light)
![VeraMem: agent memory with receipts](assets/banner-dark.png#only-dark)

VeraMem is long-term memory for AI agents that keeps track of where every memory came from.
Each memory cites the exact words it was built from, carries a label for who said them
(the user, the assistant, or a tool), and is written to a tamper-evident ledger. Tool claims
stay tool claims, private memory stays private, and an agent can check whether a memory is
safe to act on before it acts.

It runs on a single SQLite file, works offline with Ollama, and needs no server.

```bash
pip install "veramem[mcp]"
veramem init
veramem add "I'm vegetarian and I just moved to Lisbon"
veramem ask "Where do I live?"
```

![Real veramem ask output with cited memories](assets/terminal.png)

## How it works

![How VeraMem writes and reads memory](assets/how-it-works-light.png#only-light)
![How VeraMem writes and reads memory](assets/how-it-works-dark.png#only-dark)

Start with the [Quickstart](guide/quickstart.md), or read the ideas behind it in
[Concepts](guide/concepts.md).
