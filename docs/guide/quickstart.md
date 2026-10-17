# Quickstart

## Install

```bash
pip install "veramem[mcp]"     # Python 3.10+
veramem init                  # detects Ollama / OpenAI / Anthropic
veramem doctor                # checks the whole pipeline end to end
```

Without any model you can still use VeraMem. It stores and searches raw messages,
with the same access control and audit ledger. For fully local use, install
[Ollama](https://ollama.com) and run `ollama pull qwen3:8b && ollama pull nomic-embed-text`.

## Python

```python
from veramem import Memory

mem = Memory()

mem.add(
    [
        {"role": "user", "content": "I'm Priya. I moved to Lisbon last week."},
        {"role": "tool", "name": "crm", "content": "customer record: home_city=Porto"},
    ],
    user_id="priya",
)

for m in mem.get_all(user_id="priya"):
    print(m.authority.value, m.text)
# authorized  Priya moved to Lisbon in the week of 2026-10-03.
# unendorsed  According to crm, the customer's home city is Porto.

a = mem.answer("Where does Priya live?", user_id="priya")
print(a.answer, a.citations)  # Lisbon ['mem_…']

mem.feedback(a.trace_id, 1.0)  # tell VeraMem the memory helped
```

Async applications can use `AsyncMemory`, which has the same API with awaitable methods.

## Command line

```bash
veramem add "My dentist is Dr. Ortiz" -u priya
veramem search "dentist" -u priya
veramem ask "who is my dentist?" -u priya
veramem list -u priya --all            # include superseded versions
veramem verify                         # check the audit ledger
veramem forget -u priya                # erase everything about a user
```

## Web UI and REST API

```bash
veramem serve        # http://127.0.0.1:8765  (API docs at /docs)
```
