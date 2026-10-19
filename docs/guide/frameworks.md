# Agent frameworks

VeraMem ships thin adapters for LangChain, LangGraph, LlamaIndex and CrewAI, plus
plain tool functions for any tool-calling loop. Each adapter lives in
`veramem.integrations.*` and imports its framework only when you import the adapter;
`import veramem` never needs one.

Every adapter takes `user_id` when you construct it. Tool arguments and prompts never
carry a user id, so a model can't reach another user's memory. Content the agent
writes is stored as its own claim (Attested authority) unless you choose otherwise.
That keeps `check_action` honest: a value the model wrote down can't authorize a
payment or a send by itself.

The snippets below use a configured `Memory()` (see [Configuration](configuration.md)).
For a model-free try-out, use
`Memory("demo.db", llm=NoLLM(), embedder=HashEmbedder())` from `veramem.providers`.

## Plain tools (OpenAI / Anthropic tool calling)

No extra needed.

```python
from veramem import Memory
from veramem.integrations.tools import VeraMemTools

mem = Memory()
tools = VeraMemTools(mem, user_id="priya")

tools.openai_tools()  # [{"type": "function", "function": {...}}, ...]
tools.anthropic_tools()  # [{"name": ..., "description": ..., "input_schema": {...}}, ...]

# inside your loop, for each tool call the model makes:
result = tools.call("recall", '{"query": "where does priya live?"}')
tools.call("feedback", {"trace_id": result["trace_id"], "reward": 1})
```

The tools are `remember`, `recall`, `feedback` and `check_action`. `call()` rejects
unknown tools and unexpected arguments. By default `remember` accepts
`role` = `assistant | tool | document`. Pass `allow_user_role=True` only if the model
relays the user's exact words, and `infer=True` to run VeraMem's grounded extraction
when the Memory has an LLM.

## LangChain

```bash
pip install "veramem[langchain]"
```

```python
from veramem import Memory
from veramem.integrations.langchain import VeraMemChatMessageHistory, VeraMemRetriever

mem = Memory()
retriever = VeraMemRetriever(memory=mem, user_id="priya", limit=8)
docs = retriever.invoke("where does priya live?")
docs[0].metadata  # {"id", "authority", "kind", "date", "trace_id", "granularity", "score"}
retriever.feedback(1.0)  # outcome for the last search (retriever.last_trace_id)

# Chat history for RunnableWithMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_history = RunnableWithMessageHistory(
    chain,  # your prompt | model chain
    lambda session_id: VeraMemChatMessageHistory(mem, user_id="priya", session_id=session_id),
)
```

Messages map human→user, ai→assistant, tool→tool and system→system. Unknown types
are stored as tool output and never as the user. With an LLM configured, `add`
also extracts long-term memories from the conversation. Tool messages are stored
but not replayed by default (`include_tool_messages=True` changes that), because
the assistant's tool-call message isn't kept and most chat APIs reject a tool
result without it. `clear()` hides earlier turns from the history and doesn't
erase memories. Use `mem.forget(...)` to erase.

## LangGraph

```bash
pip install "veramem[langgraph]"
```

```python
from veramem import Memory
from veramem.integrations.langgraph import VeraMemStore

mem = Memory()
store = VeraMemStore(mem, user_namespace_index=1)  # namespaces look like ("memories", user_id)
graph = builder.compile(store=store)

# in a node or tool, with user_id taken from config["configurable"], never from the model:
store.put(("memories", user_id), "food", {"content": "likes ramen"})
store.get(("memories", user_id), "food").value  # {"content": "likes ramen"}
store.search(("memories", user_id), query="what food?", limit=5)
```

**Namespace → user.** Set exactly one of these options:

- `user_id="priya"` binds the store to one VeraMem user. All namespaces live in that
  user's scope.
- `user_namespace_index=i` makes `namespace[i]` the VeraMem user. Searches must
  include that component, because VeraMem never searches across users.

Each item becomes one memory. Its searchable text is `value["content"]`,
`value["text"]` or `value["memory"]`, or the fields you pass as
`index_fields`/`index=[...]`, or else the value's JSON. The namespace, key and full
value are kept in the memory's metadata, so `get` returns exactly what you put.

Values are stored with `role="assistant"` (Attested) unless you pass `role="user"`.
**Differences from LangGraph's built-in stores:**

- Overwriting a key retracts the old memory and writes a new one. `created_at`
  is preserved.
- `delete` retracts the memory, which keeps it for audit and hides it from
  retrieval. Pass `hard_delete=True` to erase it instead.
- `index=False` is accepted, but the item is still embedded.
- A query search ranks an over-fetched VeraMem hybrid result before filters and
  offsets apply, so a very selective filter can return fewer than `limit` items.
- TTL isn't supported.
- With `user_namespace_index`, `list_namespaces` lists every user's namespaces.

## LlamaIndex

```bash
pip install veramem llama-index-core
```

```python
from llama_index.core.memory import Memory as LlamaMemory
from veramem import Memory
from veramem.integrations.llamaindex import VeraMemMemoryBlock, VeraMemRetriever

mem = Memory()
retriever = VeraMemRetriever(mem, user_id="priya")
nodes = retriever.retrieve("where does priya live?")  # NodeWithScore, node_id = memory id
retriever.feedback(1.0)

# Long-term block for LlamaIndex's Memory; pass it to an agent: await agent.run(msg, memory=chat_memory)
block = VeraMemMemoryBlock(memory=mem, user_id="priya")
chat_memory = LlamaMemory.from_defaults(session_id="s1", memory_blocks=[block])
```

The block searches VeraMem with the latest user message and injects a cited context.
It stores the messages that short-term memory flushes, mapping each role to the
matching VeraMem authority.

## CrewAI

```bash
pip install veramem crewai
```

```python
from crewai import Agent, Crew
from crewai.memory import Memory as CrewMemory
from veramem import Memory
from veramem.integrations.crewai import VeraMemStorage, veramem_tools

mem = Memory()

# 1) VeraMem as the storage behind CrewAI's memory
crew = Crew(
    agents=[...], tasks=[...], memory=CrewMemory(storage=VeraMemStorage(mem, user_id="priya"))
)

# 2) VeraMem recall/feedback/authority gate as agent tools
agent = Agent(
    role="assistant", goal="...", backstory="...", tools=veramem_tools(mem, user_id="priya")
)
```

`VeraMemStorage` implements CrewAI's `StorageBackend`. Each record becomes a VeraMem
memory owned by the bound user, with hashed evidence, a ledger entry and Attested
authority. Scope, categories, importance and metadata round-trip.

CrewAI passes `search` a precomputed query embedding instead of query text. That
means VeraMem's hybrid, outcome-aware ranking can't run there, so the backend ranks
by cosine over CrewAI's stored embeddings. For VeraMem-ranked recall with feedback,
give agents `veramem_tools`. Deletes retract unless `hard_delete=True`.
