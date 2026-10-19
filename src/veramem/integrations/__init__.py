"""Framework adapters for VeraMem.

Each submodule imports its framework lazily, so importing ``veramem`` (or this
package) never requires LangChain, LangGraph, LlamaIndex or CrewAI:

    veramem.integrations.tools       plain-Python tool functions + JSON schemas (no extra)
    veramem.integrations.langchain   pip install "veramem[langchain]"
    veramem.integrations.langgraph   pip install "veramem[langgraph]"
    veramem.integrations.llamaindex  pip install veramem llama-index-core
    veramem.integrations.crewai      pip install "veramem[crewai]"

Every adapter binds ``user_id`` when it is constructed. Nothing a model writes in a
tool call or prompt can switch it to another user's memory.
"""
