"""MCP server: plug VeraMem into Claude Desktop, Cursor, VS Code or any MCP client.

    veramem mcp            # stdio transport (what desktop clients launch)

The acting user is fixed by the server environment (VERAMEM_USER, default "default"),
never chosen by the model: an LLM cannot ask for another user's memory by passing
a different id (ShareMem's user-bound preference channel).
"""

from __future__ import annotations

import os
from typing import Any

from .memory import Memory

INSTRUCTIONS = """Long-term memory with provenance.
- Call `recall` before answering anything that may depend on what the user told you before.
- Call `remember` when the user shares durable facts, preferences, plans or corrections.
  Use role="tool" for tool results and role="document" for file content; that keeps
  their claims from being treated as the user's own.
- After a task, call `feedback` with the trace_id and +1 / -1 so useful memories rank higher.
- Before acting on a remembered value (sending, paying, booking), call `check_action`;
  if it is not allowed, confirm with the user first."""


def _server_cls():
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP

        return FastMCP


def build(mem: Memory | None = None) -> Any:
    mem = mem or Memory()
    user = os.environ.get("VERAMEM_USER", "default")
    agent = os.environ.get("VERAMEM_AGENT") or None
    server = _server_cls()(name="veramem", instructions=INSTRUCTIONS)

    @server.tool()
    def remember(content: str, role: str = "user", session_id: str | None = None) -> dict:
        """Store something worth remembering. role: user | assistant | tool | document."""
        r = mem.add(
            {"role": role, "content": content}, user_id=user, agent_id=agent, session_id=session_id
        )
        return {
            "changes": [c.model_dump(mode="json") for c in r.changes],
            "rejected_ungrounded": len(r.rejected),
        }

    @server.tool()
    def recall(query: str, limit: int = 8) -> dict:
        """Search memory. Returns cited memories with dates and authority labels."""
        r = mem.search(query, user_id=user, agent_id=agent, limit=limit)
        return {
            "trace_id": r.trace_id,
            "context": r.context(),
            "memories": [
                {
                    "id": h.memory.id,
                    "text": h.memory.text,
                    "authority": h.memory.authority.value,
                    "kind": h.memory.kind.value,
                    "granularity": h.granularity,
                }
                for h in r.hits
            ],
        }

    @server.tool()
    def ask(question: str) -> dict:
        """Answer a question from memory only, with citations; abstains if unsupported."""
        a = mem.answer(question, user_id=user, agent_id=agent)
        return a.model_dump(mode="json")

    @server.tool()
    def feedback(trace_id: str, reward: float) -> dict:
        """Report whether recalled memories helped: reward from -1 (wrong) to 1 (helpful)."""
        return mem.feedback(trace_id, reward)

    @server.tool()
    def check_action(memory_ids: list[str]) -> dict:
        """Check whether memories are authorized to drive a protected action."""
        return mem.check_action(memory_ids, user_id=user, agent_id=agent)

    @server.tool()
    def correct(memory_id: str, new_text: str) -> dict:
        """Correct a memory (keeps the old version in history)."""
        return mem.update(memory_id, new_text, user_id=user, agent_id=agent).model_dump(mode="json")

    @server.tool()
    def forget(memory_id: str) -> dict:
        """Permanently erase one memory."""
        return mem.forget(memory_id, user_id=user, agent_id=agent)

    @server.tool()
    def history(memory_id: str) -> list:
        """Show how a memory changed over time, with ledger hashes."""
        return mem.history(memory_id, user_id=user, agent_id=agent)

    return server


def main() -> None:
    build().run("stdio")


if __name__ == "__main__":
    main()
