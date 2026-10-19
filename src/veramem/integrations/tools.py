"""Framework-agnostic memory tools for any tool-calling LLM loop.

No extra dependencies. ``VeraMemTools`` exposes four plain-Python functions
(``remember``, ``recall``, ``feedback``, ``check_action``) together with JSON
schemas in the OpenAI (``openai_tools()``) and Anthropic (``anthropic_tools()``)
tool formats, plus ``call(name, arguments)`` to dispatch a model's tool call.

    tools = VeraMemTools(mem, user_id="priya")
    resp = client.chat.completions.create(..., tools=tools.openai_tools())
    for tc in resp.choices[0].message.tool_calls or []:
        result = tools.call(tc.function.name, tc.function.arguments)

Security: ``user_id`` (and ``agent_id``) are bound here and are not parameters of
any tool, so a model can't read or write another user's memory. By default the
model can't claim ``role="user"`` either: whatever it writes is stored as its own
(Attested) claim, which ``check_action`` won't accept as authorization on its own.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..memory import Memory
from ._common import put_memory, require_user

TOOL_NAMES = ("remember", "recall", "feedback", "check_action")

_DESCRIPTIONS = {
    "remember": (
        "Store a durable fact, preference, plan or correction in long-term memory. "
        "Use role='tool' for tool output and role='document' for file or web content, "
        "so their claims are not treated as the user's own."
    ),
    "recall": (
        "Search long-term memory. Returns memories with ids, dates and authority labels, "
        "plus a trace_id to pass to `feedback` once you know whether they helped."
    ),
    "feedback": (
        "Report whether the memories from a recall helped: reward from -1 (wrong or "
        "harmful) to 1 (helpful). Optionally list the memory ids you actually used."
    ),
    "check_action": (
        "Before acting on remembered values (sending, paying, booking, deleting), check "
        "whether those memories are authorized to drive the action. If not allowed, "
        "confirm with the user first."
    ),
}


class VeraMemTools:
    """Memory tools bound to one user (and optionally one agent and session).

    infer: when True and the Memory has an LLM, ``remember`` runs VeraMem's grounded
        extraction instead of storing the text verbatim.
    allow_user_role: let the model store content with ``role="user"`` (Authorized).
        Leave False unless the model only ever relays the user's exact words.
    """

    def __init__(
        self,
        memory: Memory,
        *,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        recall_limit: int = 8,
        max_recall_limit: int = 50,
        infer: bool = False,
        allow_user_role: bool = False,
    ) -> None:
        self.memory = memory
        self.user_id = require_user(user_id)
        self.agent_id = agent_id
        self.session_id = session_id
        self.recall_limit = recall_limit
        self.max_recall_limit = max_recall_limit
        self.infer = infer
        self.roles = (
            ("user", "assistant", "tool", "document")
            if allow_user_role
            else ("assistant", "tool", "document")
        )
        self.last_trace_id: str | None = None

    # ---------------------------------------------------------------- tools
    def remember(self, content: str, role: str = "assistant") -> dict[str, Any]:
        """Store something worth remembering."""
        if role not in self.roles:
            return {"error": f"role must be one of {list(self.roles)}"}
        if self.infer and self.memory.has_llm:
            r = self.memory.add(
                {"role": role, "content": content},
                user_id=self.user_id,
                agent_id=self.agent_id,
                session_id=self.session_id,
            )
            return {
                "stored": [c.memory_id for c in r.changes if c.memory_id],
                "changes": [c.model_dump(mode="json") for c in r.changes],
                "rejected_ungrounded": len(r.rejected),
            }
        item = put_memory(
            self.memory,
            content,
            role=role,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
        )
        return {"stored": [item.id], "authority": item.authority.value}

    def recall(self, query: str, limit: int | None = None) -> dict[str, Any]:
        """Search memory; returns cited memories and a trace_id for feedback."""
        n = max(1, min(int(limit or self.recall_limit), self.max_recall_limit))
        r = self.memory.search(query, user_id=self.user_id, agent_id=self.agent_id, limit=n)
        self.last_trace_id = r.trace_id
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

    def feedback(
        self, trace_id: str, reward: float, cited: list[str] | None = None
    ) -> dict[str, Any]:
        """Outcome feedback for a trace that belongs to this user."""
        reward = max(-1.0, min(1.0, float(reward)))
        return self.memory.feedback(
            trace_id, reward, cited=cited, user_id=self.user_id, agent_id=self.agent_id
        )

    def check_action(self, memory_ids: list[str]) -> dict[str, Any]:
        """Authority gate: may these memories drive a protected action?"""
        return self.memory.check_action(
            list(memory_ids), user_id=self.user_id, agent_id=self.agent_id
        )

    # -------------------------------------------------------------- schemas
    def json_schemas(self) -> dict[str, dict[str, Any]]:
        """JSON Schema of each tool's arguments, by tool name."""
        return {
            "remember": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "What to remember."},
                    "role": {
                        "type": "string",
                        "enum": list(self.roles),
                        "default": "assistant",
                        "description": "Who the content comes from; decides its authority.",
                    },
                },
                "required": ["content"],
            },
            "recall": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self.max_recall_limit,
                        "default": self.recall_limit,
                    },
                },
                "required": ["query"],
            },
            "feedback": {
                "type": "object",
                "properties": {
                    "trace_id": {"type": "string", "description": "trace_id from recall."},
                    "reward": {"type": "number", "minimum": -1, "maximum": 1},
                    "cited": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ids of the memories you actually used.",
                    },
                },
                "required": ["trace_id", "reward"],
            },
            "check_action": {
                "type": "object",
                "properties": {
                    "memory_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["memory_ids"],
            },
        }

    def openai_tools(self) -> list[dict[str, Any]]:
        """Tool definitions for OpenAI-compatible chat completions."""
        return [
            {
                "type": "function",
                "function": {"name": n, "description": _DESCRIPTIONS[n], "parameters": s},
            }
            for n, s in self.json_schemas().items()
        ]

    def anthropic_tools(self) -> list[dict[str, Any]]:
        """Tool definitions in the Anthropic Messages API format."""
        return [
            {"name": n, "description": _DESCRIPTIONS[n], "input_schema": s}
            for n, s in self.json_schemas().items()
        ]

    @property
    def functions(self) -> dict[str, Callable[..., dict[str, Any]]]:
        return {
            "remember": self.remember,
            "recall": self.recall,
            "feedback": self.feedback,
            "check_action": self.check_action,
        }

    def call(self, name: str, arguments: str | dict[str, Any] | None = None) -> dict[str, Any]:
        """Run a model's tool call. `arguments` may be a dict or a JSON string.

        Unknown tools and unexpected arguments (including any attempt to pass a
        ``user_id``) are rejected rather than silently ignored.
        """
        fn = self.functions.get(name)
        if fn is None:
            return {"error": f"unknown tool {name!r}"}
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        allowed = set(self.json_schemas()[name]["properties"])
        extra = set(args) - allowed
        if extra:
            return {"error": f"unexpected arguments: {sorted(extra)}"}
        try:
            return fn(**args)
        except (ValueError, KeyError, TypeError, PermissionError) as e:
            return {"error": str(e)}


def description(name: str) -> str:
    """The model-facing description of a tool (shared by the framework adapters)."""
    return _DESCRIPTIONS[name]
