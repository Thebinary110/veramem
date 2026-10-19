"""Core data model.

Every memory in VeraMem points back to the raw evidence it came from (an Episode),
carries the *authority* of the source that produced it, lives in an explicit
access scope, and accumulates outcome statistics that drive retention.
"""

from __future__ import annotations

import threading
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


_last_now = 0.0
_now_lock = threading.Lock()


_CONTROL = set(map(chr, range(32))) | {"\x7f"}


def check_id(value: str | None, what: str = "id") -> str | None:
    """Identifiers (user, agent, team, memory, trace ids) must be printable and short.

    They are rejected rather than cleaned: silently stripping a NUL from "victim\x00"
    would turn an attacker's id into the victim's.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"invalid {what}")
    if any(c in _CONTROL for c in value):
        raise ValueError(f"invalid {what}: control characters are not allowed")
    return value


def clean_text(value: str) -> str:
    """Drop NUL bytes: meaningless in text, and rejected by PostgreSQL."""
    return value.replace("\x00", "") if "\x00" in value else value


def day(ts: float | None) -> str:
    """YYYY-MM-DD for a timestamp; never raises (stored values may be hostile)."""
    if ts is None:
        return "unknown date"
    try:
        return time.strftime("%Y-%m-%d", time.gmtime(ts))
    except (OverflowError, OSError, ValueError):
        return "unknown date"


def now() -> float:
    """Wall-clock time that never repeats or goes backwards within a process.

    Versioning and time-travel queries compare timestamps, and some platforms
    (notably Windows) have a ~15 ms clock tick, so two writes could otherwise share a
    timestamp and make an old version vanish from `as_of` queries.
    """
    global _last_now
    with _now_lock:
        t = time.time()
        if t <= _last_now:
            t = _last_now + 1e-6
        _last_now = t
        return t


class SourceType(str, Enum):
    """Who produced a piece of raw evidence."""

    USER = "user"  # the human the memory is about / acting for
    ASSISTANT = "assistant"  # the agent itself
    TOOL = "tool"  # tool / API / function output
    DOCUMENT = "document"  # ingested file, web page, email ...
    SYSTEM = "system"  # operator / developer configuration
    SHARED = "shared"  # experience transferred from another scope


class Authority(str, Enum):
    """How much a memory may be trusted to *act* on.

    Follows the source-authority taxonomy of AuthMem-Bench ("When Memory Becomes
    Authority", 2026): direct user evidence is Authorized, the agent's own claims
    are Attested, everything else (tools, documents, other users) is Unendorsed.
    Consolidation may never raise a memory's authority above its weakest evidence.
    """

    AUTHORIZED = "authorized"
    ATTESTED = "attested"
    UNENDORSED = "unendorsed"

    @property
    def rank(self) -> int:
        return {"authorized": 2, "attested": 1, "unendorsed": 0}[self.value]

    @classmethod
    def for_source(cls, source: SourceType) -> Authority:
        if source in (SourceType.USER, SourceType.SYSTEM):
            return cls.AUTHORIZED
        if source is SourceType.ASSISTANT:
            return cls.ATTESTED
        return cls.UNENDORSED

    @classmethod
    def weakest(cls, items: list[Authority]) -> Authority:
        return min(items, key=lambda a: a.rank) if items else cls.UNENDORSED


class Kind(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    EVENT = "event"
    EXPERIENCE = "experience"  # procedural lesson distilled from task outcomes
    SUMMARY = "summary"  # coarse-grained session / topic summary


class Status(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"  # replaced by a newer memory, kept for history
    RETRACTED = "retracted"  # deleted / contradicted, kept for audit
    ARCHIVED = "archived"  # low-utility, out of default retrieval


class Visibility(str, Enum):
    PRIVATE = "private"  # only the owning user (and agents they grant)
    TEAM = "team"  # members of `team_id`
    PUBLIC = "public"  # the shared experience pool (de-identified only)


class Episode(BaseModel):
    """Immutable raw evidence: one message, tool result or document chunk."""

    id: str = Field(default_factory=lambda: new_id("ep"))
    content: str
    source: SourceType = SourceType.USER
    actor: str | None = None  # name of the speaker / tool / file
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    occurred_at: float = Field(default_factory=now)  # event time
    ingested_at: float = Field(default_factory=now)  # transaction time
    sha256: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    """A pointer into an Episode: the exact span that supports a memory."""

    episode_id: str
    quote: str
    start: int = -1
    end: int = -1
    source: SourceType = SourceType.USER


class MemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: new_id("mem"))
    text: str
    kind: Kind = Kind.FACT
    user_id: str | None = None
    agent_id: str | None = None
    team_id: str | None = None
    visibility: Visibility = Visibility.PRIVATE
    authority: Authority = Authority.UNENDORSED
    evidence: list[Evidence] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    status: Status = Status.ACTIVE
    supersedes: str | None = None
    superseded_by: str | None = None
    valid_from: float | None = None  # when the fact became true in the world
    valid_to: float | None = None  # when it stopped being true (None = still true)
    created_at: float = Field(default_factory=now)
    updated_at: float = Field(default_factory=now)
    confidence: float = 1.0
    # outcome statistics (Gap A)
    uses: int = 0
    successes: float = 0.0
    failures: float = 0.0
    last_used_at: float | None = None
    shareable: bool = False  # owner consent for de-identified transfer
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def utility(self) -> float:
        """Posterior mean of a Beta(1,1) prior over 'this memory helps'."""
        return (self.successes + 1.0) / (self.successes + self.failures + 2.0)


class Hit(BaseModel):
    """One retrieval result, with the reasons it was ranked where it is."""

    memory: MemoryItem
    score: float
    signals: dict[str, float] = Field(default_factory=dict)
    granularity: str = "fact"


class SearchResult(BaseModel):
    trace_id: str
    query: str
    hits: list[Hit]
    granularity: str = "fact"

    def context(self, max_items: int | None = None, with_evidence: bool = True) -> str:
        """Render hits as a compact, citable context block for a prompt."""
        lines = []
        for h in self.hits[:max_items]:
            m = h.memory
            when = day(m.valid_from or m.created_at)
            line = f"[{m.id}] ({when}, {m.authority.value}) {m.text}"
            if with_evidence and m.evidence:
                line += f'\n    evidence: "{m.evidence[0].quote[:200]}"'
            lines.append(line)
        return "\n".join(lines)


class Operation(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


class Change(BaseModel):
    op: Operation
    memory_id: str | None = None
    text: str | None = None
    previous: str | None = None
    reason: str = ""


class AddResult(BaseModel):
    episodes: list[str]
    changes: list[Change]
    rejected: list[dict[str, Any]] = Field(default_factory=list)  # ungrounded candidates


class Answer(BaseModel):
    trace_id: str
    answer: str
    citations: list[str]
    abstained: bool = False
    used_memories: list[str] = Field(default_factory=list)
