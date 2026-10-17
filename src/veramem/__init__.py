"""VeraMem: verifiable, outcome-aware, share-safe long-term memory for AI agents."""

from .__about__ import __version__
from .aio import AsyncMemory
from .config import Config
from .memory import Memory
from .policy import AccessDenied
from .types import (
    AddResult,
    Answer,
    Authority,
    Episode,
    Evidence,
    Hit,
    Kind,
    MemoryItem,
    SearchResult,
    SourceType,
    Status,
    Visibility,
)

__all__ = [
    "__version__",
    "AsyncMemory",
    "Memory",
    "Config",
    "AccessDenied",
    "AddResult",
    "Answer",
    "Authority",
    "Episode",
    "Evidence",
    "Hit",
    "Kind",
    "MemoryItem",
    "SearchResult",
    "SourceType",
    "Status",
    "Visibility",
]
