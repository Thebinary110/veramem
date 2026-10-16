"""Configuration: explicit > environment > ~/.veramem/config.json > auto-detection.

Auto-detection order (so that `Memory()` just works):
  1. a local Ollama server with a chat model  -> fully local, no API key
  2. OPENAI_API_KEY                           -> OpenAI
  3. ANTHROPIC_API_KEY                        -> Anthropic (hash embeddings)
  4. nothing                                  -> raw-turn memory + hash embeddings
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

PREFERRED_OLLAMA = [
    "qwen3.6",
    "qwen3",
    "qwen2.5",
    "llama3.3",
    "llama3.1",
    "gemma4",
    "gemma3",
    "mistral",
    "phi4",
]
EMBED_HINTS = ("embed", "bge", "e5", "minilm", "nomic", "mxbai", "arctic")


def home() -> Path:
    return Path(os.environ.get("VERAMEM_HOME", Path.home() / ".veramem")).expanduser()


@dataclass
class Config:
    db_path: str = ""
    llm_provider: str = "auto"  # auto | ollama | openai | anthropic | none
    llm_model: str = ""
    llm_base_url: str = ""
    embed_provider: str = "auto"  # auto | ollama | openai | hash
    embed_model: str = ""
    embed_base_url: str = ""
    cache_path: str = ""  # optional: replayable LLM response cache
    telemetry: bool = False  # VeraMem never phones home; kept explicit for clarity
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> Config:
        cfg = cls()
        p = Path(path) if path else home() / "config.json"
        if p.exists():
            data = json.loads(p.read_text())
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        env = {
            "db_path": "VERAMEM_DB",
            "llm_provider": "VERAMEM_LLM",
            "llm_model": "VERAMEM_LLM_MODEL",
            "llm_base_url": "VERAMEM_LLM_BASE_URL",
            "embed_provider": "VERAMEM_EMBEDDER",
            "embed_model": "VERAMEM_EMBED_MODEL",
            "embed_base_url": "VERAMEM_EMBED_BASE_URL",
            "cache_path": "VERAMEM_CACHE",
        }
        for attr, var in env.items():
            if os.environ.get(var):
                setattr(cfg, attr, os.environ[var])
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        if not cfg.db_path:
            cfg.db_path = str(home() / "memory.db")
        return cfg

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path else home() / "config.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))
        return p


def ollama_host() -> str:
    return (
        os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        if os.environ.get("OLLAMA_HOST", "").startswith("http")
        else f"http://{os.environ.get('OLLAMA_HOST', 'localhost:11434')}"
    )


def ollama_models(host: str | None = None) -> list[str]:
    try:
        r = httpx.get(f"{host or ollama_host()}/api/tags", timeout=1.5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def pick_ollama(models: list[str]) -> tuple[str | None, str | None]:
    chat = [m for m in models if not any(h in m.lower() for h in EMBED_HINTS)]
    emb = [m for m in models if any(h in m.lower() for h in EMBED_HINTS)]
    best = None
    for pref in PREFERRED_OLLAMA:
        best = next((m for m in chat if m.startswith(pref)), None)
        if best:
            break
    return best or (chat[0] if chat else None), (emb[0] if emb else None)


def resolve(cfg: Config) -> dict[str, Any]:
    """Decide concrete providers. Returns a plain dict describing the choice."""
    llm_p, llm_m, llm_url = cfg.llm_provider, cfg.llm_model, cfg.llm_base_url
    emb_p, emb_m, emb_url = cfg.embed_provider, cfg.embed_model, cfg.embed_base_url
    models: list[str] | None = None

    def _models() -> list[str]:
        nonlocal models
        if models is None:
            models = ollama_models(llm_url or None)
        return models

    if llm_p == "auto":
        chat, _ = pick_ollama(_models())
        if chat:
            llm_p, llm_m = "ollama", llm_m or chat
        elif os.environ.get("OPENAI_API_KEY"):
            llm_p = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            llm_p = "anthropic"
        else:
            llm_p = "none"
    if emb_p == "auto":
        _, emb = (
            pick_ollama(_models()) if llm_p in ("ollama", "none", "anthropic") else (None, None)
        )
        if emb:
            emb_p, emb_m = "ollama", emb_m or emb
        elif llm_p == "openai" or (os.environ.get("OPENAI_API_KEY") and llm_p != "ollama"):
            emb_p = "openai"
        else:
            emb_p = "hash"
    return {"llm": (llm_p, llm_m, llm_url), "embed": (emb_p, emb_m, emb_url)}
