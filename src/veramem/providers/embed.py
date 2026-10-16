"""Embedding adapters. `HashEmbedder` needs nothing at all and is used offline and in tests."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import httpx
import numpy as np

from .llm import ResponseCache


class Embedder:
    name = "base"
    model = ""
    dim = 0

    def __init__(self, cache: ResponseCache | None = None):
        self.cache = cache

    def _embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 1), dtype=np.float32)
        out: list[Any] = [None] * len(texts)
        todo: list[int] = []
        if self.cache is not None:
            for i, t in enumerate(texts):
                hit = self.cache.get(self.cache.key("emb", self.name, self.model, t))
                if hit is not None:
                    out[i] = np.frombuffer(bytes.fromhex(hit), dtype=np.float32)
                else:
                    todo.append(i)
        else:
            todo = list(range(len(texts)))
        for start in range(0, len(todo), 64):
            idx = todo[start : start + 64]
            vecs = self._embed([texts[i] for i in idx])
            for i, v in zip(idx, vecs):
                a = np.asarray(v, dtype=np.float32)
                out[i] = a
                if self.cache is not None:
                    self.cache.put(
                        self.cache.key("emb", self.name, self.model, texts[i]), a.tobytes().hex()
                    )
        mat = np.vstack(out).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.dim = mat.shape[1]
        return mat / norms

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class OllamaEmbedder(Embedder):
    name = "ollama"

    def __init__(
        self, model: str = "nomic-embed-text", host: str = "http://localhost:11434", **kw: Any
    ):
        super().__init__(**kw)
        self.model, self.host = model, host.rstrip("/")
        self.http = httpx.Client(timeout=300)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        r = self.http.post(
            f"{self.host}/api/embed", json={"model": self.model, "input": texts, "truncate": True}
        )
        r.raise_for_status()
        return r.json()["embeddings"]


class OpenAICompatibleEmbedder(Embedder):
    name = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        **kw: Any,
    ):
        super().__init__(**kw)
        self.model, self.base_url = model, base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(timeout=120, headers=headers)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        r = self.http.post(
            f"{self.base_url}/embeddings", json={"model": self.model, "input": texts}
        )
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]


_WORD = re.compile(r"[a-z0-9]+")


class HashEmbedder(Embedder):
    """Feature-hashed word unigrams + bigrams + character trigrams. Zero dependencies.

    Not semantic, but surprisingly strong for lexical recall, deterministic, and fast,
    which makes it the right default for tests and for machines with no model at all.
    """

    name = "hash"
    model = "hash-v1"

    def __init__(self, dim: int = 1024, **kw: Any):
        super().__init__(**kw)
        self.dim = dim

    def _bucket(self, feat: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            words = _WORD.findall(t.lower())
            feats = [f"w:{w}" for w in words] + [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
            for w in words:
                padded = f"#{w}#"
                feats += [f"c:{padded[i : i + 3]}" for i in range(len(padded) - 2)]
            for f in feats:
                i, s = self._bucket(f)
                v[i] += s * (2.0 if f.startswith("w:") else 1.0)
            out.append(v.tolist())
        return out
