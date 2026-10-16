"""Chat model adapters. All of them speak plain HTTP via httpx; no vendor SDKs."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import httpx

Message = dict[str, str]


class LLMError(RuntimeError):
    pass


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_json(text: str) -> Any:
    """Parse the first JSON object/array in a model response, tolerating fences and chatter."""
    text = _THINK.sub("", text).strip()
    text = _FENCE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"model did not return JSON: {text[:200]!r}")


class LLM:
    name = "base"
    model = ""

    def __init__(self, cache: ResponseCache | None = None, retries: int = 2):
        self.cache = cache
        self.retries = retries
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _complete(self, messages: list[Message], json_mode: bool, max_tokens: int) -> str:
        raise NotImplementedError

    def complete(
        self, messages: list[Message], json_mode: bool = False, max_tokens: int = 2048
    ) -> str:
        key = None
        if self.cache is not None:
            key = self.cache.key(self.name, self.model, messages, json_mode, max_tokens)
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                out = self._complete(messages, json_mode, max_tokens)
                self.calls += 1
                if self.cache is not None and key is not None:
                    self.cache.put(key, out)
                return out
            except (httpx.HTTPError, LLMError) as e:
                last = e
                time.sleep(0.5 * (2**attempt))
        raise LLMError(f"{self.name} call failed: {last}") from last

    def json(self, messages: list[Message], max_tokens: int = 2048) -> Any:
        text = self.complete(messages, json_mode=True, max_tokens=max_tokens)
        try:
            return parse_json(text)
        except LLMError:
            # one repair attempt, bypassing the cache
            fixed = self._complete(
                messages
                + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "Return ONLY valid JSON, nothing else."},
                ],
                True,
                max_tokens,
            )
            return parse_json(fixed)


class OllamaLLM(LLM):
    name = "ollama"

    def __init__(
        self,
        model: str = "qwen3.6:35b",
        host: str = "http://localhost:11434",
        timeout: float = 600,
        **kw: Any,
    ):
        super().__init__(**kw)
        self.model, self.host = model, host.rstrip("/")
        self.http = httpx.Client(timeout=timeout)

    def _complete(self, messages: list[Message], json_mode: bool, max_tokens: int) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 16384},
        }
        if json_mode:
            body["format"] = "json"
        r = self.http.post(f"{self.host}/api/chat", json=body)
        if r.status_code == 400 and "think" in r.text:
            body.pop("think")
            r = self.http.post(f"{self.host}/api/chat", json=body)
        r.raise_for_status()
        d = r.json()
        if "error" in d:
            raise LLMError(d["error"])
        self.prompt_tokens += d.get("prompt_eval_count", 0)
        self.completion_tokens += d.get("eval_count", 0)
        return d["message"]["content"]


class OpenAICompatibleLLM(LLM):
    """OpenAI, Azure-compatible gateways, Groq, Together, vLLM, LM Studio, llama.cpp server ..."""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120,
        **kw: Any,
    ):
        super().__init__(**kw)
        self.model, self.base_url = model, base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(timeout=timeout, headers=headers)

    def _complete(self, messages: list[Message], json_mode: bool, max_tokens: int) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = self.http.post(f"{self.base_url}/chat/completions", json=body)
        if r.status_code == 400 and json_mode:  # some servers reject response_format
            body.pop("response_format")
            r = self.http.post(f"{self.base_url}/chat/completions", json=body)
        r.raise_for_status()
        d = r.json()
        u = d.get("usage") or {}
        self.prompt_tokens += u.get("prompt_tokens", 0)
        self.completion_tokens += u.get("completion_tokens", 0)
        return d["choices"][0]["message"]["content"] or ""


class AnthropicLLM(LLM):
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-haiku-5-5",
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 120,
        **kw: Any,
    ):
        super().__init__(**kw)
        self.model, self.base_url = model, base_url.rstrip("/")
        self.http = httpx.Client(
            timeout=timeout,
            headers={
                "x-api-key": api_key or "",
                "anthropic-version": "2023-06-01",
            },
        )

    def _complete(self, messages: list[Message], json_mode: bool, max_tokens: int) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [m for m in messages if m["role"] != "system"]
        if json_mode:
            system += "\n\nRespond with a single valid JSON value and nothing else."
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": msgs,
            "temperature": 0,
        }
        if system:
            body["system"] = system
        r = self.http.post(f"{self.base_url}/v1/messages", json=body)
        r.raise_for_status()
        d = r.json()
        u = d.get("usage") or {}
        self.prompt_tokens += u.get("input_tokens", 0)
        self.completion_tokens += u.get("output_tokens", 0)
        return "".join(b.get("text", "") for b in d.get("content", []))


class NoLLM(LLM):
    """Placeholder used when no model is configured; VeraMem then stores raw turns only."""

    name = "none"

    def _complete(self, messages: list[Message], json_mode: bool, max_tokens: int) -> str:
        raise LLMError("no LLM configured (run `veramem init`)")


class ScriptedLLM(LLM):
    """Deterministic test double: returns responses produced by a callable."""

    name = "scripted"

    def __init__(self, fn, **kw: Any):
        super().__init__(**kw)
        self.fn = fn

    def _complete(self, messages: list[Message], json_mode: bool, max_tokens: int) -> str:
        out = self.fn(messages)
        return out if isinstance(out, str) else json.dumps(out)


class ResponseCache:
    """On-disk cache of model responses, keyed by the exact request. Makes benchmarks replayable."""

    def __init__(self, path: str | Path):
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(Path(path).expanduser()), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self.lock = threading.Lock()

    @staticmethod
    def key(*parts: Any) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def get(self, k: str) -> str | None:
        with self.lock:
            r = self.db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return r[0] if r else None

    def put(self, k: str, v: str) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (k, v))
            self.db.commit()
