"""Deterministic privacy checks used before anything leaves its owner's scope.

The sharing papers (ShareMem, Collaborative Memory) rely on an LLM prompt to strip
user-specific values. Prompts leak. This module is a non-LLM gate that runs after
the abstraction step: PII patterns plus a *canary* check against the owner's own
private memories. If either fires, the candidate stays local.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass, field

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)"),
    "card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "url_with_query": re.compile(r"https?://\S+\?\S+"),
    "street_address": re.compile(
        r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln|Dr|"
        r"Drive|Way|Court|Ct|Place|Pl)\b"
    ),
    "secret": re.compile(r"(?i)\b(?:api[_-]?key|token|password|passwd|secret)\b\s*[:=]\s*\S+"),
    "long_id": re.compile(r"\b(?=[A-Z0-9-]*\d)[A-Z0-9]{2,}-?[A-Z0-9]{5,}\b"),
}

_STOP = set(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "it",
        "this",
        "that",
        "with",
        "as",
        "by",
        "from",
        "user",
        "users",
        "their",
        "they",
        "them",
        "he",
        "she",
        "his",
        "her",
        "my",
        "i",
        "you",
        "we",
        "our",
        "your",
        "has",
        "have",
        "had",
        "will",
        "would",
        "should",
        "can",
        "could",
        "not",
        "no",
        "do",
        "does",
        "did",
        "about",
        "into",
        "than",
        "then",
        "there",
        "here",
        "when",
        "where",
        "which",
        "who",
        "what",
        "how",
        "also",
        "just",
        "very",
        "more",
        "most",
        "some",
        "any",
        "all",
        "each",
        "every",
    ]
)
_WORD = re.compile(r"[a-z0-9']+")


@dataclass
class PrivacyReport:
    ok: bool
    findings: list[str] = field(default_factory=list)


def scan_pii(text: str) -> list[str]:
    hits = []
    for name, pat in PII_PATTERNS.items():
        for m in pat.finditer(text):
            if name == "phone" and sum(c.isdigit() for c in m.group()) < 8:
                continue
            hits.append(f"{name}:{m.group()[:40]}")
    return hits


def redact(text: str) -> str:
    out = text
    for name, pat in PII_PATTERNS.items():
        out = pat.sub(f"{{{name}}}", out)
    return out


def _content_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = [w for w in _WORD.findall(text.lower()) if w not in _STOP]
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def canary_hits(candidate: str, private_texts: list[str], n: int = 2) -> list[str]:
    """Content-word n-grams of the owner's private memories that survive into `candidate`.

    Capitalised tokens (names, brands, places) are checked as unigrams too.
    """
    cand = _content_ngrams(candidate, n)
    cand_uni = {w for w in _WORD.findall(candidate.lower())}
    hits: list[str] = []
    for t in private_texts:
        for g in _content_ngrams(t, n) & cand:
            hits.append(" ".join(g))
        for tok in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", t):
            if tok.lower() in cand_uni and tok.lower() not in _STOP and not _is_generic(tok):
                hits.append(tok)
    return sorted(set(hits))


_GENERIC = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "python",
    "javascript",
    "sql",
    "applies",
    "inputs",
    "procedure",
    "caution",
    "scope",
    "always",
    "never",
    "use",
    "check",
    "ask",
    "confirm",
    "when",
    "before",
    "after",
    "if",
    "the",
}


def _is_generic(tok: str) -> bool:
    return tok.lower() in _GENERIC


def check_shareable(candidate: str, private_texts: list[str]) -> PrivacyReport:
    findings = [f"pii:{h}" for h in scan_pii(candidate)]
    findings += [f"canary:{h}" for h in canary_hits(candidate, private_texts)]
    return PrivacyReport(ok=not findings, findings=findings)


def pseudonym(user_id: str, secret: bytes) -> str:
    """Revocable, non-reversible contributor id for shared provenance."""
    return "u_" + hmac.new(secret, user_id.encode(), hashlib.sha256).hexdigest()[:16]
