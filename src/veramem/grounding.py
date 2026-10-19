"""Evidence grounding: a memory may only exist if its quote is really in the source.

LLM extractors paraphrase and sometimes invent. We require each candidate memory to
cite a verbatim span, then locate that span in the episode text ourselves. Candidates
whose quote cannot be found are rejected, which removes a whole class of
hallucinated memories before they are ever stored.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def _norm(s: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", s.lower())).strip()


def locate(quote: str, text: str, threshold: float = 0.88) -> tuple[int, int, float]:
    """Find `quote` in `text`. Returns (start, end, score); start=-1 if not found.

    Exact match first, then a normalised match, then a sliding fuzzy window so
    small differences (punctuation, a dropped word) still ground.
    """
    if not quote or not text:
        return -1, -1, 0.0
    i = text.find(quote)
    if i >= 0:
        return i, i + len(quote), 1.0
    lo = text.lower().find(quote.lower().strip())
    if lo >= 0:
        return lo, lo + len(quote.strip()), 1.0

    nq, nt = _norm(quote), _norm(text)
    if not nq:
        return -1, -1, 0.0
    if nq in nt:
        return _map_back(nq, text), _map_back(nq, text) + len(quote), 0.99

    q_words, t_words = nq.split(), nt.split()
    n = len(q_words)
    if n == 0 or not t_words:
        return -1, -1, 0.0
    best, best_i = 0.0, -1
    for width in {n - 1, n, n + 1, n + 2}:
        if width <= 0:
            continue
        for s in range(0, max(1, len(t_words) - width + 1)):
            cand = " ".join(t_words[s : s + width])
            r = SequenceMatcher(None, nq, cand, autojunk=False).ratio()
            if r > best:
                best, best_i = r, s
    if best >= threshold:
        anchor = t_words[best_i]
        start = text.lower().find(anchor)
        return start, start + len(quote), best
    return -1, -1, best


def _map_back(nq: str, text: str) -> int:
    first = nq.split()[0]
    idx = text.lower().find(first)
    return max(idx, 0)


def support_ratio(claim: str, text: str) -> float:
    """Share of the claim's content words that appear in the evidence text."""
    stop = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "and",
        "in",
        "on",
        "for",
        "with",
        "user",
        "users",
        "has",
        "have",
        "their",
        "they",
        "s",
    }
    cw = [w for w in _norm(claim).split() if w not in stop and len(w) > 1]
    if not cw:
        return 1.0
    tw = set(_norm(text).split())
    return sum(1 for w in cw if w in tw or w.rstrip("s") in tw) / len(cw)


_FILLER = {
    "now",
    "currently",
    "still",
    "also",
    "today",
    "these",
    "days",
    "instead",
    "anymore",
    "update",
    "as",
    "of",
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "to",
    "and",
    "in",
    "on",
    "at",
    "for",
    "with",
    "user",
    "s",
    "has",
    "have",
    "that",
    "this",
}


def same_proposition(a: str, b: str) -> bool:
    """True only if two statements carry the same content words.

    Embedding similarity cannot tell "learning Italian" from "learning Japanese"
    (cosine > 0.95 with common embedders), so near-duplicate shortcuts must also pass
    this lexical check; anything else goes to the memory manager.
    """
    wa = {w.rstrip("s") for w in _norm(a).split() if w not in _FILLER}
    wb = {w.rstrip("s") for w in _norm(b).split() if w not in _FILLER}
    return wa == wb


# ------------------------------------------------------------ value provenance
_IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-./@]*\d[A-Za-z0-9_\-./@]*|\d{3,}")
_NAME = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_NOT_NAMES = {
    "the",
    "according",
    "user",
    "assistant",
    "tool",
    "document",
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
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "update",
    "note",
    "this",
    "that",
    "they",
    "she",
    "his",
    "her",
    "their",
    "there",
    "when",
    "what",
    "who",
    "yes",
    "and",
    "but",
    "also",
    "she's",
    "he's",
    "it's",
    "i'm",
}


def value_tokens(text: str) -> set[str]:
    """Identifier-like values (account numbers, codes, amounts) a claim depends on.

    Plausible years are excluded: models legitimately turn "last March" into a year.
    """
    out = set()
    for t in _IDENT.findall(text):
        t = t.strip("-./@").lower()
        if len(t) < 3 or (t.isdigit() and len(t) == 4 and 1900 <= int(t) <= 2200):
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):  # a normalised date
            continue
        out.add(t)
    return out


def name_tokens(text: str) -> set[str]:
    return {t.lower() for t in _NAME.findall(text) if t.lower() not in _NOT_NAMES}


def contains_token(token: str, text: str) -> bool:
    return token in text.lower()


def missing_values(text: str, sources: list[str]) -> set[str]:
    """Value tokens of `text` that appear in none of `sources`."""
    blob = "\n".join(sources).lower()
    return {t for t in value_tokens(text) if t not in blob}
