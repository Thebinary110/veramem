"""Outcome-informed ranking: contextual utility, exploration and a learned ranker.

Three increasingly general ways to use feedback, kept side by side so they can be
compared honestly (see docs/OUTCOME_LEARNING.md):

* ``global``: one Beta posterior per memory over "this memory helped". Re-ranks by
  past usefulness, regardless of the query. This is the baseline.
* ``contextual``: the same posterior, but each past outcome is weighted by how similar
  the query it came from is to the current query. A memory that helped with dentist
  questions gains little for travel questions.
* ``ranker``: a logistic model over memory-agnostic features (retrieval signals,
  authority, kind, age, length, outcome statistics). It never sees memory ids, so it
  can transfer to memories it has never been rated on.

Exploration (Thompson sampling or UCB) can be layered on the global or contextual
posterior so that uncertain memories get evaluated instead of starving.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .types import Authority, Kind, MemoryItem

CONTEXT_TAU = 0.1  # kernel width: exp((cos - 1) / tau); cos 0.9 -> 0.37, cos 0.7 -> 0.05

FEATURES = [
    "rrf",
    "dense",
    "bm25_rr",
    "entity",
    "auth_authorized",
    "auth_attested",
    "auth_unendorsed",
    "kind_fact",
    "kind_preference",
    "kind_event",
    "kind_experience",
    "kind_summary",
    "log_age_days",
    "log_len",
    "log_n_obs",
    "util_global",
    "util_ctx",
    "ctx_mass",
    "is_raw",
]


@dataclass
class Posterior:
    alpha: float = 1.0
    beta: float = 1.0
    mass: float = 0.0  # total (kernel-weighted) number of observations

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def var(self) -> float:
        a, b = self.alpha, self.beta
        return a * b / ((a + b) ** 2 * (a + b + 1))


def global_posterior(m: MemoryItem) -> Posterior:
    return Posterior(1.0 + m.successes, 1.0 + m.failures, m.successes + m.failures)


def contextual_posterior(qv: np.ndarray, rows: list[tuple[np.ndarray, float, float]]) -> Posterior:
    """rows: (query_vector, positive_mass, negative_mass) of past outcomes for one memory."""
    if not rows:
        return Posterior()
    q = np.vstack([r[0] for r in rows])
    w = np.exp((q @ qv - 1.0) / CONTEXT_TAU)
    pos = np.array([r[1] for r in rows])
    neg = np.array([r[2] for r in rows])
    return Posterior(1.0 + float(w @ pos), 1.0 + float(w @ neg), float(w @ (pos + neg)))


def explore(post: Posterior, mode: str, rng: np.random.Generator, ucb_c: float = 1.0) -> float:
    """Utility estimate used for ranking under an exploration strategy."""
    if mode == "thompson":
        return float(rng.beta(post.alpha, post.beta))
    if mode == "ucb":
        return min(1.0, post.mean + ucb_c * math.sqrt(post.var))
    return post.mean


def features(
    m: MemoryItem,
    signals: dict[str, float],
    rrf_norm: float,
    g_post: Posterior,
    c_post: Posterior,
    now_ts: float,
    is_raw: bool = False,
) -> list[float]:
    bm25_rank = signals.get("bm25_rank")
    age_days = max(0.0, (now_ts - (m.valid_from or m.created_at)) / 86400.0)
    return [
        rrf_norm,
        float(signals.get("dense", 0.0)),
        1.0 / bm25_rank if bm25_rank else 0.0,
        float(signals.get("entity", 0.0)),
        float(m.authority is Authority.AUTHORIZED),
        float(m.authority is Authority.ATTESTED),
        float(m.authority is Authority.UNENDORSED),
        float(m.kind is Kind.FACT),
        float(m.kind is Kind.PREFERENCE),
        float(m.kind is Kind.EVENT),
        float(m.kind is Kind.EXPERIENCE),
        float(m.kind is Kind.SUMMARY),
        math.log1p(age_days),
        math.log1p(len(m.text)),
        math.log1p(g_post.mass),
        g_post.mean,
        c_post.mean,
        math.log1p(c_post.mass),
        float(is_raw),
    ]


@dataclass
class LogisticRanker:
    """L2-regularised logistic regression, standardised inputs, pure numpy."""

    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=lambda: list(FEATURES))
    trained_on: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray | None = None,
        l2: float = 1.0,
        iters: int = 50,
    ) -> LogisticRanker:
        n, d = x.shape
        w = np.ones(n) if w is None else w
        mu = x.mean(axis=0)
        sd = x.std(axis=0)
        sd[sd < 1e-9] = 1.0
        z = np.hstack([(x - mu) / sd, np.ones((n, 1))])
        theta = np.zeros(d + 1)
        reg = np.full(d + 1, l2)
        reg[-1] = 0.0  # do not shrink the bias
        for _ in range(iters):  # Newton / IRLS: converges in a handful of steps
            p = 1.0 / (1.0 + np.exp(-np.clip(z @ theta, -30, 30)))
            grad = z.T @ (w * (p - y)) + reg * theta
            hess = (z.T * (w * p * (1 - p))) @ z + np.diag(reg) + 1e-9 * np.eye(d + 1)
            step = np.linalg.solve(hess, grad)
            theta -= step
            if float(np.max(np.abs(step))) < 1e-6:
                break
        return cls(
            weights=theta[:-1].tolist(),
            bias=float(theta[-1]),
            mean=mu.tolist(),
            scale=sd.tolist(),
            trained_on=int(n),
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - np.asarray(self.mean)) / np.asarray(self.scale)
        return 1.0 / (1.0 + np.exp(-np.clip(z @ np.asarray(self.weights) + self.bias, -30, 30)))

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, s: str) -> LogisticRanker:
        return cls(**json.loads(s))

    def explain(self) -> dict[str, float]:
        """Standardised coefficient per feature (sign = direction of effect)."""
        return dict(sorted(zip(self.feature_names, self.weights), key=lambda kv: -abs(kv[1])))
