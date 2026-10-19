# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - unreleased

### Added
- **Outcome-informed ranking:** contextual (query-aware) utility, damped Thompson/UCB
  exploration, and a learned logistic ranker over memory-agnostic features. The ranker is
  monotone in relevance, uses CV-chosen regularisation, and is deployed only behind a
  statistical gate. A placebo-controlled transfer benchmark accompanies it
  (`benchmarks/outcome_transfer.py`, `docs/OUTCOME_LEARNING.md`).
- `feedback(..., cited=...)`: credit exactly the memories an agent used (citation lock).
- Schema v3: `outcomes` table and trace query vectors (erased by `forget`).

- **Security:**
  - per-user API keys (hashed), OIDC/JWT, per-caller rate limits, a body size limit, CSP and security headers;
  - a STRIDE threat model, and 190+ adversarial tests that drive VeraMem with a compromised LLM.
- **Fixes found by the adversarial suite:**
  - `check_action` leaked other users' memory text;
  - anyone could rate another user's trace;
  - anyone could rewrite or delete the shared pool (now operator-only via `Memory.moderate`);
  - the shared-pool manager's output bypassed the privacy gate;
  - tool claims could borrow a user quote, and merged update text was ungrounded (now value provenance);
  - timestamp poisoning;
  - unknown roles were trusted as `user`;
  - several 500s on malformed input.
- **Scale:**
  - incremental vector cache: search after a write went from 1.4 s to 30 ms at 100k memories;
  - full-text scoping only where it's selective;
  - optional per-scope HNSW index (experimental);
  - a PostgreSQL backend that passes the whole test suite;
  - cache invalidation shared across processes.
- **Ecosystem:**
  - TypeScript SDK;
  - LangChain, LangGraph, LlamaIndex and CrewAI adapters, plus plain tool schemas.
- **Web UI:** tabs for sharing (consent, share, shared pool), access (agent grants, teams, export) and admin (keys, teams, ledger).
- **Release:** multi-arch GHCR image with SLSA provenance; coverage gate (80%) in CI.

### Changed
- The default ranking is now `contextual`. The original per-memory (`global`) utility
  lowered within-family Hit@3 by 5.9 points in the transfer experiment.

- **Memory engine:** grounded extraction (verbatim quotes verified against source),
  role-based source authority, ADD/UPDATE/DELETE/NOOP reconciliation with
  non-destructive versioning, and bi-temporal validity with `as_of` queries.
- **Retrieval:** hybrid dense + BM25 + entity retrieval with RRF, AMA-style granularity
  routing, raw episodes always fused in, and an outcome prior from feedback.
- **Answers:** citation-locked, with abstention and a sufficiency retry ladder.
- **Outcome learning:** `feedback`, `learn` (experience distillation), `consolidate`
  (archive / dedupe / summarise), and `export_trajectories` for GRPO.
- **Access control:** teams, user↔agent and agent↔resource grants with retroactive
  revocation, and owner-only writes.
- **Sharing:** two-stage experience sharing with a deterministic PII/canary gate and
  pseudonymous provenance.
- **Integrity and erasure:** hash-chained ledger with `verify`; physical erasure
  (`forget`); portable export/import.
- **Interfaces:** CLI (`veramem init/add/search/ask/serve/mcp/...`), REST API with web UI,
  MCP server, and Docker image.
- **Benchmarks:** guarantees suite (G1–G10) and a LoCoMo harness, with a mem0 adapter.
