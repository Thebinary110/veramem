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
