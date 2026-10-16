# VeraMem architecture

VeraMem is a long-term memory layer for AI agents. It is built to keep working where
existing memory systems tend to break:

| Gap | What tends to go wrong elsewhere | What VeraMem does | Grounded in |
|---|---|---|---|
| **A. Outcome-driven consolidation** | Memories are kept or dropped by heuristics or LLM whim; nothing learns from whether a memory actually *helped*. | Every retrieval is a *trace*; `feedback(trace, reward)` credits the memories that were used; utility drives ranking, retention, archival and promotion of experiences. Write-time guards stop the classic DELETE+ADD fragmentation. Trajectories export as GRPO-ready JSONL. | Memory-R1 (ACL 2026), AgeMem (ACL 2026) |
| **B. Verifiable memory** | Extracted "facts" lose the link to what was actually said; a tool's claim becomes "the user's preference" (authority collapse in 48/49 configs of AuthMem-Bench). | Every memory carries evidence pointers (verbatim quotes, checked against the raw episode), a source-authority label that consolidation can never raise, non-destructive versioning with validity time, a hash-chained audit ledger, citation-locked answers with abstention, and an action gate. | Agent Zero Memory (2026), AMA (ACL 2026), AuthMem-Bench (2026) |
| **C. Safe transfer** | Sharing is all-or-nothing; a user's private data can leak through shared pools, dedup side channels or other users' preferences. | Deterministic access predicate applied *before* ranking (never post-filtered); time-varying user↔agent / agent↔resource grants with retroactive revocation; preferences are never shareable; experiences are shared only via a two-stage pipeline (local consolidation → de-identified shared consolidation) with PII scrubbing and canary checks; shared items are labelled auxiliary and Unendorsed. | ShareMem (2026), Collaborative Memory (2025), AIM (2026) |

## Data model

- **Episode**: an immutable piece of raw evidence (a chat message, tool result or document chunk). Each one is SHA-256 hashed and records its source role, user, agent, session and event time. Every memory can be traced back to episodes.
- **MemoryItem**: a fact, preference, event, experience or summary. Fields:
  - `evidence[]`: episode id plus a verbatim quote.
  - `authority`: authorized / attested / unendorsed.
  - Scope: `user_id`, `agent_id`, `team_id`, `visibility`.
  - Bi-temporal validity: `valid_from`, `valid_to`, `created_at`.
  - `status`: active / superseded / retracted / archived.
  - Outcome stats: `uses`, `successes`, `failures`.
  - `shareable`: whether the owner has consented to sharing.
- **Trace**: one retrieval or answer, recording the ids it returned. It is the unit that outcome feedback attaches to.
- **Grant**: `user_agent`, `agent_resource` or `team_member` edges with `granted_at` / `revoked_at`. These model the time-varying bipartite graphs of Collaborative Memory.
- **Ledger**: an append-only hash chain over every write (`hash = sha256(prev_hash || canonical(entry))`). `veramem verify` recomputes it.

## Write path (`add`)

1. **Store episodes.** Raw messages are hashed and stored before anything is derived from them (P1 layering, Agent Zero).
2. **Extract.** One LLM call produces candidate memories. Each candidate must cite the message id it comes from and give a **verbatim quote**.
   - The quote is checked against the episode text with fuzzy matching (normalised ≥ 0.9). A candidate whose quote can't be located is rejected and reported in `AddResult.rejected`.
   - Rules follow the AMA constructor: facts come only from the new messages; context is reference only; explicit dates are normalised; relative dates are resolved against the message time.
3. **Authority (role-policy-v1).** The label is assigned from the role of the message that *establishes* the proposition, never by the LLM:
   - user or system → authorized
   - assistant → attested
   - tool, document or shared → unendorsed
   - A memory derived from lower-authority evidence keeps source-attributed wording (e.g. "According to tool `flights`: …"), which AuthMem-Bench found gives the best ASR/TSR. Low-authority claims are **retained**, not dropped: dropping them hurts recall.
4. **Reconcile** (Memory-R1 manager). For each candidate, retrieve its top-k neighbours *in the same scope only* (AIM scope-restricted dedup, so dedup can't leak across users). Then one LLM call chooses ADD / UPDATE / DELETE / NOOP.
   - **Guard:** a DELETE of m paired with an ADD of f where cos(m, f) > 0.75 is rewritten as UPDATE.
   - **Guard:** a lower-authority candidate cannot UPDATE or DELETE a higher-authority memory. It becomes a separate, attributed memory and a conflict is recorded (`metadata.conflicts_with`).
   - **Guard:** if cos > 0.95 with an existing memory, the result is NOOP plus added evidence.
   - **UPDATE is non-destructive** (AMA Refresher, Graphiti). A new version is written with `supersedes = old`. The old version gets `status = superseded` and `valid_to = now`, so history queries still work.
   - **DELETE is a retraction** (status retracted), not an erase. Erasure is the separate `forget()` call (see Privacy).
   - **No authority upgrade:** a merged memory's authority is the weakest authority among the evidence it keeps.
5. Every change is appended to the ledger.

## Read path (`search`)

1. **Access predicate first.** This is SQL, applied before any scoring:
   ```
   status = active
   AND ( (visibility = private AND user_id = caller)
         OR (visibility = team AND team_id ∈ teams(caller, now))
         OR (visibility = public) )
   AND (no agent grants configured OR agent_id IS NULL OR agent_id ∈ A(caller, now))
   ```
   Preferences are always private. Revoking a grant hides items retroactively.
2. **Granularity routing** (AMA). A query is routed to one or more levels:
   - fine detail → raw episodes
   - "summarise" or "overall" → summaries
   - multi-entity questions → facts plus raw ("dual")
   - default → facts
   The routing is done by a cheap rule-based classifier, so a search makes no LLM call.
3. **Hybrid candidates.** Dense cosine, BM25 (SQLite FTS5) and entity overlap are fused with reciprocal rank fusion (k = 60, as in Agent Zero Memory).
4. **Priors.**
   - Utility, Gap A: `score += λ·(utility − 0.5)`.
   - Mild recency.
   - For temporal queries ("latest", "when", "since …"), the newest valid version wins and results are ordered chronologically, not by relevance.
5. **Shared experiences** from the public pool or team are returned in a separate, labelled block. They never displace the caller's own memories (ShareMem precedence).
6. A trace is written, and `trace_id` is returned for feedback.

## Answer path (`answer`)

- **Distillation and citation lock.** The LLM first selects the memory ids it relies on (Memory-R1 distillation), then answers using only those (Agent Zero Memory citation lock).
- **Citation stripping.** Ids that were not in the retrieved, opened set are removed.
- **Abstention.** If nothing supports an answer, it abstains (`abstained = true`).
- **Sufficiency judge with retry ladder** (AMA). If the judge says "insufficient", the system tries, in order:
  1. dual channel
  2. source episodes of the retrieved facts
  3. neighbouring turns

## Outcome loop (`feedback`, `consolidate`)

- **`feedback(trace_id, reward ∈ [-1, 1])`.** Each memory cited in the trace (or, failing that, retrieved in it) gets credit:
  `successes += max(r, 0)`, `failures += max(−r, 0)`.
  The baseline-subtracted update (GRPO-style) is used when a per-user running mean is available.
- **`consolidate()`** (offline, AgeMem STM→LTM):
  - Archive memories whose utility is below 0.3 after at least 3 uses, or which haven't been used in N days and are low utility.
  - Merge near-duplicates (same scope only).
  - Summarise sessions into SUMMARY memories (the coarse granularity).
  - Distil **experiences** from successful traces (reward ≥ τ = 0.8).
- **`export_trajectories()`.** Writes JSONL in the AgeMem / Memory-R1 record format, so the manager and answer models can be fine-tuned with GRPO later.

## Sharing (`share_experiences`)

ShareMem two-stage consolidation, made safe:

1. **Eligibility.**
   - The owner has consented (`shareable = true`, or a team/org policy allows it).
   - The kind is `experience` or `fact` about the world. **Never `preference`.**
   - Outcome utility is at least τ.
2. **Abstraction.** The LLM rewrites the memory into the `## name / Applies when / Inputs {slot} / Procedure / Caution` template, with every user value replaced by a `{slot}`.
3. **Sanitisation gate**, which is deterministic and goes beyond the papers:
   - Regex PII scan (emails, phones, card/IBAN numbers, IP addresses, URLs with query strings, street addresses).
   - Canary check: no token sequence of 2+ content words from the owner's private memories and preferences may appear verbatim.
   - Anything rejected stays local, and the reason is logged.
4. **Shared reconciliation.** The candidate is reconciled against the shared pool with an independent manager, so a local operation is not replayed as-is.
5. **Shared memory properties.**
   - authority: `unendorsed`, and it can never authorise an action
   - provenance: an HMAC of the contributing user id, so it can be revoked without being re-identifiable
   - visibility: `team` or `public`

## Privacy and erasure

`forget(memory_id | user_id)` hard-deletes the memory rows, the evidence and the episodes. It writes a ledger *tombstone* containing only the id and a hash, so the chain stays verifiable without keeping any content.

## Simplicity

- `pip install veramem`, then `veramem init`. `init` auto-detects Ollama, OpenAI-compatible endpoints and Anthropic.
- One SQLite file, `~/.veramem/memory.db`. Nothing else to run.
- Entry points: `veramem serve` (REST + web UI), `veramem mcp` (Claude Desktop / Cursor / any MCP client), and the Python SDK (`Memory().add/search/answer/feedback`).
