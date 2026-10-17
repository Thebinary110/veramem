# Concepts

## Episodes and memories

An **episode** is raw evidence: one chat message, tool result or document chunk. Each
episode is immutable and SHA-256 hashed. A **memory** is distilled from episodes and
always points back to them through **evidence**: the episode id and a verbatim quote,
which VeraMem checks really occurs in the episode. Candidates whose quote can't be
found are rejected (`AddResult.rejected`).

## Source authority

The role of the message that establishes a memory decides its authority. The LLM never
decides it.

| Source role | Authority | Can drive protected actions? |
|---|---|---|
| user, system | `authorized` | yes |
| assistant | `attested` | ask the user first |
| tool, document, other users | `unendorsed` | ask the user first |

Consolidation can lower authority but never raise it: a tool's claim can't become the
user's preference. A user restating the same thing legitimately makes it `authorized`.
Use `check_action(memory_ids)` before payments, bookings or messages.

## Versions, not overwrites

When something changes ("I moved to Madrid"), VeraMem writes a new version and closes
the old one (`valid_to`). Current searches see only the new version, `history(id)` shows
the whole chain, and `search(..., as_of=timestamp)` shows what was true back then.

## Outcomes

Every `search` and `answer` returns a `trace_id`. Report how it went with
`feedback(trace_id, reward)`, where reward runs from −1 (wrong) to 1 (helpful). Memories
that keep helping rank higher. `consolidate()` archives memories that keep misleading.
`learn(task, trajectory, reward)` distils a finished task into a reusable experience.

## Scopes and sharing

- Memories are private to their user by default.
- Teams (`join_team`) and agent/resource grants (`grant_agent`, `grant_resource`)
  decide who can read what, and revocation takes effect immediately.
- Only experiences can be shared, and only with the owner's consent (`set_shareable`).
  They pass an LLM de-identification step *and* a deterministic PII/canary gate.
- Preferences are never shared.

## Integrity and erasure

- Every write is appended to a hash chain. `verify()` re-hashes the chain, every
  episode, and every memory's text.
- `forget(user_id=..., everything=True)` erases the content physically. Only hashes
  remain in the ledger.
