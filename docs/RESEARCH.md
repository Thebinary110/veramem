# Research foundations

VeraMem builds on eight papers. For each one, this page lists which technique VeraMem
uses, where it lives in the code, and where VeraMem deliberately departs from the paper.

## Gap A: outcome-driven memory consolidation

### Memory-R1: Enhancing LLM Agents to Manage and Utilize Memories via Reinforcement Learning
Yan et al., ACL 2026, [arXiv:2508.19828](https://arxiv.org/abs/2508.19828)

| Technique from the paper | In VeraMem |
|---|---|
| Memory manager choosing ADD / UPDATE / DELETE / NOOP against retrieved neighbours | `write.Writer._decide` and `prompts.RECONCILE_SYSTEM` |
| The trained manager learns to merge instead of fragmenting (DELETE+ADD → UPDATE) | Prompt says "Prefer UPDATE over DELETE+ADD"; DELETE is non-destructive and keeps the new statement as a successor |
| Answer-agent *memory distillation*: select memories, then answer | `answer.Answerer` makes one call that selects ids and answers |
| Outcome reward (EM via a frozen answer agent) | Online equivalent: `feedback()` credits the cited memories; manager decisions are logged so `export_trajectories()` can attach downstream reward to them |

**Departure:** VeraMem needs no RL training to be useful. It learns online from outcome
signals and exports GRPO-ready data for anyone who wants to train a manager.

### AgeMem: Agentic Memory, Unified Long-Term and Short-Term Memory Management
Yu et al., ACL 2026, [arXiv:2601.01885](https://arxiv.org/abs/2601.01885)

| Technique from the paper | In VeraMem |
|---|---|
| LTM tools (add, update, delete) and STM tools (summary, filter) | `add` / `update` / `delete` / `forget`; `consolidate()` writes session summaries (coarse granularity) |
| R_storage (quality of what is stored) | Grounding gate: candidates with unverifiable quotes are rejected; duplicates merge |
| R_maintenance (update or delete happened) | `consolidate()` dedupes and archives failing memories |
| R_relevance (retrieved set relevant to the task) | Outcome prior: utility re-weights ranking (`retrieval.UTILITY_BETA`) |
| Step-wise GRPO with a broadcast terminal reward | `export_trajectories()` writes the same terminal reward onto every logged decision |

## Gap B: evidence-grounded, verifiable memory

### Agent Zero Memory: Provenance-Aware Long-Term Memory for LLM Agents
Wu & Zhu, 2026, [arXiv:2608.29606](https://arxiv.org/abs/2608.29606)

| Technique from the paper | In VeraMem |
|---|---|
| P1 layering: raw sources separate from distilled memory | `Episode` (immutable, SHA-256) vs `MemoryItem` |
| Provenanced items ⟨origin, time, evidence pointer⟩ | `Evidence(episode_id, quote, start, end, source)` on every memory |
| Citation-locked answers (cite only what was opened; otherwise abstain) | `answer.Answerer` strips citations to ids that were never shown, and abstains |
| Dense + lexical retrieval fused with RRF (k = 60) | `retrieval._rrf` over dense, BM25 (FTS5) and entity channels |
| Non-destructive knowledge updates | UPDATE writes a new version; the old one gets `valid_to` and stays queryable with `as_of=` |

**Beyond the paper:**
- Quotes are *verified* against the source text, not just stored.
- The whole store is tamper-evident through a hash-chained ledger (`ledger.py`).

### AMA: Adaptive Memory via Multi-Agent Collaboration
Huang et al., ACL 2026 Findings, [arXiv:2601.20352](https://arxiv.org/abs/2601.20352), [code](https://github.com/Sherlockwz/AMA) (Apache-2.0)

| Technique from the paper | In VeraMem |
|---|---|
| Granularities: raw text, facts, episodes | Raw episodes, facts, and summaries |
| Retriever routes queries by intent | `retrieval.route()` is a rule-based router, so no LLM call is needed at query time |
| Judge: sufficiency and conflict checks, then a retry ladder | `answer.Answerer` retries: dual channel, then source turns, then neighbouring turns |
| Refresher: conservative updates to outdated memories | Reconcile prompt rule "If unsure, ADD"; updates are versioned |
| Constructor rules (facts come only from new input, dates are normalised) | `prompts.EXTRACT_SYSTEM` rules 4 and 5 |

**Departure:** raw turns are *always* a low-weight retrieval channel, not only a fallback.
In VeraMem's own benchmark, the extraction LLM silently skipped 4 of 8 identical-looking
facts. A memory system must never let an extraction miss make information unfindable.

### When Memory Becomes Authority: Benchmarking Authority Collapse at the Memory Consolidation Boundary
2026, [arXiv:2608.01679](https://arxiv.org/abs/2608.01679)

| Technique from the paper | In VeraMem |
|---|---|
| role-policy-v1: user → Authorized, assistant → Attested, tool → Unendorsed | `types.Authority.for_source`, assigned in code, never by the LLM |
| "The label follows the source that establishes the proposition" | The extractor must name the source message and quote it; the label comes from that message's role |
| Seven prohibited upgrades (R2F, P2R, C2O, MIX, O2I, R2P, S2D) | No-upgrade rules in `write.Writer._update`; non-user sources cannot create preferences; all seven are tested in `benchmarks/guarantees.py` G4 |
| Gold label plus source-attributed text gives the best ASR/TSR | Both are stored: the structured `authority` field *and* attributed wording ("According to crm, ...") |
| Avoid blanket distrust (it kills task success) | `check_action()` asks for confirmation only on low-authority memories; it doesn't drop them |
| Keep low-authority claims rather than omitting them (recall) | The extraction prompt asks for attributed extraction from tool and assistant messages |

## Gap C: safe cross-agent and cross-user experience transfer

### ShareMem: Learning from Others, Acting for You: Cross-User Memory Sharing for LLM Agents
Hu et al., 2026, [arXiv:2609.32511](https://arxiv.org/abs/2609.32511)

| Technique from the paper | In VeraMem |
|---|---|
| Two-stage consolidation (local, then shared) with add / replace / none | `learning.Learner._manage_experience` runs both stages; `sharing.Sharer.share` triggers stage 2 |
| Admission only from well-rewarded trajectories | `learn()` requires reward ≥ 0.8, or ≤ −0.5 for lessons from failure |
| Experience template with `{slot}` abstraction | `prompts.EXPERIENCE_SYSTEM` (template verbatim) |
| Preferences stay user-bound; a SHARE-ALL configuration caused 3–11% misuse | Hard invariant: preferences can never be marked shareable |
| Current-user data overrides shared experience | Shared hits are labelled "auxiliary", returned after the user's own memories, and never displace them |

**Beyond the paper:** a deterministic PII and canary gate runs after the LLM abstraction
(`privacy.check_shareable`). Shared items carry no evidence pointers into private
episodes, only an HMAC pseudonym of the contributor, which is withdrawn on erasure.

### Collaborative Memory: Multi-User Memory Sharing in LLM Agents with Dynamic Access Control
Rezazadeh et al., 2025, [arXiv:2505.18279](https://arxiv.org/abs/2505.18279)

| Technique from the paper | In VeraMem |
|---|---|
| Time-varying bipartite graphs G_UA(t) and G_AR(t) | `grants` table with `granted_at` / `revoked_at` |
| Admissible set M(u,a,t) = {m : A(m) ⊆ A(u,t) ∧ R(m) ⊆ R(a,t)}, evaluated at read time | `policy.read_filter` and `policy.episode_filter`, compiled to SQL |
| Private and shared tiers | `visibility` is private, team or public |
| Write policies that transform before sharing | The sharing pipeline: abstraction, then the privacy gate, then a shared manager |

### AIM: A Privacy-Aware Interoperable Memory Framework for Multi-Agent Multi-User LLM Systems
2026, [arXiv:2609.12320](https://arxiv.org/abs/2609.12320)

| Technique from the paper | In VeraMem |
|---|---|
| Visibility predicate V(u,m), enforced at the index level (pre-filter) | Same: SQL `WHERE` before scoring, never post-filtered |
| Scope-restricted dedup (prevents leakage through the dedup signal) | `write.Writer._scope_where` |
| Write ownership (only the creator may update or delete) | `policy.check_write` |
| Adversarial cross-user queries | `benchmarks/guarantees.py` G1 |
| Interoperable, system-agnostic service | REST, MCP, a portable `veramem.export.v1` JSON format, and the Python SDK |

**Beyond the paper:**
- Team scopes, which the paper lists as future work.
- Physical erasure: `secure_delete`, an FTS rebuild, and `VACUUM`.
