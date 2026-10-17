# Benchmarks

Everything on this page can be reproduced on one machine, with no API keys.

## Setup used for the published numbers

| | |
|---|---|
| Hardware | NVIDIA GB10 (DGX Spark), 119 GB unified memory |
| LLM for every system, plus answerer and judge | `qwen3.6:35b` via Ollama, thinking **off**, temperature 0 |
| Embedder for every system | `nomic-embed-text` (768-d) via Ollama |
| VeraMem | this repository (version in `src/veramem/__about__.py`) |
| mem0 | `mem0ai==2.2.1` (PyPI), its best OSS configuration: `[nlp,extras]` (spaCy entities + BM25), embedded Qdrant |

Parity notes:
- **Thinking mode.** mem0's Ollama adapter does not disable qwen3 "thinking". The harness
  forces `think=False` for both systems (`benchmarks/adapters._disable_thinking`). Without
  this, mem0 took about 70 s per `add`.
- **Timestamps.** mem0 OSS rejects per-message timestamps ("platform-only"). Its adapter
  therefore writes the session date into the message text *and* keeps it in metadata,
  where it appears in the answer context. This helps mem0; it was not required.
- **Roles.** mem0 accepts only `user` / `assistant` / `system`. Tool and document messages
  are passed as `user` messages prefixed with `[tool output]` / `[document output]`, so
  the source is still visible to its extractor.

## 1. Guarantees benchmark (`benchmarks/guarantees.py`)

This turns each gap into a measurable test. A capability a system does not offer is
reported as **unsupported**, never as a pass.

| Test | What is measured | Derived from |
|---|---|---|
| G1 isolation | 8 users with private canary codes; 336 cross-user queries (direct, generic, canary string, prompt-injection). Leak = any foreign canary in results. Also: own-recall. | MUMBench (AIM) |
| G2 dedup side channel | Two users store an identical fact; the second must get their own copy, not be silently merged into the first. | AIM §3.5 |
| G3 revocation | Memory written via an agent; revoke the user↔agent grant; it must disappear, then reappear after re-grant. | Collaborative Memory Eq. 3 |
| G4 authority collapse | 7 prohibited transitions × 5 synthetic unique values (35 pairs, 70 histories). H⁻: a tool, document or assistant carries the value. H⁺: the user states it. **Upgrade** = a retained H⁻ memory that presents the value as the user's own (machine label, or an LLM judge on the text when the system has no label). Ret⁻ / Ret⁺ = retention. | AuthMem-Bench |
| G5 knowledge update | 8 entities, each changing value three times; is the current value ranked #1, and is a stale value ranked above it? | LongMemEval KU, AMA |
| G6 outcome adaptation | 6 questions whose correct memory is outranked by lexically closer distractors; 5 rounds of top-1 retrieval with ±1 feedback. | Memory-R1 / AgeMem reward signal |
| G7 tamper evidence | Edit a stored memory directly in the backend; is it detected? | Agent Zero Memory P2 |
| G8 evidence coverage | Share of stored memories with a resolvable source span. | Agent Zero Memory Def. 1 |
| G9 erasure | After deleting a user, scan every storage file (bytes + decoded SQLite) for the user's secret. | GDPR Art. 17 |
| G10 footprint | Search p50 latency, external services, telemetry default. | |

**Caveats:**
- G4–G6 use synthetic data written for this suite, following each paper's protocol. They
  are not the papers' official datasets: AuthMem-Bench and MUMBench have no public
  release yet. The LLM-judge prompt is in the source, and every judgement is cached in
  `benchmarks/runs/`.

## 2. LoCoMo (`benchmarks/locomo.py`)

The protocol used by the Mem0, Memory-R1 and AMA papers:
- Ingest every session with its date.
- Answer every non-adversarial question (categories 1–4; 1,540 questions in total).
- Score with LLM-judge accuracy (J), token F1 and BLEU-1.

**Same answerer for every system.** Each system returns its top-k (k = 20) memories, and
one fixed answer prompt and model produce the answer. Differences therefore come only from
what each memory system stores and retrieves. `veramem-full` additionally reports
VeraMem's own `answer()` pipeline, which adds the citation lock and the sufficiency retry
ladder.

**Comparison with published numbers.** Papers report LoCoMo with GPT-4o-mini or GPT-4.1
judges. Our judge is the local `qwen3.6:35b`, so absolute numbers aren't directly
comparable with papers. The head-to-head numbers on this page are: same judge, same
answerer, same data.

<!-- RESULTS -->
