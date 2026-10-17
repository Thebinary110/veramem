# What VeraMem learns from feedback

**Short answer:** VeraMem's learned ranker improves retrieval on task families it has never
received feedback for: **+2.8 points Hit@3 [95% CI +1.9, +3.9]** on held-out LoCoMo
conversations. It beats a shuffled-reward placebo by the same margin, and uninformative
feedback cannot change rankings, because the ranker is deployed only when it passes a
statistical gate. The per-memory utility that VeraMem shipped with first does **not**
help: within a family it *lowers* Hit@3 by 5.9 points.

This supports the claim **"outcome-informed memory ranking that transfers to unseen
tasks"**. It does **not** support "a learned consolidation policy" like Memory-R1 or
AgeMem: deciding what to write, update or delete is prompted, not learned, in VeraMem.

## The question

Does feedback teach the system which memories improve *future* task success, or does
it only re-rank memories by how useful they were before? Re-ranking by past usefulness
cannot help on tasks whose memories have never been rated, so the test is
**transfer to held-out task families**, with controls that separate learning from
exposure.

## Methods compared

| Method | What it does |
|---|---|
| `none` | Retrieval only (dense + BM25 + entity, reciprocal rank fusion) |
| `global` | One Beta posterior per memory over "this memory helped", used to re-rank (the original VeraMem mechanism) |
| `contextual` | The same posterior, but each past outcome is weighted by the similarity between its query and the current query (kernel `exp((cos−1)/0.1)`) |
| `contextual+ts` | Contextual, plus damped Thompson-sampling exploration while feedback is collected |
| `ranker` | A logistic model over memory-agnostic features: retrieval signals, authority, kind, age, length, outcome statistics and contextual utility. It is residual over retrieval relevance, relevance features are constrained to be non-negative, its regularisation is chosen by grouped cross-validation, and it is **deployed only if** its out-of-fold log-loss beats the relevance-only model with 95% confidence (bootstrap over traces). When it isn't deployed, ranking falls back to `contextual`. |

The feature list is in `src/veramem/ranking.py`. No feature identifies a memory, so
whatever the ranker learns can apply to memories it has never seen.

## Experiment (`benchmarks/outcome_transfer.py`)

- **Data.** LoCoMo: 10 long conversations between different people, 1,540 non-adversarial
  questions, each with gold evidence turns. Memories are the raw turns
  (`infer=False`), so each memory maps to exactly one gold turn id. No LLM is involved
  anywhere in the experiment.
- **Task families = conversations.** Each of 5 seeds splits them 5/5. Every memory in the 5
  held-out conversations is unseen during feedback.
- **Within-family held-out.** Questions in the feedback conversations are split 50/50. A
  held-out question is dropped if it is a near-duplicate (cosine > 0.9) of any feedback
  question in the same conversation (11–25 dropped per seed).
- **Agent.** The agent retrieves the top 10 and cites the top 3. Reward is +1 if a cited
  memory is a gold turn, otherwise −1. Credit goes to the 3 cited memories.
  2 epochs over the feedback questions (690–870 feedback events per seed).
- **Conditions:**
  - `genuine`
  - `shuffled`: an exact permutation of the same seed's genuine reward sequence. The
    reward distribution is preserved and its association with memories is broken.
  - `noisy`: 20% of rewards flipped.
- **Leakage controls.** Evaluation searches use `record=False` and give no feedback. The
  ranker trains only on traces from the feedback families.
- **Metrics.** Hit@3 (the agent's task success), MRR@10, nDCG@10. Seed means come with a
  95% t-interval, and differences come with a paired bootstrap 95% CI over questions
  (all seeds pooled).

## Results (run 2, final design)

### Held-out task families: every memory unseen

| Method | Condition | Hit@3 [95% CI over seeds] | Δ vs no feedback [95% CI] | Δ vs shuffled [95% CI] |
|---|---|---|---|---|
| none | — | 0.386 [0.361, 0.411] | | |
| global | genuine | 0.386 | +0.000 | +0.000 |
| contextual | genuine | 0.386 | +0.000 | +0.000 |
| **ranker** | **genuine** | **0.414 [0.387, 0.442]** | **+0.028 [+0.019, +0.039]** | **+0.028 [+0.019, +0.039]** |
| ranker | noisy (20%) | 0.397 [0.374, 0.420] | +0.011 [+0.005, +0.017] | +0.011 [+0.005, +0.017] |
| ranker | shuffled | 0.386 | +0.000 | |

`global` and `contextual` are exactly zero by construction: they have no statistics
for unseen memories.

### Within-family held-out questions

| Method | Condition | Hit@3 | Δ vs no feedback [95% CI] | Δ vs shuffled [95% CI] |
|---|---|---|---|---|
| none | — | 0.382 [0.353, 0.411] | | |
| **global** (original default) | genuine | 0.323 | **−0.059 [−0.079, −0.039]** | +0.015 [−0.006, +0.035] |
| global | shuffled | 0.309 | −0.074 [−0.093, −0.054] | |
| global | noisy | 0.313 | −0.069 [−0.089, −0.050] | |
| contextual | genuine | 0.383 | +0.001 [−0.003, +0.004] | −0.001 [−0.005, +0.004] |
| contextual+ts | genuine | 0.389 | +0.007 [+0.002, +0.012] | +0.004 [−0.001, +0.009] |
| **ranker** | **genuine** | **0.422 [0.394, 0.450]** | **+0.040 [+0.025, +0.056]** | **+0.038 [+0.023, +0.054]** |
| ranker | noisy | 0.390 | +0.008 [−0.002, +0.017] | +0.005 [−0.004, +0.016] |
| ranker | shuffled | 0.384 | +0.002 [−0.002, +0.007] | |

MRR@10 and nDCG@10 move in the same direction everywhere. See the
[full table](results/outcome-transfer-2026-10-17/run2-report.md).

### The deployment gate

| Feedback | Seeds where the ranker was deployed |
|---|---|
| genuine | 5 / 5 |
| noisy (20% flipped) | 2 / 5 |
| shuffled | 0 / 5 |

### What the ranker learned (standardised coefficients, mean over 5 seeds)

| Feature | Coefficient | Consistent sign across seeds? |
|---|---|---|
| contextual utility | +0.46 | yes |
| contextual evidence mass | +0.26 | yes |
| BM25 reciprocal rank | +0.23 | yes |
| log length of the memory | +0.20 | yes |
| log number of ratings | −0.09 | yes |
| authority, kind, raw-turn flags | 0.00 | constant in this data |

On unseen memories the outcome features are constant, so **the transfer comes from
re-weighting generic retrieval signals**: exact lexical matches and longer, more
informative turns are more often the evidence. Within a family, contextual utility,
meaning outcomes on similar past questions, adds to that.

### Cost

Feedback adds no model calls. With the feedback phase taking about 3.5 s per 700–870
events, search latency stays at about 2–3 ms p50 at this scale for every method.

## Negative results and limitations

1. **Per-memory ("global") utility hurts.** With question-level rewards credited to
   3 cited memories, only one of which is usually the evidence, per-memory counts are
   too noisy, and the strong re-ranking prior (`β = 3`) amplifies the noise. This was the
   original default. **VeraMem now defaults to `contextual`,** which is neutral to
   slightly positive.
2. **The first version of the ranker was unsafe** (run 1,
   [report](results/outcome-transfer-2026-10-17/run1-report.md)).
   - Trained on shuffled rewards, an unconstrained ranker learned a *negative*
     relevance weight from chance correlations and collapsed Hit@3 from 0.39 to 0.13 in
     a smoke run. Monotonicity constraints and the CV-based gate fixed this.
   - Run 1 still lost 10 points within-family under shuffled feedback, because the
     gate's fallback was `global`. Run 2 falls back to `contextual`.
   - Both design changes were made *after* seeing run 1 and are reported as such.
3. **The effect is modest.** +2.8 points Hit@3 on held-out families is about +7%
   relative. With 20% noisy feedback it shrinks to +1.1, and the gate deploys in only
   2 of 5 seeds.
4. **This is not counterfactual credit.** "Cited in a successful answer" is not proof
   that a memory caused the success. Counterfactual credit assignment is future work.
5. **A static re-weighting was not compared.** Because the transferable signal is
   retrieval re-weighting (BM25, length), a hand-tuned static boost might capture part
   of the gain without any feedback.
6. **The agent and reward are simulated.** The cite-top-3 agent and the gold-turn reward
   are a proxy. With an LLM agent and an LLM judge the effect could be larger or smaller.
7. **Only one dataset was used.** Authority and kind features never varied, so this
   experiment says nothing about them.
8. **The write side is not learned.** Outcome signals affect ranking and archival only.
   `export_trajectories()` provides data for training a write policy with GRPO, but
   none is trained here.

## Reproduce

```bash
ollama pull nomic-embed-text
python -m benchmarks.outcome_transfer --seeds 5 --epochs 2   # ~15 min, no LLM needed
```

## Use it

```python
mem = Memory()                      # ranking="contextual" by default
r = mem.search("...", user_id="u")
mem.feedback(r.trace_id, +1, cited=[r.hits[0].memory.id])   # what your agent used
...
mem.train_ranker()                  # deploys only if it beats retrieval with 95% confidence
mem = Memory(ranking="ranker")      # falls back to contextual if the gate is closed
```
