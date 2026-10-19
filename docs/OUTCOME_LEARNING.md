# What VeraMem learns from feedback

**Short answer, after two follow-up experiments:**

- **The learned ranker transfers to unseen task families, but on LoCoMo a static
  weighting does better.** With genuine feedback the ranker improves Hit@3 on held-out
  conversations by +2.1 points [95% CI +1.1, +3.1]. A fixed re-weighting of the same
  retrieval features, tuned once on labelled data and given no feedback at all, improves
  it by +3.4 [+2.2, +4.6]. On this data the ranker learns retrieval *calibration* (mainly
  "longer turns tend to be the evidence"), and a tuned default captures that calibration
  better.
- **Where the useful signal is not a retrieval signal, feedback is what finds it.** In a
  controlled benchmark where tools and the assistant make conflicting claims, the ranker
  learns from feedback alone to prefer user-stated (authorized) memories. On attribute
  families it never saw, Hit@1 goes from 0.14 to 1.00. A static weighting tuned on
  retrieval features cannot learn this unless someone already knows the domain.
- **Uninformative feedback never changes rankings.** Shuffled rewards left the deployment
  gate closed on 15 of 15 seeds across both experiments.
- **The original per-memory ("global") utility does not help.** Within a family it lowers
  Hit@3 by 5.9 points. The default is now `contextual`.

This supports **"outcome-informed ranking that learns non-retrieval signals such as source
authority and transfers them"**. It does **not** support "feedback beats a well-tuned
retriever on LoCoMo", and it is **not** a learned consolidation policy like Memory-R1 or
AgeMem: deciding what to write, update or delete is prompted, not learned.

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

## Results (run 2; the follow-ups below have the current numbers)

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

## Follow-up 1: a static baseline (no feedback)

Is the ranker's transfer *learning*, or calibration a fixed setting could provide? The
`static-tuned` method re-weights exactly the features the ranker relied on (BM25 rank and
memory length). Its two weights are grid-searched per seed on the feedback families'
questions using **gold** labels, which is strictly more information than feedback
provides, and then frozen for evaluation (`--static-baseline`). Results are from the current
code, 5 seeds (raw data in
[results/ranker-validation-2026-10-19](results/ranker-validation-2026-10-19/static-baseline-report.md)):

| Split | Method | Hit@3 | Δ vs no feedback [95% CI] |
|---|---|---|---|
| held-out families | none | 0.386 | |
| held-out families | ranker, genuine feedback | 0.407 | +0.021 [+0.011, +0.031] |
| held-out families | ranker, shuffled feedback | 0.386 | +0.000 |
| held-out families | **static-tuned (no feedback)** | **0.420** | **+0.034 [+0.022, +0.046]** |
| within families | ranker, genuine feedback | 0.410 | +0.027 [+0.012, +0.042] |
| within families | **static-tuned (no feedback)** | **0.426** | **+0.043 [+0.026, +0.060]** |

The tuning picked a small length boost (weight 0.25, no BM25 boost) on 4 of 5 seeds. The
ranker's held-out gain here (+0.021) is lower than in run 2 (+0.028). That is the same
method on the current code, after the retrieval changes made since run 2.

**Conclusion:** on LoCoMo, the transferable part of the feedback signal is retrieval
calibration, and a tuned static default beats learning it online. The defaults were *not*
changed to that length boost: it was tuned on one dataset, and adopting it would need
validation elsewhere first.

## Follow-up 2: can the ranker learn source authority? (`benchmarks/authority_ranking.py`)

LoCoMo never varies authority or kind, so the first experiment couldn't test those
features. In this controlled benchmark:

- There are 8 attribute families (airline, home city, diet, ...), split 4/4 into feedback
  families and held-out families.
- Each user has their own statement of the value, plus two tool claims and one assistant
  suggestion with wrong values. The wrong ones are worded to match the question better.
- Memories go through the real write path, so authority and kind are assigned by
  production code.

| Held-out attribute families | Hit@1 [95% CI over 5 seeds] |
|---|---|
| retrieval only | 0.144 [0.015, 0.273] |
| ranker, genuine feedback | **1.000** |
| ranker, shuffled feedback | 0.144 (gate closed on 5/5 seeds) |

Learned coefficients were consistent across seeds:

| Feature | Coefficient |
|---|---|
| authorized | +1.5 to +2.1 |
| unendorsed | −0.6 to −0.9 |
| attested | −0.8 to −1.2 |
| preference | +0.35 to +0.66 |

The perfect score reflects a clean synthetic signal. This shows the authority and kind
features are wired correctly and can be learned and transferred. It does not show how
large the effect would be on real traffic.

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
5. **A static re-weighting does better on LoCoMo** (follow-up 1). The ranker's value there
   is limited to calibrating retrieval, which a tuned default does better.
6. **The agent and reward are simulated.** The cite-top-3 agent and the gold-turn reward
   are a proxy. With an LLM agent and an LLM judge the effect could be larger or smaller.
7. **Authority and kind features were tested only on synthetic data** (follow-up 2). Real
   effect sizes are unmeasured.
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
mem = Memory()  # ranking="contextual" by default
r = mem.search("...", user_id="u")
mem.feedback(r.trace_id, +1, cited=[r.hits[0].memory.id])  # what your agent used
...
mem.train_ranker()  # deploys only if it beats retrieval with 95% confidence
mem = Memory(ranking="ranker")  # falls back to contextual if the gate is closed
```
