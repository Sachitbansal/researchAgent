# Evaluation

What was measured, how, and what the numbers actually say — including the ones that
came out worse than hoped. `docs/ARCHITECTURE.md` §13 summarises the design; this is
the results document it points at.

The current numbers are the **final evaluation of 2026-10-25**, the first section after
the question set. The agent runs after it are **earlier runs**, kept as history and dated.
All numbers are reproduced verbatim from `eval/results/*.json`, and none is re-run for
this document: the agent model is a hosted service and `eval/build_corpus.py` queries a
live arXiv, so a re-run would produce different numbers and leave the prose disagreeing
with the JSON beside it.

## The corpus these numbers describe

11 papers · 406 chunks · 48 figures. Two deliberately **adjacent** topics, indexed into
one pool so cross-topic distractors are present by construction — trivially disjoint
topics (astrophysics vs. baking) would make retrieval look good for a reason that has
nothing to do with the system.

| topic tag | query |
|---|---|
| `sparse_mixture_of_experts_routing_in_transformer_models` | sparse mixture-of-experts routing in transformer models |
| `structured_pruning_and_quantization_for_efficient_transforme` | structured pruning and quantization for efficient transformer inference |

<details>
<summary>The 11 papers (arXiv ids as stored)</summary>

`2310_00811v1` Sparse Backpropagation for MoE Training ·
`2410_18035v1` MiLoRA ·
`2412_11629v1` QPruner ·
`2502_01705v4` Progressive Binarization with Semi-Structured Pruning ·
`2503_06184v1` Sample-aware Adaptive Structured Pruning ·
`2505_18232v4` Two-Stage Regularization-Based Structured Pruning (TRSP) ·
`2506_18952v2` LLMs on a Budget? Say HOLA ·
`2605_17598v1` Mixture of Experts for Low-Resource LLMs ·
`2607_08027v1` Structured Pruning via Power Transform ·
`2607_16427v1` Multi-level Context Modeling for Expert Selection ·
`2608_15102v2` A Declarative-Procedural Perspective on Expert Routing

</details>

## Question set and splits

18 questions in `eval/questions.json`, hand-annotated with gold chunks, gold papers,
required facts, and the tools a reasonable agent should reach for. Five categories:
single-paper (5), multi-paper (4), conflicting (3), not-in-corpus (1) and
absent-but-fetchable (5). There are no figure-dependent questions; the multimodal path is
demonstrated in `examples/cold_start.md` but not scored.

**12 questions tune, 6 are held out.** The tuning split has 8 answerable questions and 4
absent ones; the held-out split has 4 and 2. Every threshold sweep and tuning decision
touched only the tuning split. The held-out split is run only at the end, and reported
separately. That is the only defensible way to report a threshold that was itself chosen
from data.

At M8 the set was 15 questions in four categories, 10 tuning and 5 held out. The earlier
runs below say which set they used.

### Absent but fetchable (added for CRAG phase 1)

The set now holds **18 questions, 6 held out**, so the held-out share is still a third.
The new category is `absent_fetchable`. Its subject is absent from the corpus but well
covered on arXiv, so the right outcome is to abstain **and** propose a fetch for the user
to approve (`docs/DECISIONS.md` D14). Abstention is scored as before, and the proposal
is scored separately.

| question | split | subject | origin |
|---|---|---|---|
| `ni02` | tuning | 4-bit quantization of vision transformers | re-tagged from `not_in_corpus` |
| `ni03` | held out | client drift in federated learning | re-tagged from `not_in_corpus` |
| `af01` | tuning | few-step diffusion sampling | new |
| `af02` | tuning | over-smoothing in graph networks | new |
| `af03` | held out | sim-to-real transfer for robot policies | new |

- **Annotation.** `expect_fetch_proposal: true`. `expect_abstention` stays `true`, so the
  re-tag alone does not change how the M8 gate sweeps score `ni02`/`ni03` (`previous_category`
  records their history). The af questions were written before any score was probed. They
  do join the tuning split, so a re-run of a gate sweep now sees 12 tuning questions.
- **`ni01` stays a pure abstention control.** It carries no `expect_fetch_proposal`, so
  it is not scored for proposals either way. Proposing a fetch for sourdough is not wrong:
  the admission gate's job is to find that arXiv has nothing.
- **Scoring.** `eval.metrics.fetch_proposal_detected(run)` reads the structured
  `fetch_proposal` field that `AgentLoop.run` derives (D18; since D34, from whether the
  final answer cites anything). The runner reports `fetch_proposal_on_fetchable` (the target)
  and `fetch_proposal_on_answerable` (false alarms: a needless fetch button).
- **Measured.** First on the tuning split ("CRAG phase 2" and later earlier runs), then
  on both splits in the final evaluation below. At phase 1 there were no numbers, because
  the prompt rule that acts on the hint was about to change behaviour.
- **The live fetch path is not in the automated eval.** Record/replay of arXiv responses
  is benched. `scripts/crag_fetch_check.py` runs one real `search_literature` on a
  throwaway copy of the data dir and proves the real `data/index` is byte-identical
  afterwards.

## Final evaluation (2026-10-25)

The full agent under the shipped configuration: hybrid retrieval on, contextual headers
off, per-tool call limits (D4), the D34 fetch trigger, and the system prompt's "How to
search" rule (plain keyword phrases, sub-queries for multi-part questions; D35). Same corpus as every run here: 11 papers, 406 chunks.
Written to `eval/results/final_tuning.json` and `final_held_out.json`, with 0 harness
errors on either split. `search_literature` was called 0 times in 18 runs, and the run
reported the index unchanged afterwards.

| metric | tuning (12 q) | held out (6 q) |
|---|---|---|
| fact coverage (answerable) | **1.000** | **1.000** |
| abstention accuracy | **1.000** (12 of 12) | **1.000** (6 of 6) |
| — abstains on absent (answer text) | 1.000 | 1.000 |
| — false abstention on answerable | 0.000 | 0.000 |
| citations resolved | 0.971 | **1.000** |
| gold-paper hit rate | **1.000** (8 of 8) | **1.000** (4 of 4) |
| MRR | 0.906 | 0.446 |
| recall@1 / @3 / @5 | 0.150 / 0.217 / 0.321 | 0.042 / 0.167 / 0.333 |
| `fetch_proposal_on_fetchable` | **1.000** (3 of 3) | **1.000** (2 of 2) |
| `fetch_proposal_on_answerable` (false proposals) | 0.125 (1 of 8) | **0.000** (0 of 4) |
| tool trace matched | 0.917 | 1.000 |
| mean iterations | 4.92 | 2.83 |
| median time per question | 46.6 s | 29.2 s |

Over time, on comparable metrics (all agent-level; M8 used the 15-question set):

| run | split | fact coverage | MRR | recall@5 | fetch on fetchable | false fetch |
|---|---|---|---|---|---|---|
| M8 (2026-09-09) | tuning, 10 q | 1.000 | 0.660 | 0.238 | — | — |
| CRAG phase 2 | tuning, 12 q | 1.000 | 0.750 | 0.342 | 0.667 | 0.000 |
| retrieval v2 | tuning, 12 q | 0.969 | 0.781 | 0.321 | 1.000 | 0.125 |
| "How to search" prompt (`tuning_split_prompts_v3.json`) | tuning, 12 q | 1.000 | 0.643 | 0.258 | 0.667 | 0.125 |
| **final** | tuning, 12 q | **1.000** | **0.906** | **0.321** | **1.000** | **0.125** |
| M8 (2026-09-09) | held out, 5 q | 0.917 | 0.458 | 0.250 | — | — |
| **final** | held out, 6 q | **1.000** | **0.446** | **0.333** | **1.000** | **0.000** |

**A short, honest reading:**

- **The generation side holds on held-out data.** Every answerable question got all its
  facts, every abstention was right, every held-out citation resolved, and every
  answerable question reached a gold paper.
- **Corrective RAG works on both splits.** All 5 absent-but-fetchable questions got a
  fetch proposal, including the two held-out ones (ni03, af03), which had never been run.
- **One false proposal, and it is the cost of D34.** sp01 answered correctly (3 of 3
  facts) but wrote no inline citations, so the card appeared. D34 reads citations, not
  correctness. An uncited answer is a defect anyway: `citations_resolved` skips it, so the
  card is the only place it shows.
- **Citations resolved 0.971 is one citation.** It is a mean over the 7 tuning answers
  that cite anything. mp03 cited `2502_01705v4__c0001`, a real chunk that the agent had not
  retrieved in that run, so 4 of its 5 citations resolved.
- **MRR is mostly noise at this n.** The "How to search" run and the final run differ only
  in the D34 trigger, which does not touch retrieval, yet tuning MRR moved from 0.643 to
  0.906. That is run-to-run variation of a hosted model over 8 questions. Held-out MRR is
  flat against M8 (0.446 vs 0.458): three of its four answerable questions first saw a
  gold chunk at rank 3 to 5.
- **Recall@5 is up on both splits against M8** (0.238 → 0.321 tuning, 0.250 → 0.333
  held out). It cannot be credited to the retrieval upgrade: the CRAG phase 2 run, before
  hybrid existed, already reached 0.342. The agent's keyword queries are where hybrid adds
  least (see "Retrieval upgrade"), and the differences are within one or two questions'
  worth of noise.
- **Sub-queries are now used.** 17 of 51 `retrieve_evidence` calls passed `queries` (14 of
  42 tuning, 3 of 9 held out, counted from the run's traces), against 0 of 42 in the
  retrieval-v2 run. No arm isolates what they are worth.
- **Not measured here:** long-term memory (the eval never runs in a thread), the live
  fetch itself (no arXiv call is made), and run-to-run spread (one run per split).

## Earlier runs

The agent runs and analyses from here through "Retrieval upgrade" were made before the final
configuration existed. They are kept because the decisions in `docs/DECISIONS.md` cite
them. The sweeps and ablations after them (relevance threshold, rerank, admission) are
still the measurements behind the shipped values.

### M8 results (2026-09-09, 15 questions)

These results were measured under the old global 8-iteration cap (`agent.max_iterations: 8`), before per-tool call limits replaced it (`docs/DECISIONS.md` D4).

| metric | tuning (10 q) | held out (5 q) |
|---|---|---|
| fact coverage | **1.000** | **0.917** |
| abstention accuracy | **1.000** | **1.000** |
| — abstains on absent topics | 1.000 | 1.000 |
| — false abstention on answerable | 0.000 | 0.000 |
| citations resolved | **1.000** | **1.000** |
| gold-paper hit rate | **1.000** | **1.000** |
| MRR | 0.660 | 0.458 |
| recall@1 / @3 / @5 | 0.088 / 0.175 / 0.238 | 0.042 / 0.208 / 0.250 |
| tool trace matched | 0.900 | 1.000 |
| harness errors | 0 | 0 |

The generation-side metrics hold up on held-out data. The retrieval-side ones drop
(MRR 0.660 → 0.458), which is what a 10-question tuning split should be expected to do
— it is a small sample and the threshold was fitted on it.

### CRAG phase 2 (2026-10-25, earlier run)

The tuning split was run once with the phase-2 prompt rule (`docs/DECISIONS.md` D24–D27),
written to `eval/results/tuning_split_crag2.json`. The cited `tuning_split.json` is
untouched. This is now 12 questions, with af01 and af02 added. The run uses per-tool limits
(D4), not the old 8-iteration cap, so the comparison is not like for like. To guarantee
the eval made no arXiv calls, it ran on a throwaway copy of the data with an arXiv
cooldown planted. `search_literature` was called **0 times** in all 12 runs. Afterwards,
`search_literature`'s description gained one sentence (leave `categories` unset, D25).
That text is sent with every request, but the tool was never called here, so no rerun
was made.

| metric | baseline (10 q, cited) | phase 2, same 10 q | phase 2, all 12 q |
|---|---|---|---|
| fact coverage | 1.000 | 1.000 | 1.000 |
| abstention accuracy | 1.000 | 1.000 | 1.000 |
| citations resolved | 1.000 | 1.000 | 1.000 |
| `fetch_proposal_on_fetchable` | — | — | **0.667** (2 of 3) |
| `fetch_proposal_on_answerable` (false proposals) | — | — | **0.000** (0 of 8) |

- **Proposals land on af01 and af02.** Both answers name what was searched for and the two
  held topics, point to the "Fetch papers" button, and answer nothing from the model's own
  knowledge.
- **ni02 got no proposal, and the cause is the gate, not the derivation.** Its last
  unfiltered retrieve ("vision transformer quantization accuracy") passed, on chunks from
  the corpus's LLM-quantization papers. One earlier retrieve did return `propose_fetch` at
  −3.83, but a later passing call clears it, as D18 and D24 intend. The answer abstains
  correctly but still tells the user to use the "Fetch papers" button, which does not
  appear without a proposal. The prompt rule leaks into abstentions that have none. The baseline run of ni02 also cited 6 chunks
  from those papers. The relevance gate cannot tell "quantization of ViTs" from
  "quantization of LLMs" at −3.0.
- **ni01 (sourdough) proposed a fetch.** It is unscored by design (see above). The
  admission gate is what should then find that arXiv has nothing.
- **One regression worth knowing about:** sp03 answered correctly (facts 3/3) but with no
  inline citations, where the baseline had 5. `citations_resolved` skips answers with no
  citations, so the 1.000 above does not show it. This is one run of a hosted model, not a
  measured trend.

### Why recall@5 is low while gold-paper hit rate is 1.000 (M8 analysis)

These look contradictory. Part of the gap is a ceiling effect, but — measured rather
than assumed — **less of it than is comfortable.**

Gold sets hold 5 or 6 chunks each (mean 5.92 over the 12 answerable questions; the three
`not_in_corpus` questions have no gold chunks and are scored by abstention instead).
`recall@k` counts gold chunks among the first *k* chunks the agent saw across the whole
conversation, so with `k=5` against ~6 gold chunks the ceiling is **0.847**, not 1.0.

Measured recall@5 is 0.238. Against a ceiling of 0.847 that is roughly 28% of what was
achievable — so the ceiling explains some of the gap and **not most of it**. The first
five chunks the agent sees contain about 1.4 of the ~6 chunks an annotator marked as
supporting.

What keeps this from being a failure is what the same runs show alongside it: gold-paper
hit rate 1.000, fact coverage 1.000/0.917, and every citation resolving. The agent is
reliably reaching the right papers and producing correct, supported answers — it is
just not reaching the *specific* chunks the annotation names, in the first five. Two
things drive that, and `eval/metrics.py` flags the first in its own docstring:

- **Gold sets cannot be exhaustive.** A neighbouring chunk from the right paper often
  supports the claim just as well and is scored as a miss.
- **Chunk boundaries are arbitrary** with respect to where a fact sits. Section-aware
  chunking at ~350 tokens splits an argument across chunks that a human annotator
  labelled as one.

Read recall@5 as a relative signal between configurations, never as an absolute — and
read gold-paper hit rate as the metric that reflects whether the system worked.

## Retrieval upgrade (2026-10-25, earlier runs)

New measurements, beside the cited ones, which are untouched. Retrieval-only, local, tuning
split only (12 questions: 8 answerable, 4 absent). Decisions are D29–D33 in
`docs/DECISIONS.md`.

### Where the loss was (`eval/results/retrieval_diagnosis.json`)

| query | pool recall @5 / @10 / @20 / @40 | after rerank @1 / @3 / @5 | gold in pool | median gold bi-rank |
|---|---|---|---|---|
| raw question | 0.108 / 0.108 / 0.254 / 0.400 | 0.042 / 0.083 / 0.146 | 19 / 47 | 53 |
| agent's first query (traces) | 0.104 / 0.171 / 0.279 / 0.621 | 0.087 / 0.175 / 0.237 | 29 / 47 | 32 |

Most gold chunks never reach the 40-slot pool, so rerank cannot rescue them. Of the
misses, the cross-encoder would put 7 (raw) and 3 (agent) in the top 5 if they had been
shortlisted. The SparseMixer case is different. `2310_00811v1__c0023` is always in the
pool (bi-rank 3–16, 16 phrasings), and wording decides whether it reaches the top 5.

### Ablation (`eval/results/retrieval_ablation_v2.json`)

Raw questions, `k=10` returned, gate at −3.0. Each arm uses its own scratch index;
`data/index` is untouched.

| arm | r@1 | r@3 | r@5 | r@10 | MRR | pool@40 | paper hit | answers (answerable) | abstains (absent) | median latency |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.042 | 0.083 | 0.125 | 0.125 | 0.344 | 0.400 | 0.875 | 0.875 | 1.000 | 1510 ms |
| +headers | 0.042 | 0.062 | 0.083 | 0.083 | 0.312 | 0.317 | 0.875 | 0.875 | 1.000 | 1515 ms |
| **+hybrid (shipped)** | **0.062** | **0.125** | **0.188** | **0.229** | **0.479** | **0.567** | 0.875 | 0.875 | **1.000** | 1487 ms |
| headers+hybrid | 0.062 | 0.125 | 0.188 | 0.229 | 0.479 | 0.567 | 0.875 | 0.875 | 1.000 | 1474 ms |
| +multi (retrieving) | 0.042 | 0.083 | 0.104 | 0.167 | 0.338 | 0.421 | 0.875 | 0.875 | 1.000 | 3049 ms |
| hybrid+multi (all pairs) | 0.042 | 0.104 | 0.167 | 0.250 | 0.410 | 0.504 | 0.875 | 0.875 | 1.000 | 4293 ms |
| hybrid+multi (retrieving) | 0.042 | 0.104 | 0.167 | 0.250 | 0.410 | 0.504 | 0.875 | 0.875 | 1.000 | 3253 ms |

The same arms on each question's first agent query, replayed from the main checkout's
traces (10 tuning questions; af01/af02 have no trace):

| arm | r@5 | r@10 | MRR | pool@40 | paper hit |
|---|---|---|---|---|---|
| baseline | 0.237 | 0.321 | 0.656 | 0.621 | 1.000 |
| +headers | 0.196 | 0.279 | 0.542 | 0.537 | 0.875 |
| +hybrid | 0.279 | 0.362 | 0.656 | 0.662 | 1.000 |
| headers+hybrid | 0.279 | 0.362 | 0.667 | 0.683 | 0.875 |

- **Hybrid is the only arm that helps.** Its abstention holds at 1.000, and its latency is
  flat because the pool stays 40.
- **Headers hurt the dense pool.** With hybrid they add nothing. They ship off.
- **Multi-query was measured with a deterministic split, not an LLM.** It uses the
  question plus its clauses, split at sentence ends and ", and <wh-word>" (7 of 12
  questions split). The second clause is often a pronoun fragment ("what does it propose
  to fix that"), so this is a lower bound. It lowered MRR, because the clauses share one
  40-slot pool. Rerank cost: 4364 ms over 120 pairs for every pair, against 1428 ms for
  one query (3.1×). Scoring only the sub-queries that found a candidate costs 3323 ms over
  93 pairs (2.3×), and ships.
- **SparseMixer, before → after** (`eval/results/sparsemixer_case_v2.json`): single
  queries are unchanged, with c0023 in the top 5 for 9 of 16 phrasings both times.
  Broad wording ("SparseMixer results", "limitations", "… results benchmarks") ranks it #13–18. Paired with a
  specific sub-query in `queries`, it ranks #1, #1 and #5: top 5 in 3 of 3.

### Thresholds re-checked (`threshold_sweep_v2.json`, `bi_threshold_sweep_v2.json`)

The cross-encoder gate stays at −3.0. It is still the balanced best (0.933; −2.0 ties at
a lower MRR). The bi-encoder fallback stays at 0.55, but on the 12-question split its best
balance is 0.400: it abstains on 1 of 4 absent questions, since ni02, af01 and af02 score
cosines ≥ 0.71. That number comes from the larger split, not from hybrid: the max cosine
is identical either way. Nothing changed, and nothing is pending.

### Agent run, tuning split (`eval/results/retrieval_v2.json`)

Final config (hybrid on, headers off, `multi_query_rerank: retrieving`), with an arXiv
cooldown planted so that no fetch could reach arXiv. `search_literature` was called 0
times. One run of a hosted model.

| metric | baseline (10 q, cited) | CRAG phase 2 (12 q) | retrieval v2 (12 q) |
|---|---|---|---|
| fact coverage | 1.000 | 1.000 | 0.969 |
| abstention accuracy | 1.000 | 1.000 | 1.000 |
| abstains on absent (answer text) | 1.000 | 1.000 | 0.750 |
| citations resolved | 1.000 | 1.000 | 1.000 |
| gold-paper hit rate | 1.000 | 1.000 | 1.000 |
| MRR | 0.660 | 0.750 | 0.781 |
| recall@1 / @3 / @5 | 0.088 / 0.175 / 0.238 | 0.108 / 0.150 / 0.342 | 0.108 / 0.196 / 0.321 |
| `fetch_proposal_on_fetchable` | — | 0.667 | 1.000 |
| `fetch_proposal_on_answerable` | — | 0.000 | 0.125 |
| median time per question | 33.7 s | 29.7 s | 36.5 s |

- **No clear agent-level gain.** MRR and recall move within one question's worth of noise
  against CRAG phase 2. The offline MRR gain does not show at the agent level, because the
  agent writes keyword queries, where hybrid adds less (+0.042 recall@5, MRR unchanged).
- **The agent never used `queries`.** None of 42 `retrieve_evidence` calls passed it. The
  tool description alone did not change behaviour, so multi-query's agent-level effect is
  unmeasured. *Later, a "How to search" rule in the system prompt got the agent to use
  sub-queries in 7 of 46 calls (`tuning_split_prompts_v3.json`).*
- **af01: "abstains on absent" 0.75 is a text-marker miss, not a gate failure.** *Since
  fixed: "does not cover" is now an abstention marker.* All three
  retrieves failed the gate (top rerank −8.95, −5.46, −3.98, all below −3.0). It proposed
  a fetch, and its answer opens "The indexed corpus does not cover this topic". That
  phrase is not in `eval/metrics.py`'s marker list, so `abstention_accuracy`, which also
  reads the gate, stays 1.000. No threshold is involved.
- **sp03: the false fetch proposal (0.125 = 1 of 8).** Its first seven retrieves passed
  (top rerank +0.02, +0.56, −1.87, −2.80, −2.79, −2.02, +4.25). The hybrid first call
  returned four QPruner gold chunks, against three for dense alone. The agent then
  kept searching with quoted boolean syntax. Its last two calls failed at −3.47 and −6.21,
  and the last unfiltered failure (−6.21, well beyond the 0.25 rephrase margin) proposed a
  fetch. The answer led with QLoRA, which a corpus chunk mentions, and had no citations.
  No swept threshold fixes this: −6.21 passes only at −8.0, where absent abstention falls
  to 0.25. This is agent behaviour (one hosted-model run), and it
  interacts with D18/D24's "last unfiltered call" rule.
- **sp02: facts 4/4 → 3/4, citations 5 → 1.** Attributable to hybrid. On the agent's query
  BM25 added three chunks from an off-target paper that clear the gate (−1.47 to −2.18).
  Dense alone returned the one gold chunk plus two off-target ones, and neither arm
  returned the other gold chunks.
- **ni02 now proposes a fetch.** In phase 2 it did not, so `fetch_proposal_on_fetchable`
  is 3 of 3.

## Relevance threshold sweep

Cross-encoder scores are uncalibrated logits spanning roughly −11..+11, so the gate
threshold can only come from data. Swept −8..+2 on the tuning split, scored by harmonic
mean so a threshold cannot win one half by abandoning the other:

| threshold | answers when answerable | abstains when absent | balanced | gold-paper hit |
|---|---|---|---|---|
| −8.0 | 1.000 | 0.500 | 0.667 | 1.000 |
| −6.0 | 1.000 | 0.500 | 0.667 | 0.875 |
| −4.0 | 0.875 | 0.500 | 0.636 | 0.875 |
| **−3.0** | **0.875** | **1.000** | **0.933** | 0.875 |
| −2.0 | 0.875 | 1.000 | 0.933 | 0.875 |
| −1.0 | 0.750 | 1.000 | 0.857 | 0.750 |
| 0.0 | 0.500 | 1.000 | 0.667 | 0.500 |
| +2.0 | 0.125 | 1.000 | 0.222 | 0.125 |

`0.0` was the placeholder the code shipped with before tuning. It refuses **half** the
answerable questions — a bug that no unit test could have caught, because the gate was
behaving exactly as written. −3.0 and −2.0 tie; −3.0 was taken as the more permissive of
the two.

## The rerank ablation — a marginal gain, and what the ablation missed

This is the result that came out worse than hoped, and then turned out to be measuring
the wrong thing. Both halves are recorded here, because the second half only surfaced
by acting on the first.

Same 10 questions, retrieval only, rerank on vs. off:

| | recall@1 | recall@3 | recall@5 | MRR | gold-paper hit | median latency |
|---|---|---|---|---|---|---|
| rerank **on** | 0.042 | 0.083 | 0.125 | 0.344 | 0.875 | 1548 ms |
| rerank **off** | 0.042 | 0.063 | 0.108 | 0.317 | **1.000** | **21 ms** |
| delta | 0 | +0.021 | +0.017 | **+0.027** | **−0.125** | **+1527 ms** |

Reranking buys **+0.027 MRR for +1527 ms** — a 74× latency increase. On gold-paper hit
rate it is *worse*, 0.875 against 1.000.

### Both differences are below the noise floor

The ablation scores the 8 answerable questions in the tuning split. At n=8, one
question moving from rank 2 to rank 1 shifts MRR by **0.0625**. The measured difference
is **0.0271** — less than half the smallest move a single question can make. It is not
a small effect; it is sub-resolution.

The gold-paper hit difference is the same story from the other side: 0.875 vs 1.000 is
**7/8 vs 8/8**, exactly one question.

So neither the gain nor the harm is measurable here. The only robust number in the
table is the cost: **74×**, which is not noise.

### Acting on that read was wrong, and the correction is the finding

The obvious conclusion — rerank is not earning its keep at 406 chunks, so turn it off —
was tested before being applied. It does not survive, because **the ablation never
measured abstention.** It reports recall, MRR, latency and gold-paper hit. It says
nothing about whether the system still refuses questions the corpus cannot answer.

The relevance gate is what produces abstention, and it thresholds whatever score the
retrieval stage produced. With rerank on that is a cross-encoder logit (−11..+11, gated
at −3.0). With rerank off it is a cosine (0..1), and every cosine clears −3.0 — so
turning rerank off silently disables the gate entirely:

```
rerank=True   ABSENT topic ("sourdough fermentation")  -> 0 chunks, sufficient=False
rerank=False  ABSENT topic ("sourdough fermentation")  -> 5 chunks, sufficient=True
```

Giving the bi-encoder its own tuned threshold does not recover it. Swept 0.0–0.80 on the
same 10 tuning questions (`eval/results/bi_threshold_sweep.json`):

| gate | best threshold | answers when answerable | abstains when absent | balanced |
|---|---|---|---|---|
| bi-encoder (rerank off) | 0.55 | 1.000 | **0.500** | **0.667** |
| cross-encoder (rerank on) | −3.0 | 0.875 | **1.000** | **0.933** |

No cosine threshold reaches the cross-encoder's separation. The thresholds that abstain
correctly (0.75+) collapse answerable coverage to 0.125 — refusing 7 of 8 answerable
questions. The ranges genuinely overlap: an absent topic scores 0.57, while real
answerable questions sit at 0.55–0.65.

That is a principled result rather than a tuning accident. Cosine measures overlap
between two independently-encoded vectors. The cross-encoder attends over query and
passage together, so it can represent "this passage does not answer this question" —
a judgement a dot product between separate encodings cannot make.

**Rerank stays enabled.** Not because the architecture says so, but because the metric
it earns its cost on is abstention, and the ablation simply never looked there. What
the MRR ablation does establish is narrower and still true: at this corpus size,
reranking does not measurably improve *ordering*. It is paying 1527 ms for the gate,
not for the ranking.

Two things this leaves open, neither done:
- **The ablation should measure abstention.** It is the metric rerank actually carries,
  and it is absent from `eval/ablation.py`.
- **`k_retrieve` is untouched at 40.** Rerank cost scales with the shortlist, so a
  smaller pool may keep the gate at a fraction of the latency. That is a measurement
  nobody has taken.

A latent bug surfaced on the way, independent of this decision. `retrieve_evidence`
falls back to bi-encoder scores when the reranker fails at runtime, but kept gating at
the cross-encoder's −3.0 — so a reranker that failed in production would silently
disable abstention with no error. The gate now selects its threshold by which scorer
actually ran, and `bi_encoder_relevance_threshold` is tuned to 0.55 for that path, with
its limits documented above.

## Admission threshold sweep

`search_literature` scores `(query, title + abstract)` with the cross-encoder before any
download (`docs/DECISIONS.md` D16). The threshold was swept offline, with no arXiv calls,
over 175 labelled pairs against the 11 indexed abstracts (`eval/tune_admission.py`,
`eval/results/admission_sweep.json`), using the tuning split only.

- **Positives:** each topic's query against its own papers (topic-shaped), and each
  tuning question against its gold papers (question-shaped).
- **Absent negatives:** the absent and absent-fetchable tuning questions, plus five
  off-domain queries from `eval/topics.yaml`, against every abstract.
- **Cross-topic negatives:** each topic's query against the other topic's papers (and the
  question-shaped equivalent). Adjacent but distinct is the hard case.

Scored by harmonic mean of the three rates, as above:

| threshold | admits positive | rejects absent | rejects cross-topic | balanced | topic-shaped | question-shaped |
|---|---|---|---|---|---|---|
| −8.0 | 0.885 | 0.990 | 0.800 | 0.885 | 0.783 | 0.877 |
| −7.0 | 0.846 | 1.000 | 0.920 | **0.918** | 0.889 | 0.885 |
| −5.0 | 0.769 | 1.000 | 0.960 | 0.898 | 0.968 | 0.812 |
| −4.0 | 0.692 | 1.000 | 0.980 | 0.866 | 0.968 | 0.724 |
| **−3.0** | 0.615 | 1.000 | 1.000 | 0.828 | **1.000** | 0.600 |
| 0.0 | 0.577 | 1.000 | 1.000 | 0.804 | 1.000 | 0.522 |
| +2.0 | 0.385 | 1.000 | 1.000 | 0.652 | 0.968 | 0.000 |

- **Topic-shaped pairs separate perfectly** anywhere in (−3.94, +1.23]. `search_literature`
  is called with topic phrases, so −3.0, the permissive end of that gap, was taken.
- **The hardest adjacent negative** (the pruning query against an MoE routing paper)
  sits 0.94 below −3.0, and n is 11 per group.
- **The cost is on question-shaped queries.** A long question scores far lower against
  a whole abstract than a topic phrase does, and only 61.5% of those positives are
  admitted at −3.0. The pooled optimum, −7.0, would admit 3 of the 11 topic-shaped
  cross-topic papers instead. A gate that keeps adjacent papers out also rejects some
  useful ones. They are listed in `papers_rejected` with their scores, never silently
  dropped.

## What else the numbers do not say

- **n is small.** 12 tuning and 6 held-out questions (10 and 5 at M8). One question
  moves a rate over the 8 answerable tuning questions by 0.125, and over the 4 held-out
  ones by 0.25. Every metric here should be read with that in mind.
- **One annotator.** Gold chunks and required facts were annotated once, by the same
  person who wrote the questions. No inter-annotator agreement was measured.
- **Tool trace matched 0.900 on tuning at M8** (0.917 in the final run), and the M8 miss is instructive: on `cf02`
  ("is expert specialisation coherent or unstable?") the agent answered from two
  retrieval calls and never ran `check_evidence_consistency`, which the annotation
  expected. It reached a well-cited, correct answer anyway (fact coverage 1.0, 9/9
  citations resolved) — but it got there without the verification step the system
  prompt asks for on multi-source answers. Nudging that behaviour is prompt work that
  has not been done.
- **Temperature is 0.0**, but the provider is not bit-reproducible, so a re-run will
  vary somewhat regardless.
- **The corpus is not frozen.** `eval/build_corpus.py` queries live arXiv; the same
  topics will return a different 11 papers later.

## Reproducing

```bash
python main.py eval --out final_tuning                # tuning split
python main.py eval --held-out --out final_held_out   # held-out split
python scripts/m8_gate.py                             # both splits plus the rerank ablation
```

Without `--out`, each split writes its own default file. A full eval makes LLM calls. The
eval never sends a fetch approval, so it reaches arXiv only if the agent breaks its prompt
rule and calls `search_literature` unasked; the earlier CRAG runs planted an arXiv cooldown
to rule even that out.

Threshold sweep: `python -m eval.tune_threshold`. Ablation alone: `python -m eval.ablation`.
Results land in `eval/results/`.

Retrieval upgrade (local, no LLM, no arXiv), each writing its own new file:

```bash
python eval/diagnose_retrieval.py --traces ../researchAgent/logs/traces   # retrieval_diagnosis.json
python eval/diagnose_retrieval.py --case-only --out eval/results/sparsemixer_case_v2.json
python eval/retrieval_ablation.py --traces ../researchAgent/logs/traces   # retrieval_ablation_v2.json
python -m eval.tune_threshold --out threshold_sweep_v2.json
python -m eval.tune_bi_threshold --out bi_threshold_sweep_v2.json
python main.py eval --out retrieval_v2                                    # one LLM run
```
