# Evaluation metrics, explained

What each metric in the eval harness calculates, in plain terms, and why it is there.
Numbers and methodology are in [`EVALUATION.md`](EVALUATION.md); the definitions are in
`eval/metrics.py` and `eval/runner.py`.

A useful way to hold the whole set: you ask a librarian a question. Did they fetch the
right books? Did they answer correctly from them? And did they behave sensibly —
including admitting when the library does not have what you asked for?

That gives three layers, and there are seven metrics rather than one because each layer
catches a failure the others cannot. A system can score perfectly on retrieval and still
invent its answer, or write a well-cited answer whose citations point nowhere.

## Setup the metrics rely on

Each question in `eval/questions.json` is hand-annotated with:

- **gold chunks** — the chunks that contain the answer
- **gold papers** — the papers those chunks come from
- **expected facts** — the statements a good answer must make
- **expected tools** — the tools a sensible agent would reach for
- **`expect_abstention`** — whether the correct response is to decline

"Retrieved" means every chunk the agent saw across the whole conversation, in the order
it first saw them.

---

## Layer 1 — Did it find the right evidence?

### Recall@k (k = 1, 3, 5)

**In plain terms:** of the correct chunks, how many were among the first *k* the agent saw?

Six gold chunks, two of them in the first five → recall@5 = 2/6 = 0.33.

**Why:** tells you whether the answer-bearing text actually reached the model early.

**Result:** 0.238 tuning, 0.250 held-out. Partly a ceiling effect — gold sets hold 5–6
chunks, so five slots cannot hold them all and the ceiling is about 0.85 — but only
partly. See `EVALUATION.md` for the full reading.

### MRR — Mean Reciprocal Rank

**In plain terms:** how far down the list is the *first* correct chunk? Position 1 scores 1,
position 2 scores ½, position 4 scores ¼, never found scores 0. Averaged over questions.

**Why:** rewards putting a good chunk at the top, which is what the model reads first.

**Result:** 0.660 tuning — the first correct chunk sits around position 1–2 on average.
0.458 held-out.

### Gold-paper hit rate

**In plain terms:** ignoring exact chunks, did anything from the right *paper* reach the
model?

**Why:** gold chunks cannot be exhaustive. A neighbouring paragraph from the right paper
often answers just as well, and chunk-level recall scores that as a miss. For an agent
whose job is to cite papers, this is the fairer signal.

**Result:** 1.000 on both splits.

---

## Layer 2 — Is the answer right and honest?

### Fact coverage

**In plain terms:** how many of the expected facts does the answer actually state?

**How:** each expected fact and each answer sentence is embedded; a fact counts as covered
if some sentence reaches cosine similarity ≥ 0.62. This is a meaning match, not a word
match — the facts are written as paraphrases, so exact string matching would mark correct
answers wrong. It is also deterministic, which keeps threshold sweeps comparable.

**Result:** 1.000 tuning, 0.917 held-out.

### Citations resolved

**In plain terms:** every `[chunk_id]` the answer cites — does it point at a chunk the
agent actually retrieved?

**Why:** catches fabricated sources. An answer can be fluent and heavily cited and still
cite things that do not exist. An unresolvable citation is worse than none: it reads as a
source but points nowhere.

**Result:** 1.000 on both splits. This metric caught a real defect during development — the
original citation format produced ids that resolved to nothing (0 of 4 on a sample
answer). Changing the format took the same answer to 4 of 4.

---

## Layer 3 — Did it behave correctly?

### Abstention accuracy

**In plain terms:** did it decline *exactly* when it should have? Declining a question the
corpus cannot answer is correct. Answering a question it can answer is correct. Declining
an answerable question is also an error.

**How:** a response counts as abstaining if the relevance gate fired — no chunk cleared the
threshold — or if the answer contains a refusal phrase such as "cannot answer" or
"no evidence". The gate is the mechanism; the phrase check catches the failure being
tested, which is the model answering anyway from its own background knowledge.

**Result:** 1.000 on both splits, with no false abstentions on answerable questions.

### Tool trace match

**In plain terms:** did the agent use at least the tools a sensible agent would?

**How:** checks that the expected tools are a subset of those actually called, and
records which were missing or unexpected. Deliberately soft: the agent is allowed to reach
the right answer another way, so a mismatch is flagged for a person to read rather than
counted as a failure.

**Result:** 0.900 tuning, 1.000 held-out. The one miss, `cf02`, answered a
conflicting-evidence question correctly but without running the contradiction check the
annotation expected.

---

## Rigour

**Held-out split.** Every threshold was tuned on the 10 tuning questions. The 5 held-out
questions ran once, at the end. Without that separation the numbers would be graded on the
same questions they were fitted to.

**Noise floor.** With 8 answerable questions, a single question moving from rank 2 to rank
1 shifts MRR by 0.0625. Differences smaller than that are not measurable on this set — which
is why the rerank ablation's +0.027 MRR is reported as noise rather than a gain.

## Known limitations of the metrics themselves

- **Fact coverage shares a model with retrieval.** It judges with the same BGE embedder the
  system retrieves with, so judge and system are not independent. An LLM judge, or a
  separate embedding model, would remove that coupling.
- **Abstention detection is a phrase list.** An answer saying "there is no evidence for X,
  but …" and then answering would count as a refusal.
- **One annotator.** Gold chunks, facts and expected tools were written by one person; no
  agreement between annotators was measured.
- **No figure-dependent questions.** The multimodal path is demonstrated in
  `examples/cold_start.md` but not scored by any metric here.
- **No "absent but fetchable" category.** Questions outside the corpus all expect
  abstention, including topics arXiv covers well, so the eval rewards declining over
  calling `search_literature`.
