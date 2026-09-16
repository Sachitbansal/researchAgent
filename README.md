# Agentic research assistant over scientific papers

Ask a research question; the agent decides which tools it needs, retrieves evidence
from a corpus of arXiv papers, reads figures when the text is not enough, checks its
own draft for unsupported claims, and answers with a citation on every claim — or tells
you it cannot answer, when the corpus does not support one.

It is **topic-agnostic**: a topic is a runtime argument, not a configuration. Point it
at a subject it has never seen and it collects, indexes and answers from scratch — see
[`examples/cold_start.md`](examples/cold_start.md), a run from empty index to cited
answer on protein structure prediction, a field neither eval topic touches.

No agent framework. The tool loop is written directly against the provider's
OpenAI-compatible API, because how the agentic system is structured is itself part of
what this project set out to evaluate.

## What it does

| | |
|---|---|
| **Collection** | Plans an arXiv query with an LLM, fetches PDFs, dedups against what is already held |
| **Extraction** | PyMuPDF text + caption-driven figure rendering; a vision model describes each figure, and the description is embedded *with* its caption |
| **Retrieval** | Two stage — bi-encoder over the full index (k=40), cross-encoder rerank on that shortlist only, a per-chunk relevance gate, then ±1 neighbour expansion |
| **Agent** | One hand-written loop, five tools, an iteration cap and a forced final answer |
| **Verification** | An NLI model checks drafted claims against retrieved text, and flags contradictions between sources |

The five tools are `retrieve_evidence`, `search_literature`, `analyze_corpus`,
`inspect_figure` and `check_evidence_consistency` — specified in
[`docs/TOOLS.md`](docs/TOOLS.md).

## Setup

Tested on **Python 3.10.12**, Linux.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads three models from HuggingFace (~500 MB total): the bi-encoder
`BAAI/bge-small-en-v1.5`, the cross-encoder `ms-marco-MiniLM-L-6-v2`, and
`nli-deberta-v3-small`. They are cached afterwards.

### Environment

Create a `.env` in the repo root:

```bash
OPENROUTER_API_KEY=sk-or-...      # required — every LLM call goes through OpenRouter
OPENROUTER_APP_NAME=sciagent      # optional, used in the attribution header
```

| variable | required | purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | yes | LLM, vision and utility calls |
| `OPENROUTER_APP_NAME` | no | OpenRouter attribution header, defaults to `sciagent` |
| `SCIAGENT_CONFIG` | no | path to an alternative `config.yaml`; same as `--config` |

### GPU (optional)

Three models run locally: the bi-encoder, the cross-encoder reranker and the NLI
cross-encoder. `compute.device` in `config.yaml` governs all three.

`auto` (the default) uses CUDA when torch can reach a device and falls back to CPU
otherwise, so the project runs unchanged on a machine with no NVIDIA driver. Naming a
device explicitly (`cuda`, `cuda:0`, `cpu`) is a hard assertion instead: an unreachable
device fails at config resolution rather than quietly dropping to CPU, because a silent
fallback turns a driver problem into an unexplained slowdown.

Check what is actually in use, and time the three models on it:

```bash
python scripts/check_gpu.py              # exits 0 on CUDA, 1 on CPU
python scripts/check_gpu.py --skip-bench # report the device only
```

Measured on the same 406-chunk corpus (278 tokens mean), 16-thread CPU vs an RTX 3050
6GB laptop GPU, same texts in one process:

| stage | when it runs | CPU | GPU | saved |
|---|---|---|---|---|
| embed 406 chunks | index time, once per new chunk | 23.9s | 3.6s | 20.3s |
| rerank 40 pairs | per query | 1.35s | 0.20s | 1.15s |
| NLI 30 pairs | per groundedness check | 1.18s | 0.32s | 0.86s |
| CUDA context init | once per process | — | 2.2s | −2.2s |

Which nets out, per command, as roughly:

| command | saved |
|---|---|
| `index`, cold (406 new chunks) | ~18s |
| `index`, incremental (one paper, ~37 chunks) | ~0s — the init costs more than it saves |
| `ask`, one question | ~0s — likewise |
| `eval`, 30 questions in one process | ~58s |

Break-even is about **45 newly embedded chunks**, or **2 questions in a single process**.
Below that the CUDA context init is not repaid. The GPU is worth having for the eval
harness; it is not what makes indexing fast — the figure-description model is.

**Installing the driver on Ubuntu.** The CUDA torch wheel is already installed; what is
missing on a fresh machine is the host driver. Nothing in the project changes — with
`compute.device: auto`, the next run picks the GPU up on its own.

```bash
ubuntu-drivers devices              # shows the recommended package for this card
sudo ubuntu-drivers install         # installs it
sudo reboot
```

If Secure Boot is enabled (`mokutil --sb-state`), the install prompts for a one-time
password and the reboot stops at a blue **MOK Manager** screen. Choose *Enroll MOK* →
*Continue* → *Yes*, enter that password, and reboot again. Skipping this step leaves the
kernel module unsigned and unloadable, and `nvidia-smi` keeps failing with the driver
apparently installed — which looks like a torch problem and is not one.

On hybrid Intel + NVIDIA laptops the Intel chip keeps driving the display; the discrete
GPU is used for compute only, and no Xorg or PRIME configuration is needed.

## Usage

Three subcommands.

### Index a topic

```bash
python main.py index "sparse mixture-of-experts routing in transformer models"
```

Searches arXiv, downloads the PDFs, extracts text and figures, describes each figure
with a vision model, then embeds and indexes everything. Re-running is cheap: papers
already held are not re-downloaded, embeddings are cached by content hash, and figure
descriptions are cached by image hash.

```
--max-results N     papers to fetch (default: collection.max_results_default)
--categories ...    restrict to arXiv categories, e.g. cs.CL cs.LG
--no-describe       skip vision calls; figures index on caption text alone
--rebuild           rebuild the vector index from scratch
```

Indexing a second topic adds to the *same* index — papers are distinguished by
`topic_tags`, and a paper found under two topics gains both tags.

### Ask a question

```bash
python main.py ask "What load-balancing losses do these papers use for MoE routing?"
python main.py ask "..." --show-tools     # print each tool call the agent made
```

Prints the answer, then the iteration count, tool-call count, context size, and the
path to a full JSONL trace of every tool call under `logs/traces/`.

### Run the eval

```bash
python main.py eval               # the 10-question tuning split
python main.py eval --held-out    # the 5 held-out questions
python main.py eval --limit 3     # a quick smoke run (saved separately, so it
                                  # cannot overwrite a full run's results)
```

Results land in `eval/results/`. Full write-up in
[`docs/EVALUATION.md`](docs/EVALUATION.md).

### Working against a separate corpus

`--config` points the whole system at a different `config.yaml`, and a config with
different `paths.*` gives an entirely separate corpus — used by the cold-start demo so
it cannot disturb the eval corpus:

```bash
python main.py --config data_demo/config.yaml ask "..."
```

## Results in brief

On 11 papers / 406 chunks, over 15 hand-annotated questions:

| | tuning (10 q) | held out (5 q) |
|---|---|---|
| fact coverage | 1.000 | 0.917 |
| abstention accuracy | 1.000 | 1.000 |
| citations resolved | 1.000 | 1.000 |
| gold-paper hit rate | 1.000 | 1.000 |
| MRR | 0.660 | 0.458 |

The rerank stage buys **+0.027 MRR for +1527 ms** at this corpus size — a difference
below the noise floor of an 8-question sample. It stays enabled anyway, for a reason the
ablation could not see: the relevance gate thresholds whatever score retrieval produced,
so turning rerank off swaps a cross-encoder logit for a cosine and silently disables
abstention. A separately tuned cosine gate does not recover it. The measurement, and
what it leaves open, is in
[`docs/EVALUATION.md`](docs/EVALUATION.md#the-rerank-ablation--a-marginal-gain-and-what-the-ablation-missed).

## Examples

- [`examples/worked_examples.md`](examples/worked_examples.md) — an abstention, a
  conflicting-evidence case, and an ordinary multi-paper synthesis, lifted from scored
  eval runs
- [`examples/cold_start.md`](examples/cold_start.md) — a topic from a different field
  entirely, taken from nothing to an answered question, with every tool call shown.
  Regenerate with `python scripts/m9_cold_start.py --fresh`; it builds into
  `data_demo/`, so it cannot disturb the eval corpus in `data/`.

## Layout

```
main.py                CLI: index / ask / eval
config.yaml            every tunable, with the measurement behind each value
src/
  config.py            config + secrets loading
  llm_client.py        the OpenRouter wrapper — retries, image attachment
  common/              storage, records, tokenization
  corpus/              arXiv query planning, collection, manifest, chunk store
  extraction/          PDF text, chunking, figure rendering, vision descriptions
  retrieval/           embedder, FAISS index, indexer, cross-encoder reranker
  tools/               the five agent-facing tools
  agent/               tool registry, conversation, trace, the loop
  analysis/            NLI
  prompts/             system prompt and tool descriptions, as files
eval/                  question set, annotation, metrics, tuning, ablation
scripts/               one verification gate per milestone, m0..m8, plus the M9 demos
tests/                 unit and smoke tests
docs/                  the specs — these are authority, not background reading
```

`eval/` imports `src/`, never the reverse. That is why `main.py` sits at the repo root:
`eval` is one of its subcommands.

## Configuration

Everything tunable is in `config.yaml`, and each value carries the measurement that
produced it. The ones most worth knowing:

| key | value | why |
|---|---|---|
| `chunking.max_tokens` | 445 | min of the two encoder budgets; the cross-encoder binds |
| `retrieval.k_retrieve` / `k_final` | 40 / 5 | wide cheap pool, narrow expensive context |
| `retrieval.relevance_threshold` | −3.0 | swept −8..+2 on the tuning split only |
| `retrieval.bi_encoder_relevance_threshold` | 0.55 | the gate when the cross-encoder did not run; a degraded fallback, not an alternative |
| `agent.max_iterations` | 8 | with a forced final answer when the cap is hit |
| `agent.max_context_tokens` | 60000 | past this the oldest tool results are elided |
| `nli.min_pair_similarity` | 0.75 | below it, NLI scores unrelated sentences as confident contradictions |

Models are set per role (`agent_model`, `vision_model`, `utility_model`). The agent
model must accept **both tools and images** — `inspect_figure` attaches a figure to the
agent's own conversation on a follow-up turn, so a text-only model breaks that path.

## Tests

```bash
python -m pytest tests/ -q          # 191 tests
python scripts/m5_gate.py           # one milestone's verification gate
```

If pytest fails at collection with an import error from something outside this repo,
a system `PYTHONPATH` is leaking foreign pytest plugins into the run (ROS does this).
`env -u PYTHONPATH python -m pytest tests/ -q` isolates it.

Unit tests are not the main safety net here. Each milestone ends at a **gate** —
`scripts/mN_gate.py` — that runs the real pipeline on real data and prints pass/fail per
claim. Nearly every real defect in this project was caught by a gate or by reading a
gate's output, not by an assertion; `docs/ARCHITECTURE.md` §15 lists them.

## Documentation

| | |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | the design, its rejected alternatives, and §15 on how AI agents were used to build it |
| [`docs/TOOLS.md`](docs/TOOLS.md) | the five tools — signatures, return shapes, failure modes |
| [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) | records, ids, hashes, on-disk layout |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | what was measured and what the numbers do not say |
| [`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md) | the milestones and their verification gates |

## Known limitations

- **Small eval set.** 15 questions, one annotator. Every number above should be read
  with that in mind.
- **No index eviction.** Removing from a flat index with a positional id map implies a
  full rebuild. Bounded instead by `max_papers_per_topic`. `IndexIDMap` is the fix if it
  ever matters.
- **Tables are images.** No structural table parsing; a table is rendered and described
  like a figure.
- **arXiv only**, and its rate limits are real. `export.arxiv.org` throttles by IP and
  then answers HTTP 429 to *every* query, not just the one that tripped it. A burst of
  indexing runs earns a cooling-off period measured in tens of minutes, during which no
  collection can proceed. Indexing and asking are unaffected once a corpus exists,
  since both work off the local index.
