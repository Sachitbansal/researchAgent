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

## Architecture

Two phases. **Indexing** is slow and runs once per topic; **asking** is cheap and reads
what indexing produced. They share nothing but files on disk.

```
index "<topic>"
  arXiv query planned by an LLM, cached by topic hash
  → search, dedup against the manifest, download new PDFs
  → PyMuPDF text; figure regions rendered and described by a vision model
  → section-aware chunks (350 tokens, max 445)
  → BGE embeddings → FAISS IndexFlatIP + a positional chunk_id map

ask "<question>"
  agent loop ── chooses among five tools, up to 8 iterations
       │
       ├─ retrieve_evidence          bi-encoder top-40 → cross-encoder rerank
       │                             → relevance gate → top-5 + neighbours
       ├─ search_literature          fetch and index new papers mid-question
       ├─ analyze_corpus             stats, timeline, KMeans clustering
       ├─ inspect_figure             attach a figure image to the conversation
       └─ check_evidence_consistency NLI: contradictions, groundedness
```

### Design choices

**No agent framework.** The loop is written directly against the provider's API. How the
agentic system is structured is part of what is being evaluated, and a framework hides
it. The whole loop is ~200 lines.

**The agent chooses its tools.** There is no routing logic — the model gets five schemas
and decides. Most questions need only `retrieve_evidence`; some need none of them.

**Statelessness is handled explicitly.** `/chat/completions` stores nothing between
requests, so the entire message array is rebuilt and re-sent every iteration.
`agent/conversation.py` owns that array and enforces the API's pairing rules in code —
the assistant turn is appended verbatim before its results so `tool_call_id`s stay
intact. Past a token budget the oldest tool results have their *content* replaced with a
placeholder; the messages stay, because deleting one orphans its `tool_call` and the API
rejects the whole request.

**Two-stage retrieval.** A bi-encoder scores the whole index cheaply; the cross-encoder
only ever rescores its top 40. Reranking the corpus directly would defeat the point.

**Abstention is a mechanism, not a prompt instruction.** Chunks scoring below a tuned
relevance threshold are withheld, and the tool reports `sufficient_evidence: false`.
Naive top-k always returns something, which is what lets a model answer confidently from
irrelevant text.

**Every threshold is measured.** `config.yaml` carries the sweep or measurement behind
each number rather than a plausible-looking constant.

Full detail, including rejected alternatives, is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

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
| `SCIAGENT_CONFIG` | no | use a different `config.yaml`; paths live in it, so this selects a separate corpus |

### GPU (optional)

Three models run locally — the bi-encoder, the cross-encoder reranker and the NLI
model. `compute.device` in `config.yaml` governs all three.

`auto` (the default) uses CUDA when torch can reach a device and falls back to CPU, so
the project runs unchanged with no NVIDIA driver present. Naming a device explicitly
(`cuda`, `cpu`) is a hard assertion instead: an unreachable device fails at config
resolution rather than quietly dropping to CPU, because a silent fallback turns a driver
problem into an unexplained slowdown.

```bash
python scripts/check_gpu.py              # reports the device and times the models
python scripts/check_gpu.py --skip-bench # report the device only
```

It is worth having for indexing and the eval harness, and makes no difference to a
single question — the CUDA context costs more to start than one query saves.

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
python main.py ask "..." --show-tools     # after the answer, one line per tool call
python main.py ask "..." --quiet          # no live progress
```

While it runs, each step is printed to **stderr** as it happens — the model thinking,
every tool call with its arguments, what came back and how long it took:

```
[1] thinking...
[1] → retrieve_evidence(query='sparse mixture-of-experts load balancing') ... 8 chunks, 6796ms
[2] thinking...
[2] → check_evidence_consistency(mode='groundedness', ...) ... 20 claims, grounded 0.5, 45571ms
[3] answering
```

The answer itself goes to stdout, so `ask "..." > answer.txt` still captures the answer
alone. Afterwards it prints the iteration count, tool-call count, context size, and the
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

### Show it in a browser

A one-page demo of the same agent, for showing the system to someone rather than
driving it:

```bash
python web/server.py                 # then open http://127.0.0.1:8000
python web/server.py --port 9000 --host 0.0.0.0
```

The answer renders as formatted text with every citation numbered and resolvable to its
paper, the agent's steps stream into the margin as it works, and a second tab lists
every paper in the corpus. It needs the models and the index — this is a local demo, not
a static site that can be hosted on its own.

`web/` sits outside `src/` and imports it, exactly as `eval/` does. Nothing in `src/`
knows it exists: the system being evaluated is still the CLI.

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
