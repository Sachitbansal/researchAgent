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

**Just want to see it work?** After [Setup](#setup), run `python web/server.py` and open
<http://127.0.0.1:8000> (after a one-time `pip install -r web/requirements.txt` and frontend
build). It is the same agent with a page to look at, including a live view of the steps it
takes. See [Show it in a browser](#show-it-in-a-browser).

## What it does

| | |
|---|---|
| **Collection** | Plans an arXiv query with an LLM, fetches PDFs, dedups against what is already held |
| **Extraction** | PyMuPDF text + caption-driven figure rendering; a vision model describes each figure, and the description is embedded *with* its caption |
| **Retrieval** | Two stage — bi-encoder over the full index (k=40), cross-encoder rerank on that shortlist only, a per-chunk relevance gate, then ±1 neighbour expansion |
| **Agent** | One hand-written loop, five tools, a call limit per tool (10; `search_literature` 5), two termination backstops and a forced final answer |
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
  agent loop ── chooses among five tools, up to 10 calls each (search_literature 5)
       │
       ├─ retrieve_evidence          bi-encoder top-40 + BM25 top-40 → RRF pool of 40
       │                             (per sub-query, up to 3) → cross-encoder rerank
       │                             → relevance gate → top-5 + neighbours
       ├─ search_literature          fetch and index new papers mid-question
       ├─ analyze_corpus             stats, timeline, KMeans clustering
       ├─ inspect_figure             attach a figure image to the conversation
       └─ check_evidence_consistency NLI: contradictions, groundedness
```

### Design choices

**No agent framework.** The loop is written directly against the provider's API. How the
agentic system is structured is part of what is being evaluated, and a framework hides
it. The loop itself (`src/agent/loop.py`) is about 500 lines, most of it comments explaining why.

**The agent chooses its tools.** There is no routing logic — the model gets five schemas
and decides. Most questions need only `retrieve_evidence`; some need none of them.

**Statelessness is handled explicitly.** `/chat/completions` stores nothing between
requests, so the entire message array is rebuilt and re-sent every iteration.
`agent/conversation.py` owns that array and enforces the API's pairing rules in code —
the assistant turn is appended verbatim before its results so `tool_call_id`s stay
intact. Past a token budget the oldest tool results have their *content* replaced with a
placeholder; the messages stay, because deleting one orphans its `tool_call` and the API
rejects the whole request.

**Two-stage retrieval.** A bi-encoder and a hand-written BM25 score the whole index
cheaply. Their rankings are fused by rank (RRF) into a pool of 40, and the cross-encoder
only ever rescores that pool. Reranking the corpus directly would defeat the point.

**Abstention is a mechanism, not a prompt instruction.** Chunks scoring below a tuned
relevance threshold are withheld, and the tool reports `sufficient_evidence: false`.
Naive top-k always returns something, which is what lets a model answer confidently from
irrelevant text.

**Every threshold is measured.** `config.yaml` carries the sweep or measurement behind
each number rather than a plausible-looking constant.

Full detail, including rejected alternatives, is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Setup

Python 3.10+. Developed and tested on **Python 3.10.12**, Linux.

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
already held are not re-downloaded, embeddings are cached by the hash of the exact
embedder input, and figure
descriptions are cached by image hash.

```
--max-results N     papers to fetch (default: collection.max_results_default)
--categories ...    restrict to arXiv categories, e.g. cs.CL cs.LG
--no-describe       skip vision calls; figures index on caption text alone
--rebuild           rebuild the vector index from scratch
```

Indexing a second topic adds to the *same* index — papers are distinguished by
`topic_tags`, and a paper found under two topics gains both tags.

`python main.py reindex` re-embeds the chunks already held, with no arXiv call. It is the
step to run after changing what the first stage embeds, e.g. `retrieval.contextual_headers`.
Unchanged inputs hit the cache.

### Ask a question

```bash
python main.py ask "What load-balancing losses do these papers use for MoE routing?"
python main.py ask "..." --show-tools     # after the answer, one line per tool call
python main.py ask "..." --quiet          # no live progress
```

Questions are independent unless you put them in a **thread**, where each new question is
sent with the earlier questions, answers and a one-line note per tool call
(`docs/ARCHITECTURE.md` §7.1):

```bash
python main.py ask "What load-balancing losses do these papers use?" --new-thread   # prints the thread id
python main.py ask "Which of those also report downstream results?" --thread <id>
python main.py threads                    # id, turns, last updated, title
```

Threads live in `data/threads.sqlite3`, shared with the browser UI, so a chat started in
one can be continued in the other. The eval never uses them.

A thread run also uses **long-term memory** (`docs/ARCHITECTURE.md` §7.2). After the answer,
the claims it cites that NLI finds entailed by their passages are remembered, along with
preferences you stated outright ("only papers from 2024 onward"). A later chat on the same
subject is given the relevant ones with their passages, to check and cite. Memories fade
when unused and are replaced when contradicted. `ask` prints a `--- memory:` line. In the
browser UI you can see, edit, add (preferences only), delete and restore them on the
**Memories** page.

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
alone. Afterwards it prints the iteration count, tool-call count, context size, why the
loop stopped, the calls made per tool, and the path to a full JSONL trace of every tool call under `logs/traces/`.

**When the corpus lacks the subject**, the agent does not fetch on its own. It says what
it searched for and which held topics came closest, and the run ends with a line like
`--- fetch proposal: "<query>"  (re-ask with --approve-fetch)`. To approve, continue the
same thread (`docs/DECISIONS.md` D24–D27):

```bash
python main.py ask --thread <id> --approve-fetch "<query>"   # fetch, then answer the previous question
```

### Run the eval

```bash
python main.py eval               # the 10-question tuning split
python main.py eval --held-out    # the 5 held-out questions
python main.py eval --limit 3     # a quick smoke run (saved separately, so it
                                  # cannot overwrite a full run's results)
python main.py eval --out NAME    # write eval/results/NAME.json instead
```

Results land in `eval/results/`. Full write-up in
[`docs/EVALUATION.md`](docs/EVALUATION.md).

### Show it in a browser

A three-pane chat UI over the same agent, for showing the system to someone rather than
driving it from a terminal. You set it up once:

```bash
source .venv/bin/activate
pip install -r web/requirements.txt          # fastapi, uvicorn, httpx — web/ only
cd frontend && npm install && npm run build  # Node 20.19+ or 22.12+ (Vite 8); writes frontend/dist/
cd ..
```

Then run:

```bash
python web/server.py
```

It loads the models first, which takes about ten seconds, then prints:

```
loading models and the index ...
ready — 11 paper(s) indexed
open http://127.0.0.1:8000
```

Open that address. Stop it with `Ctrl-C`. If `frontend/dist/` is missing, the API still
runs, and the page tells you to run the build command.

```
--port N        listen on a different port (default: 8000)
--host H        default 127.0.0.1, this machine only.
                --host 0.0.0.0 makes it reachable from another machine on the network
--config PATH   is NOT a flag here; use SCIAGENT_CONFIG=... python web/server.py instead
```

**Frontend development.** Keep `python web/server.py` running, then
`cd frontend && npm run dev` and open the address Vite prints (default
<http://127.0.0.1:5173>). Vite proxies `/api` to `127.0.0.1:8000`. Set
`SCIAGENT_BACKEND=http://127.0.0.1:PORT` if the server listens on another port.
`npm run typecheck` runs `tsc` alone; `npm run check:markdown` runs the answer
renderer's checks (math, citations, escaping). Styling is Tailwind v4, with the palette,
type scale and radii as tokens in `frontend/src/styles.css`; answers render math with KaTeX.

**Left: chats and corpus.** *New chat*; *Chats*, the stored threads, newest first, each
one deletable; *Samples*; and *Corpus*: every indexed paper grouped by the topic it was
collected under, with the arXiv query the LLM planned for that topic. Per paper it lists
title, authors, year, PDF filename, chunk and figure counts, and an arXiv link. A chat is
created by its first question and named after it. Opening one redraws its answers,
citations and per-answer timeline from the server. The three **Sample** chats are
hand-written fixtures, not stored threads, so asking from one starts a new chat. Their
citations point at real passages, but the agent did not write them.

**Centre: the conversation.** Each question goes to the same agent as `python main.py
ask --thread`, through the same config. There is no second code path, and no answer can
appear here that the CLI would not also give. **The agent sees the earlier questions and
answers in this chat**, and a note of the tools each one used, but not the passages. It
retrieves those again, and the page says so under the composer.
While the agent works, a status line shows what it is doing: thinking, which tool it is
calling, or writing the answer. The answer arrives as formatted text. Every citation is
numbered like a paper's references, and clicking it opens the cited passage with an arXiv
link. A reference list sits under the answer. Figures the agent looked at with
`inspect_figure` appear inline.

**Right: the agent's working.** *Tool calls* is a turn-by-turn timeline of the selected
answer, live while it runs. For each call it shows the tool, its arguments, what came back
(expandable to the summarised result) and how long it took, plus skipped calls and stops.
It ends with the run's stats: turns, tool calls, context tokens, why it stopped, and the
trace's run id. *Sources* lists the cited passages per paper, as retrieval returned them.
It also shows how many earlier turns were sent with the question. *Memory* has two parts. *Across chats* lists the long-term memories used in the selected
answer, each with its chunk citations and whether it was cited, and those written from it
(new, seen again, replaced). Below them are every active memory, with uses, when last
used, strength and a delete button, and a collapsed *Forgotten* list giving why each one
went, plus a *Manage memories* link. *In this chat* is the short-term part: the tool-call
notes a later question in this chat is shown. On narrow screens the panel becomes a drawer.

**Memories (`#/memories`).** Open it from the user menu at the bottom left ("You"; there is
one user and no sign-in), which also holds the theme. It lists every long-term memory
(`docs/DECISIONS.md` D23), with a filter over the text:

- *Preferences*: add one ("Only consider papers from 2024 onward"), or edit or delete one.
  A restatement of an existing preference refreshes it instead of adding a copy.
- *Findings*: each with its passages as chips (they open the citation card), its uses,
  when it was last used, a strength meter with how many days it has left unused, and the
  chat it came from. Edit or delete one. An edit is re-checked by NLI against the
  finding's passages. If they don't support the new wording, it is kept but flagged
  "Edited by you; not supported by its passages", and the agent is told so. Findings
  can't be added by hand: each must rest on passages an answer cited.
- *Forgotten and replaced* (collapsed): why each one went (faded, over the cap, replaced
  by a newer finding, its passage changed, deleted by you), when it is deleted for good,
  and a *Restore* button. Restore counts as a use. A finding whose passage has changed
  can't be restored, and a row past `purge_after_days` is gone.

It needs Python, the models and an index, so it cannot be hosted as a static site. Run it
on the machine that has the corpus. It holds one agent and answers one question at a time.
A second question asked meanwhile is told straight away that the agent is busy. Closing
the tab does not cancel a run. The question finishes, and its trace is written, before the
agent takes another.

**The Fetch papers button.** When the corpus has no papers on what you asked, the answer
says so, and a card under it names the gap and the closest topics the corpus holds. The
card has an editable query and a **Fetch papers** button. Clicking it is your approval:
nothing is ever fetched without it. The fetch runs as the chat's next turn. The agent calls
`search_literature`, an admission gate drops arXiv results that do not match the query (the
step says how many), and it then answers your original question with citations to the new
papers. Once used, the button is gone. Only the latest answer's card can fetch.

That `search_literature` step can sit in the timeline for minutes: fetching PDFs,
extracting them, describing figures and embedding all happen inside that one tool call.
Fetched papers stay in the corpus. For a demo, index the topic beforehand.

`web/` and `frontend/` sit outside `src/`. `web/` imports `src/`, exactly as `eval/` does.
Nothing in `src/` knows either exists: the system being evaluated is still the CLI.

API, for reference: `GET /api/papers` (the corpus); `POST /api/ask` with
`{"question": ..., "thread_id"?: ...}`, or `{"thread_id": ..., "approve_fetch": {"query": ...}}`
to run an approved fetch (SSE; the `answer` frame carries `fetch_proposal`: `thread` first, naming the chat, which is new if
no `thread_id` was sent; then `step`…, then `answer` or `failed`, then `memory`
`{written, refreshed, invalidated, used, forgotten, dropped}`, then `done`; or else
`busy`, `done`); `GET`/`POST /api/threads`; `GET`/`DELETE /api/threads/<id>`;
`GET /api/memories` (`?all=1` adds the forgotten, with reasons; each row carries strength,
days left, passages and whether its chat still exists, plus the forgetting `settings`);
`POST /api/memories` `{"kind": "preference", "text": ...}`; `PATCH /api/memories/<id>`
`{"text": ...}`; `DELETE /api/memories/<id>`; `POST /api/memories/<id>/restore`;
`GET /api/figures/<paper_id>/<file>`; and `GET /api/status`.

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

The retrieval upgrade (2026-10-25) added BM25 with RRF, which raised retrieval-only MRR
from 0.344 to 0.479 on the tuning split with abstention unchanged. Contextual headers and
multi-query were built and measured, and did not help offline. One agent run showed no
clear change. See [`docs/EVALUATION.md`](docs/EVALUATION.md#retrieval-upgrade-2026-10-25).

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
main.py                CLI: index / ask (--thread) / threads / eval
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
  memory/              threads in SQLite, history for a thread's next question, long-term memory
  analysis/            NLI
  prompts/             system prompt, thread-history prompts and tool descriptions, as files
eval/                  question set, annotation, metrics, tuning, ablation
web/                   the browser UI's server — FastAPI, streams agent steps as SSE
frontend/              the browser UI — Vite + React + TypeScript, Tailwind v4, KaTeX; built into frontend/dist/
scripts/               one verification gate per milestone, m0..m8, plus the M9 demos
tests/                 unit and smoke tests
docs/                  the specs — these are authority, not background reading
```

`eval/` imports `src/`, never the reverse. That is why `main.py` sits at the repo root:
`eval` is one of its subcommands. `web/` and `frontend/` sit outside `src/` for the same
reason and follow the same rule — see
[`docs/FUTURE_WORK.md`](docs/FUTURE_WORK.md#web-is-a-demo-shim-outside-the-system-under-evaluation).

## Configuration

Everything tunable is in `config.yaml`, and each value carries the measurement that
produced it. The ones most worth knowing:

| key | value | why |
|---|---|---|
| `chunking.max_tokens` | 445 | min of the two encoder budgets; the cross-encoder binds |
| `retrieval.k_retrieve` / `k_final` | 40 / 5 | wide cheap pool, narrow expensive context |
| `retrieval.relevance_threshold` | −3.0 | swept −8..+2 on the tuning split only |
| `retrieval.bi_encoder_relevance_threshold` | 0.55 | the gate when the cross-encoder did not run (on the dense cosine, never the RRF score); a degraded fallback, not an alternative |
| `retrieval.hybrid` / `rrf_k` | true / 60 | BM25 beside the bi-encoder, fused by RRF; pool recall@40 0.400 → 0.567 on the tuning split |
| `retrieval.contextual_headers` | false | `title — section` headers for the first stage; measured to lower pool recall here, so off. Toggling needs `main.py reindex` |
| `retrieval.max_subqueries` / `multi_query_rerank` | 3 / `retrieving` | `retrieve_evidence(queries=[...])`; each candidate is scored only against the sub-queries that found it, since scoring all pairs cost 3.1× one query |
| `agent.max_calls_per_tool` | 10 | per tool, per question; an exhausted tool is no longer offered |
| `agent.tool_call_limits` | `search_literature: 5` | it downloads papers: slow, and trips arXiv's per-IP 429 |
| `agent.max_no_progress_turns` | 3 | consecutive turns with every call refused → forced final answer |
| `agent.max_context_tokens` | 60000 | past this the oldest tool results are elided |
| `agent.thread_history_tokens` | 9000 | budget for a thread's earlier turns; the oldest go first, whole, so history cannot crowd out new evidence |
| `paths.threads_db` | `data/threads.sqlite3` | conversation threads and long-term memories, shared by the CLI and the browser UI |
| `memory.inject_k` / `inject_min_cosine` / `inject_tokens` | 4 / 0.60 / 2500 | findings recalled per thread run, with their passages; cosine measured on 11 questions |
| `memory.duplicate_cosine` / `related_cosine` | 0.85 / 0.80 | refresh vs NLI conflict check; contradiction wins, because swaps sit closer than paraphrases |
| `memory.half_life_days` / `min_strength` / `max_active` | 30 / 0.125 / 500 | decay: used once and unused for 90 days is forgotten. Confirmed by the user |
| `memory.purge_after_days` | 180 | forgotten and replaced memories are deleted for good after this, and can't be restored. Confirmed by the user |
| `nli.min_pair_similarity` | 0.75 | below it, NLI scores unrelated sentences as confident contradictions |

Models are set per role (`agent_model`, `vision_model`, `utility_model`). The agent
model must accept **both tools and images** — `inspect_figure` attaches a figure to the
agent's own conversation on a follow-up turn, so a text-only model breaks that path.

## Tests

```bash
python -m pytest tests/ -q          # 474 tests
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
| [`docs/research/`](docs/research/) | the 2026 agentic-engineering landscape (memory, frameworks, agentic RAG), mapped to what was built, and a one-page framework comparison |
