# Framework comparison: why this agent's loop is hand-rolled

One page, for explaining the choice. Facts about frameworks are as of **2026-10-24**; the
sources and versions are in [`agentic-landscape-2026.md`](agentic-landscape-2026.md) §1.3
and §2. Where this page says what a framework would *hide* here, that is about where *our*
behaviour would have to live, not a claim about the framework's internals. Specifics I
could not confirm are marked **[unverified]**.

## The options

| | Core idea | What it would have given this project | What it would have cost or hidden here | OpenRouter |
|---|---|---|---|---|
| **Hand-rolled loop** (ours) | A plain loop against `/chat/completions`. It sends the messages and five tool schemas, runs the tool calls, appends the results and repeats. `src/agent/loop.py` (553 lines) plus `conversation.py` (249) | Everything is visible and testable. Every policy below is code we wrote and can point to | Nothing hidden. The cost is that we built threads (D12), memory (D19–D23) and approval (D26) ourselves. There are no checkpoints or time travel, and tracing is JSONL only | Native. The provider is a config value (ARCHITECTURE §11) |
| **LangGraph** | A `StateGraph` of nodes with typed state and reducers, plus a checkpointer | Durable per-step checkpoints. `interrupt()` / `Command(resume=)` approval that survives restarts. Time travel. Thread memory and a cross-thread Store | The loop becomes a graph, and message plumbing, retries and tool-call parsing sit behind Runnables. Each of our policies would become custom graph code next to the prebuilt agent **[unverified how cleanly]**: the image on a follow-up user turn (§9), elision that keeps messages and releases chunks (D13), per-tool limits with two backstops, and the forced final answer (D4). LangSmith self-hosting is Enterprise-only | Yes, `ChatOpenAI(base_url=...)` |
| **Pydantic AI** | A typed `Agent[Deps, Output]`. Tools are typed functions with validated arguments | Validated tool arguments, OTel-native tracing, and durable execution via Temporal, DBOS and others. Tool approval for human-in-the-loop | It hides schema generation and retry-on-validation. Of the five, it hides the least, but the same four policies would need porting and re-testing **[unverified where they would hook in]** | Yes, provider-agnostic. Exact OpenRouter class **[unverified]** |
| **OpenAI Agents SDK** | `Agent` plus a `Runner` loop, with handoffs, guardrails and sessions | SQLite and other session stores. `needs_approval` with a serialisable `RunState` for pause and resume | Its best extras (compaction, hosted tools, server conversations) are OpenAI Responses-API-only. Tracing is on by default and sends to OpenAI. Handoffs are multi-agent, which we rule out. Our limits, backstops and forced answer would live in `Runner` configuration or hooks **[unverified]** | Through LiteLLM / Any-LLM adapters |
| **LlamaIndex Workflows** | Event-driven `@step` functions over a serialisable `Context`, with prebuilt `FunctionAgent` / `ReActAgent` | Transparent steps, a serialisable `Context` and Phoenix instrumentation | The pull is its RAG layer, and that would replace measured code: bi-encoder top-40 then rerank (§7), and the **per-scorer gate thresholds** (a cross-encoder logit of −3.0, or a cosine of 0.55 on the fallback path). Those live inside `retrieve_evidence`, so a pure loop framework would not touch them; LlamaIndex's retrievers would. It would also replace `coverage_gap` (D15) | Yes (OpenAILike / OpenRouter integrations) |

## Why hand-rolled here

- **The loop is the thing being evaluated.** `.claude/CLAUDE.md` rules out adopting a
  framework without a logged decision with the user, and ARCHITECTURE §14 lists it as a
  non-goal. A framework would hide the very structure under review.
- **The provider forces a non-standard turn shape.** On OpenRouter's OpenAI-compatible API,
  a `tool` message's content is a string, and image parts are only valid on a `user`
  message. So `inspect_figure` returns a path, and the loop attaches the image on a
  follow-up user turn (ARCHITECTURE §9). A framework's standard tool-result path would
  have to be bent around this.
- **The loop policies are the product, and they are small.** The policies are:
  - per-tool call limits instead of an iteration cap
  - two backstops: every tool exhausted, or 3 turns with nothing run
  - the forced final answer (D4, TOOLS.md "Loop control")
  - elision that keeps the messages, so no `tool_call` is orphaned, and releases elided
    chunks (D13)

  Each is a few lines here, and each would be an override somewhere else.
- **The features frameworks sell were cheap to build.** Each was built from parts the
  project already had:
  - **Threads:** stdlib SQLite (D12).
  - **Long-term memory:** the existing embedder and NLI model, with no new service (D19).
  - **Human approval:** a new thread turn (D26), not a suspended run, so no checkpoint
    store is needed.
- **Abstention depends on details a default stack lacks.** The gate thresholds the score
  of whichever scorer ran, and reranking is what makes abstention work (EVALUATION.md,
  rerank ablation). A generic RAG layer would bring its own defaults.

## When I'd switch, and to what

| Trigger | Why the current design stops fitting | Pick |
|---|---|---|
| **Approval pauses that must survive restarts** (approve mid-run, several approvals per run) | Today, approval works only because it is a new turn. A paused run would need its state saved and resumed | **LangGraph**: a checkpointer plus `interrupt()` / `Command(resume=)` is the most direct fit |
| **Multi-hour runs** (deep research mode) | A crash or deploy would lose the run. Nothing checkpoints | **Pydantic AI** with durable execution (Temporal or DBOS). It hides the least and is OTel-native. LangGraph if time travel matters more |
| **Many concurrent users** | One run at a time (D6), single-user SQLite (D12), no auth (D23), and arXiv's 429 limit is per IP | The fix is serving first: workers, a shared database and a shared arXiv rate limit. Add **LangGraph** with a Postgres checkpointer only if run state must move between workers |

Adoption stays benched (`docs/FUTURE_WORK.md`). If any trigger fires, log the
switch as a decision, along with what it loses: the image turn, elision, the per-scorer
gate and the forced answer.

## Memory products vs our hand-rolled memory

| Product | What it is | What we borrowed | Why we didn't adopt it (D19) |
|---|---|---|---|
| **Mem0** | Extract facts, then an LLM picks ADD / UPDATE / DELETE / NOOP against similar memories | The add / refresh / supersede decision. Ours makes it with cosine plus NLI and **no LLM call** (D22) | An LLM call per decision, plus a vector-DB service |
| **Zep / Graphiti** | A temporal knowledge graph in which a contradiction *invalidates* the old fact rather than deleting it | Invalidate, don't delete: a superseded memory is soft-deleted with a reason and `superseded_by`, and can be restored (D20, D23). Ours is hard-purged after 180 days, and is not bi-temporal | A graph database and a server, for one user |
| **Letta** | A stateful agent server. The agent edits its own memory blocks | Nothing directly. The nearest parallel is that our preferences are always injected (D22), for a reason of our own: a cosine gate would rarely surface them | The loop moves inside Letta, and the loop is what is being evaluated |
| **LangMem** | Memory tools on the LangGraph Store | Nothing | A LangChain/LangGraph dependency, which CLAUDE.md rules out without a logged decision. Also no release for a year |
| **Ours** | SQLite in `data/threads.sqlite3`. Stores grounded findings with chunk hashes, and stated preferences. Injected by the loop with chunk text, not through a sixth tool (ARCHITECTURE §7.2) | — | — |

## Likely interview questions

**Why didn't you use LangChain?**
How the agentic loop is structured was itself under evaluation, so a framework would hide
the answer. The provider also needs images on a follow-up user turn, and the loop
policies (per-tool limits, backstops, forced answer, elision) are small code that a
framework would make into overrides. I'd switch to LangGraph if approvals had to survive
restarts.

**How does your agent avoid hallucinating?**
- **Abstention is a mechanism.** Chunks below a tuned cross-encoder threshold are
  withheld, and the tool returns `sufficient_evidence: false`.
- **Every claim carries a `[chunk_id]`**, and the eval checks that each one resolves.
- **NLI groundedness is advisory, not a gate** (§8).
- **When the corpus lacks the subject**, the agent proposes a fetch rather than answering
  from its own knowledge (D25).

Abstention accuracy was 1.000 on both splits in every recorded run. Citation resolution
skips uncited answers, so an uncited answer never shows in it; since D34 such an answer
gets a Fetch card instead, which is where the final run's one false proposal came from.

**How does memory forget?**
Automatically, in four ways (D20):
- it is superseded by a confident contradiction
- it goes stale when a cited chunk's hash changes
- it decays: `strength = (1 + ln uses) × 0.5^(days/30)`, forgotten below 0.125, weakest
  first past 500
- re-finding a memory refreshes it

A "use" means injected *and* cited. Forgetting is a soft delete, so the user can restore a
memory until the hard purge at 180 days. Preferences don't decay.

**What would you change at 100× scale?**
- **Corpus.** Flat FAISS is fine at 40k chunks: an estimated 48 ms even at 1M. Past about
  100k vectors I'd move to IVF or HNSW (FUTURE_WORK). `IndexIDMap` would make eviction
  safe (§6).
- **Users.** Replace the one-run lock (D6) and single-user SQLite (D12) with workers and a
  shared database. Rate-limit arXiv centrally, since its 429 is per IP.
- **Cost.** Prompt caching, and a context budget under qwen's 32k price tier (benched).

**How do you know it works?**
- **The question set.** 18 paraphrased questions in 5 categories, hand-labelled with gold
  chunks, with 6 held out until tuning stopped.
- **Results.** The final run (2026-10-25) is in EVALUATION.md, "Final evaluation". Every
  earlier run is kept there too, marked as such.
- **Retrieval.** Hybrid BM25 + dense raised retrieval-only MRR from 0.344 to 0.479 on the
  tuning split; contextual headers made it worse and ship off (D29–D33).
- **Gaps.** Long-term memory has no eval yet. Each number is a single run of a hosted
  model, at n = 12 and 6.

**Why must the user approve fetches?**
Auto-fetch pollutes the shared corpus, and arXiv returns *something* even for topics it
barely covers (D14). So the agent proposes, and a "Fetch papers" click comes back as a
structured thread turn (D26). Even then, an admission gate scores each abstract before
anything is downloaded (D16).
