# Future work and known limitations

What to build next, in order, then what was tried and did not pay off, then the known
limitations. Each limitation is a decision that was made and could be revisited, not an
oversight. The decision log (`docs/DECISIONS.md`) points here for everything benched.

## Good to have later, ranked

1. **Self-check and revise, plus a PDF hidden-text filter.** NLI groundedness is already
   computed but only advisory. One revision turn when a claim comes back `neutral` or
   `contradiction` would use it. PDF text is untrusted input: arXiv manuscripts have been
   found hiding instructions to reviewers (`docs/research/agentic-landscape-2026.md` §5.3).
2. **Prompt caching and a smaller context.** `agent.max_context_tokens` is 60k, which puts
   late turns in the agent model's >32k price tier, and every request re-sends the same
   system prompt and tool schemas. Cache reads cost about a tenth.
3. **An MCP server.** A thin `mcp_server/` importing `src/` would expose the five tools to
   other clients. `search_literature` writes to the shared corpus, so it needs the same
   approval there.
4. **A tracing UI.** Traces are JSONL only. OTel spans into a local viewer (Phoenix) would
   make long runs easier to read than the browser's timeline.
5. **A long-context baseline.** The eval corpus is about 160–180k tokens and fits the agent
   model's window. An `eval/`-only arm that puts it all in the prompt shows the ceiling
   retrieval is chasing.
6. **A citation graph.** Snowballing from a seed paper (Semantic Scholar), as a mode of
   `search_literature`, would find later work that disputes a claim.
7. **History summarisation.** Thread turns past the 9k budget are dropped whole. A summary
   would keep early constraints, but it must keep the cited chunk ids.
8. **A reranker swap.** `ms-marco-MiniLM-L-6-v2` is trained on web queries. A scientific
   reranker might rank better, but the gate threshold and the admission threshold would
   both have to be re-swept.
9. **arXiv record/replay.** The live fetch path, including the approval turn, is checked
   only by hand on a throwaway copy of the data (`scripts/crag_fetch_check.py`). Recorded
   responses would make it an automated test.
10. **Small fixes.** Query-truncation telemetry (below); a floor check on
    `agent.tool_call_limits` and a warning for unknown tool names in it; chunk ids in the
    elision placeholder; one reserved slot per sufficient sub-query in multi-query.

## Tried, measured, and not worth it here

- **Contextual headers ship off.** Embedding `title — section` with each chunk lowered
  dense pool recall@40 from 0.400 to 0.317 on raw questions, and from 0.621 to 0.537 on the
  agent's queries. With hybrid on it was a wash. Likely causes: every chunk of a paper
  looks alike, and many detected sections are noise (`I(X; Y ) =`). Title-only headers
  and a cleaner section detector are untried (D30).
- **Multi-query did not help offline, and its agent-level effect is unmeasured.**
  Splitting questions deterministically lowered retrieval-only MRR from 0.479 to 0.410,
  because the sub-queries share one 40-slot pool. The agent ignored `queries` entirely
  until a "How to search" rule went into the system prompt. Then it used them in 7 of 46
  `retrieve_evidence` calls (`tuning_split_prompts_v3.json`), and in 17 of 51 in the final
  run. No run isolates what that use is worth. It is the only fix for the SparseMixer case,
  where broad wording hides a specific chunk (D32).
- **Rerank does not measurably improve ordering at this size.** +0.027 MRR for +1.5 s,
  below what one question can move. It stays, because it is what makes the gate work
  (`docs/EVALUATION.md`).
- **The bi-encoder fallback gate is weak.** On the 12-question tuning split, a cosine of
  0.55 abstains on only 1 of 4 absent questions. No cosine threshold does better. It is a
  degraded path for when the cross-encoder fails, not an alternative (D33).

## Known limitations

- **Figure images are not shipped with the repo.** They live in `data/cache/figures/`
  (gitignored), and `chunks.jsonl` stores their paths as absolute paths. On a fresh clone
  the text index works fully, but `inspect_figure` returns `image_missing` until the corpus
  is re-indexed on that machine. The fix is to store image paths relative to
  `paths.figures` and track the eval corpus's figures (about 2.6 MB).

### Query truncation has no telemetry, and the budget is over-provisioned

`retrieval.max_query_tokens: 64` reserves 64 of the cross-encoder's 512-token window for
the query, which is what forces `chunking.max_tokens` down to 445. Across 63
`retrieve_evidence` calls in `logs/traces/` (measured at M9): longest query 29 tokens, p95
22, mean 13.0, and **0** hit the limit. So the reserve gives up ~35 tokens of chunk budget
for headroom nothing used. And if a query ever exceeds 64, it is cut in
`reranker.py:_pair()` with nothing logged. Record the pre-truncation token count on the
trace first, then set the budget from data.

### The rerank on/off ablation does not measure abstention

`eval/ablation.py` reports recall, MRR, latency and gold-paper hit rate, none of which is
what reranking earns its cost on: the relevance gate thresholds the cross-encoder's score.
Acting on that ablation alone produced a wrong conclusion once (`docs/EVALUATION.md`). The
retrieval ablation v2 (`eval/retrieval_ablation.py`) does report abstention for every arm,
but keeps rerank on throughout.

### What the retrieval upgrade left open (D29–D33)

- **Multi-query fuses by max score into one shared pool.** One sub-query's strong chunks
  can fill `k`. The note names a sub-query that cleared the gate but was crowded out. Its
  rerank costs 2.3× one query.
- **Hybrid widens what passes the gate.** In the retrieval-v2 agent run, BM25 surfaced
  off-target chunks on sp02 that cleared −3.0. On the fallback path, RRF order is worse
  than dense order (MRR 0.213 vs 0.317); the spec asked for RRF order there.
- **`rrf_k` (60) and BM25's `k1`/`b` are textbook defaults**, not tuned.
- **`k_retrieve` has never been swept.** Fixed at 40. Rerank latency scales with the
  pool, so a smaller one might keep the gate at a fraction of the cost.

### Thresholds that rest on very little data

`rephrase_margin` and `admission_threshold` are measured, and both are thin. Distance below
the gate does not separate "bad wording" from "missing literature": the one answerable miss
sits further down than four absent topics, so the margin is held under the nearest absent
miss rather than fitted. The admission gate separates topic-shaped queries perfectly on
11 + 11 pairs, but question-shaped queries lose 38% of their positives at −3.0. The card's
query is a keyword phrase rewritten from the question (D28), but a user who edits it into a
full question meets that loss.

### The fetch rules are prompt rules

`coverage_gap.suggested_next` says whether to reword or describe the gap, and the system
prompt tells the agent to follow it. Nothing enforces "one reworded retrieve" or the ban on
searching unasked; only the eval checks them. The card itself now follows the answer: it
appears exactly when the answer cites nothing (D34). So a correct answer written without
citations also gets a card. Since the user approves every fetch, whether the agent would
search well on its own is not tested.

### Long-term memory is unevaluated, and its NLI filter is strict

The eval never runs in a thread, so nothing measures whether memory helps. That needs a
multi-chat eval: a second question that should build on a first.

- **Small samples.** Its thresholds rest on 32 paraphrase/swap pairs, 496 distinct pairs
  and 11 questions (`scripts/tune_memory.py`). The decay numbers are not measured at all.
- **Low write yield.** About a third of extracted claims pass the entailment filter (32 of
  93). The premise is one sentence, so a claim spanning two cannot be entailed.
  Two-sentence windows are the obvious next step.
- **Misses on the same subject.** In this narrow corpus, a question's own claims and other
  answers' claims overlap in cosine, so `inject_min_cosine` trades misses for stray
  injections.

### Flat index, no eviction

`IndexFlatIP` is exhaustive and exact: 19 µs at 406 vectors, ~48 ms extrapolated to 1M.
Past roughly 100k vectors this wants IVF or HNSW, at the cost of a recall variable that
cannot be measured meaningfully on 11 papers. Removal is a non-goal for the same reason:
`faiss_id_map` is positional, so eviction implies a full rebuild
(`docs/ARCHITECTURE.md` §6).

### No general Python execution tool

`analyze_corpus` performs bounded data analysis (clustering, metadata statistics) rather
than exposing arbitrary code execution. A deliberate choice, not an omission.

### One user, one run at a time

The browser UI holds one agent behind a lock, so it answers one question at a time (D6),
and a second asker is told it is busy. Chats live in `data/threads.sqlite3` with no owner,
there is no authentication, and it binds to localhost by default. Serving many users would
need workers, a shared database and a shared arXiv rate limit
(`docs/research/framework-comparison.md`).

## `web/` and `frontend/` sit outside the system under evaluation

The CLI is the system. How the agentic system is structured was itself under evaluation,
and a UI layer is one more place for behaviour to hide. So the browser UI is kept apart:

- It lives outside `src/`. `web/` (the FastAPI server) imports `src/`, the same direction
  as `eval/`, and `frontend/` (the React page) only talks to `web/` over HTTP. Nothing in
  `src/` references either, so removing both changes nothing about the agent
  (`.claude/CLAUDE.md`, "No UI inside `src/`").
- Its dependencies are its own. FastAPI, uvicorn and httpx are pinned in
  `web/requirements.txt`; React, Vite, TypeScript, Tailwind and KaTeX in
  `frontend/package.json`. The root `requirements.txt`, which pins the system under
  evaluation, is unchanged. The answer renderer is hand-written rather than a markdown
  library (D1, D2, D7, D8).
- It runs the same `AgentLoop` with the same registry and config as `python main.py ask`.
  A browser chat is a thread: the server passes the question and its `thread_id` to
  `loop.run`, and the loop adds the history itself, exactly as `main.py ask --thread`
  does. No answer can appear in the browser that the CLI would not also give.
