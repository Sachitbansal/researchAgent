# Known limitations and future work

Things that are incomplete, over-provisioned, or unmeasured. Recorded deliberately:
each one is a decision that was made and could be revisited, not an oversight.

## Query truncation has no telemetry, and the budget is over-provisioned

`retrieval.max_query_tokens: 64` reserves 64 of the cross-encoder's 512-token window for
the query, which is what forces `chunking.max_tokens` down to 445. Measured across all
63 `retrieve_evidence` calls in `logs/traces/`:

| | tokens |
|---|---|
| longest query ever generated | 29 |
| p95 | 22 |
| mean | 13.0 |
| queries that hit the 64-token limit | **0 (0.0%)** |

Two separate problems:

- **Over-provisioned.** Reserving 64 when p95 is 22 gives up ~35 tokens of chunk budget
  for headroom nothing has used. Chunks could be ~480 tokens instead of 445.
- **Silent when it fires.** If a query ever does exceed 64 it is cut in
  `reranker.py:_pair()` with nothing logged — no trace field, no warning. Retrieval would
  quietly degrade on exactly the longest, most specific queries, which are the ones most
  likely to be reformulations after a failed search.

The telemetry is the more valuable fix and the cheaper one: record the pre-truncation
token count on the trace record, then the budget can be set from data rather than from
caution. Raising the limit without telemetry just moves an unmeasured number.

## The rerank ablation does not measure abstention

`eval/ablation.py` reports recall, MRR, latency and gold-paper hit rate. None of those is
what reranking actually earns its cost on here — the relevance gate thresholds the
cross-encoder's score, so rerank is what makes abstention work. Acting on the ablation
alone produced a wrong conclusion once already (see `docs/EVALUATION.md`).

## `k_retrieve` has never been swept

Fixed at 40. Rerank latency scales with the shortlist, so a smaller pool might keep the
gate at a fraction of the 1527 ms. No measurement has been taken.

## The "absent but fetchable" category is measured on the tuning split only

The tuning split's fetchable questions have been run once with the phase-2 prompt
(`docs/EVALUATION.md`, CRAG phase 2). The held-out one has not. The live fetch path,
including the approval turn, is still checked only by hand, on a throwaway copy of the
data. Record/replay of arXiv responses would make it an automated test, and is benched.

## `rephrase_margin` and `admission_threshold` rest on very little data

Both are measured, and both are thin. Distance below the gate does not separate "bad
wording" from "missing literature": the one answerable miss sits further down than four
absent topics. So the margin is held under the nearest absent miss (0.27 logits) rather
than fitted. The admission gate separates topic-shaped queries perfectly on 11 + 11
pairs, but question-shaped queries lose 38% of their positives at the chosen −3.0. The
proposed query is now condensed to a keyword phrase from the user's question (D28), but a
user who edits the card into a full question still meets that loss. Both values are
confirmed by the user, and neither has been re-measured on more data.

## Rephrase vs. fetch is still a hint the model follows

Both answer the same signal: the corpus cannot answer this. `coverage_gap.suggested_next`
says which, and `src/prompts/system.md` tells the agent to follow it. Nothing enforces the
"one reworded retrieve" or the ban on searching unasked; only the eval checks them. The
user approves every fetch (`docs/DECISIONS.md` D14–D18, D24–D27), so whether the agent
*would* choose to search well on its own is no longer tested.

## Long-term memory is unevaluated, and its NLI filter is strict

Long-term memory exists (`docs/ARCHITECTURE.md` §7.2, `docs/DECISIONS.md` D19, D20), but the
eval never runs in a thread, so nothing measures whether it helps. That needs a multi-chat
eval: a second question that should build on a first.

- **Small samples.** Its thresholds rest on 32 paraphrase/swap pairs, 496 distinct pairs
  and 11 questions (`scripts/tune_memory.py`). The decay numbers are not measured at all.
- **Low write yield.** Only about a third of extracted claims pass the entailment filter (32 of 93).
  The premise is one sentence, so a claim spanning two sentences cannot be entailed.
  Two-sentence windows are the obvious next step.
- **Misses on the same subject.** In this narrow corpus, a question's own claims and
  other answers' claims overlap in cosine, so `inject_min_cosine` trades misses for
  stray injections.
- **Turns are dropped, not summarised.** Turns that fall out of the short-term budget are
  still dropped, not summarised.

## Flat index, no eviction

`IndexFlatIP` is exhaustive and exact: 19 µs at 406 vectors, ~48 ms extrapolated to 1M.
Past roughly 100k vectors this wants IVF or HNSW, at the cost of a recall variable that
cannot be measured meaningfully on 11 papers. Removal is a non-goal for the same
underlying reason — `faiss_id_map` is positional, so eviction implies a full rebuild.

## No general Python execution tool

`analyze_corpus` performs bounded data analysis (clustering, metadata statistics) rather
than exposing arbitrary code execution. A deliberate choice, not an omission.

## `web/` is a demo shim, outside the system under evaluation

`.claude/CLAUDE.md` says "No UI. CLI only", and that constraint exists because how the
agentic system is structured is itself being graded — a UI layer is one more place for
behaviour to hide. `web/` was added anyway, deliberately, to show the system to someone
who is not going to read a terminal.

What keeps the constraint's intent intact:

- It lives outside `src/`: `web/` (the server) imports it, the same direction as
  `eval/`, and `frontend/` (the page) only talks to `web/` over HTTP. Nothing in `src/`
  references either, so removing both directories changes nothing about the agent.
- Its dependencies are its own. FastAPI, uvicorn and httpx (for tests) are pinned in
  `web/requirements.txt`, and React, Vite and TypeScript in `frontend/package.json`. The
  root `requirements.txt`, which pins the system under evaluation, is unchanged. The
  answer renderer is still hand-written rather than a markdown library
  (`docs/DECISIONS.md`, D1, D2, D7).
- It runs the same `AgentLoop` with the same registry and config as `python main.py
  ask`. There is no second code path to keep in sync, and no answer can appear in the
  browser that the CLI would not also give. A browser chat is a thread. The server passes
  the question and its `thread_id` to `loop.run`, and the loop adds the history itself,
  exactly as `main.py ask --thread` does.

What it is not: it holds one shared agent behind a lock, so it serves one question at a
time (a second asker is told it is busy) and is not a server for more than one person.
Chats are stored as threads in `data/threads.sqlite3`, shared with the CLI. They have no
owner, because there is one user. It has no authentication and binds to localhost by
default. Neither the single-user limit nor the
missing authentication is worth fixing unless it stops being a demo.
