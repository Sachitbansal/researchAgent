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

## The "absent but fetchable" category exists but has not been run

`ni02` and `ni03` are now `absent_fetchable`, alongside three new questions, and are
scored by `fetch_proposal_detected` (`docs/EVALUATION.md`; `docs/DECISIONS.md` D23). No
agent run has scored them yet: the prompt rule that acts on `coverage_gap` is phase 2,
and numbers taken before it would be stale on arrival. The live fetch path is also only
checked by hand (`scripts/crag_fetch_check.py`). Record/replay of arXiv responses would
make it an automated test, and is benched.

## `rephrase_margin` and `admission_threshold` rest on very little data

Both are measured, and both are thin. Distance below the gate does not separate "bad
wording" from "missing literature": the one answerable miss sits further down than four
absent topics. So the margin is held under the nearest absent miss (0.27 logits) rather
than fitted. The admission gate separates topic-shaped queries perfectly on 11 + 11
pairs, but question-shaped queries lose 38% of their positives at the chosen −3.0. If
phase 2 hands `search_literature` long questions, either the proposed query should be
condensed to a topic phrase first or the threshold re-swept on that shape.

A filtered retrieval can also yield a false proposal. `coverage_gap` is emitted on any
gate failure, including calls with `topic_filter` or `chunk_types`, so a run whose last
call was a figure-only search that found no figure proposes a fetch even when the
unfiltered corpus covers the subject. Two fixes for phase 2: carry the filter in the gap,
or have the loop ignore filtered gaps.

## Abstention vs. literature search is resolved by prompt, not by rule

Both answer the same signal: the corpus cannot answer this. Which one fires is left to
the model via `src/prompts/system.md`. In every eval run the agent chose to abstain and
*offer* to search rather than search, so `search_literature` is implemented and tested
but has no end-to-end transcript. CRAG phase 1 turns the offer into data: a
`coverage_gap` hint and a loop-derived `fetch_proposal` for the user to approve
(`docs/DECISIONS.md` D20–D24). The prompt rule that uses them is phase 2.

## No cross-session memory

Each `main.py ask` is an independent process; question 2 knows nothing about question 1.
Within a question, context is fully preserved and budget-managed (`conversation.py`).
Making it conversational means holding one `Conversation` across questions — roughly
30–50 lines — plus deciding whether chunk dedup stays per-question or per-conversation.

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
  browser that the CLI would not also give. Each question goes to `loop.run` bare. Earlier
  turns of a browser chat are never pasted in.

What it is not: it holds one shared agent behind a lock, so it serves one question at a
time (a second asker is told it is busy) and is not a server for more than one person. It
stores no chats. Chat history lives in the page until a reload, because where it should be
kept is undecided and will be designed with conversation memory. It has no
authentication and binds to localhost by default. Neither the single-user limit nor the
missing authentication is worth fixing unless it stops being a demo.
