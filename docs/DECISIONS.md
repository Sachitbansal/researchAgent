# Decision log

Every build decision: what, who decided, why, what was rejected. If a decision had no real
reason (a default, nobody answered), it says so. Agent design lives in
`docs/ARCHITECTURE.md`; changes to it are logged here with a pointer.

---

## D1 — Web backend: FastAPI
*2026-10-24 · `web/`*
- **By:** Claude, recommended. User asked which of Express / FastAPI / Django suits LLM work and later accepted FastAPI.
- **Why:** agent, local models and FAISS are Python, so the server must be too. Native SSE streaming of agent steps. Pydantic shapes. Auto `/docs`.
- **Rejected:** Express (would need a second Python server); Django (ORM/admin, and there is no DB); Litestar/Starlette (no advantage); stdlib `http.server` (outgrown).
- **Consequence:** `fastapi`, `uvicorn`, `httpx` pinned in `web/requirements.txt`, kept apart from the core `requirements.txt`.

## D2 — Frontend: Vite + React + TypeScript
*2026-10-24 · `frontend/`*
- **By:** Claude. User said "Next.js / React, whichever is best" and accepted.
- **Why:** local app with a Python backend, so no SSR needed. Static build that FastAPI serves: one process, one command.
- **Rejected:** Next.js (SSR/Vercel strengths unused; adds a Node server). Revisit if it is ever hosted publicly.
- **`@types/react` + `@types/react-dom` approved (By: user).** Dev-only type definitions `tsc` needs to check JSX.

## D3 — Three-pane conversational layout
*2026-10-24 · `frontend/src/`*
- **By:** user. Why: should feel like an agentic chat UI (Claude Desktop). Chats on the left, conversation in the middle, tool calls and memory on the right.
- **Consequence:** Memory pane is a placeholder. Each question is still answered independently; the UI says so. *Superseded by D12: chats are threads with short-term memory, and the Memory pane shows each answer's notes.*

## D4 — Per-tool call limits replace the global iteration cap
*2026-10-24 · `src/agent/loop.py`, `config.yaml` agent · changes ARCHITECTURE §7, TOOLS.md "Loop control"*
- **Decision:** no `max_iterations`. Each tool may run 10 times per question; `search_literature` 5. No absolute ceiling.
- **By:** user. Why: a new topic needs 2–3 retrieves, an overview, a literature search, then more retrieves, and one global budget of 8 gets eaten up. Each tool gets its own headroom. `search_literature` gets 5 because it downloads papers (slow, rate-limited), never below 5. No ceiling, because the per-tool limits already bound the loop.
- **Fact noted:** the old cap counted model turns, not tool calls (one turn can make several calls).
- **Backstops:** force the answer when every tool is exhausted, or after 3 consecutive turns with nothing run (refused calls cost no budget, so a stuck model would otherwise loop forever). Proposed by Claude; N=3 confirmed by user.
- **Consequence:** worst case 45 tool calls in one question. Repeated searches trip arXiv's 429, and the cooldown makes them fail fast.

## D5 — Asking is `POST /api/ask`, read as a fetch stream

*2026-10-24 · `web/api.py`, `frontend/src/api.ts`*

- **Decision:** one stateless POST that answers with `text/event-stream`. The frontend reads it with `fetch` and a stream reader.
- **By:** Claude (subagent).
- **Why:** `EventSource` reconnects when a stream ends and would silently re-run the question. A POST body also keeps the question out of URLs and logs.
- **Rejected:** GET + `EventSource` (the old demo's approach, which needed a manual close to avoid re-runs); WebSockets (two-way, and nothing needs that).
- **Consequence:** frames are `step`* then `answer` | `failed`, then `done`, or else `busy`, `done`. *Since D12, a `thread` frame `{thread_id, created, title}` comes first; `busy` creates no thread.*

## D6 — One run at a time, and the worker thread owns it

*2026-10-24 · `web/api.py`*

- **Decision:** the lock is taken non-blocking. If it is held, the reply is an immediate `busy` frame. `loop.run` runs on a thread that releases the lock and resets `on_event` itself.
- **By:** Claude (subagent), within the brief.
- **Why:** a closed tab ends the stream but not the run. The question finishes, and its trace is written, before the agent is free again. No disconnect can leak the lock.
- **Rejected:** a queue of waiting askers. That needs fairness and timeouts and has no demo value. Cancelling the run on disconnect was also rejected, because `AgentLoop` has no cancellation point and adding one would change `src/`.

## D7 — Answer markdown: the old hand-written renderer, ported, plus KaTeX

*2026-10-24, math 2026-10-25 · `frontend/src/markdown.ts`*

- **Decision:** port the old app.js renderer to TypeScript. No markdown library.
- **By:** Claude (subagent). The brief preferred this.
- **Why:** it already handles what the agent emits, and it owns citation numbering. A library would need a plugin to do the same thing.
- **Rejected:** `marked`/`markdown-it` plus a sanitiser (two dependencies).
- **Math (By: user, `katex` 0.19.1 pinned):** `$…$`, `$$…$$`, `\(…\)` and `\[…\]` render with `katex.renderToString` (`throwOnError: false`, `trust: false`). Code spans and math are cut out before the citation, emphasis and link passes and put back last, so `_`, `*` and `[` inside them are never touched. Inline `$` follows the pandoc rule (no space just inside either dollar, no digit after the closing one), so "$5 and $10" stays text. KaTeX's output is the only unescaped HTML in an answer. Memory texts (Memory tab, Memories page) get math only, no citations or emphasis. Citation-card excerpts stay plain: they are PDF text, not model output. Checks: `npm run check:markdown`.
- **Also:** a code span holding only citations (`` `[id__c0001]` ``) is read as those citations; any other code span keeps its contents literal, so an id inside real code no longer becomes a button.

## D8 — Styling: Tailwind v4 with the design tokens in `@theme`

*2026-10-24, Tailwind 2026-10-25 · `frontend/src/styles.css`, the TSX, `frontend/vite.config.ts`*

- **Decision:** Tailwind v4 through `@tailwindcss/vite` (both 4.3.3, pinned). The D11 palette, the Plex fonts, the 4px spacing base, the type sizes, the radii and the breakpoints are `@theme` tokens, and Tailwind's own colours, sizes and radii are switched off, so utilities can only use the design's values. Dark mode re-points the same variables under `prefers-color-scheme` (unless `data-theme="light"`) and under `data-theme="dark"`, as before; a `dark:` variant follows the same rule. Components are styled with utilities in the TSX; a small `@layer components` keeps the patterns that read badly as utilities (answer prose and math, buttons, chips, the timeline rows, the input focus ring).
- **By:** user, cosmetic: "no point sticking with plain CSS if Tailwind is cleaner and not a big step". Same visual direction as D11; the port was checked against the previous build element by element (computed styles and boxes) in both themes and at three widths.
- **Rejected:** `@tailwindcss/typography` (a further dependency; the answer prose is a dozen rules), and component libraries.
- **Theme storage (by user):** only the light/dark/system choice is saved in `localStorage`, wrapped in try/catch. Chats are never stored in the browser.

## D9 — Figures served by `/api/figures/<paper_id>/<file>` only

*2026-10-24 · `web/api.py`*

- **Decision:** two plain-name path segments, resolved under `paths.figures`, must still be inside it after symlinks. Anything else is a 404. An `image` step carries the URL, or `null` for a file outside that directory.
- **By:** Claude (subagent), within the brief.
- **Why:** it matches how figures are written to disk and cannot name any other file.
- **Rejected:** a `StaticFiles` mount over `data/`, or a raw path query parameter.

## D10 — Sample chats as a frontend fixture

*2026-10-24 · `frontend/src/fixtures/sampleChats.ts`*

- **Decision:** three hand-written chats, tagged "Sample" in the UI. Their chunk ids and excerpts are real, copied from `data/index/chunks.jsonl`.
- **By:** user.
- **Why:** to show the panes populated before anyone has asked a question.
- **Consequence:** this is the one file under `frontend/src` that names topics, because it describes the shipped corpus. Delete it and its one import in `App.tsx` to remove the samples.

## D11 — Visual direction: a quiet documentation-style tool

*2026-10-24 · `frontend/src/styles.css` (tokens, since D8's Tailwind port), `frontend/index.html`*

- **Decision:** neutral slate surfaces with one link-blue accent (#2563EB, #6EA8FE in dark) used for links, citations, the send button and focus. Text is IBM Plex Sans throughout, with Plex Mono for ids and arguments. Answers are set at 16px/1.7 with a 68ch measure. Spacing is on a 4px scale; radii are 4, 6 and 8px. Every text pair meets WCAG AA in both themes.
- **By:** Claude (subagent), from the `ui-ux-pro-max` skill: its "Minimalism & Swiss" style, its "Knowledge Base/Documentation" palette, and the Plex Sans half of its "Developer Mono" pairing. The user asked for professional and restrained.
- **Why:** this is a research tool people read in for a long time, so it should feel like a good docs site and not a landing page. One accent keeps the citations the clearest thing on the page.
- **Rejected:** the skill's first suggestion (teal and orange with Plus Jakarta Sans), which reads as SaaS marketing; the first pass's serif answers and highlighter-yellow brand, which looked closer to the cream-and-serif "AI" look; and purple/blue gradients and glass effects, which the user excluded.
- **No strong reason** for Plex over Inter, which the skill ranked just behind it. Plex was already loaded and has a matching mono.

## D12 — Conversation memory v1: short-term, per thread

*2026-10-25 · `src/memory/`, `src/agent/loop.py`, `web/api.py`, `main.py` · adds ARCHITECTURE §7.1*

- **By:** user. "Simple hybrid": each prior turn is sent as its question and answer, verbatim, plus a one-line note per tool call. There are no LLM summaries. Stored in SQLite at `data/threads.sqlite3`, with a 9k-token history budget (`agent.thread_history_tokens`). The oldest turns are dropped whole.
- **Why SQLite:** stdlib, no server, a single file, transactions, and SQL queries for listing and ordering.
- **Rejected:** JSON files (no transactions; concurrent CLI and web writes race), Postgres and Redis (a server for a single-user tool), a vector DB (v1 has no long-term recall).
- **Consequence:** `thread_id=None` is exactly the old behaviour, and the eval harness never passes one (a test asserts it). Notes are assistant messages, never replayed tool messages, so no `tool_call` pairing can break. *Since D19 the system prompt carries one memory rule for every run.*

## D13 — "Seen = still visible" chunk dedup

*2026-10-25 · `src/tools/retrieve_evidence.py`, `src/agent/conversation.py`*

- **By:** user, who wanted dedup to span the whole thread. Claude found that elision hid a result's text while its chunks stayed marked as seen, so they could not be fetched again. Fixed by releasing the elided chunks: `Conversation.last_released` feeds `EvidenceRetriever.release()`.
- **Rule:** the seen set holds only chunks whose text is in the context now. A run starts from the chunks visible in its conversation.
- **Rejected:** seeding the seen set from the chunk ids in prior turns' notes. That would suppress chunks whose text the model no longer has.
- **Consequence:** history is notes only, so in practice dedup spans the current question. A history strategy that replays results would widen it with no code change.

## D14 — CRAG trigger: the agent proposes a fetch, the user approves it

*2026-10-25 · `src/tools/retrieve_evidence.py`, `src/agent/fetch_proposal.py`; the button is phase 2*

- **Decision:** when retrieval cannot answer, the agent proposes a literature fetch, and a "Fetch papers" button in the UI runs it on the user's approval.
- **By:** user.
- **Why:** the relevance gate already makes the agent abstain, but nothing helped it go and get the missing literature.
- **Rejected:** auto-fetch, because it pollutes the shared corpus and arXiv returns *something* even for topics it barely covers. Plain abstention was also rejected, because it leaves the gap for the user to diagnose.

## D15 — A structured `coverage_gap` hint, plus a prompt rule

*2026-10-25 · `src/tools/retrieve_evidence.py`, `config.yaml` retrieval*

- **Decision:** when the gate fails, `retrieve_evidence` returns `coverage_gap` with the query, best score, the threshold and scorer that actually ran, up to 3 closest topics, `suggested_next` (`rephrase` | `propose_fetch`) and `proposed_query`. It is data, not a call, so tools still never call tools.
- **By:** user. The prompt rule that tells the agent how to act on the hint is phase 2 (`src/prompts/system.md` is untouched here).
- **Margins (by Claude, measured; confirmed by user 2026-10-25):** `rephrase_margin` is 0.25 logits and `bi_encoder_rephrase_margin` is 0.0. They are per scorer for the same reason the thresholds are.
- **Why so small:** no failed retrieval, at any distance, was recovered by rewording the same request. The one answerable miss (mp03) sits 2.51 below, further than four absent topics. A wrong `rephrase` silently suppresses the proposal; a wrong `propose_fetch` is a button the user can ignore.
- **Resolved by Claude:** the hint is absent on non-gate paths (empty filter, no candidates, all already returned). `proposed_query` is the retrieve query itself, because `search_literature` plans its own arXiv syntax.

## D16 — Admission gate: abstracts scored by the cross-encoder before download

*2026-10-25 · `src/corpus/admission.py`, `src/corpus/collect.py`, `src/tools/search_literature.py`*

- **Decision:** `search_literature` scores `(query, title + abstract)` for every arXiv result with the retriever's cross-encoder (no new model) and never downloads, indexes or tags papers below `collection.admission_threshold`. They are reported in `papers_rejected`. If every result is rejected, the tool says arXiv does not cover the topic well, and this is a result, not an error. A broken cross-encoder admits everything with `admission_gate: "skipped: <reason>"`.
- **By:** user. The threshold is measured offline (`eval/tune_admission.py`, no arXiv calls): -3.0, the permissive end of the perfect separation on topic-shaped pairs.
- **Trade-off:** question-shaped queries score far lower against whole abstracts, so only 61.5% of those positives are admitted at -3.0. The pooled sweep's -7.0 would let 3 of 11 adjacent cross-topic papers in instead. -3.0 confirmed by user 2026-10-25.
- **Resolved by Claude:** the gate also covers dedup hits, so a rejected paper already held is not tagged with the new topic. `main.py index` calls `collect` directly and stays ungated, because it is an explicit user command, not the agent. `eval/build_corpus.py` goes through `LiteratureSearcher`, so it is now gated. The selection rule (topic-shaped pairs first) was adopted after the pooled sweep had recommended -7.0, and both numbers are recorded.

## D17 — Eval: an `absent_fetchable` category, plus a manual check on a throwaway copy

*2026-10-25 · `eval/questions.json`, `eval/metrics.py`, `eval/runner.py`, `scripts/crag_fetch_check.py`*

- **Decision:** a new category, `absent_fetchable`, with `expect_fetch_proposal: true`. `ni02` and `ni03` are re-tagged into it, keeping their ids, `previous_category` and `expect_abstention: true`. Three new questions are added: af01 and af02 tuning, af03 held out (18 total, 6 held out). `ni01` stays a pure abstention control. The live search path is exercised by hand with `scripts/crag_fetch_check.py`, which copies the data dir to a temp dir, runs one `search_literature` with 3 results, verifies the real `data/index` hashes are unchanged, and deletes the copy.
- **By:** user.
- **Rejected / benched:** record/replay of arXiv responses for an automated end-to-end test.
- **Note:** the full agent eval has not been run on the new category. Phase 2 changes the agent's behaviour, so no numbers exist yet.

## D18 — Proposal mechanism: a structured `fetch_proposal` field derived by the loop

*2026-10-25 · `src/agent/fetch_proposal.py`, `src/agent/loop.py` (one field)*

- **Decision:** `AgentLoop.run` returns `fetch_proposal: {query, searched_for, best_score, closest_topics} | null`. It is derived from the run's last `retrieve_evidence` call when that call's `coverage_gap` says `propose_fetch` and `search_literature` was not called afterwards. A later passing retrieval clears it. `eval.metrics.fetch_proposal_detected` only reads this field.
- **By:** user.
- **Rejected:** a text marker the agent writes, because it can be forgotten or garbled.
- **Consequence:** the `loop.py` change is one import and one return field, kept small because the memory branch edits the same file. (D28 adds a rewrite of the proposal's query after this derivation; the derivation itself stays pure.)

## D19 — Long-term memory: grounded findings across chats

*2026-10-25 · `src/memory/longterm.py`, `memory_store.py`, `src/agent/loop.py`, `web/api.py` · adds ARCHITECTURE §7.2, DATA_SCHEMA §10*

- **By:** user. Two things are remembered: findings (a claim, its chunk_ids and their `content_hash`es, the source question, thread and date) and preferences the user states explicitly. Nothing episodic is kept. Memory is written after the answer, only for a thread run that passed the gate, and only from claims NLI labels entailed. Duplicates and conflicts are settled by cosine plus NLI, with no LLM call. The loop injects memories, with their chunk text, so the agent can cite. Storage is the threads SQLite file. `thread_id=None` never touches memory, and an AST test enforces this.
- **Rejected:** Mem0 (an LLM call per add/update/delete decision, and a vector-DB service); Zep/Graphiti (a graph database and server, for one user); Letta (the agent manages its own memory, so the loop structure under evaluation moves into the framework); LangMem (a LangChain/LangGraph dependency, ruled out by CLAUDE.md); a sixth `recall` tool (the one-agent, five-tool limit, and the agent would have to know to ask).
- **Consequence:** the "memories are leads, chunks are the evidence" rule lives only in `memory_block.md`, so it is sent only when memories are injected. `system.md` is unchanged, and a no-thread (eval) request stays byte-identical to pre-D12.

## D20 — Automatic forgetting

*2026-10-25 · `src/memory/forgetting.py`, `config.yaml` memory*

- **By:** user: "otherwise it keeps adding to memory and never forgets old memories, and the user has to go through the pain of deleting what they don't care about. It should be handled automatically by the new context and new results the agent produces."
- **Paths:** superseded (a confident contradiction from a newer finding); stale (a cited chunk's hash changed or the chunk is gone; checked at read time and in the sweep); decay plus cap (`strength = (1 + ln use_count) × 0.5^(days/half_life)`, below `min_strength` or past `max_active`, weakest first, swept after each write); refresh (a re-finding resets recency); and a manual delete as override. Everything is soft-deleted with a reason and hard-purged after `purge_after_days`.
- **A use** is injected **and** cited. Injected but uncited does not count. Being found counts as the first use.
- **Confirmed by the user (2026-10-25):** `half_life_days` 30, `max_active` 500, `min_strength` 0.125 (= 0.5^(90/30): used once and untouched for 90 days), `purge_after_days` 180.

## D21 — Grounding checks the passage-worded claim; the stored claim names its paper

*2026-10-25 · `src/prompts/memory_extract.md`, `src/memory/longterm.py`*

- **By:** Claude, after the first smoke run wrote 0 of 12 claims. Two measured causes. (1) The prompt makes every finding name its paper, so two papers' numbers never merge or read as a contradiction, and this NLI model then scores a supported claim at 0.001 entailment instead of 0.993. (2) The supporting sentence was the 10th to 12th of its chunk, past `nli.max_sentences_per_chunk` (8).
- **Decision:** extraction returns `text` (attributed: stored, embedded, conflict-checked) and `check` (the same assertion as the passage words it: NLI-grounded). Attribution is checked by the citation instead. The premise pool is `memory.premise_sentences_per_chunk` (64), but NLI still scores only the top `premises_per_claim`, so cost is embeddings, not NLI pairs.
- **Rejected:** unattributed stored claims ("the coefficient is 0.01" from one paper would contradict, and supersede, "the coefficient is 0.02" from another; argued, not measured); lowering `entailment_threshold` (a tuned value shared with the tool).
- **Effect, re-measured on the 12 eval answers:** 32 of 93 extracted claims entailed (34%), up from 20 of 97 (21%). The `min_pair_similarity` floor drops 2 of 34.
- **Needs user confirmation:** this departs from "the same premise selection as groundedness mode" in pool size only.

## D22 — A contradiction outranks a refresh; preferences are always injected and never decay

*2026-10-25 · `src/memory/longterm.py`, `src/memory/forgetting.py`*

- **By:** Claude. Measured (`eval/results/memory_thresholds.json`): number- or method-swapped claims sit closer than paraphrases (median cosine 0.91 vs 0.86). A duplicate test run first would refresh a contradiction and merge its chunks into the old memory. So every memory at or above `related_cosine` is NLI-checked first, and a confident contradiction (both directions: 27/32 swaps caught, 0 of 8 distinct pairs ≥ 0.80 flagged, 0 of 32 paraphrases; either direction flags 2 paraphrases) supersedes.
- **Preferences:** injected on every thread run (a cosine gate would rarely surface "only 2024 onward"). They are exempt from decay and the cap, because they are never cited, so the use rule would decay every one. They go by contradiction, by restatement (refresh) or by hand.
- **Write trigger:** "the gate passed" means some `retrieve_evidence` call returned `sufficient_evidence: true` with chunks. An answer that cites only injected chunks still counts its uses but writes nothing new.
- **Deleting a chat keeps its memories** (`source_thread_id` has no foreign key). Memory is meant to outlive the chat.

## D23 — A Memories page: the user sees, edits, adds, deletes and restores memories

*2026-10-25 · `frontend/src/components/MemoriesView.tsx`, `UserMenu.tsx`, `web/api.py`, `src/memory/longterm.py`, `memory_store.py` · DATA_SCHEMA §10*

- **By:** user: "Like Claude, I should be able to edit and see my memories … have a memories option where the person can see all memories and edit or delete them himself." It is reached from a user menu at the bottom left (one user, no auth), which also holds the theme.
- **Edit:** the text is re-embedded on the passage side (no query prefix). Editing is not a use. A finding is re-grounded with the writer's own `ground()`, with the edited text as the hypothesis. Entailed → `verified`, and the chunk hashes it was checked against are stored. Not entailed → **saved anyway**, with `verified=false` and `user_edited=true`. It is still injected, and the memory block marks it "(edited by user, unverified)" (`memory_block.md` says what that means). **Tentative (Claude), user to confirm.** Measured on the smoke run: an edit that keeps the paper's name but copies the passage's wording was entailed (0.98), while "0.01" for the passage's "1e-2" was not (0.0006). So attributed or paraphrased edits often come back unverified (D21).
- **Add:** preferences only (`POST /api/memories`). They are settled like extracted ones: a restatement refreshes, a contradiction supersedes. Findings can't be added by hand, because they need cited chunk evidence.
- **Restore:** clears `forgotten_at`/`invalidated_at`, the reason and `superseded_by`, and resets `last_used_at` to now. Refused (409) for a finding whose passage hash still differs, because the next recall would only invalidate it again. Restoring a superseded finding leaves both sides of the contradiction active; that is the user's call. Hard-purged rows are gone.
- **Schema:** `origin`, `user_edited`, `verified`, `edited_at` and `forget_detail` (`below_min_strength` | `over_cap`, so the page can say "over the cap"). Each is added by `ALTER TABLE ADD COLUMN` when missing, tolerating a concurrent opener. Old rows read as agent-made, unedited and verified.
- **Concurrency:** an edit or an add is refused (409 `agent_busy`) while a question runs, because it would share the run's embedder and NLI model (the fast tokenizer is not thread-safe). Delete and restore touch no model and stay open.
- **Rejected:** blocking unverified edits (the user owns their memory); re-running reconciliation on an edit (it would let an edit silently supersede other memories).

## D24 — CRAG phase 2: filtered gaps are ignored (option A)

*2026-10-25 · `src/agent/fetch_proposal.py`, `src/tools/retrieve_evidence.py`*

- **By:** user, choosing option A of the two fixes `docs/FUTURE_WORK.md` listed.
- **Rule:** the proposal is derived from the run's last **unfiltered** `retrieve_evidence` (no `topic_filter`, no `chunk_types`). Filtered calls are skipped entirely: they neither propose nor clear. If that call passed, or there was none, there is no proposal. A `search_literature` after it clears the proposal, as before.
- **Why:** a filtered search failing says nothing about whether the corpus as a whole lacks the subject. A figure-only search that finds no figure would otherwise put a fetch button under a question the corpus covers.
- **`coverage_gap.filtered: bool`** is added so the agent and the UI can tell. The derivation reads filtering from the call's arguments, because a passing call has no gap.
- **Resolved by Claude:** an unfiltered `empty_index` also proposes a fetch of its query. Without this, the cold start would be unreachable: no scorer runs, so there is no gap, no button, and the agent may not search on its own.

## D25 — The prompt rule: describe the gap, never fetch unasked

*2026-10-25 · `src/prompts/system.md`, `src/prompts/tools/retrieve_evidence.md`, `search_literature.md`*

- **By:** user. The agent acts on `suggested_next`. On `rephrase` it makes one reworded retrieve. On `propose_fetch` it does not call `search_literature`: it says what it searched for, which held topics came closest, and that the "Fetch papers" button fetches, and it never answers from its own knowledge.
- **`search_literature` only after approval,** which the approval message states explicitly. After fetching, the agent retrieves again and answers the original question with citations.
- **Lines that contradicted the rule were edited, not appended to:** "use `search_literature` when the corpus lacks the subject" (system prompt and both tool descriptions).
- **Resolved by Claude:** two clauses beyond the spec. If `coverage_gap.filtered` is true, retry once unfiltered; otherwise the agent would mention a button that D24 never shows. On `arxiv_coverage: "poor"`, say arXiv does not cover the subject well.
- **Resolved by Claude, after the end-to-end check:** `search_literature`'s description now says to leave `categories` unset. In the live run the agent passed `["cs.CV","cs.LG"]`, which were ANDed onto the planner's own category filter (the `.claude/CLAUDE.md` pitfall). It was harmless there only because the sets matched.
- **Consequence:** the system prompt, and so the eval's prompt, changed. Measured in `docs/EVALUATION.md` (CRAG phase 2).

## D26 — Approval is a structured `approve_fetch`, worded by a server-side template

*2026-10-25 · `web/api.py`, `main.py`, `src/agent/fetch_proposal.py`, `src/agent/loop.py`, `src/prompts/fetch_approval*.md`*

- **By:** user. `POST /api/ask` takes `approve_fetch: {query}` with a `thread_id`, in place of a question. The server builds the agent's message from `src/prompts/fetch_approval.md`. It is a normal thread turn, so short-term memory (D12) gives the agent the question it is to answer. The CLI has the same thing: `main.py ask --thread ID --approve-fetch "<query>"`.
- **Why a template, not UI text:** the wording lives in one prompt file and is diffed like every other prompt. The browser and the CLI cannot drift apart.
- **Validation** (`approval_request`, shared): a thread is required, the query is whitespace-collapsed, must be non-empty, and is capped at `agent.fetch_query_max_chars` (300). Sending both a question and an approval is refused (`question_and_approval`). All of it is checked before the stream opens, so errors are a 400, not a frame.
- **Stored:** the turn's question is the short form from `fetch_approval_title.md` ("Fetch papers on "…""). It is a prompt file too, because later turns replay it to the model as history. Its stats carry `approved_fetch: {query}`. Every turn's stats carry `fetch_proposal`, which is also in the `answer` frame, so a reloaded chat still shows the button. The CLI prints `--- fetch proposal: "<query>"  (re-ask with --approve-fetch)`.
- **Resolved by Claude:** an approval run whose retrieval still fails re-derives the same proposal ("as now"). The edges ignore it: no card in the UI and no CLI line on a turn that carries `approved_fetch`. Long-term memory sees the template text as the turn's question.

## D27 — The "Fetch papers" card

*2026-10-25 · `frontend/src/components/FetchCard.tsx`, `ChatView.tsx`, `format.ts`*

- **By:** user. It sits under an answer with a `fetch_proposal`. It says "The corpus has no papers on *X*. Closest topics held: …" (tags mapped to their corpus labels). It has an editable query with a visible label, a primary **Fetch papers** button, and a note that fetching takes a minute or more and adds papers to the corpus permanently.
- **Click = approval.** It sends `approve_fetch` in the same chat. While it runs, the card says it is fetching, and the streamed steps show `search_literature`. Afterwards it says the fetch was approved, and the button is gone for good, because the next turn's stats carry `approved_fetch`.
- **Resolved by Claude:** the card is actionable only on the **latest** answer of a real (non-sample) chat while nothing runs, since the approval asks the agent to answer "my previous question". On an older answer it says a later question followed. While another run is going, the button is disabled with a reason. If the approval run failed, the card says so and to ask again, rather than claiming success. The stored turn still carries `approved_fetch`, so the button stays gone.
- **Admission verdict in the step summary:** `search_literature`'s outcome line now shows how many papers the gate rejected and "arXiv covers this poorly" when `arxiv_coverage` is `poor`.

## D28 — Fetch query rewritten from the user's question

*2026-10-25 · `src/agent/fetch_query.py`, `src/prompts/fetch_query.md`, `src/agent/loop.py`, `config.yaml` agent*

- **By:** user, option (a). When a run ends with a proposal, one `utility` call rewrites the user's question into a short keyword phrase. That phrase becomes `fetch_proposal.query`. The agent's last unfiltered retrieve query is kept as `agent_query`, and `searched_for` is unchanged. `query_source` is `"rewritten"`, or `"agent"` on a fallback.
- **Why:** the agent's queries drift. In the phase-2 check, "4-bit post-training quantization of ViTs" proposed `vision image classification transformer ViT`. The admission gate scores against the fetched query, so it would have admitted generic ViT papers.
- **Rejected:** the first failed query, because it is still the agent's wording, not the user's. The verbatim question, because at -3.0 the gate admits only 61.5% of relevant papers for question-shaped queries (D16). Keeping the query as it was.
- **Guards:** no call without a proposal, and the call is made once, with no retry. An LLM error, an empty reply, more than one line, a sentence or question shape (a `?`, closing `.`/`!`, an opening question word, more than `agent.fetch_query_max_words` (12) words), or more than `agent.fetch_query_max_chars` all fall back to the agent query with `rewrite_error`. Nothing raises. The derivation stays pure, and the rewrite is a separate helper that tests drive with a stub client.
- **Threads (resolved by Claude):** the prompt always gets the current turn's question, plus the thread's previous stored question as context, so "what about 2-bit?" and approval turns resolve their subject. The prompt tells the model to ignore the earlier question when the current one names its own subject. An approval run that re-derives a proposal also pays for one rewrite, which nothing shows (D26).
- **Card:** still editable. Its label reads "Search arXiv for (suggested from your question)" when `query_source` is `"rewritten"`.

---

## Benched questions

- **Framework for memory / multi-turn:** LangGraph or others versus hand-rolled. The comparison now exists at `docs/research/framework-comparison.md`, with the triggers for switching. Adoption stays benched: threads (D12), long-term memory (D19) and fetch approval (D26) are hand-rolled.
- **Record/replay of arXiv responses** for an automated CRAG end-to-end test (D17). Until then the live path is checked by hand on a throwaway copy.
- **Roadmap, in the user's order:** history summarisation, prompt caching plus a smaller context, and a tracing UI. (Long-term research memory: D19–D22. The framework comparison is written up in `docs/research/framework-comparison.md`; adopting one stays benched.)
- **Unordered:** retrieval upgrade (diagnose, then contextual headers, hybrid BM25 and multi-query); self-check and revise, plus a PDF hidden-text filter; a long-context baseline eval; minor items (MCP, citation graph, reranker swap, a floor check on limits, a warning for unknown tool names in limits, citations wrapped in backticks rendering as code).
