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

## D3 — Three-pane conversational layout
*2026-10-24 · `frontend/src/`*
- **By:** user. Why: should feel like an agentic chat UI (Claude Desktop). Chats on the left, conversation in the middle, tool calls and memory on the right.
- **Consequence:** Memory pane is a placeholder. Each question is still answered independently; the UI says so.

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
- **Consequence:** frames are `step`* then `answer` | `failed`, then `done`, or else `busy`, `done`.

## D6 — One run at a time, and the worker thread owns it

*2026-10-24 · `web/api.py`*

- **Decision:** the lock is taken non-blocking. If it is held, the reply is an immediate `busy` frame. `loop.run` runs on a thread that releases the lock and resets `on_event` itself.
- **By:** Claude (subagent), within the brief.
- **Why:** a closed tab ends the stream but not the run. The question finishes, and its trace is written, before the agent is free again. No disconnect can leak the lock.
- **Rejected:** a queue of waiting askers. That needs fairness and timeouts and has no demo value. Cancelling the run on disconnect was also rejected, because `AgentLoop` has no cancellation point and adding one would change `src/`.

## D7 — Answer markdown: the old hand-written renderer, ported

*2026-10-24 · `frontend/src/markdown.ts`*

- **Decision:** port the old app.js renderer to TypeScript. No markdown library.
- **By:** Claude (subagent). The brief preferred this.
- **Why:** it already handles what the agent emits, and it owns citation numbering. A library would need a plugin to do the same thing.
- **Rejected:** `marked`/`markdown-it` plus a sanitiser (two dependencies).
- **Pending (approved by user):** render `$…$` and `$$…$$` with the pinned `katex` package, with math spans protected from the citation and emphasis passes and `throwOnError: false`. The `npm install` was blocked by the session's permission check, so for now math still shows as plain text.

## D8 — Styling: hand-written CSS with design tokens

*2026-10-24 · `frontend/src/styles.css`*

- **Decision:** one stylesheet with CSS custom properties, and light/dark themes under `prefers-color-scheme` plus a `data-theme` override.
- **By:** Claude (subagent). **No strong reason** over Tailwind, beyond having fewer dependencies and the old demo already being tokenised CSS. The palette and type are in D11.
- **Rejected:** Tailwind (a dependency and a build plugin), and component libraries (the brief said no bloat).
- **Theme storage (by user):** only the light/dark/system choice is saved in `localStorage`, wrapped in try/catch. Chats are never stored in the browser.
- **Pending:** the user has approved switching to Tailwind v4 (`@tailwindcss/vite`), on the grounds that there is "no point sticking with plain CSS if Tailwind is cleaner and not a big step". The `npm install` was blocked by the session's permission check, so the switch is not done yet.

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

*2026-10-24 · `frontend/src/styles.css`, `frontend/index.html`*

- **Decision:** neutral slate surfaces with one link-blue accent (#2563EB, #6EA8FE in dark) used for links, citations, the send button and focus. Text is IBM Plex Sans throughout, with Plex Mono for ids and arguments. Answers are set at 16px/1.7 with a 68ch measure. Spacing is on a 4px scale; radii are 4, 6 and 8px. Every text pair meets WCAG AA in both themes.
- **By:** Claude (subagent), from the `ui-ux-pro-max` skill: its "Minimalism & Swiss" style, its "Knowledge Base/Documentation" palette, and the Plex Sans half of its "Developer Mono" pairing. The user asked for professional and restrained.
- **Why:** this is a research tool people read in for a long time, so it should feel like a good docs site and not a landing page. One accent keeps the citations the clearest thing on the page.
- **Rejected:** the skill's first suggestion (teal and orange with Plus Jakarta Sans), which reads as SaaS marketing; the first pass's serif answers and highlighter-yellow brand, which looked closer to the cream-and-serif "AI" look; and purple/blue gradients and glass effects, which the user excluded.
- **No strong reason** for Plex over Inter, which the skill ranked just behind it. Plex was already loaded and has a matching mono.

## D20 — CRAG trigger: the agent proposes a fetch, the user approves it

*2026-10-25 · `src/tools/retrieve_evidence.py`, `src/agent/fetch_proposal.py`; the button is phase 2*

- **Decision:** when retrieval cannot answer, the agent proposes a literature fetch, and a "Fetch papers" button in the UI runs it on the user's approval.
- **By:** user.
- **Why:** the relevance gate already makes the agent abstain, but nothing helped it go and get the missing literature.
- **Rejected:** auto-fetch, because it pollutes the shared corpus and arXiv returns *something* even for topics it barely covers. Plain abstention was also rejected, because it leaves the gap for the user to diagnose.

## D21 — A structured `coverage_gap` hint, plus a prompt rule

*2026-10-25 · `src/tools/retrieve_evidence.py`, `config.yaml` retrieval*

- **Decision:** when the gate fails, `retrieve_evidence` returns `coverage_gap` with the query, best score, the threshold and scorer that actually ran, up to 3 closest topics, `suggested_next` (`rephrase` | `propose_fetch`) and `proposed_query`. It is data, not a call, so tools still never call tools.
- **By:** user. The prompt rule that tells the agent how to act on the hint is phase 2 (`src/prompts/system.md` is untouched here).
- **Margins (by Claude, measured, need user confirmation):** `rephrase_margin` is 0.25 logits and `bi_encoder_rephrase_margin` is 0.0. They are per scorer for the same reason the thresholds are.
- **Why so small:** no failed retrieval, at any distance, was recovered by rewording the same request. The one answerable miss (mp03) sits 2.51 below, further than four absent topics. A wrong `rephrase` silently suppresses the proposal; a wrong `propose_fetch` is a button the user can ignore.
- **Resolved by Claude:** the hint is absent on non-gate paths (empty filter, no candidates, all already returned). `proposed_query` is the retrieve query itself, because `search_literature` plans its own arXiv syntax.

## D22 — Admission gate: abstracts scored by the cross-encoder before download

*2026-10-25 · `src/corpus/admission.py`, `src/corpus/collect.py`, `src/tools/search_literature.py`*

- **Decision:** `search_literature` scores `(query, title + abstract)` for every arXiv result with the retriever's cross-encoder (no new model) and never downloads, indexes or tags papers below `collection.admission_threshold`. They are reported in `papers_rejected`. If every result is rejected, the tool says arXiv does not cover the topic well, and this is a result, not an error. A broken cross-encoder admits everything with `admission_gate: "skipped: <reason>"`.
- **By:** user. The threshold is measured offline (`eval/tune_admission.py`, no arXiv calls): -3.0, the permissive end of the perfect separation on topic-shaped pairs.
- **Trade-off:** question-shaped queries score far lower against whole abstracts, so only 61.5% of those positives are admitted at -3.0. The pooled sweep's -7.0 would let 3 of 11 adjacent cross-topic papers in instead. Needs user confirmation.
- **Resolved by Claude:** the gate also covers dedup hits, so a rejected paper already held is not tagged with the new topic. `main.py index` calls `collect` directly and stays ungated, because it is an explicit user command, not the agent. `eval/build_corpus.py` goes through `LiteratureSearcher`, so it is now gated. The selection rule (topic-shaped pairs first) was adopted after the pooled sweep had recommended -7.0, and both numbers are recorded.

## D23 — Eval: an `absent_fetchable` category, plus a manual check on a throwaway copy

*2026-10-25 · `eval/questions.json`, `eval/metrics.py`, `eval/runner.py`, `scripts/crag_fetch_check.py`*

- **Decision:** a new category, `absent_fetchable`, with `expect_fetch_proposal: true`. `ni02` and `ni03` are re-tagged into it, keeping their ids, `previous_category` and `expect_abstention: true`. Three new questions are added: af01 and af02 tuning, af03 held out (18 total, 6 held out). `ni01` stays a pure abstention control. The live search path is exercised by hand with `scripts/crag_fetch_check.py`, which copies the data dir to a temp dir, runs one `search_literature` with 3 results, verifies the real `data/index` hashes are unchanged, and deletes the copy.
- **By:** user.
- **Rejected / benched:** record/replay of arXiv responses for an automated end-to-end test.
- **Note:** the full agent eval has not been run on the new category. Phase 2 changes the agent's behaviour, so no numbers exist yet.

## D24 — Proposal mechanism: a structured `fetch_proposal` field derived by the loop

*2026-10-25 · `src/agent/fetch_proposal.py`, `src/agent/loop.py` (one field)*

- **Decision:** `AgentLoop.run` returns `fetch_proposal: {query, searched_for, best_score, closest_topics} | null`. It is derived from the run's last `retrieve_evidence` call when that call's `coverage_gap` says `propose_fetch` and `search_literature` was not called afterwards. A later passing retrieval clears it. `eval.metrics.fetch_proposal_detected` only reads this field.
- **By:** user.
- **Rejected:** a text marker the agent writes, because it can be forgotten or garbled.
- **Consequence:** the `loop.py` change is one import and one return field, kept small because the memory branch edits the same file.

---

## Benched questions

- **Chat history storage:** undecided. Design it together with conversation memory. Until then, chats live only in page memory, seeded with the samples (D10). The server stores nothing (D5).
- **Framework for memory / multi-turn:** LangGraph or others versus hand-rolled. Compare the options with the user when that work starts.
- **`@types/react`, `@types/react-dom`:** dev-only TypeScript type definitions that `tsc` needs to check JSX. Installed; awaiting the user's approval.
- **Record/replay of arXiv responses** for an automated CRAG end-to-end test: benched (D23). The live path is checked by hand on a throwaway copy instead.
