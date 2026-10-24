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

## D12 — Conversation memory v1: short-term, per thread

*2026-10-25 · `src/memory/`, `src/agent/loop.py`, `web/api.py`, `main.py` · adds ARCHITECTURE §7.1*

- **By:** user. "Simple hybrid": each prior turn is sent as its question and answer, verbatim, plus a one-line note per tool call. There are no LLM summaries. Stored in SQLite at `data/threads.sqlite3`, with a 9k-token history budget (`agent.thread_history_tokens`). The oldest turns are dropped whole.
- **Why SQLite:** stdlib, no server, a single file, transactions, and SQL queries for listing and ordering.
- **Rejected:** JSON files (no transactions; concurrent CLI and web writes race), Postgres and Redis (a server for a single-user tool), a vector DB (v1 has no long-term recall).
- **Consequence:** `thread_id=None` is exactly the old behaviour, and the eval harness never passes one (a test asserts it). Notes are assistant messages, never replayed tool messages, so no `tool_call` pairing can break.

## D13 — "Seen = still visible" chunk dedup

*2026-10-25 · `src/tools/retrieve_evidence.py`, `src/agent/conversation.py`*

- **By:** user, who wanted dedup to span the whole thread. Claude found that elision hid a result's text while its chunks stayed marked as seen, so they could not be fetched again. Fixed by releasing the elided chunks: `Conversation.last_released` feeds `EvidenceRetriever.release()`.
- **Rule:** the seen set holds only chunks whose text is in the context now. A run starts from the chunks visible in its conversation.
- **Rejected:** seeding the seen set from the chunk ids in prior turns' notes. That would suppress chunks whose text the model no longer has.
- **Consequence:** history is notes only, so in practice dedup spans the current question. A history strategy that replays results would widen it with no code change.

---

## Benched questions

- **Framework for memory / multi-turn:** LangGraph or others versus hand-rolled. Compare the options with the user when that work starts. (v1 in D12 is hand-rolled.)
