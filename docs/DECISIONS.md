# Decision log

Every decision about how the project is built, who made it, and why. Where a decision had
no real reasoning behind it — a default taken, a coin flip, nobody asked — that is written
down too, so a later reader does not mistake an accident for a constraint and defend it.

`docs/ARCHITECTURE.md` holds the design of the agent itself. This file covers everything
around it — the serving layer, the frontend, tooling — plus any later change to the
agent's design, with a pointer to the section it changed.

Each entry: **Decision · Decided by · Why · Rejected · Consequences.**

---

## D1 — Backend for the browser UI: FastAPI

*2026-10-24 · `web/server.py`*

**Decided by:** Claude (agent), recommending. The user asked which backend fits agentic /
LLM work, offering Express, FastAPI or Django, and did not pick. Earlier the user said
"whichever you think best" about the frontend. So this is the agent's call, not the
user's, and it carries no user rationale.

**Why:**
- The agent, its three local models (bi-encoder, cross-encoder, NLI) and the FAISS index
  are Python. The server has to run in that process, or else proxy to one.
- Streaming each agent step (Server-Sent Events) is a first-class pattern: an async
  generator plus `StreamingResponse`.
- Pydantic request/response models match the project rule that tools return structured
  dicts, and `/docs` gives an OpenAPI page for free.
- It is the default choice in the LLM-serving ecosystem, so examples and help are easy to
  find.

**Rejected:**
- **Express (Node).** Python would still have to run somewhere, so this means two servers
  and IPC between them for no gain.
- **Django.** Its value is the ORM, admin and auth. This project has no database. It would
  be weight with nothing to carry.
- **Litestar / bare Starlette.** Technically comparable. Smaller ecosystem (Litestar) or
  more hand-rolling (Starlette). No reason to prefer them.
- **Keep stdlib `http.server`.** It worked for a one-page demo. A multi-pane app with
  chats, figure serving and typed APIs would mean re-writing a framework by hand.

**Consequences:**
- New dependencies `fastapi`, `uvicorn`, and `httpx` (test client only). They are pinned
  in `web/requirements.txt`, not the core `requirements.txt`, because the core file pins
  the system under evaluation and the web layer is not part of it.
- `docs/FUTURE_WORK.md` previously said `web/` "adds no dependency". That is no longer
  true, and the doc is updated.

## D2 — Frontend: Vite + React + TypeScript

*2026-10-24 · `frontend/`*

**Decided by:** Claude (agent). The user offered "Next.js / React, whichever you think is
best".

**Why:**
- The page talks to a local Python backend. Nothing benefits from server-side rendering.
- Vite builds to static files that FastAPI serves itself. The demo stays one process,
  started with one command.
- TypeScript, so the event and answer shapes coming off the SSE stream are typed once and
  checked everywhere.

**Rejected:**
- **Next.js.** Its strengths are SSR, routing conventions and Vercel deployment, and none
  of them apply. It would mean a second (Node) server alongside Python, or a static export
  that discards most of what Next adds. Worth revisiting only if the app is ever hosted
  publicly, or if Next.js experience is wanted for its own sake. That second reason is
  legitimate, but it is not a technical one.

## D3 — Three-pane conversational layout

*2026-10-24 · `frontend/src/`*

**Decided by:** user.

**Why (user's):** it should look and feel like an agentic conversational UI, as Claude
Code does in Claude Desktop:
- left: chats and chat history
- centre: the conversation
- right: tool calls, memory and similar

The user noted that not all of it is backed by features yet ("later we will do that").
The layout should exist now so the app looks finished and has places for those features
to land.

**Consequences:**
- The **memory** pane is a labelled placeholder. The agent has no cross-session memory
  (`docs/FUTURE_WORK.md`, "No cross-session memory").
- **Each question is still answered independently.** A chat groups questions for the
  user. It does not give the agent the earlier turns, and the UI says so. Feeding history
  into the agent would change the system under evaluation, and would let the browser
  produce answers the CLI cannot. That is a separate decision, not a UI detail.

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

---

## Benched questions

- **Chat history storage:** undecided. It will be designed together with conversation memory. Until then, chats live in the page's memory and are lost on reload, alongside the sample fixtures (D10). The server stores nothing (D5). The earlier D4 proposal of JSON files in `logs/web_chats/` was withdrawn by the user.
