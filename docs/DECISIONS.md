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

---

## Benched questions

- **Chat history storage:** undecided. Design it together with conversation memory. Until then the UI uses in-memory chats and sample fixtures.
- **Framework for memory / multi-turn:** LangGraph or others versus hand-rolled. Compare the options with the user when that work starts.
