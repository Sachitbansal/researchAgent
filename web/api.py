"""
The browser UI's HTTP API: the corpus, figure images, chat threads, one streamed question.

In:  a builder returning (AgentLoop, Config), called once at startup; the loop's
     ThreadStore (`loop.store`) holds the threads, its LongTermMemory (`loop.memory`) the
     memories. Then HTTP requests.
Out: a FastAPI app. POST /api/ask streams a `thread` frame, the loop's own events as SSE,
     the answer, then a `memory` frame once memory is written (D12, D19); with
     `approve_fetch` it runs the user's approved fetch instead. Memories are listed,
     edited, added (preferences only), deleted and restored under /api/memories for the
     Memories page (D23). Lives outside src/ and imports it, as eval/.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Optional, Tuple
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from web.corpus_view import read_corpus

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIST = ROOT / "frontend" / "dist"

# A path segment that cannot be "..", ".", hidden, or carry a separator. Figure files are
# written as <figures>/<paper_id>/<paper_id>__fNN.png, so two such segments name all of
# them, and nothing else.
SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_+-][A-Za-z0-9._+-]*$")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

# Only these leave the server. loop.run also returns the Conversation object, which is
# not JSON-serialisable, and trace_records, which the steps already cover.
ANSWER_FIELDS = ("answer", "iterations", "tool_calls", "context_tokens", "stopped_because",
                 "run_id", "thread_id", "turn_seq", "fetch_proposal", "approved_fetch")

MISSING_BUILD = """<!doctype html><meta charset="utf-8"><title>sciagent</title>
<body style="font:16px/1.6 system-ui,sans-serif;max-width:38rem;margin:4rem auto;padding:0 1rem">
<h1 style="font-size:1.3rem">The frontend has not been built yet</h1>
<p>The API is up, but there is no page to serve. Build it once:</p>
<pre style="background:#8881;padding:.8rem 1rem;border-radius:8px">cd frontend && npm install && npm run build</pre>
<p>then reload. For frontend development, run <code>npm run dev</code> in
<code>frontend/</code> instead and open the address it prints.</p></body>"""


class ApproveFetch(BaseModel):
    """The user's click on "Fetch papers": the query to fetch, possibly edited by them."""

    query: str


class AskBody(BaseModel):
    """Module level, not inside create_app: with postponed annotations FastAPI resolves
    the model by name, and a local class would turn the body into a query parameter.

    Either a question, or `approve_fetch` with a thread_id. The approval's wording is
    built here from src/prompts/fetch_approval.md, never sent by the page (D26)."""

    question: Optional[str] = None
    thread_id: Optional[str] = None
    approve_fetch: Optional[ApproveFetch] = None


class NewThreadBody(BaseModel):
    title: Optional[str] = None


class MemoryEditBody(BaseModel):
    text: str


class NewMemoryBody(BaseModel):
    kind: str
    text: str


Builder = Callable[[], Tuple[Any, Any]]


def _err(status: int, error: str, detail: str) -> JSONResponse:
    """The tool convention, carried over: errors are structured, never a bare string."""
    return JSONResponse({"error": error, "detail": detail}, status_code=status)


def sse(event: str, payload: Dict[str, Any]) -> str:
    """One Server-Sent Events frame. `default=str` because tool args come from the model
    and a stray non-JSON value must degrade to text, not end the stream."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def figure_url(image_path: Any, figures_root: Path) -> Optional[str]:
    """The URL the browser can load an image from, or None when it is not ours to serve.

    Only files inside the figures directory get one. inspect_figure can also open a file
    the user named anywhere on disk; that one is shown to the model, not to the page.
    """
    try:
        rel = Path(str(image_path)).resolve().relative_to(Path(figures_root).resolve())
    except (ValueError, OSError, TypeError):
        return None
    if len(rel.parts) != 2 or not all(SAFE_SEGMENT.match(part) for part in rel.parts):
        return None
    return f"/api/figures/{quote(rel.parts[0])}/{quote(rel.parts[1])}"


def resolve_figure(figures_root: Path, paper_id: str, filename: str) -> Optional[Path]:
    """The file behind /api/figures/<paper_id>/<filename>, or None.

    Belt and braces: each segment must be a plain name, and the resolved path — after
    symlinks — must still sit inside the figures directory.
    """
    if not (SAFE_SEGMENT.match(paper_id) and SAFE_SEGMENT.match(filename)):
        return None
    root = Path(figures_root).resolve()
    candidate = (root / paper_id / filename).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None
    if candidate.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    return candidate


def step_payload(event: Dict[str, Any], text_chars: int, figures_root: Path,
                 started: float) -> Dict[str, Any]:
    """A loop event as the browser receives it: results summarised, images given a URL.

    Full chunk text would be kilobytes per frame; the trace summariser is already the
    project's answer to that, and it is config-truncated.
    """
    from agent.trace import summarise_result

    payload = dict(event)
    if "result" in payload:
        payload["result"] = summarise_result(payload["result"], text_chars)
    if payload.get("kind") == "image":
        payload["image_url"] = figure_url(payload.get("image_path"), figures_root)
    payload["t_ms"] = int((time.perf_counter() - started) * 1000)
    return payload


def final_event(result: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """`answer` with the whitelisted fields, or `failed` with whatever partial answer
    the loop had written before it stopped."""
    if "error" in result:
        partial = result.get("partial") or {}
        return "failed", {
            "error": result["error"],
            "detail": result.get("detail", ""),
            "answer": partial.get("answer", "") or "",
            "run_id": partial.get("run_id"),
            "thread_id": result.get("thread_id"),
            "turn_seq": result.get("turn_seq"),
        }
    return "answer", {field: result.get(field) for field in ANSWER_FIELDS}


def thread_payload(thread: Dict[str, Any], figures_root: Path) -> Dict[str, Any]:
    """A stored thread as the page redraws it: image steps get their URL back.

    Steps are stored without URLs (src/ knows nothing of this server), so figure_url is
    applied here, on the way out, with the same rule as a live stream.
    """
    turns = []
    for turn in thread.get("turns") or []:
        steps = []
        for step in turn.get("steps") or []:
            if isinstance(step, dict) and step.get("kind") == "image":
                step = {**step, "image_url": figure_url(step.get("image_path"), figures_root)}
            steps.append(step)
        turns.append({**turn, "steps": steps})
    return {**thread, "turns": turns}


def _store_status(result: Dict[str, Any]) -> int:
    return {"bad_thread_id": 400, "thread_not_found": 404}.get(result.get("error"), 503)


def create_app(builder: Builder, dist_dir: Path = DEFAULT_DIST) -> FastAPI:
    """The app. `builder` loads the models and returns (loop, config); tests pass a stub."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Once, before the first request: the bi-encoder, cross-encoder and NLI models
        # take seconds to load and the index has to be read from disk.
        loop, config = builder()
        app.state.loop = loop
        app.state.config = config
        # One store, owned by the loop: the loop appends turns, these routes read them.
        app.state.store = getattr(loop, "store", None)
        # Long-term memory: the loop reads and writes it; these routes list and delete.
        app.state.memory = getattr(loop, "memory", None)
        app.state.memory_store = getattr(app.state.memory, "store", None)
        yield

    app = FastAPI(title="sciagent", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def bad_body(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's own 422 body is {"detail": [...]}; every error here is {error, detail}.
        problems = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg')}"
                             for e in exc.errors())
        return _err(422, "bad_request", problems or "the request body is not valid")
    # One agent, one question at a time. The retriever's dedup state and the loop's
    # on_event slot are both per-run, so two runs at once would interleave their events
    # into each other's streams. Acquired without blocking: a second asker is told the
    # agent is busy straight away rather than left staring at a stalled stream.
    app.state.lock = threading.Lock()
    # The thread the current run is appending to, so it cannot be deleted mid-run.
    app.state.running_thread = None
    dist = Path(dist_dir)

    # ---------------------------------------------------------------- corpus, status

    @app.get("/api/papers")
    def papers(request: Request) -> Dict[str, Any]:
        return read_corpus(request.app.state.config)

    @app.get("/api/status")
    def status(request: Request) -> Dict[str, Any]:
        return {"busy": request.app.state.lock.locked()}

    @app.get("/api/figures/{paper_id}/{filename}")
    def figure(request: Request, paper_id: str, filename: str) -> Any:
        path = resolve_figure(Path(request.app.state.config.paths.figures), paper_id, filename)
        if path is None:
            return _err(404, "not_found", "no such figure")
        return FileResponse(path)

    # ---------------------------------------------------------------- threads

    def _store(request: Request) -> Any:
        return request.app.state.store

    def _no_store() -> JSONResponse:
        return _err(503, "no_thread_store", "this server was started without a thread store")

    @app.get("/api/threads")
    def list_threads(request: Request) -> Any:
        store = _store(request)
        if store is None:
            return _no_store()
        result = store.list_threads()
        return _err(503, result["error"], result.get("detail", "")) if "error" in result else result

    @app.post("/api/threads")
    def create_thread(request: Request, body: Optional[NewThreadBody] = None) -> Any:
        store = _store(request)
        if store is None:
            return _no_store()
        result = store.create_thread((body.title if body else None) or "")
        if "error" in result:
            return _err(503, result["error"], result.get("detail", ""))
        return JSONResponse(result, status_code=201)

    @app.get("/api/threads/{thread_id}")
    def get_thread(request: Request, thread_id: str) -> Any:
        store = _store(request)
        if store is None:
            return _no_store()
        result = store.get_thread(thread_id)
        if "error" in result:
            return _err(_store_status(result), result["error"], result.get("detail", ""))
        return thread_payload(result, Path(request.app.state.config.paths.figures))

    @app.delete("/api/threads/{thread_id}")
    def delete_thread(request: Request, thread_id: str) -> Any:
        store = _store(request)
        if store is None:
            return _no_store()
        state = request.app.state
        if state.lock.locked() and state.running_thread == thread_id:
            return _err(409, "thread_busy", "the agent is answering in this chat; wait for it to finish")
        result = store.delete_thread(thread_id)
        if "error" in result:
            return _err(_store_status(result), result["error"], result.get("detail", ""))
        return result

    # ---------------------------------------------------------------- long-term memory

    def _memory_status(result: Dict[str, Any]) -> int:
        return {"bad_memory_id": 400, "empty_text": 400, "text_too_long": 400,
                "bad_memory_kind": 400, "findings_need_evidence": 400,
                "memory_not_found": 404, "memory_inactive": 409, "memory_active": 409,
                "evidence_changed": 409}.get(result.get("error"), 503)

    def _with_models(state: Any, work: Callable[[], Dict[str, Any]], status: int = 200) -> Any:
        """Run an edit or an add, which embed (and NLI-check) on the run's own models, only
        while no run holds them: the fast tokenizer is not safe to share across threads,
        and a collision would break the run's retrieval, not just the edit. Refused, not
        queued, as a second question is (D23)."""
        if not state.lock.acquire(blocking=False):
            return _err(409, "agent_busy", "the agent is answering a question; try again when it finishes")
        try:
            result = work()
        finally:
            state.lock.release()
        return _memory_result(result, status(result) if callable(status) else status)

    def _no_memory() -> JSONResponse:
        return _err(503, "no_memory_store", "this server was started without long-term memory")

    def _memory_result(result: Dict[str, Any], status: int = 200) -> Any:
        if "error" in result:
            return _err(_memory_status(result), result["error"], result.get("detail", ""))
        return JSONResponse(result, status_code=status)

    def _thread_ids(state: Any) -> Optional[set]:
        """Ids of the chats that still exist, so a memory can link to its source chat.
        None when they cannot be read: the page then shows the link without the check."""
        store = state.store
        if store is None:
            return None
        listed = store.list_threads()
        if "error" in listed:
            return None
        return {t["id"] for t in listed.get("threads") or []}

    @app.get("/api/memories")
    def list_memories(request: Request, all: int = 0) -> Any:  # noqa: A002 - the query name
        """Active memories, most recently used first; `?all=1` adds the forgotten and
        invalidated ones with their reasons. Strength is the decay score now (D20). With
        a LongTermMemory behind it, each row also carries what the Memories page shows
        (LongTermMemory.catalogue, D23) and the reply carries the forgetting `settings`."""
        state = request.app.state
        if state.memory_store is None:
            return _no_memory()
        catalogue = getattr(state.memory, "catalogue", None)
        if callable(catalogue):
            result = catalogue(_thread_ids(state))
            if "error" in result:
                return _err(503, result["error"], result.get("detail", ""))
            rows = result["memories"]
            if not all:
                rows = [m for m in rows if m.get("active")]
            return {"memories": rows, "settings": result.get("settings", {})}
        result = state.memory_store.list(include_inactive=bool(all))
        if "error" in result:
            return _err(503, result["error"], result.get("detail", ""))
        memories = result["memories"]
        strengths = {}
        strength_of = getattr(state.memory, "strengths", None)
        if callable(strength_of):
            strengths = strength_of([m for m in memories if m.get("active")])
        return {"memories": [{**m, "strength": strengths.get(m["id"])} for m in memories]}

    @app.post("/api/memories")
    def add_memory(request: Request, body: NewMemoryBody) -> Any:
        """A preference typed by the user. Findings cannot be added by hand: a finding
        is only ever a claim NLI found in the passages it cites (D19, D23)."""
        state = request.app.state
        if state.memory_store is None or state.memory is None:
            return _no_memory()
        if body.kind == "finding":
            return _err(400, "findings_need_evidence",
                        "findings are kept only from answers their passages support; add a preference instead")
        if body.kind != "preference":
            return _err(400, "bad_memory_kind", f"kind must be preference, got {body.kind!r}")
        return _with_models(state, lambda: state.memory.add_preference(body.text),
                            lambda r: 201 if r.get("outcome") == "added" else 200)

    @app.patch("/api/memories/{memory_id}")
    def edit_memory(request: Request, memory_id: str, body: MemoryEditBody) -> Any:
        """The user's rewording. A finding is re-checked against its passages by NLI; an
        unsupported edit is kept, marked unverified (D23). Not a use."""
        state = request.app.state
        if state.memory_store is None or state.memory is None:
            return _no_memory()
        return _with_models(state, lambda: state.memory.edit(memory_id, body.text))

    @app.post("/api/memories/{memory_id}/restore")
    def restore_memory(request: Request, memory_id: str) -> Any:
        """Undo a forget or an invalidation; recency resets to now (D23)."""
        state = request.app.state
        if state.memory_store is None or state.memory is None:
            return _no_memory()
        return _memory_result(state.memory.restore(memory_id))

    @app.delete("/api/memories/{memory_id}")
    def delete_memory(request: Request, memory_id: str) -> Any:
        """Manual forget: soft, with reason `user`, never injected again (D20)."""
        state = request.app.state
        if state.memory_store is None:
            return _no_memory()
        result = state.memory_store.forget_one(memory_id, "user")
        if "error" in result:
            return _err(_memory_status(result), result["error"], result.get("detail", ""))
        return result

    # ---------------------------------------------------------------- ask

    def run_question(state: Any, question: str, thread_id: str,
                     push: Callable[[Optional[str], Dict[str, Any]], None],
                     approval: Optional[Dict[str, Any]] = None) -> None:
        """The whole run, on a worker thread that owns the lock until it is over.

        The thread, not the response, owns the run: a browser that closes the tab ends
        the stream but not the question, which finishes (and writes its trace, as every
        run does) before the agent is freed.

        Order (D19): the answer frame goes out first, then long-term memory is written,
        still under the lock so the next question cannot read memory halfway through a
        write, then the `memory` frame. Release comes before `done`, so a client that
        asks again the moment it sees `done` is never told the agent is busy. A memory
        failure is reported in its frame and never turns an answered run into a failure.
        """
        loop, config = state.loop, state.config
        text_chars = int(config.agent.trace_text_chars)
        figures_root = Path(config.paths.figures)
        started = time.perf_counter()

        def on_event(event: Dict[str, Any]) -> None:
            push("step", step_payload(event, text_chars, figures_root, started))

        kind, payload = "failed", {"error": "loop_crashed", "detail": "the run did not finish", "answer": ""}
        sent = False
        try:
            loop.on_event = on_event
            try:
                # The question as typed, plus the thread it belongs to. Earlier turns are
                # added by the loop itself from the store (memory.history), never pasted
                # in here, so the browser and `main.py ask --thread` send the same thing.
                if approval is None:
                    result = loop.run(question, thread_id=thread_id)
                else:
                    # The approved fetch is an ordinary turn of the same thread, so the
                    # history gives the agent the question it is to answer (D26).
                    result = loop.run(question, thread_id=thread_id, approval=approval)
            except Exception as exc:  # the loop is meant not to raise; a page must not hang
                result = {"error": "loop_crashed", "detail": f"{type(exc).__name__}: {exc}",
                          "thread_id": thread_id}
            finally:
                loop.on_event = None
            kind, payload = final_event(result)
            push(kind, payload)
            sent = True
            remember = getattr(loop, "remember", None)
            if callable(remember):
                try:
                    report = remember(result)
                except Exception as exc:  # remember is meant not to raise
                    report = {"error": "memory_write_failed", "detail": f"{type(exc).__name__}: {exc}"}
                if report is not None:
                    push("memory", report)
        finally:
            state.running_thread = None
            state.lock.release()
            if not sent:
                push(kind, payload)
            push("done", {})
            push(None, {})

    @app.post("/api/ask")
    async def ask(request: Request, body: AskBody) -> Any:
        """POST, not GET + EventSource: EventSource reconnects on its own when a stream
        ends, which would silently re-run the whole question.

        Every question belongs to a thread (docs/DECISIONS.md D12). Without a thread_id a
        new one is created, named after the question, and the first frame says which.
        Frames: `thread`, `step`*, `answer` | `failed`, `memory`?, `done`; or `busy`, `done`.

        `approve_fetch: {query}` instead of a question runs the fetch the user approved,
        in the thread whose answer proposed it. All of it is checked before the stream
        opens, so a bad approval is a plain 400, not a frame.
        """
        state = request.app.state
        question = (body.question or "").strip()
        approval: Optional[Dict[str, Any]] = None
        if body.approve_fetch is not None:
            if question:
                return _err(400, "question_and_approval",
                            "send a question or an approved fetch, not both")
            if body.thread_id is None:
                return _err(400, "approval_needs_thread",
                            "an approved fetch continues the chat that proposed it; send its thread_id")
            from agent.fetch_proposal import approval_request

            approval = approval_request(body.approve_fetch.query, state.config)
            if "error" in approval:
                return _err(400, approval["error"], approval.get("detail", ""))
            question = approval["question"]
        if not question:
            return _err(400, "empty_question", "type a question first")

        if state.store is None:
            return _no_store()
        thread_id, title = body.thread_id, ""
        if thread_id is not None:
            # Checked before the stream opens, so a bad id is a plain 400/404, not a frame.
            found = state.store.get_thread(thread_id)
            if "error" in found:
                return _err(_store_status(found), found["error"], found.get("detail", ""))
            title = found["title"]

        async def stream() -> Any:
            if not state.lock.acquire(blocking=False):
                # Before any thread is created: a refused question leaves nothing behind.
                yield sse("busy", {"detail": "the agent is answering another question"})
                yield sse("done", {})
                return

            target, created, name = thread_id, False, title
            if target is None:
                try:
                    from memory.threads import title_from

                    made = state.store.create_thread(
                        title_from(question, int(state.config.agent.thread_title_chars)))
                except BaseException:
                    state.lock.release()
                    raise
                if "error" in made:
                    state.lock.release()
                    yield sse("failed", {"error": made["error"], "detail": made.get("detail", ""),
                                         "answer": "", "run_id": None})
                    yield sse("done", {})
                    return
                target, created, name = made["id"], True, made["title"]
            state.running_thread = target

            aloop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            # Queued, not yielded: a yield here would be a point where a disconnect could
            # close the stream with the lock held and no worker to release it.
            queue.put_nowait(("thread", {"thread_id": target, "created": created, "title": name}))

            def push(event: Optional[str], payload: Dict[str, Any]) -> None:
                try:
                    aloop.call_soon_threadsafe(queue.put_nowait, (event, payload))
                except RuntimeError:
                    pass  # the event loop has closed (server shutting down); nobody to tell

            # No await between acquiring and starting, so nothing can cancel us in the
            # gap and leave the lock held with no thread to release it.
            try:
                threading.Thread(target=run_question, args=(state, question, target, push, approval),
                                 name="agent-run", daemon=True).start()
            except BaseException:
                state.running_thread = None
                state.lock.release()
                raise

            while True:
                event, payload = await queue.get()
                if event is None:
                    return
                yield sse(event, payload)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---------------------------------------------------------------- the page

    @app.get("/{full_path:path}", include_in_schema=False)
    def page(full_path: str) -> Any:
        # An unknown /api/ route is a 404 with a body the client can read, not the SPA
        # shell with a 200 that would parse as garbage.
        if full_path == "api" or full_path.startswith("api/"):
            return _err(404, "not_found", f"no route /{full_path}")
        index = dist / "index.html"
        if not index.is_file():
            return HTMLResponse(MISSING_BUILD, status_code=503)
        if full_path:
            root = dist.resolve()
            candidate = (root / full_path).resolve()
            if candidate.is_relative_to(root) and candidate.is_file():
                return FileResponse(candidate)
        return FileResponse(index)

    return app
