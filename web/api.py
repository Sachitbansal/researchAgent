"""
The browser UI's HTTP API: the corpus, figure images, and one streamed question.

In:  a builder returning (AgentLoop, Config), called once at startup, and HTTP requests.
Out: a FastAPI app. POST /api/ask streams the loop's own events as SSE, then the answer.
     Stateless: no chat is stored. Lives outside src/ and imports it, as eval/ does; the
     question goes to loop.run bare, exactly as `python main.py ask` sends it.
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
ANSWER_FIELDS = ("answer", "iterations", "tool_calls", "context_tokens", "stopped_because", "run_id")

MISSING_BUILD = """<!doctype html><meta charset="utf-8"><title>sciagent</title>
<body style="font:16px/1.6 system-ui,sans-serif;max-width:38rem;margin:4rem auto;padding:0 1rem">
<h1 style="font-size:1.3rem">The frontend has not been built yet</h1>
<p>The API is up, but there is no page to serve. Build it once:</p>
<pre style="background:#8881;padding:.8rem 1rem;border-radius:8px">cd frontend && npm install && npm run build</pre>
<p>then reload. For frontend development, run <code>npm run dev</code> in
<code>frontend/</code> instead and open the address it prints.</p></body>"""


class AskBody(BaseModel):
    """Module level, not inside create_app: with postponed annotations FastAPI resolves
    the model by name, and a local class would turn the body into a query parameter."""

    question: str


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
        }
    return "answer", {field: result.get(field) for field in ANSWER_FIELDS}


def create_app(builder: Builder, dist_dir: Path = DEFAULT_DIST) -> FastAPI:
    """The app. `builder` loads the models and returns (loop, config); tests pass a stub."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Once, before the first request: the bi-encoder, cross-encoder and NLI models
        # take seconds to load and the index has to be read from disk.
        loop, config = builder()
        app.state.loop = loop
        app.state.config = config
        yield

    app = FastAPI(title="sciagent", lifespan=lifespan)
    # One agent, one question at a time. The retriever's dedup state and the loop's
    # on_event slot are both per-run, so two runs at once would interleave their events
    # into each other's streams. Acquired without blocking: a second asker is told the
    # agent is busy straight away rather than left staring at a stalled stream.
    app.state.lock = threading.Lock()
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

    # ---------------------------------------------------------------- ask

    def run_question(state: Any, question: str,
                     push: Callable[[Optional[str], Dict[str, Any]], None]) -> None:
        """The whole run, on a worker thread that owns the lock until it is over.

        The thread, not the response, owns the run: a browser that closes the tab ends
        the stream but not the question, which finishes (and writes its trace, as every
        run does) before the agent is freed. Release comes before the last frames, so a
        client that asks again the moment it sees `done` is never told the agent is busy.
        """
        loop, config = state.loop, state.config
        text_chars = int(config.agent.trace_text_chars)
        figures_root = Path(config.paths.figures)
        started = time.perf_counter()

        def on_event(event: Dict[str, Any]) -> None:
            push("step", step_payload(event, text_chars, figures_root, started))

        kind, payload = "failed", {"error": "loop_crashed", "detail": "the run did not finish", "answer": ""}
        try:
            loop.on_event = on_event
            try:
                # The bare question, exactly as `python main.py ask` passes it. Nothing
                # from earlier in the browser's chat is ever pasted in (docs/DECISIONS.md, D3).
                result = loop.run(question)
            except Exception as exc:  # the loop is meant not to raise; a page must not hang
                result = {"error": "loop_crashed", "detail": f"{type(exc).__name__}: {exc}"}
            finally:
                loop.on_event = None
            kind, payload = final_event(result)
        finally:
            state.lock.release()
            push(kind, payload)
            push("done", {})
            push(None, {})

    @app.post("/api/ask")
    async def ask(request: Request, body: AskBody) -> Any:
        """POST, not GET + EventSource: EventSource reconnects on its own when a stream
        ends, which would silently re-run the whole question.

        Stateless on purpose. Where chat history lives is undecided (docs/DECISIONS.md,
        benched questions) — it will be designed with conversation memory — so the
        server keeps nothing between questions beyond the trace every run writes.
        """
        question = body.question.strip()
        if not question:
            return _err(400, "empty_question", "type a question first")

        state = request.app.state

        async def stream() -> Any:
            if not state.lock.acquire(blocking=False):
                yield sse("busy", {"detail": "the agent is answering another question"})
                yield sse("done", {})
                return

            aloop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def push(event: Optional[str], payload: Dict[str, Any]) -> None:
                try:
                    aloop.call_soon_threadsafe(queue.put_nowait, (event, payload))
                except RuntimeError:
                    pass  # the event loop has closed (server shutting down); nobody to tell

            # No await between acquiring and starting, so nothing can cancel us in the
            # gap and leave the lock held with no thread to release it.
            try:
                threading.Thread(target=run_question, args=(state, question, push),
                                 name="agent-run", daemon=True).start()
            except BaseException:
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
