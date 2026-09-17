"""
A local demo server for the CLI: one page that asks a question and shows the corpus.

In:  HTTP requests. Out: the static page, /api/papers from the manifest, and /api/ask as
     a Server-Sent Events stream of the agent's progress followed by its answer.
Lives outside src/ and imports it, exactly as eval/ does. Nothing in src/ knows this
exists — the system under evaluation is still the CLI (.claude/CLAUDE.md, Hard
constraints). Standard library only; no web framework, no new dependency.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

STATIC = Path(__file__).resolve().parent / "static"
# An explicit map, not a path join: a request can only ever reach these three files.
SERVED = {
    "/": (STATIC / "index.html", "text/html; charset=utf-8"),
    "/static/style.css": (STATIC / "style.css", "text/css; charset=utf-8"),
    "/static/app.js": (STATIC / "app.js", "text/javascript; charset=utf-8"),
}

# The agent holds the bi-encoder, cross-encoder and NLI models, so it is built once at
# startup and shared. `LOCK` serialises whole requests: the retriever's dedup state and
# the loop's on_event slot are both per-conversation, and two questions in flight at
# once would interleave their events into each other's streams.
LOCK = threading.Lock()
AGENT: Dict[str, Any] = {}


def build_agent() -> None:
    """Load the models and the index once, before the first question arrives."""
    from agent.loop import AgentLoop
    from agent.tool_registry import build_full_registry
    from config import CFG
    from retrieval.embedder import Embedder
    from tools.retrieve_evidence import EvidenceRetriever

    CFG.paths.ensure()
    retriever = EvidenceRetriever(CFG, embedder=Embedder(CFG))
    AGENT["config"] = CFG
    AGENT["loop"] = AgentLoop(
        build_full_registry(retriever=retriever, config=CFG), config=CFG, retriever=retriever
    )


def read_corpus() -> Dict[str, Any]:
    """The indexed papers and the topics they were collected under, for the corpus view.

    Read fresh on every request rather than cached: indexing from the CLI while the
    server is up should show up on the next reload, not on the next restart.
    """
    from config import CFG

    path = CFG.paths.manifest
    if not path.exists():
        return {"papers": [], "topics": [], "error": "no index yet — run: python main.py index \"<topic>\""}

    manifest = json.loads(path.read_text(encoding="utf-8"))
    raw_topics = manifest.get("topics") or {}
    papers: List[Dict[str, Any]] = []
    for record in manifest.get("papers", {}).values():
        pdf_path = record.get("pdf_path") or ""
        papers.append({
            "paper_id": record.get("paper_id"),
            "arxiv_id": record.get("arxiv_id"),
            "title": record.get("title") or record.get("paper_id"),
            "authors": record.get("authors") or [],
            "year": record.get("year"),
            "categories": record.get("categories") or [],
            "topic_tags": record.get("topic_tags") or [],
            "n_chunks": record.get("n_chunks", 0),
            "n_figures": record.get("n_figures", 0),
            "pdf_name": Path(pdf_path).name if pdf_path else "",
            "pdf_url": record.get("pdf_url") or "",
            "parse_status": record.get("parse_status"),
        })
    papers.sort(key=lambda p: (str(p["title"]).lower()))

    topics = [
        {
            "tag": tag,
            # The topic as it was typed, which reads better than the slug it became.
            "label": (body or {}).get("query") or tag.replace("_", " "),
            "arxiv_query": (body or {}).get("arxiv_query", ""),
            "count": sum(1 for p in papers if tag in p["topic_tags"]),
        }
        for tag, body in raw_topics.items()
    ]
    topics.sort(key=lambda t: -t["count"])
    return {
        "papers": papers,
        "topics": topics,
        "embedding_model": manifest.get("embedding_model", ""),
        "updated_at": manifest.get("updated_at", ""),
    }


class Handler(BaseHTTPRequestHandler):
    """Three routes: the page, the corpus, and one streamed question."""

    server_version = "sciagent-demo"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")

    # ---- helpers

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Dict[str, Any], code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _sse(self, event: str, payload: Dict[str, Any]) -> bool:
        """Write one SSE frame. False once the browser has gone away."""
        try:
            self.wfile.write(f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    # ---- routes

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's naming
        route = urlparse(self.path)
        path = route.path.rstrip("/") or "/"

        if path in SERVED:
            file_path, content_type = SERVED[path]
            if not file_path.exists():
                self._send(404, b"missing static file", "text/plain; charset=utf-8")
                return
            self._send(200, file_path.read_bytes(), content_type)
            return

        if path == "/api/papers":
            self._send_json(read_corpus())
            return

        if path == "/api/ask":
            question = (parse_qs(route.query).get("q") or [""])[0].strip()
            self.stream_answer(question)
            return

        self._send(404, b"no route here", "text/plain; charset=utf-8")

    def stream_answer(self, question: str) -> None:
        """Run one question, sending each loop event as it happens, then the answer."""
        from agent.trace import summarise_result

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        if not question:
            self._sse("failed", {"error": "empty_question", "detail": "type a question first"})
            self._sse("done", {})
            return

        config = AGENT["config"]
        text_chars = int(config.agent.trace_text_chars)
        alive = threading.Event()
        alive.set()

        def on_event(event: Dict[str, Any]) -> None:
            # Full chunk text would be kilobytes per frame; the trace summariser is
            # already the project's answer to that, and it is config-truncated.
            payload = dict(event)
            if "result" in payload:
                payload["result"] = summarise_result(payload["result"], text_chars)
            if not self._sse("step", payload):
                alive.clear()

        loop = AGENT["loop"]
        with LOCK:
            loop.on_event = on_event
            try:
                result = loop.run(question)
            except Exception as exc:  # the loop is meant not to raise; a page must not 500
                result = {"error": "loop_crashed", "detail": f"{type(exc).__name__}: {exc}"}
            finally:
                loop.on_event = None

        if not alive.is_set():
            return

        if "error" in result:
            partial = result.get("partial") or {}
            self._sse("failed", {
                "error": result["error"],
                "detail": result.get("detail", ""),
                "answer": partial.get("answer", ""),
            })
        else:
            # Whitelisted: loop.run also returns the Conversation object, which is not
            # JSON-serialisable and has no business on the wire.
            self._sse("answer", {
                "answer": result["answer"],
                "iterations": result["iterations"],
                "tool_calls": result["tool_calls"],
                "context_tokens": result["context_tokens"],
                "stopped_because": result["stopped_because"],
                "run_id": result["run_id"],
            })
        self._sse("done", {})


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Load the models, then answer requests until interrupted."""
    print("loading models and the index ...", flush=True)
    build_agent()
    corpus = read_corpus()
    print(f"ready — {len(corpus['papers'])} paper(s) indexed", flush=True)
    print(f"open http://{host}:{port}", flush=True)
    httpd = ThreadingHTTPServer((host, port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo web UI for the research assistant.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.host, args.port)
