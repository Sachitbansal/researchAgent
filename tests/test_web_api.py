"""
Offline tests for the browser UI's API (web/api.py) against a stubbed agent.

In:  a FastAPI TestClient over create_app, with a stub loop that emits scripted events
     and a config rooted at tmp_path — no models, no network, no real index.
Out: assertions on the corpus shape, the SSE order, that the server stores no chats, the
     busy and failed paths, and that figure serving cannot be walked out of its directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from web.api import create_app, figure_url, resolve_figure  # noqa: E402

PAPER = "2101_00001v1"
CHUNK = f"{PAPER}__c0003"


class StubLoop:
    """Stands in for AgentLoop: emits the loop's event kinds through on_event, returns a
    result shaped like loop.run's — including a non-serialisable object, as the real
    Conversation is."""

    def __init__(self, figure_path: str, fail: bool = False) -> None:
        self.on_event = None
        self.questions: List[str] = []
        self.figure_path = figure_path
        self.fail = fail
        self.on_event_during_run: List[Any] = []

    def _emit(self, kind: str, **fields: Any) -> None:
        if self.on_event is not None:
            self.on_event({"kind": kind, **fields})

    def run(self, question: str) -> Dict[str, Any]:
        self.questions.append(question)
        self.on_event_during_run.append(self.on_event)
        self._emit("thinking", iteration=0)
        self._emit("tool_start", iteration=0, tool_name="retrieve_evidence", args={"query": question})
        self._emit("tool_end", iteration=0, tool_name="retrieve_evidence", latency_ms=12, result={
            "chunks": [{"chunk_id": CHUNK, "paper_id": PAPER, "score": 4.2,
                        "chunk_type": "text", "text": "x" * 5000, "extra": "dropped"}],
            "sufficient_evidence": True,
        })
        self._emit("image", iteration=0, image_path=self.figure_path)
        if self.fail:
            return {"error": "provider_error", "detail": "upstream 502",
                    "partial": {"answer": "half an answer", "run_id": "run-x"}}
        self._emit("answering", iteration=1)
        return {
            "answer": f"Widgets are blue [{CHUNK}].",
            "question": question,
            "iterations": 2,
            "tool_calls": 1,
            "context_tokens": 1234,
            "stopped_because": "answered",
            "run_id": "run-1",
            "conversation": object(),   # must never reach the wire
        }


def _config(tmp_path: Path) -> SimpleNamespace:
    index = tmp_path / "index"
    figures = tmp_path / "cache" / "figures"
    (figures / PAPER).mkdir(parents=True)
    (figures / PAPER / f"{PAPER}__f01.png").write_bytes(b"\x89PNG fake")
    index.mkdir()
    # Neutral, invented topic: no real topic name belongs in a source file outside eval/.
    (index / "manifest.json").write_text(json.dumps({
        "embedding_model": "stub-model",
        "updated_at": "2026-01-01",
        "topics": {"topic_a": {"query": "topic a", "arxiv_query": "all:a"}},
        "papers": {PAPER: {
            "paper_id": PAPER, "arxiv_id": "2101.00001v1", "title": "On Widgets",
            "authors": ["A. Author"], "year": 2021, "categories": ["cs.LG"],
            "topic_tags": ["topic_a"], "n_chunks": 9, "n_figures": 1,
            "pdf_path": "/x/papers/2101_00001v1.pdf", "pdf_url": "https://arxiv.org/pdf/2101.00001v1",
            "parse_status": "ok",
        }},
    }), encoding="utf-8")
    return SimpleNamespace(
        paths=SimpleNamespace(manifest=index / "manifest.json", figures=figures, logs=tmp_path / "logs"),
        agent=SimpleNamespace(trace_text_chars=50),
    )


@pytest.fixture
def setup(tmp_path):
    config = _config(tmp_path)
    loop = StubLoop(str(config.paths.figures / PAPER / f"{PAPER}__f01.png"))
    app = create_app(lambda: (loop, config), dist_dir=tmp_path / "no_dist")
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, loop=loop, config=config, app=app)


def _frames(body: str) -> List[tuple]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def _ask(client, question: str):
    response = client.post("/api/ask", json={"question": question})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return _frames(response.text)


# ---------------------------------------------------------------- corpus


def test_papers_endpoint_shape(setup):
    data = setup.client.get("/api/papers").json()
    assert set(data) >= {"papers", "topics", "embedding_model", "updated_at"}
    paper = data["papers"][0]
    assert paper["paper_id"] == PAPER and paper["pdf_name"] == "2101_00001v1.pdf"
    assert {"title", "authors", "year", "n_chunks", "n_figures", "pdf_url", "topic_tags"} <= set(paper)
    assert data["topics"] == [{"tag": "topic_a", "label": "topic a", "arxiv_query": "all:a", "count": 1}]


def test_papers_without_an_index_is_structured(setup):
    setup.config.paths.manifest.unlink()
    data = setup.client.get("/api/papers").json()
    assert data["papers"] == [] and "error" in data


# ---------------------------------------------------------------- ask


def test_ask_streams_steps_then_answer_then_done(setup):
    frames = _ask(setup.client, "  What colour are widgets?  ")
    events = [e for e, _ in frames]

    assert events[-2:] == ["answer", "done"]
    assert set(events[:-2]) == {"step"}
    kinds = [p["kind"] for e, p in frames if e == "step"]
    assert kinds == ["thinking", "tool_start", "tool_end", "image", "answering"]

    tool_end = next(p for e, p in frames if e == "step" and p["kind"] == "tool_end")
    chunk = tool_end["result"]["chunks"][0]
    assert len(chunk["text"]) == 50 and "extra" not in chunk          # summarised
    image = next(p for e, p in frames if e == "step" and p["kind"] == "image")
    assert image["image_url"] == f"/api/figures/{PAPER}/{PAPER}__f01.png"

    answer = frames[-2][1]
    assert set(answer) == {"answer", "iterations", "tool_calls", "context_tokens",
                           "stopped_because", "run_id"}
    assert answer["run_id"] == "run-1"

    # The loop got the bare question, and its observer was set during the run only.
    assert setup.loop.questions == ["What colour are widgets?"]
    assert setup.loop.on_event_during_run[0] is not None and setup.loop.on_event is None
    assert not setup.app.state.lock.locked()

    served = setup.client.get(image["image_url"])
    assert served.status_code == 200 and served.content.startswith(b"\x89PNG")


def test_ask_is_stateless(setup):
    """Chat storage is undecided (docs/DECISIONS.md), so the server must keep none: each
    question reaches the agent bare, and nothing is written under logs/."""
    _ask(setup.client, "first question")
    _ask(setup.client, "second question")
    assert setup.loop.questions == ["first question", "second question"]
    logs = setup.config.paths.logs
    assert not logs.exists() or not any(logs.rglob("*"))
    assert setup.client.get("/api/chats").status_code == 404


def test_busy_when_the_lock_is_held(setup):
    setup.app.state.lock.acquire()
    try:
        frames = _ask(setup.client, "anything")
    finally:
        setup.app.state.lock.release()
    assert [e for e, _ in frames] == ["busy", "done"]
    assert setup.loop.questions == []
    assert setup.client.get("/api/status").json() == {"busy": False}


def test_failed_run_streams_failed_with_the_partial_answer(setup):
    setup.loop.fail = True
    frames = _ask(setup.client, "will fail")
    assert [e for e, _ in frames][-2:] == ["failed", "done"]
    failed = frames[-2][1]
    assert failed["error"] == "provider_error" and failed["answer"] == "half an answer"
    assert failed["detail"] == "upstream 502"
    assert not setup.app.state.lock.locked() and setup.loop.on_event is None


def test_loop_that_raises_still_frees_the_agent(setup):
    def boom(question):
        raise RuntimeError("kaput")
    setup.loop.run = boom
    frames = _ask(setup.client, "q")
    assert frames[-2][0] == "failed" and frames[-2][1]["error"] == "loop_crashed"
    assert not setup.app.state.lock.locked() and setup.loop.on_event is None


def test_empty_question_rejected(setup):
    response = setup.client.post("/api/ask", json={"question": "   "})
    assert response.status_code == 400 and response.json()["error"] == "empty_question"
    assert setup.loop.questions == []


# ---------------------------------------------------------------- figures


def test_figure_endpoint_rejects_traversal(setup, tmp_path):
    c = setup.client
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"secret")
    for url in (f"/api/figures/{PAPER}/..%2F..%2Fsecret.png",
                "/api/figures/%2E%2E/secret.png",
                "/api/figures/../secret.png",
                f"/api/figures/{PAPER}/%2Fetc%2Fpasswd",
                f"/api/figures/{PAPER}/missing.png",
                "/api/figures/x"):
        response = c.get(url)
        assert response.status_code == 404, url
        assert response.content != b"secret"


def test_resolve_figure_directly(setup, tmp_path):
    root = setup.config.paths.figures
    assert resolve_figure(root, PAPER, f"{PAPER}__f01.png") is not None
    assert resolve_figure(root, "..", "secret.png") is None
    assert resolve_figure(root, PAPER, "../../secret.png") is None
    assert resolve_figure(root, PAPER, str(tmp_path / "secret.png")) is None
    # A symlink inside the figures directory that points out of it is still refused.
    (tmp_path / "outside.png").write_bytes(b"x")
    (root / PAPER / "link.png").symlink_to(tmp_path / "outside.png")
    assert resolve_figure(root, PAPER, "link.png") is None


def test_figure_url_only_for_files_inside_the_figures_dir(setup, tmp_path):
    root = setup.config.paths.figures
    assert figure_url(root / PAPER / "a.png", root) == f"/api/figures/{PAPER}/a.png"
    assert figure_url(tmp_path / "elsewhere.png", root) is None
    assert figure_url(root / PAPER / ".." / ".." / "x.png", root) is None
    assert figure_url(None, root) is None


# ---------------------------------------------------------------- the page


def test_missing_build_explains_itself_and_api_404s_stay_json(setup):
    page = setup.client.get("/")
    assert page.status_code == 503 and "npm run build" in page.text
    missing = setup.client.get("/api/nope")
    assert missing.status_code == 404 and missing.json()["error"] == "not_found"


def test_built_frontend_is_served_with_spa_fallback(tmp_path):
    config = _config(tmp_path)
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>shell</html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    app = create_app(lambda: (StubLoop(""), config), dist_dir=dist)
    with TestClient(app) as client:
        assert client.get("/").text == "<html>shell</html>"
        assert client.get("/assets/app.js").text == "console.log(1)"
        assert client.get("/some/client/route").text == "<html>shell</html>"
