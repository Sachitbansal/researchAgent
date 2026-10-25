"""
Offline tests for the browser UI's API (web/api.py) against a stubbed agent.

In:  a FastAPI TestClient over create_app, with a stub loop that emits scripted events
     and appends turns to a real ThreadStore, and a config rooted at tmp_path — no models,
     no network, no real index.
Out: assertions on the corpus shape, the SSE order (`thread` first), the threads routes,
     the busy and failed paths, and that figure serving cannot be walked out of its directory.
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

from memory.threads import ThreadStore  # noqa: E402
from web.api import create_app, figure_url, resolve_figure  # noqa: E402

PAPER = "2101_00001v1"
CHUNK = f"{PAPER}__c0003"


class StubLoop:
    """Stands in for AgentLoop: emits the loop's event kinds through on_event, appends the
    turn to its store when given a thread (as AgentLoop does), and returns a result shaped
    like loop.run's — including a non-serialisable object, as the real Conversation is."""

    def __init__(self, figure_path: str, store: Any = None, fail: bool = False) -> None:
        self.on_event = None
        self.questions: List[str] = []
        self.thread_ids: List[Any] = []
        self.figure_path = figure_path
        self.store = store
        self.fail = fail
        self.on_event_during_run: List[Any] = []
        self.approvals: List[Any] = []

    def _emit(self, kind: str, **fields: Any) -> None:
        if self.on_event is not None:
            self.on_event({"kind": kind, **fields})

    def _save(self, thread_id: Any, question: str, answer: str, stats: Dict[str, Any]) -> Any:
        if thread_id is None or self.store is None:
            return None
        saved = self.store.append_turn(
            thread_id, question, answer, stats=stats, notes=["retrieve_evidence(...) → 1 chunk"],
            steps=[{"kind": "thinking", "iteration": 0},
                   {"kind": "image", "iteration": 0, "image_path": self.figure_path}],
            run_id=stats.get("run_id"))
        return saved.get("seq")

    def run(self, question: str, thread_id: Any = None, **kwargs: Any) -> Dict[str, Any]:
        # Only an approved fetch passes `approval`; a plain question passes nothing more.
        if "approval" in kwargs:
            self.approvals.append(kwargs["approval"])
        self.questions.append(question)
        self.thread_ids.append(thread_id)
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
            seq = self._save(thread_id, question, "half an answer",
                             {"run_id": "run-x", "error": "provider_error"})
            return {"error": "provider_error", "detail": "upstream 502",
                    "partial": {"answer": "half an answer", "run_id": "run-x"},
                    "thread_id": thread_id, "turn_seq": seq}
        self._emit("answering", iteration=1)
        answer = f"Widgets are blue [{CHUNK}]."
        seq = self._save(thread_id, question, answer, {"run_id": "run-1", "iterations": 2})
        return {
            "answer": answer,
            "question": question,
            "iterations": 2,
            "tool_calls": 1,
            "context_tokens": 1234,
            "stopped_because": "answered",
            "run_id": "run-1",
            "thread_id": thread_id,
            "turn_seq": seq,
            "fetch_proposal": {"query": "absent subject", "searched_for": "absent subject",
                               "best_score": -7.5, "closest_topics": ["topic_a"]},
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
        agent=SimpleNamespace(trace_text_chars=50, thread_title_chars=24,
                              fetch_query_max_chars=40),
        # The real prompt files: the approval's wording must come from them (D26).
        prompt=lambda name: (ROOT / "src" / "prompts" / f"{name}.md").read_text(encoding="utf-8"),
    )


@pytest.fixture
def setup(tmp_path):
    config = _config(tmp_path)
    store = ThreadStore(tmp_path / "threads.sqlite3")
    loop = StubLoop(str(config.paths.figures / PAPER / f"{PAPER}__f01.png"), store=store)
    app = create_app(lambda: (loop, config), dist_dir=tmp_path / "no_dist")
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, loop=loop, config=config, app=app, store=store)


def _frames(body: str) -> List[tuple]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def _ask(client, question: str, thread_id: Any = None):
    body: Dict[str, Any] = {"question": question}
    if thread_id is not None:
        body["thread_id"] = thread_id
    response = client.post("/api/ask", json=body)
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


def test_ask_streams_thread_then_steps_then_answer_then_done(setup):
    frames = _ask(setup.client, "  What colour are widgets?  ")
    events = [e for e, _ in frames]

    assert events[0] == "thread"
    assert events[-2:] == ["answer", "done"]
    assert set(events[1:-2]) == {"step"}
    kinds = [p["kind"] for e, p in frames if e == "step"]
    assert kinds == ["thinking", "tool_start", "tool_end", "image", "answering"]

    tool_end = next(p for e, p in frames if e == "step" and p["kind"] == "tool_end")
    chunk = tool_end["result"]["chunks"][0]
    assert len(chunk["text"]) == 50 and "extra" not in chunk          # summarised
    image = next(p for e, p in frames if e == "step" and p["kind"] == "image")
    assert image["image_url"] == f"/api/figures/{PAPER}/{PAPER}__f01.png"

    answer = frames[-2][1]
    assert set(answer) == {"answer", "iterations", "tool_calls", "context_tokens",
                           "stopped_because", "run_id", "thread_id", "turn_seq",
                           "fetch_proposal", "approved_fetch"}
    assert answer["run_id"] == "run-1"
    assert answer["thread_id"] == frames[0][1]["thread_id"] and answer["turn_seq"] == 1

    # The loop got the question as typed, and its observer was set during the run only.
    assert setup.loop.questions == ["What colour are widgets?"]
    assert setup.loop.on_event_during_run[0] is not None and setup.loop.on_event is None
    assert not setup.app.state.lock.locked()

    served = setup.client.get(image["image_url"])
    assert served.status_code == 200 and served.content.startswith(b"\x89PNG")


def test_the_server_never_pastes_history_into_the_question(setup):
    """History is the loop's job (memory.history), from the store. The server passes the
    question as typed plus the thread id, and writes nothing under logs/ itself."""
    first = _ask(setup.client, "first question")
    tid = first[0][1]["thread_id"]
    _ask(setup.client, "second question", thread_id=tid)
    assert setup.loop.questions == ["first question", "second question"]
    assert setup.loop.thread_ids == [tid, tid]
    logs = setup.config.paths.logs
    assert not logs.exists() or not any(logs.rglob("*"))
    assert setup.client.get("/api/chats").status_code == 404


# ---------------------------------------------------------------- threads


def test_ask_without_a_thread_creates_one_named_after_the_question(setup):
    frames = _ask(setup.client, "How are widgets coloured in practice, at scale?")
    event, payload = frames[0]
    assert event == "thread" and payload["created"] is True
    assert payload["title"] == "How are widgets…"
    tid = payload["thread_id"]

    listed = setup.client.get("/api/threads").json()["threads"]
    assert [t["id"] for t in listed] == [tid]
    assert listed[0]["title"] == "How are widgets…"          # thread_title_chars=24
    assert listed[0]["n_turns"] == 1

    thread = setup.client.get(f"/api/threads/{tid}").json()
    turn = thread["turns"][0]
    assert turn["question"] == "How are widgets coloured in practice, at scale?"
    assert turn["answer"] == f"Widgets are blue [{CHUNK}]."
    assert turn["seq"] == 1 and turn["notes"] and turn["stats"]["run_id"] == "run-1"
    image = next(s for s in turn["steps"] if s["kind"] == "image")
    assert image["image_url"] == f"/api/figures/{PAPER}/{PAPER}__f01.png"


def test_ask_with_a_thread_appends_a_turn(setup):
    tid = _ask(setup.client, "first")[0][1]["thread_id"]
    frames = _ask(setup.client, "second", thread_id=tid)
    assert frames[0] == ("thread", {"thread_id": tid, "created": False, "title": "first"})
    assert frames[-2][1]["turn_seq"] == 2

    thread = setup.client.get(f"/api/threads/{tid}").json()
    assert [t["question"] for t in thread["turns"]] == ["first", "second"]
    assert len(setup.client.get("/api/threads").json()["threads"]) == 1


def test_a_failed_run_in_a_thread_is_still_a_turn(setup):
    setup.loop.fail = True
    frames = _ask(setup.client, "will fail")
    failed = frames[-2][1]
    assert frames[-2][0] == "failed" and failed["thread_id"] == frames[0][1]["thread_id"]
    assert failed["turn_seq"] == 1
    turn = setup.client.get(f"/api/threads/{failed['thread_id']}").json()["turns"][0]
    assert turn["answer"] == "half an answer" and turn["stats"]["error"] == "provider_error"


def test_ask_with_a_bad_or_unknown_thread_is_refused_before_streaming(setup):
    bad = setup.client.post("/api/ask", json={"question": "q", "thread_id": "../etc"})
    assert bad.status_code == 400 and bad.json()["error"] == "bad_thread_id"
    missing = setup.client.post("/api/ask", json={"question": "q", "thread_id": "0" * 32})
    assert missing.status_code == 404 and missing.json()["error"] == "thread_not_found"
    assert setup.loop.questions == []
    assert not setup.app.state.lock.locked()


def test_threads_routes_create_list_get_delete(setup):
    c = setup.client
    made = c.post("/api/threads", json={"title": "  A named   chat "})
    assert made.status_code == 201
    first = made.json()
    assert first["title"] == "A named chat" and first["n_turns"] == 0
    second = c.post("/api/threads").json()                  # title is optional
    assert second["title"] == "Untitled chat"

    listed = c.get("/api/threads").json()["threads"]
    assert [t["id"] for t in listed] == [second["id"], first["id"]]   # newest first
    got = c.get(f"/api/threads/{first['id']}").json()
    assert got["id"] == first["id"] and got["turns"] == []

    deleted = c.delete(f"/api/threads/{first['id']}")
    assert deleted.status_code == 200 and deleted.json()["deleted"] == first["id"]
    assert c.get(f"/api/threads/{first['id']}").status_code == 404
    assert c.delete(f"/api/threads/{first['id']}").status_code == 404
    assert [t["id"] for t in c.get("/api/threads").json()["threads"]] == [second["id"]]


def test_threads_routes_validate_ids(setup):
    for url in ("/api/threads/not-hex", "/api/threads/" + "G" * 32, "/api/threads/%27%3B--"):
        assert setup.client.get(url).status_code == 400, url
        assert setup.client.delete(url).status_code == 400, url
    assert setup.client.get("/api/threads/" + "0" * 32).json()["error"] == "thread_not_found"


def test_delete_refuses_the_thread_a_run_is_writing_to(setup):
    tid = setup.store.create_thread("t")["id"]
    setup.app.state.lock.acquire()
    setup.app.state.running_thread = tid
    try:
        response = setup.client.delete(f"/api/threads/{tid}")
    finally:
        setup.app.state.running_thread = None
        setup.app.state.lock.release()
    assert response.status_code == 409 and response.json()["error"] == "thread_busy"
    assert setup.client.delete(f"/api/threads/{tid}").status_code == 200


def test_threads_need_a_store(tmp_path):
    config = _config(tmp_path)
    app = create_app(lambda: (StubLoop(""), config), dist_dir=tmp_path / "no_dist")
    with TestClient(app) as client:
        assert client.get("/api/threads").status_code == 503
        assert client.post("/api/ask", json={"question": "q"}).status_code == 503


def test_busy_when_the_lock_is_held(setup):
    setup.app.state.lock.acquire()
    try:
        frames = _ask(setup.client, "anything")
    finally:
        setup.app.state.lock.release()
    assert [e for e, _ in frames] == ["busy", "done"]
    assert setup.loop.questions == []
    assert setup.client.get("/api/status").json() == {"busy": False}
    # A refused question leaves no empty thread behind.
    assert setup.client.get("/api/threads").json()["threads"] == []


def test_failed_run_streams_failed_with_the_partial_answer(setup):
    setup.loop.fail = True
    frames = _ask(setup.client, "will fail")
    assert [e for e, _ in frames][-2:] == ["failed", "done"]
    failed = frames[-2][1]
    assert failed["error"] == "provider_error" and failed["answer"] == "half an answer"
    assert failed["detail"] == "upstream 502"
    assert not setup.app.state.lock.locked() and setup.loop.on_event is None


def test_loop_that_raises_still_frees_the_agent(setup):
    def boom(question, thread_id=None):
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


# ---------------------------------------------------------------- long-term memory


class StubMemoryStore:
    """The two calls the routes make, over a dict."""

    def __init__(self) -> None:
        self.rows = {
            "a" * 32: {"id": "a" * 32, "kind": "finding", "text": "Widgets are blue.",
                       "chunk_ids": [CHUNK], "use_count": 2, "last_used_at": 10.0,
                       "active": True, "forget_reason": None},
            "b" * 32: {"id": "b" * 32, "kind": "finding", "text": "Widgets were red.",
                       "chunk_ids": [CHUNK], "use_count": 1, "last_used_at": 5.0,
                       "active": False, "forget_reason": "superseded"},
        }

    def list(self, include_inactive: bool = False) -> Dict[str, Any]:
        return {"memories": [dict(r) for r in self.rows.values() if include_inactive or r["active"]]}

    def forget_one(self, memory_id: str, reason: str = "user") -> Dict[str, Any]:
        from memory.memory_store import valid_memory_id

        if not valid_memory_id(memory_id):
            return {"error": "bad_memory_id", "detail": "bad"}
        row = self.rows.get(memory_id)
        if row is None or not row["active"]:
            return {"error": "memory_not_found", "detail": "gone"}
        row.update(active=False, forget_reason=reason)
        return {"forgotten": memory_id, "reason": reason}


class MemoryStubLoop(StubLoop):
    """StubLoop plus long-term memory: remember() is called after the answer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.memory = SimpleNamespace(store=StubMemoryStore(),
                                      strengths=lambda rows: {r["id"]: 0.5 for r in rows})
        self.remembered: List[Any] = []
        self.lock_held_while_remembering: List[bool] = []
        self.app = None
        self.boom = False

    def remember(self, result: Dict[str, Any]) -> Any:
        self.remembered.append(result.get("answer"))
        if self.app is not None:
            self.lock_held_while_remembering.append(self.app.state.lock.locked())
        if self.boom:
            raise RuntimeError("writer exploded")
        return {"written": [{"id": "c" * 32, "text": "Widgets are blue."}], "refreshed": [],
                "invalidated": [], "used": [], "forgotten": [], "dropped": {}}


@pytest.fixture
def mem_setup(tmp_path):
    config = _config(tmp_path)
    store = ThreadStore(tmp_path / "threads.sqlite3")
    loop = MemoryStubLoop(str(config.paths.figures / PAPER / f"{PAPER}__f01.png"), store=store)
    app = create_app(lambda: (loop, config), dist_dir=tmp_path / "no_dist")
    loop.app = app
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, loop=loop, app=app)


def test_memory_frame_comes_after_the_answer_and_before_done(mem_setup):
    frames = _ask(mem_setup.client, "What colour are widgets?")
    events = [e for e, _ in frames]
    assert events[0] == "thread" and events[-3:] == ["answer", "memory", "done"]
    memory = frames[-2][1]
    assert set(memory) >= {"written", "refreshed", "invalidated"}
    assert memory["written"][0]["text"] == "Widgets are blue."
    # Written after the answer was produced, and still under the run lock.
    assert mem_setup.loop.remembered == ["Widgets are blue [" + CHUNK + "]."]
    assert mem_setup.loop.lock_held_while_remembering == [True]
    assert not mem_setup.app.state.lock.locked()


def test_a_memory_failure_never_fails_the_answer(mem_setup):
    mem_setup.loop.boom = True
    frames = _ask(mem_setup.client, "q")
    events = [e for e, _ in frames]
    assert events[-3:] == ["answer", "memory", "done"]
    assert frames[-2][1]["error"] == "memory_write_failed" and "exploded" in frames[-2][1]["detail"]
    assert not mem_setup.app.state.lock.locked()


def test_no_memory_frame_without_long_term_memory(setup):
    events = [e for e, _ in _ask(setup.client, "q")]
    assert "memory" not in events


def test_memories_routes(mem_setup):
    c = mem_setup.client
    active = c.get("/api/memories").json()["memories"]
    assert [m["id"] for m in active] == ["a" * 32] and active[0]["strength"] == 0.5
    everything = c.get("/api/memories?all=1").json()["memories"]
    assert {m["id"]: m["forget_reason"] for m in everything} == {"a" * 32: None, "b" * 32: "superseded"}
    assert everything[1]["strength"] is None

    gone = c.delete("/api/memories/" + "a" * 32)
    assert gone.status_code == 200 and gone.json() == {"forgotten": "a" * 32, "reason": "user"}
    assert c.get("/api/memories").json()["memories"] == []
    assert c.delete("/api/memories/" + "a" * 32).status_code == 404
    assert c.delete("/api/memories/not-an-id").status_code == 400


def test_memories_routes_need_long_term_memory(setup):
    assert setup.client.get("/api/memories").status_code == 503
    assert setup.client.delete("/api/memories/" + "a" * 32).status_code == 503


# ---------------------------------------------------------------- approved fetch (D26)

def _approve(client, body: Dict[str, Any]):
    return client.post("/api/ask", json=body)


def test_the_answer_frame_carries_the_fetch_proposal(setup):
    frames = _ask(setup.client, "What about the absent subject?")
    answer = dict(frames)["answer"]
    assert answer["fetch_proposal"]["query"] == "absent subject"
    assert answer["fetch_proposal"]["closest_topics"] == ["topic_a"]
    assert "conversation" not in answer


def test_an_approval_runs_the_prompt_file_text_in_the_thread(setup):
    from agent.fetch_proposal import approval_request

    tid = setup.store.create_thread("t")["id"]
    response = _approve(setup.client, {"thread_id": tid, "approve_fetch": {"query": " absent  subject "}})
    assert response.status_code == 200
    frames = _frames(response.text)
    assert frames[0] == ("thread", {"thread_id": tid, "created": False, "title": "t"})
    template = (ROOT / "src" / "prompts" / "fetch_approval.md").read_text(encoding="utf-8")
    expected = template.strip().replace("{query}", "absent subject")
    assert setup.loop.questions == [expected]
    assert setup.loop.thread_ids == [tid]
    approval = setup.loop.approvals[0]
    assert approval == approval_request("absent subject", setup.config)
    assert approval["display"].endswith('"absent subject"')


def test_a_plain_question_passes_no_approval(setup):
    _ask(setup.client, "an ordinary question")
    assert setup.loop.approvals == []


def test_an_approval_is_validated_before_the_stream_opens(setup):
    tid = setup.store.create_thread("t")["id"]
    cases = [
        ({"approve_fetch": {"query": "x"}}, "approval_needs_thread"),
        ({"thread_id": tid, "approve_fetch": {"query": "   "}}, "empty_fetch_query"),
        ({"thread_id": tid, "approve_fetch": {"query": "y" * 41}}, "fetch_query_too_long"),
        ({"thread_id": tid, "question": "and also this", "approve_fetch": {"query": "x"}},
         "question_and_approval"),
    ]
    for body, code in cases:
        response = _approve(setup.client, body)
        assert response.status_code == 400, body
        assert response.json()["error"] == code
    assert _approve(setup.client, {"thread_id": tid, "approve_fetch": {}}).status_code == 422
    assert _approve(setup.client, {"thread_id": "0" * 32, "approve_fetch": {"query": "x"}}).status_code == 404
    assert setup.loop.questions == []
    assert _approve(setup.client, {}).json()["error"] == "empty_question"
