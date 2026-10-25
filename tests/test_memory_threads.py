"""
Offline tests for conversation memory v1 (docs/DECISIONS.md D12, D13).

In:  a ThreadStore on tmp_path, a stubbed LLM that replays scripted responses, stub and
     real retrievers over a tiny tmp corpus. No network, no models.
Out: assertions on store CRUD/ordering/cascade/errors/transactions, history building and
     its budget, valid message pairing, "seen = still visible" dedup, and that the eval
     harness never runs inside a thread.
"""

from __future__ import annotations

import ast
import sqlite3
import threading
from pathlib import Path

import pytest

from agent.conversation import Conversation, ConversationError
from agent.loop import AgentLoop
from common.tokenization import TokenCounter
from llm_client import LLMError
from memory.history import build_history, build_notes, note_message, tool_note
from memory.threads import ThreadStore, title_from, valid_thread_id
from test_m5_agent import (  # noqa: F401  (cfg is a fixture)
    ScriptedClient,
    _config_at,
    cfg,
    ok_result,
    registry_with,
    text_response,
    tool_response,
)

ROOT = Path(__file__).resolve().parent.parent


class Clock:
    """Strictly increasing timestamps, so ordering tests cannot tie."""

    def __init__(self) -> None:
        self.t = 1_000.0

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


@pytest.fixture()
def store(tmp_path):
    return ThreadStore(tmp_path / "threads.sqlite3", now=Clock())


def _turns_rows(path: Path, thread_id: str) -> int:
    with sqlite3.connect(str(path)) as conn:
        return conn.execute("SELECT COUNT(*) FROM turns WHERE thread_id = ?", (thread_id,)).fetchone()[0]


# ===================================================================== the store

def test_create_append_get_rename_delete(store):
    made = store.create_thread("  First   question ")
    assert valid_thread_id(made["id"]) and made["title"] == "First question"
    assert made["n_turns"] == 0

    one = store.append_turn(made["id"], "q1", "a1 [p__c0001]", stats={"iterations": 2},
                            notes=["retrieve_evidence(...) → 1 chunk [p__c0001]"],
                            steps=[{"kind": "thinking", "iteration": 0}], run_id="r1")
    two = store.append_turn(made["id"], "q2", "a2")
    assert (one["seq"], two["seq"]) == (1, 2)

    thread = store.get_thread(made["id"])
    assert thread["n_turns"] == 2
    first = thread["turns"][0]
    assert first["question"] == "q1" and first["answer"] == "a1 [p__c0001]"
    assert first["stats"] == {"iterations": 2} and first["run_id"] == "r1"
    assert first["notes"] == ["retrieve_evidence(...) → 1 chunk [p__c0001]"]
    assert first["steps"] == [{"kind": "thinking", "iteration": 0}]
    assert thread["turns"][1]["notes"] == [] and thread["turns"][1]["steps"] == []

    assert store.rename_thread(made["id"], "Renamed")["title"] == "Renamed"
    assert store.get_thread(made["id"])["title"] == "Renamed"

    assert store.delete_thread(made["id"]) == {"deleted": made["id"], "turns_deleted": 2}
    assert store.get_thread(made["id"])["error"] == "thread_not_found"
    assert store.list_threads() == {"threads": []}


def test_threads_are_listed_most_recently_updated_first(store):
    a = store.create_thread("a")["id"]
    b = store.create_thread("b")["id"]
    c = store.create_thread("c")["id"]
    assert [t["id"] for t in store.list_threads()["threads"]] == [c, b, a]

    store.append_turn(a, "q", "answer")          # a is now the most recently updated
    listed = store.list_threads()["threads"]
    assert [t["id"] for t in listed] == [a, c, b]
    assert listed[0]["n_turns"] == 1


def test_delete_cascades_to_turns(store):
    keep = store.create_thread("keep")["id"]
    gone = store.create_thread("gone")["id"]
    for i in range(3):
        store.append_turn(gone, f"q{i}", f"a{i}")
    store.append_turn(keep, "q", "a")

    assert store.delete_thread(gone)["turns_deleted"] == 3
    assert _turns_rows(store.path, gone) == 0
    assert _turns_rows(store.path, keep) == 1


def test_foreign_key_cascade_holds_on_its_own(store):
    """The pragma is per connection; with it on, deleting the parent row alone cascades."""
    tid = store.create_thread("t")["id"]
    store.append_turn(tid, "q", "a")
    conn = store._connect()
    try:
        conn.execute("DELETE FROM threads WHERE id = ?", (tid,))
    finally:
        conn.close()
    assert _turns_rows(store.path, tid) == 0


@pytest.mark.parametrize("bad", [
    "", "abc", "x" * 32, "A" * 32, "0" * 31, "0" * 33, "'; DROP TABLE threads; --",
    None, 42, "../../etc/passwd",
])
def test_invalid_ids_are_refused_before_any_query(store, bad):
    store.create_thread("survivor")
    for result in (store.get_thread(bad), store.append_turn(bad, "q", "a"),
                   store.delete_thread(bad), store.rename_thread(bad, "x")):
        assert result["error"] == "bad_thread_id"
    assert len(store.list_threads()["threads"]) == 1


def test_missing_thread_is_structured(store):
    missing = "0" * 32
    assert store.get_thread(missing)["error"] == "thread_not_found"
    assert store.delete_thread(missing)["error"] == "thread_not_found"
    assert store.append_turn(missing, "q", "a")["error"] == "thread_not_found"
    assert _turns_rows(store.path, missing) == 0, "a refused append writes nothing"


def test_a_failed_write_rolls_back_everything_in_it(store):
    def work(conn):
        conn.execute("INSERT INTO threads (id, title, created_at, updated_at) "
                     "VALUES (?, 'half', 1, 1)", ("f" * 32,))
        raise sqlite3.IntegrityError("simulated failure after the first insert")

    result = store._write(work)
    assert result["error"] == "thread_store_unavailable"
    assert store.list_threads() == {"threads": []}


def test_concurrent_appends_get_distinct_consecutive_seqs(tmp_path):
    store = ThreadStore(tmp_path / "threads.sqlite3")
    tid = store.create_thread("busy")["id"]
    results = []

    def append(i):
        results.append(store.append_turn(tid, f"q{i}", f"a{i}"))

    workers = [threading.Thread(target=append, args=(i,)) for i in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert all("error" not in r for r in results), results
    assert sorted(r["seq"] for r in results) == list(range(1, 9))


def test_a_corrupt_file_returns_errors_and_never_raises(tmp_path):
    path = tmp_path / "threads.sqlite3"
    path.write_bytes(b"this is not a sqlite database at all " * 64)
    store = ThreadStore(path)
    for result in (store.list_threads(), store.create_thread("x"),
                   store.get_thread("0" * 32), store.append_turn("0" * 32, "q", "a"),
                   store.delete_thread("0" * 32)):
        assert result["error"] == "thread_store_unavailable"
        assert "detail" in result


def test_unreadable_json_columns_degrade_per_field(store):
    tid = store.create_thread("t")["id"]
    store.append_turn(tid, "q", "a", stats={"x": 1})
    with sqlite3.connect(str(store.path)) as conn:
        conn.execute("UPDATE turns SET steps_json = '{not json', notes_json = '\"a string\"'")
    turn = store.get_thread(tid)["turns"][0]
    assert turn["steps"] == [] and turn["notes"] == [] and turn["stats"] == {"x": 1}


@pytest.mark.parametrize("key", ["thread_history_tokens", "thread_note_arg_chars",
                                 "thread_title_chars"])
@pytest.mark.parametrize("value", [0, -1, "9000", True, None])
def test_bad_memory_settings_fail_at_load(tmp_path, monkeypatch, key, value):
    from config import ConfigError

    with pytest.raises(ConfigError, match=key):
        _config_at(tmp_path, monkeypatch, **{key: value})


def test_title_from_cuts_at_a_word_boundary():
    assert title_from("  short   question ", 72) == "short question"
    long = "word " * 40
    cut = title_from(long, 20)
    assert cut.endswith("…") and len(cut) <= 21 and not cut[:-1].endswith(" ")
    assert title_from("", 20) == "Untitled chat"


# ===================================================================== notes and history

def _record(tool, args, chunk_ids=(), error=None, summary=None):
    return {"tool_name": tool, "args": args, "chunk_ids_returned": list(chunk_ids),
            "error": error, "result_summary": summary or {}}


def test_notes_name_the_call_and_what_came_back():
    notes = build_notes([
        _record("retrieve_evidence", {"query": "load balancing", "k": 5}, ["p__c0001", "q__c0002"]),
        _record("analyze_corpus", {"operation": "stats"},
                summary={"operation": "stats", "summary": "11 papers, 2 topics"}),
        _record("retrieve_evidence", {"query": "absent"},
                summary={"chunks": [], "sufficient_evidence": False}),
        _record("search_literature", {"topic": "x"}, error="arxiv_rate_limited"),
    ], width=80)
    assert notes[0] == ('retrieve_evidence(query="load balancing", k=5) → '
                        "2 chunks [p__c0001, q__c0002]")
    assert notes[1] == 'analyze_corpus(operation="stats") → 11 papers, 2 topics'
    assert notes[2].endswith("→ 0 chunks, insufficient")
    assert notes[3].endswith("→ error: arxiv_rate_limited")


def test_long_arguments_are_clipped_but_chunk_ids_are_not():
    ids = [f"p__c{i:04d}" for i in range(8)]
    note = tool_note(_record("check_evidence_consistency", {"text": "x" * 5000}, ids), width=40)
    assert len(note.split(" → ")[0]) <= len("check_evidence_consistency()") + 40
    assert all(chunk_id in note for chunk_id in ids)


def _turn(q, a, notes=()):
    return {"question": q, "answer": a, "notes": list(notes)}


def test_history_is_verbatim_question_note_answer_per_turn(cfg):
    header = cfg.prompt("thread_notes")
    turns = [_turn("Q1?", "A1 [p__c0001].", ["retrieve_evidence(...) → 1 chunk [p__c0001]"]),
             _turn("Q2?", "A2.")]
    built = build_history(turns, TokenCounter(cfg), 9000, header)

    messages = built["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "assistant", "user", "assistant"]
    assert messages[0]["content"] == "Q1?"
    assert messages[1]["content"] == note_message(turns[0]["notes"], header)
    assert messages[1]["content"].startswith(header.strip())
    assert "- retrieve_evidence(...) → 1 chunk [p__c0001]" in messages[1]["content"]
    assert messages[2]["content"] == "A1 [p__c0001]."
    assert messages[3:] == [{"role": "user", "content": "Q2?"},
                            {"role": "assistant", "content": "A2."}]
    assert all("tool_calls" not in m for m in messages)
    assert built["turns_used"] == 2 and built["turns_dropped"] == 0


def test_budget_drops_whole_oldest_turns_first(cfg):
    counter = TokenCounter(cfg)
    header = cfg.prompt("thread_notes")
    turns = [_turn(f"question {i} " * 5, f"answer {i} " * 30, [f"note {i}"]) for i in range(4)]

    def cost(turn):
        return (counter.count(turn["question"]) + counter.count(turn["answer"])
                + counter.count(note_message(turn["notes"], header)))

    # Exactly the newest two fit: the two oldest go, whole.
    budget = cost(turns[3]) + cost(turns[2])
    built = build_history(turns, counter, budget, header)
    assert built["turns_used"] == 2 and built["turns_dropped"] == 2
    assert built["tokens"] == budget
    kept_questions = [m["content"] for m in built["messages"] if m["role"] == "user"]
    assert kept_questions == [turns[2]["question"], turns[3]["question"]]

    # One token short of two turns: only the newest survives. Never half a turn.
    built = build_history(turns, counter, budget - 1, header)
    assert built["turns_used"] == 1
    assert [m["content"] for m in built["messages"]] == [
        turns[3]["question"], note_message(turns[3]["notes"], header), turns[3]["answer"]]

    # Nothing fits: no history at all, rather than a fragment.
    assert build_history(turns, counter, 1, header)["messages"] == []


def test_a_turn_without_an_answer_is_skipped(cfg):
    built = build_history([_turn("Q1", "A1"), _turn("Q2", "  ")], TokenCounter(cfg), 9000,
                          cfg.prompt("thread_notes"))
    assert [m["content"] for m in built["messages"]] == ["Q1", "A1"]
    assert built["turns_skipped_empty"] == 1


def test_conversation_refuses_tool_messages_in_history(cfg):
    with pytest.raises(ConversationError):
        Conversation("sys", "q", cfg, history=[{"role": "tool", "tool_call_id": "x", "content": "{}"}])
    with pytest.raises(ConversationError):
        Conversation("sys", "q", cfg, history=[
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}])


# ===================================================================== the loop, in a thread

def _thread_loop(cfg, store, client, handler=None, retriever=None):
    return AgentLoop(registry_with(cfg, handler or (lambda **kw: ok_result(["pA__c0001"]))),
                     client=client, config=cfg, store=store, retriever=retriever)


def _assert_valid_pairing(messages):
    """Every assistant tool_call answered by exactly one tool message, before the next turn."""
    pending = []
    for message in messages:
        if message["role"] == "tool":
            assert message["tool_call_id"] in pending
            pending.remove(message["tool_call_id"])
        else:
            assert not pending, f"unanswered tool_calls {pending} before a {message['role']}"
            pending = [c["id"] for c in message.get("tool_calls") or []]
    assert not pending


def test_a_thread_run_saves_its_turn_and_the_next_run_sees_it(cfg, store):
    tid = store.create_thread("t")["id"]
    first = ScriptedClient(
        tool_response([("c1", "retrieve_evidence", {"query": "load balancing"})]),
        text_response("Answer one [pA__c0001]."),
    )
    result = _thread_loop(cfg, store, first).run("What losses?", thread_id=tid)
    assert result["thread_id"] == tid and result["turn_seq"] == 1

    turn = store.get_thread(tid)["turns"][0]
    assert turn["question"] == "What losses?" and turn["answer"] == "Answer one [pA__c0001]."
    assert turn["notes"] == ['retrieve_evidence(query="load balancing") → 1 chunk [pA__c0001]']
    assert [s["kind"] for s in turn["steps"]] == [
        "thinking", "tool_start", "tool_end", "thinking", "answering"]
    tool_end = turn["steps"][2]
    assert len(tool_end["result"]["chunks"][0]["text"]) <= int(cfg.agent.trace_text_chars)
    assert all(isinstance(s["t_ms"], int) for s in turn["steps"])
    assert turn["stats"]["tool_calls"] == 1 and turn["stats"]["stopped_because"] == "answered"
    assert turn["stats"]["history_turns"] == 0 and turn["run_id"] == result["run_id"]

    second = ScriptedClient(
        tool_response([("c1", "retrieve_evidence", {"query": "downstream tasks"})]),
        text_response("Answer two."),
    )
    result = _thread_loop(cfg, store, second).run("Which of those?", thread_id=tid)
    assert result["turn_seq"] == 2

    sent = second.requests[0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "assistant", "user"]
    assert sent[1]["content"] == "What losses?"
    assert "1 chunk [pA__c0001]" in sent[2]["content"]
    assert sent[3]["content"] == "Answer one [pA__c0001]."
    assert sent[-1] == {"role": "user", "content": "Which of those?"}
    assert cfg.prompt("thread_history").strip() in sent[0]["content"]
    _assert_valid_pairing(second.requests[-1]["messages"])
    assert store.get_thread(tid)["turns"][1]["stats"]["history_turns"] == 1


def test_the_current_question_is_never_dropped(tmp_path, monkeypatch):
    cfg = _config_at(tmp_path, monkeypatch, thread_history_tokens=5)
    store = ThreadStore(tmp_path / "threads.sqlite3", now=Clock())
    tid = store.create_thread("t")["id"]
    store.append_turn(tid, "an old question " * 20, "an old answer " * 50)

    client = ScriptedClient(text_response("ok"))
    question = "the current question " * 40     # far over the history budget on its own
    result = _thread_loop(cfg, store, client).run(question, thread_id=tid)

    sent = client.requests[0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user"]
    assert sent[-1]["content"] == question.strip()
    # No history survived, so the history instruction is not added either.
    assert cfg.prompt("thread_history").strip() not in sent[0]["content"]
    assert store.get_thread(tid)["turns"][1]["stats"]["history_turns_dropped"] == 1
    assert result["turn_seq"] == 2


def test_a_failed_thread_run_saves_its_own_partial_answer_not_the_previous_one(cfg, store):
    tid = store.create_thread("t")["id"]
    store.append_turn(tid, "Q1", "The PREVIOUS answer.")
    client = ScriptedClient(LLMError("provider_error", "upstream 502"))
    result = _thread_loop(cfg, store, client).run("Q2", thread_id=tid)

    assert result["error"] == "provider_error"
    assert result["partial"]["answer"] == ""
    assert result["thread_id"] == tid and result["turn_seq"] == 2
    saved = store.get_thread(tid)["turns"][1]
    assert saved["answer"] == ""
    assert saved["stats"]["error"] == "provider_error" and saved["stats"]["detail"] == "upstream 502"


def test_an_unknown_thread_fails_before_calling_the_model(cfg, store):
    client = ScriptedClient(text_response("never"))
    result = _thread_loop(cfg, store, client).run("q", thread_id="0" * 32)
    assert result["error"] == "thread_not_found" and client.requests == []
    result = _thread_loop(cfg, None, client).run("q", thread_id="0" * 32)
    assert result["error"] == "no_thread_store" and client.requests == []


# ===================================================================== no thread = today's behaviour

class SpyStore:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append(name)
            return {"error": "should_not_be_called", "detail": name}
        return record


def test_a_run_without_a_thread_touches_no_store_and_sends_the_plain_prompt(cfg):
    spy = SpyStore()
    client = ScriptedClient(text_response("answer"))
    loop = _thread_loop(cfg, spy, client)
    result = loop.run("  a question  ")

    assert spy.calls == []
    assert result["thread_id"] is None and result["turn_seq"] is None
    limits = loop._limits()
    assert client.requests[0]["messages"] == [
        {"role": "system", "content": loop._system_prompt(limits)},
        {"role": "user", "content": "a question"},
    ]
    assert "thread_history" not in client.requests[0]["messages"][0]["content"]
    assert cfg.prompt("thread_history").strip() not in client.requests[0]["messages"][0]["content"]


def _calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            yield name, node


def test_the_eval_harness_never_runs_inside_a_thread():
    """Eval questions must each be answered alone: no store or long-term memory handed to
    the loop, no thread passed to run, no memory written, and no thread or memory module
    imported, anywhere under eval/ (D12, D19)."""
    files = sorted((ROOT / "eval").glob("*.py"))
    assert any(f.name == "runner.py" for f in files)
    for path in files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for name, node in _calls(tree):
            keywords = {k.arg for k in node.keywords}
            if name == "AgentLoop":
                assert "store" not in keywords, f"{path.name} gives AgentLoop a store"
                assert "memory" not in keywords, f"{path.name} gives AgentLoop long-term memory"
                assert len(node.args) < 6, f"{path.name} passes AgentLoop a positional store"
            if name == "run":
                assert "thread_id" not in keywords, f"{path.name} passes run a thread_id"
            assert name not in ("remember", "after_run", "recall", "build_long_term_memory"), \
                f"{path.name} calls long-term memory ({name})"
        for banned in ("memory.threads", "ThreadStore", "memory.longterm", "LongTermMemory",
                       "memory.memory_store", "MemoryStore"):
            assert banned not in source, f"{path.name} mentions {banned}"


# ===================================================================== seen = still visible

def test_elision_reports_the_chunk_ids_it_took_out_of_view(tmp_path, monkeypatch):
    cfg = _config_at(tmp_path, monkeypatch, max_context_tokens=50)
    conversation = Conversation("system", "question", cfg)
    conversation.add_assistant(tool_response([("c1", "retrieve_evidence", {"query": "a"})]))
    conversation.add_tool_result("c1", ok_result(["pA__c0001", "pB__c0002"]))
    assert conversation.visible_chunk_ids() == ["pA__c0001", "pB__c0002"]

    assert conversation.elide_if_needed() == 1
    assert conversation.last_released == ["pA__c0001", "pB__c0002"]
    assert conversation.visible_chunk_ids() == []

    conversation.elide_if_needed()          # nothing left to elide: nothing released
    assert conversation.last_released == []


def test_an_elided_chunk_is_released_and_can_be_returned_again(tmp_path, monkeypatch):
    """End to end with the real retriever: the first result is elided, so its chunks are
    released, and a rephrased query gets them back instead of being told they were
    already returned (the within-question bug D13 fixes)."""
    from test_m4_retrieval import build_corpus, retriever as make_retriever

    cfg = _config_at(tmp_path, monkeypatch, max_context_tokens=50)
    build_corpus(cfg, {"pA": (["alpha"], [(f"alpha chunk {i} about routing", "text")
                                          for i in range(5)])})
    retriever = make_retriever(cfg, default=2.0)
    results = []

    def handler(**kwargs):
        out = retriever.retrieve(**kwargs)
        results.append(out)
        return out

    client = ScriptedClient(
        tool_response([("c1", "retrieve_evidence", {"query": "routing", "k": 2})]),
        tool_response([("c2", "retrieve_evidence", {"query": "routing, rephrased", "k": 2})]),
        text_response("done"),
    )
    AgentLoop(registry_with(cfg, handler), client=client, config=cfg,
              retriever=retriever).run("q")

    first = [c["chunk_id"] for c in results[0]["chunks"]]
    second = [c["chunk_id"] for c in results[1]["chunks"]]
    assert first and second == first, "released chunks are fetchable again"
    assert "already returned" not in results[1]["note"]


def test_without_elision_a_visible_chunk_stays_suppressed(cfg):
    """The control: nothing elided, so the dedup still holds within the question."""
    from test_m4_retrieval import build_corpus, retriever as make_retriever

    build_corpus(cfg, {"pA": (["alpha"], [(f"alpha chunk {i} about routing", "text")
                                          for i in range(5)])})
    retriever = make_retriever(cfg, default=2.0)
    results = []

    def handler(**kwargs):
        out = retriever.retrieve(**kwargs)
        results.append(out)
        return out

    client = ScriptedClient(
        tool_response([("c1", "retrieve_evidence", {"query": "routing", "k": 2})]),
        tool_response([("c2", "retrieve_evidence", {"query": "routing, rephrased", "k": 2})]),
        text_response("done"),
    )
    AgentLoop(registry_with(cfg, handler), client=client, config=cfg,
              retriever=retriever).run("q")
    first = {c["chunk_id"] for c in results[0]["chunks"]}
    second = {c["chunk_id"] for c in results[1]["chunks"]}
    assert first and second and not (first & second)


def test_a_thread_run_starts_with_an_empty_seen_set(cfg, store):
    """Prior turns leave notes, not chunk text, so nothing is visible and nothing is
    suppressed when a thread's next question starts."""
    class Retriever:
        def __init__(self):
            self.resets = []

        def reset(self, visible=()):
            self.resets.append(list(visible))

    tid = store.create_thread("t")["id"]
    store.append_turn(tid, "Q1", "A1 [pA__c0001]", notes=["retrieve_evidence() → 1 chunk [pA__c0001]"])
    retriever = Retriever()
    _thread_loop(cfg, store, ScriptedClient(text_response("a")), retriever=retriever).run(
        "Q2", thread_id=tid)
    assert retriever.resets == [[]]


def test_retriever_reset_and_release_keep_seen_equal_to_visible(cfg):
    from tools.retrieve_evidence import EvidenceRetriever

    class Never:
        def encode_query(self, query):
            raise AssertionError("not used")

    retriever = EvidenceRetriever(cfg, embedder=Never(), reranker=Never())
    retriever.reset(["a", "b", "c"])
    assert retriever.returned == {"a", "b", "c"}
    retriever.release(["b", "not-there"])
    assert retriever.returned == {"a", "c"}
    retriever.reset()
    assert retriever.returned == set()


def test_failed_turn_answer_is_marked_partial_in_history(cfg):
    turns = [
        {"question": "q ok", "answer": "full answer", "notes": [], "stats": {}},
        {"question": "q bad", "answer": "half an answ", "notes": [],
         "stats": {"error": "rate_limited"}},
    ]
    built = build_history(turns, TokenCounter(cfg), 9000, cfg.prompt("thread_notes"),
                          partial_marker=cfg.prompt("thread_partial"))
    answers = [m["content"] for m in built["messages"] if m["role"] == "assistant"]
    assert answers[0] == "full answer"
    assert "rate_limited" in answers[1] and answers[1].endswith("half an answ")
