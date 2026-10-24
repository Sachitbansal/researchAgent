"""
Offline tests for M5: dispatch, the message-array contract, per-tool call limits and
termination backstops, loop detection, traces.

In:  a stubbed LLM that replays scripted responses — no network, no models.
Out: assertions that the conversation the loop re-sends is one the API would accept, and
     that the loop stops for the right reason.
"""

import json

import pytest

from agent.conversation import Conversation, ConversationError
from agent.loop import NO_PROGRESS_NUDGE, TOOLS_EXHAUSTED_NUDGE, AgentLoop, _canonical
from agent.tool_registry import RETRIEVE_EVIDENCE_PARAMETERS, ToolRegistry
from agent.trace import TraceWriter, chunk_ids_of, summarise_result
from config import load_config
from llm_client import LLMError, LLMResponse


def _config_at(tmp_path, monkeypatch, **agent_overrides):
    """A Config rooted at tmp_path, optionally with agent settings overridden.

    Overrides are written into the YAML before loading rather than poked into the loaded
    object: config sections are deliberately read-only, and a test that routes around
    that is testing something the real system cannot do.
    """
    import yaml

    import config as config_module

    repo_root = config_module.REPO_ROOT
    raw = yaml.safe_load((repo_root / "config.yaml").read_text(encoding="utf-8"))
    raw["agent"].update(agent_overrides)
    local = tmp_path / "config.yaml"
    local.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setattr(config_module, "REPO_ROOT", tmp_path)
    built = load_config(local)
    built.paths.ensure()
    built.paths.prompts = repo_root / "src" / "prompts"
    return built


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    return _config_at(tmp_path, monkeypatch)


@pytest.fixture()
def tiny_context_cfg(tmp_path, monkeypatch):
    """A config whose context budget is small enough that elision must fire."""
    return _config_at(tmp_path, monkeypatch, max_context_tokens=50)


def tool_response(calls, model="stub"):
    """An assistant turn asking for tool calls. `calls` is [(id, name, args_dict), ...]."""
    raw = {
        "model": model,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": call_id, "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args)}}
                    for call_id, name, args in calls
                ],
            },
            "finish_reason": "tool_calls",
        }],
    }
    from llm_client import _parse_response
    return _parse_response(raw)


def text_response(text):
    from llm_client import _parse_response
    return _parse_response({
        "model": "stub",
        "choices": [{"message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
    })


class ScriptedClient:
    """Replays a list of responses and records every request it was given."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, messages, tools=None, role="agent", **kwargs):
        self.requests.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        if not self.responses:
            return text_response("done")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def registry_with(cfg, handler, name="retrieve_evidence"):
    registry = ToolRegistry(cfg)
    registry.register(name, RETRIEVE_EVIDENCE_PARAMETERS, handler)
    return registry


def ok_result(chunk_ids=("pA__c0000",)):
    return {
        "chunks": [{"chunk_id": cid, "paper_id": cid.split("__")[0], "paper_title": "T",
                    "page": 1, "section": "S", "chunk_type": "text", "score": 1.0,
                    "text": "some evidence text " * 20}
                   for cid in chunk_ids],
        "sufficient_evidence": True,
        "note": "",
        "n_candidates_considered": 40,
    }


# ------------------------------------------------------------------ the message array

def test_the_provider_gets_the_whole_conversation_every_call(cfg):
    """The API is stateless: nothing is incremental, the full array goes every time."""
    client = ScriptedClient(
        tool_response([("call_1", "retrieve_evidence", {"query": "routing"})]),
        text_response("Final answer [pA:pA__c0000]."),
    )
    loop = AgentLoop(registry_with(cfg, lambda **kw: ok_result()), client=client, config=cfg)
    loop.run("what do the papers say?")

    first, second = client.requests[0]["messages"], client.requests[1]["messages"]
    assert [m["role"] for m in first] == ["system", "user"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    assert second[:2] == first, "the earlier turns are re-sent unchanged, not referenced"


def test_tool_call_id_round_trips_into_the_tool_message(cfg):
    """tool_call_id is the only link between a result and the call that asked for it."""
    client = ScriptedClient(
        tool_response([("call_xyz", "retrieve_evidence", {"query": "q"})]),
        text_response("done"),
    )
    loop = AgentLoop(registry_with(cfg, lambda **kw: ok_result()), client=client, config=cfg)
    loop.run("q")

    messages = client.requests[1]["messages"]
    assistant = next(m for m in messages if m["role"] == "assistant")
    tool = next(m for m in messages if m["role"] == "tool")
    assert assistant["tool_calls"][0]["id"] == "call_xyz"
    assert tool["tool_call_id"] == "call_xyz"


def test_parallel_tool_calls_each_get_their_own_result(cfg):
    """An assistant turn with two tool_calls needs two tool messages or the API 400s."""
    client = ScriptedClient(
        tool_response([("call_a", "retrieve_evidence", {"query": "one"}),
                       ("call_b", "retrieve_evidence", {"query": "two"})]),
        text_response("done"),
    )
    loop = AgentLoop(registry_with(cfg, lambda **kw: ok_result()), client=client, config=cfg)
    loop.run("q")

    tools = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools] == ["call_a", "call_b"]


def test_conversation_refuses_to_emit_an_unanswered_tool_call(cfg):
    conversation = Conversation("sys", "q", cfg)
    conversation.add_assistant(tool_response([("call_1", "retrieve_evidence", {"query": "x"})]))
    with pytest.raises(ConversationError, match="no result yet"):
        conversation.messages()


def test_conversation_refuses_a_result_for_an_unknown_id(cfg):
    conversation = Conversation("sys", "q", cfg)
    conversation.add_assistant(tool_response([("call_1", "retrieve_evidence", {"query": "x"})]))
    with pytest.raises(ConversationError, match="no unanswered tool_call"):
        conversation.add_tool_result("call_wrong", {"ok": True})


def test_tool_result_content_is_a_string(cfg):
    """Tools return dicts; the wire format takes a string."""
    conversation = Conversation("sys", "q", cfg)
    conversation.add_assistant(tool_response([("call_1", "retrieve_evidence", {"query": "x"})]))
    message = conversation.add_tool_result("call_1", {"sufficient_evidence": True})
    assert isinstance(message["content"], str)
    assert json.loads(message["content"])["sufficient_evidence"] is True


# ---------------------------------------------------------------------- context budget

def test_elision_replaces_content_but_keeps_the_message(tiny_context_cfg):
    """Deleting a tool message would orphan its assistant tool_call and the API would
    reject the request, so the content is replaced instead."""
    conversation = Conversation("sys", "q", tiny_context_cfg)
    for index in range(4):
        call_id = f"call_{index}"
        conversation.add_assistant(tool_response([(call_id, "retrieve_evidence", {"query": str(index)})]))
        conversation.add_tool_result(call_id, ok_result([f"pA__c{index:04d}"] * 5))

    before = len(conversation)
    elided = conversation.elide_if_needed()

    assert elided > 0
    assert len(conversation) == before, "no message may be removed"
    tools = [m for m in conversation.messages() if m["role"] == "tool"]
    assert any(m["content"].startswith("[earlier result elided") for m in tools)
    assert all(m.get("tool_call_id") for m in tools), "pairing survives"


def test_placeholder_names_the_papers_so_the_agent_knows_it_searched(tiny_context_cfg):
    conversation = Conversation("sys", "q", tiny_context_cfg)
    conversation.add_assistant(tool_response([("c1", "retrieve_evidence", {"query": "x"})]))
    conversation.add_tool_result("c1", ok_result(["pA__c0001", "pB__c0002"]))
    conversation.elide_if_needed()

    tool = [m for m in conversation.messages() if m["role"] == "tool"][0]
    assert "pA" in tool["content"] and "pB" in tool["content"]
    assert "already ran this search" in tool["content"]


# ------------------------------------------------------------------------- the loop

def test_tool_is_dispatched_by_name_with_parsed_arguments(cfg):
    seen = {}

    def handler(**kwargs):
        seen.update(kwargs)
        return ok_result()

    client = ScriptedClient(
        tool_response([("c1", "retrieve_evidence", {"query": "moe routing", "k": 3})]),
        text_response("answer"),
    )
    AgentLoop(registry_with(cfg, handler), client=client, config=cfg).run("q")
    assert seen == {"query": "moe routing", "k": 3}


def test_loop_stops_when_the_model_answers(cfg):
    client = ScriptedClient(text_response("Immediate answer."))
    result = AgentLoop(registry_with(cfg, lambda **kw: ok_result()), client=client,
                       config=cfg).run("q")
    assert result["answer"] == "Immediate answer."
    assert result["iterations"] == 1
    assert result["tool_calls"] == 0
    assert result["tool_call_counts"] == {"retrieve_evidence": 0}
    assert result["stopped_because"] == "answered"


class EndlessClient:
    """Answers every request with a fresh tool-call turn built by `make(turn)`, forever.

    The guard turns a regression in the termination backstops into a failure instead of
    a hung test run. The forced final call (no tools offered) gets a text answer.
    """

    GUARD = 200

    def __init__(self, make, final="Partial answer from what I have."):
        self.make = make
        self.final = final
        self.requests = []

    def complete(self, messages, tools=None, role="agent", **kwargs):
        self.requests.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        if len(self.requests) > self.GUARD:
            raise AssertionError("the loop did not terminate")
        if tools is None:
            return text_response(self.final)
        return self.make(len(self.requests) - 1)


def offered(request):
    return sorted(t["function"]["name"] for t in (request["tools"] or []))


def tool_messages(messages):
    return [m for m in messages if m["role"] == "tool"]


def assert_every_call_answered(messages):
    requested = [c["id"] for m in messages for c in (m.get("tool_calls") or [])]
    answered = [m["tool_call_id"] for m in tool_messages(messages)]
    assert sorted(requested) == sorted(answered)


def two_tool_registry(cfg, handler):
    registry = ToolRegistry(cfg)
    registry.register("retrieve_evidence", RETRIEVE_EVIDENCE_PARAMETERS, handler)
    registry.register("analyze_corpus", RETRIEVE_EVIDENCE_PARAMETERS, handler)
    return registry


def test_eleventh_call_to_one_tool_is_refused(cfg):
    """Ten distinct calls run; the eleventh is answered with tool_limit_reached."""
    limit = int(cfg.agent.max_calls_per_tool)
    calls = []
    client = ScriptedClient(*[
        tool_response([(f"c{i}", "retrieve_evidence", {"query": f"q{i}"})])
        for i in range(limit + 1)
    ], text_response("answer"))
    registry = two_tool_registry(cfg, lambda **kw: calls.append(kw) or ok_result())
    result = AgentLoop(registry, client=client, config=cfg).run("q")

    assert len(calls) == limit
    assert result["tool_call_counts"] == {"retrieve_evidence": limit, "analyze_corpus": 0}
    last = tool_messages(client.requests[-1]["messages"])[-1]
    assert last["tool_call_id"] == f"c{limit}"
    assert json.loads(last["content"])["error"] == "tool_limit_reached"
    assert "retrieve_evidence" in last["content"] and str(limit) in last["content"]


def test_parallel_calls_past_the_limit_in_one_turn_are_each_refused(cfg):
    """Calls 10, 11 and 12 arrive in one turn: 10 runs, 11 and 12 are refused, and
    every tool_call id still gets a result, or the next request would be invalid."""
    limit = int(cfg.agent.max_calls_per_tool)
    calls = []
    client = ScriptedClient(
        *[tool_response([(f"c{i}", "retrieve_evidence", {"query": f"q{i}"})])
          for i in range(limit - 1)],
        tool_response([(f"c{i}", "retrieve_evidence", {"query": f"q{i}"})
                       for i in range(limit - 1, limit + 2)]),
        text_response("answer"),
    )
    events = []
    registry = two_tool_registry(cfg, lambda **kw: calls.append(kw) or ok_result())
    result = AgentLoop(registry, client=client, config=cfg, on_event=events.append).run("q")

    assert len(calls) == limit
    assert result["tool_call_counts"]["retrieve_evidence"] == limit
    assert result["tool_calls"] == limit
    messages = client.requests[-1]["messages"]
    assert_every_call_answered(messages)
    by_id = {m["tool_call_id"]: json.loads(m["content"]) for m in tool_messages(messages)}
    assert "error" not in by_id[f"c{limit - 1}"]
    assert by_id[f"c{limit}"]["error"] == by_id[f"c{limit + 1}"]["error"] == "tool_limit_reached"
    skipped = [e for e in events if e["kind"] == "skipped"]
    assert len(skipped) == 2 and all("limit" in e["reason"] for e in skipped)


def test_an_exhausted_tool_is_not_offered_on_the_next_call(tmp_path, monkeypatch):
    cfg = _config_at(tmp_path, monkeypatch, max_calls_per_tool=2)
    client = ScriptedClient(
        tool_response([("c0", "retrieve_evidence", {"query": "a"})]),
        tool_response([("c1", "retrieve_evidence", {"query": "b"})]),
        text_response("answer"),
    )
    AgentLoop(two_tool_registry(cfg, lambda **kw: ok_result()), client=client,
              config=cfg).run("q")

    assert offered(client.requests[0]) == ["analyze_corpus", "retrieve_evidence"]
    assert offered(client.requests[1]) == ["analyze_corpus", "retrieve_evidence"]
    assert offered(client.requests[2]) == ["analyze_corpus"]


def test_per_tool_override_changes_one_tools_limit(tmp_path, monkeypatch):
    cfg = _config_at(tmp_path, monkeypatch, max_calls_per_tool=3,
                     tool_call_limits={"analyze_corpus": 1})
    client = ScriptedClient(
        tool_response([("c0", "analyze_corpus", {"query": "a"})]),
        text_response("answer"),
    )
    loop = AgentLoop(two_tool_registry(cfg, lambda **kw: ok_result()), client=client, config=cfg)
    loop.run("q")
    assert offered(client.requests[1]) == ["retrieve_evidence"]
    system = client.requests[0]["messages"][0]["content"]
    assert "{tool_call_budget}" not in system
    assert "3 calls per tool (except analyze_corpus: 1)" in system


def test_shipped_config_gives_search_literature_five_calls(cfg):
    """The user's override: search_literature is slow and rate-limited, so 5, not 10."""
    assert int(cfg.agent.max_calls_per_tool) == 10
    assert int(cfg.agent.tool_call_limits.search_literature) == 5

    calls = []
    registry = ToolRegistry(cfg)
    registry.register("retrieve_evidence", RETRIEVE_EVIDENCE_PARAMETERS, lambda **kw: ok_result())
    registry.register("search_literature", RETRIEVE_EVIDENCE_PARAMETERS,
                      lambda **kw: calls.append(kw) or {"papers_added": []})
    client = ScriptedClient(*[
        tool_response([(f"s{i}", "search_literature", {"query": f"q{i}"})]) for i in range(6)
    ], text_response("answer"))
    result = AgentLoop(registry, client=client, config=cfg).run("q")

    assert len(calls) == 5
    assert result["tool_call_counts"] == {"retrieve_evidence": 0, "search_literature": 5}
    assert offered(client.requests[5]) == ["retrieve_evidence"]
    last = json.loads(tool_messages(client.requests[-1]["messages"])[-1]["content"])
    assert last["error"] == "tool_limit_reached" and "5" in last["detail"]
    system = client.requests[0]["messages"][0]["content"]
    assert "10 calls per tool (except search_literature: 5)" in system


def test_repeating_one_call_forever_stops_on_no_progress(cfg):
    """The first call runs; every repeat is refused, so after N refused turns the loop
    forces an answer instead of running until some cap that no longer exists."""
    n = int(cfg.agent.max_no_progress_turns)
    calls = []
    client = EndlessClient(
        lambda t: tool_response([(f"c{t}", "retrieve_evidence", {"query": "same"})]))
    events = []
    result = AgentLoop(registry_with(cfg, lambda **kw: calls.append(kw) or ok_result()),
                       client=client, config=cfg, on_event=events.append).run("q")

    assert len(calls) == 1
    assert result["stopped_because"] == "no_progress"
    assert result["iterations"] == 1 + n
    assert result["tool_call_counts"] == {"retrieve_evidence": 1}
    assert result["repeated_calls"] == n
    assert result["answer"] == "Partial answer from what I have."
    final = client.requests[-1]
    assert final["tools"] is None, "no tools offered on the forced final call"
    assert final["messages"][-1] == {"role": "user", "content": NO_PROGRESS_NUDGE}
    cap = [e for e in events if e["kind"] == "cap"]
    assert len(cap) == 1 and cap[0]["reason"] == "no_progress"


def test_calling_a_nonexistent_tool_forever_stops_on_no_progress(cfg):
    n = int(cfg.agent.max_no_progress_turns)
    client = EndlessClient(lambda t: tool_response([(f"c{t}", "made_up_tool", {"x": t})]))
    result = AgentLoop(registry_with(cfg, lambda **kw: ok_result()), client=client,
                       config=cfg).run("q")

    assert result["stopped_because"] == "no_progress"
    assert result["iterations"] == n
    assert result["tool_calls"] == 0
    assert result["tool_call_counts"] == {"retrieve_evidence": 0}
    assert result["trace_records"] == []
    assert client.requests[-1]["tools"] is None
    assert_every_call_answered(client.requests[-1]["messages"])


def test_exhausting_every_tool_stops_with_tools_exhausted(tmp_path, monkeypatch):
    cfg = _config_at(tmp_path, monkeypatch, max_calls_per_tool=3)
    client = EndlessClient(lambda t: tool_response([
        (f"r{t}", "retrieve_evidence", {"query": f"q{t}"}),
        (f"a{t}", "analyze_corpus", {"query": f"q{t}"}),
    ]))
    events = []
    result = AgentLoop(two_tool_registry(cfg, lambda **kw: ok_result()), client=client,
                       config=cfg, on_event=events.append).run("q")

    assert result["stopped_because"] == "tools_exhausted"
    assert result["iterations"] == 3
    assert result["tool_call_counts"] == {"retrieve_evidence": 3, "analyze_corpus": 3}
    assert result["tool_calls"] == 6 == len(result["trace_records"])
    final = client.requests[-1]
    assert final["tools"] is None
    assert final["messages"][-1] == {"role": "user", "content": TOOLS_EXHAUSTED_NUDGE}
    assert result["answer"] == "Partial answer from what I have."
    assert [e["reason"] for e in events if e["kind"] == "cap"] == ["tools_exhausted"]


def test_refused_calls_do_not_consume_a_tools_budget(tmp_path, monkeypatch):
    """Repeated, malformed, unknown and over-limit calls are all free."""
    from llm_client import _parse_response

    cfg = _config_at(tmp_path, monkeypatch, max_calls_per_tool=2)
    bad = _parse_response({
        "model": "stub",
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "bad", "type": "function",
             "function": {"name": "retrieve_evidence", "arguments": "{not json"}}]},
            "finish_reason": "tool_calls"}],
    })
    client = ScriptedClient(
        tool_response([("c0", "retrieve_evidence", {"query": "a"})]),
        tool_response([("c1", "retrieve_evidence", {"query": "a"}),     # repeat
                       ("c2", "ghost_tool", {"query": "a"})]),          # unknown
        bad,                                                             # bad JSON
        tool_response([("c3", "retrieve_evidence", {"query": "b"}),     # 2nd real call
                       ("c4", "retrieve_evidence", {"query": "c"})]),   # over the limit
        text_response("answer"),
    )
    result = AgentLoop(two_tool_registry(cfg, lambda **kw: ok_result()), client=client,
                       config=cfg).run("q")

    assert result["stopped_because"] == "answered"
    assert result["tool_call_counts"] == {"retrieve_evidence": 2, "analyze_corpus": 0}
    assert result["tool_calls"] == 2 == len(result["trace_records"])
    assert_every_call_answered(client.requests[-1]["messages"])


def test_a_dispatched_call_that_errors_still_counts(cfg):
    """An error from inside the tool is a real dispatch: it ran, so it is budgeted."""
    def explode(**kwargs):
        raise RuntimeError("boom")

    client = ScriptedClient(
        tool_response([("c1", "retrieve_evidence", {"query": "x"})]), text_response("a"))
    result = AgentLoop(registry_with(cfg, explode), client=client, config=cfg).run("q")
    assert result["tool_call_counts"] == {"retrieve_evidence": 1}


def test_stale_max_iterations_is_rejected_at_load(tmp_path, monkeypatch):
    from config import ConfigError

    with pytest.raises(ConfigError, match="max_calls_per_tool"):
        _config_at(tmp_path, monkeypatch, max_iterations=8)


@pytest.mark.parametrize("key, value", [
    ("max_calls_per_tool", 0), ("max_calls_per_tool", True), ("max_no_progress_turns", -1),
    ("max_no_progress_turns", "3"), ("tool_call_limits", {"search_literature": 0}),
])
def test_bad_termination_settings_fail_at_load(tmp_path, monkeypatch, key, value):
    from config import ConfigError

    with pytest.raises(ConfigError, match=key):
        _config_at(tmp_path, monkeypatch, **{key: value})


def test_loop_detection_injects_a_notice_instead_of_re_executing(cfg):
    calls = []

    def handler(**kwargs):
        calls.append(kwargs)
        return ok_result()

    same = {"query": "identical"}
    client = ScriptedClient(
        tool_response([("c1", "retrieve_evidence", same)]),
        tool_response([("c2", "retrieve_evidence", same)]),
        text_response("answer"),
    )
    result = AgentLoop(registry_with(cfg, handler), client=client, config=cfg).run("q")

    assert len(calls) == 1, "the repeat must not run the tool again"
    assert result["repeated_calls"] == 1
    tool_messages = [m for m in client.requests[-1]["messages"] if m["role"] == "tool"]
    assert any("already made this exact call" in m["content"] for m in tool_messages)


def test_argument_order_does_not_defeat_loop_detection():
    assert _canonical("t", {"a": 1, "b": 2}) == _canonical("t", {"b": 2, "a": 1})


def test_unknown_tool_name_returns_an_error_result_not_a_crash(cfg):
    client = ScriptedClient(
        tool_response([("c1", "hallucinated_tool", {"x": 1})]),
        text_response("recovered"),
    )
    result = AgentLoop(registry_with(cfg, lambda **kw: ok_result()), client=client,
                       config=cfg).run("q")

    assert result["answer"] == "recovered"
    tool_message = [m for m in client.requests[1]["messages"] if m["role"] == "tool"][0]
    assert "unknown_tool" in tool_message["content"]
    assert "retrieve_evidence" in tool_message["content"], "it is told what does exist"


def test_a_tool_that_raises_is_contained(cfg):
    def explode(**kwargs):
        raise RuntimeError("boom")

    client = ScriptedClient(
        tool_response([("c1", "retrieve_evidence", {"query": "x"})]),
        text_response("still answered"),
    )
    result = AgentLoop(registry_with(cfg, explode), client=client, config=cfg).run("q")
    assert result["answer"] == "still answered"
    tool_message = [m for m in client.requests[1]["messages"] if m["role"] == "tool"][0]
    assert "tool_crashed" in tool_message["content"]


def test_malformed_tool_arguments_are_reported_back(cfg):
    from llm_client import _parse_response

    bad = _parse_response({
        "model": "stub",
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "retrieve_evidence", "arguments": "{not json"}}]},
            "finish_reason": "tool_calls"}],
    })
    client = ScriptedClient(bad, text_response("recovered"))
    result = AgentLoop(registry_with(cfg, lambda **kw: ok_result()), client=client,
                       config=cfg).run("q")

    assert result["answer"] == "recovered"
    tool_message = [m for m in client.requests[1]["messages"] if m["role"] == "tool"][0]
    assert "bad_arguments" in tool_message["content"]


def test_llm_failure_returns_an_error_dict_with_partial_progress(cfg):
    client = ScriptedClient(LLMError("llm_unreachable", "network down"))
    result = AgentLoop(registry_with(cfg, lambda **kw: ok_result()), client=client,
                       config=cfg).run("q")
    assert result["error"] == "llm_unreachable"
    assert "partial" in result and result["partial"]["run_id"]


def test_empty_question_is_rejected(cfg):
    result = AgentLoop(registry_with(cfg, lambda **kw: ok_result()),
                       client=ScriptedClient(), config=cfg).run("   ")
    assert result["error"] == "empty_question"


def test_retriever_dedup_state_is_reset_per_question(cfg):
    class Retriever:
        def __init__(self):
            self.resets = 0

        def reset(self):
            self.resets += 1

    retriever = Retriever()
    loop = AgentLoop(registry_with(cfg, lambda **kw: ok_result()),
                     client=ScriptedClient(text_response("a")), config=cfg,
                     retriever=retriever)
    loop.run("q1")
    loop.client = ScriptedClient(text_response("b"))
    loop.run("q2")
    assert retriever.resets == 2


# ---------------------------------------------------------------------------- tracing

def test_a_trace_record_is_written_per_tool_call(cfg):
    client = ScriptedClient(
        tool_response([("c1", "retrieve_evidence", {"query": "routing"})]),
        text_response("answer"),
    )
    result = AgentLoop(registry_with(cfg, lambda **kw: ok_result(["pA__c0000", "pB__c0001"])),
                       client=client, config=cfg).run("q", question_id="Q7")

    assert len(result["trace_records"]) == 1
    record = result["trace_records"][0]
    assert set(record) == {
        "run_id", "question_id", "iteration", "tool_name", "args", "result_summary",
        "chunk_ids_returned", "error", "latency_ms", "timestamp",
    }
    assert record["question_id"] == "Q7"
    assert record["iteration"] == 0
    assert record["args"] == {"query": "routing"}
    assert record["chunk_ids_returned"] == ["pA__c0000", "pB__c0001"]
    assert record["error"] is None

    from common.storage import read_jsonl
    assert [r["run_id"] for r in read_jsonl(cfg.paths.trace_file(result["run_id"]))] == \
        [result["run_id"]]


def test_trace_summary_is_truncated(cfg):
    limit = int(cfg.agent.trace_text_chars)
    summary = summarise_result(ok_result(["pA__c0000"]), limit)
    assert len(summary["chunks"][0]["text"]) <= limit
    assert summary["chunks"][0]["chunk_id"] == "pA__c0000"


def test_chunk_ids_of_handles_a_result_without_chunks():
    assert chunk_ids_of({"error": "no_results"}) == []
    assert chunk_ids_of({}) == []


# --------------------------------------------------------------------------- progress

def test_progress_events_report_each_step_in_order(cfg):
    """on_event sees the loop's steps as they happen, in the order they happen."""
    client = ScriptedClient(
        tool_response([("c1", "retrieve_evidence", {"query": "routing"})]),
        text_response("answer"),
    )
    events = []
    result = AgentLoop(registry_with(cfg, lambda **kw: ok_result(["pA__c0000"])),
                       client=client, config=cfg, on_event=events.append).run("q")

    assert [e["kind"] for e in events] == [
        "thinking", "tool_start", "tool_end", "thinking", "answering",
    ]
    assert events[1]["tool_name"] == "retrieve_evidence"
    assert events[1]["args"] == {"query": "routing"}
    assert events[2]["result"]["chunks"][0]["chunk_id"] == "pA__c0000"
    assert isinstance(events[2]["latency_ms"], int)
    assert result["answer"] == "answer"


def test_a_repeated_call_is_reported_as_skipped(cfg):
    """A call that never dispatches emits `skipped`, not an unpaired tool_start."""
    call = ("c1", "retrieve_evidence", {"query": "same"})
    client = ScriptedClient(
        tool_response([call]), tool_response([call]), text_response("answer"),
    )
    events = []
    AgentLoop(registry_with(cfg, lambda **kw: ok_result()), client=client, config=cfg,
              on_event=events.append).run("q")

    kinds = [e["kind"] for e in events]
    assert kinds.count("tool_start") == kinds.count("tool_end") == 1
    assert "skipped" in kinds


def test_malformed_arguments_are_reported_as_skipped(cfg):
    """The other path that never dispatches: a line is never left open for it either."""
    from llm_client import _parse_response

    bad = _parse_response({
        "model": "stub",
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "retrieve_evidence", "arguments": "{not json"}}]},
            "finish_reason": "tool_calls"}],
    })
    events = []
    result = AgentLoop(registry_with(cfg, lambda **kw: ok_result()),
                       client=ScriptedClient(bad, text_response("recovered")), config=cfg,
                       on_event=events.append).run("q")

    assert result["answer"] == "recovered"
    kinds = [e["kind"] for e in events]
    assert kinds.count("tool_start") == kinds.count("tool_end") == 0
    assert kinds.count("skipped") == 1


def test_a_broken_reporter_cannot_end_a_run(cfg):
    """An observer is for watching a run, so its failure must not stop one."""
    def explode(event):
        raise RuntimeError("reporter is broken")

    client = ScriptedClient(
        tool_response([("c1", "retrieve_evidence", {"query": "q"})]),
        text_response("answer"),
    )
    result = AgentLoop(registry_with(cfg, lambda **kw: ok_result()), client=client,
                       config=cfg, on_event=explode).run("q")
    assert result["answer"] == "answer"
    assert result["tool_calls"] == 1


def test_progress_lines_are_one_per_event(cfg):
    """tool_start opens a line and tool_end closes it; every other kind closes its own."""
    import io

    from agent.progress import ConsoleReporter

    stream = io.StringIO()
    reporter = ConsoleReporter(stream=stream, config=cfg)
    reporter({"kind": "thinking", "iteration": 0})
    reporter({"kind": "tool_start", "iteration": 0, "tool_name": "retrieve_evidence",
              "args": {"query": "protein folding"}})
    reporter({"kind": "tool_end", "iteration": 0, "tool_name": "retrieve_evidence",
              "result": ok_result(["pA__c0000"]), "latency_ms": 840})
    reporter({"kind": "answering", "iteration": 1})

    lines = stream.getvalue().splitlines()
    assert lines[0] == "[1] thinking..."
    assert lines[1].startswith("[1] → retrieve_evidence(query='protein folding') ... ")
    assert lines[1].endswith("1 chunk, 840ms")
    assert lines[2] == "[2] answering"


@pytest.mark.parametrize("reason, phrase", [
    ("tools_exhausted", "budget is used"), ("no_progress", "without a tool actually running"),
])
def test_progress_names_why_the_loop_stopped(cfg, reason, phrase):
    import io

    from agent.progress import ConsoleReporter

    stream = io.StringIO()
    ConsoleReporter(stream=stream, config=cfg)({"kind": "cap", "iteration": 4, "reason": reason})
    line = stream.getvalue().strip()
    assert line.startswith("[5] ") and phrase in line and "forcing an answer" in line
