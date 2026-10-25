"""
Offline tests for the fetch-query rewrite (D28): the proposal's query is rewritten from
the user's question in one `utility` call, and falls back to the agent's query on junk.

In:  scripted LLM stubs, a tmp-rooted config, and a tmp ThreadStore. No network.
Out: assertions on the rewritten proposal's fields, the fallbacks, the call count, the
     prompt file being the prompt, and the earlier question riding along in a thread.
"""

from __future__ import annotations

import pytest

from agent.fetch_query import build_prompt, clean_phrase, rewrite_fetch_query
from agent.loop import AgentLoop
from llm_client import LLMError
from memory.threads import ThreadStore
from test_crag_phase2 import GAP_RESULT
from test_m5_agent import (  # noqa: F401  (cfg is a fixture)
    ScriptedClient,
    _config_at,
    cfg,
    ok_result,
    registry_with,
    text_response,
    tool_response,
)

QUESTION = "How do recent methods quantize vision transformers to 4 bits after training?"
PROPOSAL = {"query": "vision image classification transformer ViT",
            "searched_for": "vision image classification transformer ViT",
            "best_score": -8.1, "closest_topics": ["topic_a"]}
GOOD = "4-bit post-training quantization vision transformers"


def rewrite(reply, cfg, question=QUESTION, previous=None):  # noqa: F811
    client = ScriptedClient(reply)
    return rewrite_fetch_query(dict(PROPOSAL), question, previous, client=client,
                               config=cfg), client


# ------------------------------------------------------------------ the helper

def test_a_good_rewrite_replaces_the_query_and_keeps_the_agents(cfg):  # noqa: F811
    result, client = rewrite(text_response(f'  "{GOOD}"\n'), cfg)
    assert result["query"] == GOOD
    assert result["query_source"] == "rewritten"
    assert result["agent_query"] == PROPOSAL["query"]
    assert result["searched_for"] == PROPOSAL["searched_for"]
    assert result["closest_topics"] == ["topic_a"] and result["best_score"] == -8.1
    assert "rewrite_error" not in result
    assert len(client.requests) == 1
    assert client.requests[0]["kwargs"]["max_retries"] == 1  # one call, no transport retry


@pytest.mark.parametrize("reply", [
    LLMError("llm_timeout", "request timed out"),
    text_response(""),
    text_response("   "),
    text_response("How do methods quantize vision transformers to 4 bits?"),
    text_response("Methods quantize vision transformers to 4 bits after training."),
    text_response("What methods quantize ViTs to 4 bits"),
    text_response("4-bit ViT quantization\nThis phrase keeps the bit-width."),
    text_response("one two three four five six seven eight nine ten eleven twelve thirteen"),
], ids=["llm_error", "empty", "blank", "question", "sentence", "question_word",
        "multi_line", "too_many_words"])
def test_junk_or_a_failure_falls_back_to_the_agent_query(cfg, reply):  # noqa: F811
    result, _ = rewrite(reply, cfg)
    assert result["query"] == PROPOSAL["query"]
    assert result["agent_query"] == PROPOSAL["query"]
    assert result["query_source"] == "agent"
    assert result["rewrite_error"]


def test_a_broken_client_falls_back_instead_of_raising(cfg):  # noqa: F811
    class Broken:
        def complete(self, *args, **kwargs):
            raise RuntimeError("stub exploded")

    result = rewrite_fetch_query(dict(PROPOSAL), QUESTION, client=Broken(), config=cfg)
    assert result["query_source"] == "agent" and "stub exploded" in result["rewrite_error"]


def test_the_rewrite_is_capped_by_fetch_query_max_chars(tmp_path, monkeypatch):
    capped = _config_at(tmp_path, monkeypatch, fetch_query_max_chars=30)
    assert clean_phrase("x" * 30, capped) == ("x" * 30, "")
    result, _ = rewrite(text_response(GOOD), capped)  # 52 characters
    assert len(GOOD) > 30
    assert result["query"] == PROPOSAL["query"] and result["query_source"] == "agent"
    assert "limit is 30" in result["rewrite_error"]


def test_the_prompt_is_loaded_from_its_file(cfg, tmp_path):  # noqa: F811
    real = build_prompt(QUESTION, None, cfg)
    assert QUESTION in real and "{question}" not in real and "{previous_question}" not in real

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "fetch_query.md").write_text("REWRITE <{question}> AFTER <{previous_question}>\n")
    cfg.paths.prompts = prompts
    _, client = rewrite(text_response(GOOD), cfg, question="a {previous_question} q",
                        previous="earlier")
    sent = client.requests[0]["messages"]
    # Filled in one pass: a placeholder inside the question is not substituted again.
    assert sent == [{"role": "user",
                     "content": "REWRITE <a {previous_question} q> AFTER <earlier>\n"}]


def test_a_missing_prompt_file_falls_back(cfg, tmp_path):  # noqa: F811
    cfg.paths.prompts = tmp_path
    result, client = rewrite(text_response(GOOD), cfg)
    assert result["query_source"] == "agent" and "prompt not found" in result["rewrite_error"]
    assert client.requests == []


# ------------------------------------------------------------------ in the loop

def test_the_loop_rewrites_a_proposal_with_one_utility_call(cfg):  # noqa: F811
    roles = []

    class Recording(ScriptedClient):
        def complete(self, messages, tools=None, role="agent", **kwargs):
            roles.append(role)
            return super().complete(messages, tools=tools, role=role, **kwargs)

    client = Recording(tool_response([("c1", "retrieve_evidence", {"query": PROPOSAL["query"]})]),
                       text_response("The corpus has no papers on this."),
                       text_response(GOOD))
    loop = AgentLoop(registry_with(cfg, lambda **kw: dict(GAP_RESULT)), client=client,
                     config=cfg)
    result = loop.run(QUESTION)
    assert roles == ["agent", "agent", "utility"]
    assert result["fetch_proposal"]["query"] == GOOD
    assert result["fetch_proposal"]["agent_query"] == "absent subject"
    assert QUESTION in client.requests[-1]["messages"][0]["content"]


def test_no_rewrite_call_without_a_proposal(cfg):  # noqa: F811
    client = ScriptedClient(tool_response([("c1", "retrieve_evidence", {"query": "held"})]),
                            text_response("Answered [pA__c0000]."))
    loop = AgentLoop(registry_with(cfg, lambda **kw: ok_result()), client=client, config=cfg)
    result = loop.run("A question the corpus answers?")
    assert result["fetch_proposal"] is None
    assert len(client.requests) == 2  # the agent's two turns, and nothing after


def test_a_thread_follow_up_sends_the_previous_question_as_context(cfg, tmp_path):  # noqa: F811
    store = ThreadStore(tmp_path / "threads.sqlite3")
    tid = store.create_thread("t")["id"]
    store.append_turn(tid, QUESTION, "The corpus lacks it.")
    client = ScriptedClient(tool_response([("c1", "retrieve_evidence", {"query": "2-bit ViT"})]),
                            text_response("Still nothing."),
                            text_response("2-bit post-training quantization vision transformers"))
    loop = AgentLoop(registry_with(cfg, lambda **kw: dict(GAP_RESULT)), client=client,
                     config=cfg, store=store)
    result = loop.run("What about 2-bit?", thread_id=tid)

    prompt = client.requests[-1]["messages"][0]["content"]
    assert "What about 2-bit?" in prompt and QUESTION in prompt
    proposal = result["fetch_proposal"]
    assert proposal["query"] == "2-bit post-training quantization vision transformers"
    assert proposal["query_source"] == "rewritten"
    # The turn is saved after the rewrite, so a reloaded chat shows the same card.
    assert store.get_thread(tid)["turns"][1]["stats"]["fetch_proposal"] == proposal


def test_the_first_turn_of_a_thread_has_no_earlier_question(cfg, tmp_path):  # noqa: F811
    store = ThreadStore(tmp_path / "threads.sqlite3")
    tid = store.create_thread("t")["id"]
    client = ScriptedClient(tool_response([("c1", "retrieve_evidence", {"query": "x"})]),
                            text_response("Nothing."), text_response(GOOD))
    loop = AgentLoop(registry_with(cfg, lambda **kw: dict(GAP_RESULT)), client=client,
                     config=cfg, store=store)
    loop.run(QUESTION, thread_id=tid)
    prompt = client.requests[-1]["messages"][0]["content"]
    assert prompt.splitlines()[-2].strip() == "Earlier question:"



def test_approval_run_skips_the_rewrite(cfg):  # noqa: F811
    """An approval turn's proposal is shown nowhere, so it must not pay for a rewrite."""
    roles = []

    class Recording(ScriptedClient):
        def complete(self, messages, tools=None, role="agent", **kwargs):
            roles.append(role)
            return super().complete(messages, tools=tools, role=role, **kwargs)

    client = Recording(tool_response([("c1", "retrieve_evidence", {"query": PROPOSAL["query"]})]),
                       text_response("The corpus has no papers on this."))
    loop = AgentLoop(registry_with(cfg, lambda **kw: dict(GAP_RESULT)), client=client,
                     config=cfg)
    result = loop.run(QUESTION, approval={"query": "absent subject", "display": "Fetch"})
    assert "utility" not in roles
    assert result["fetch_proposal"]["query"] == "absent subject"
