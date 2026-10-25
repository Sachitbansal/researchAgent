"""
Offline tests for CRAG phase 2: filtered gaps ignored, the approval round-trip, the CLI.

In:  synthetic trace records, a tmp corpus with stub embedder/reranker, a ThreadStore on
     tmp_path and a scripted LLM, and main.py with its model-loading imports stubbed.
Out: assertions that only the last unfiltered retrieve decides the proposal (D24), that
     the approval's wording comes from its prompt files and is validated (D26), that turn
     stats carry the proposal and the approval, and the CLI's flag and proposal line.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.fetch_proposal import approval_request, derive_fetch_proposal, is_filtered
from agent.loop import AgentLoop
from memory.threads import ThreadStore
from test_crag import TableReranker, corpus, retrieve  # noqa: F401  (corpus is a fixture)
from test_m5_agent import (  # noqa: F401  (cfg is a fixture)
    ScriptedClient,
    cfg,
    ok_result,
    registry_with,
    text_response,
    tool_response,
)

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ trace records

def gap(query="absent subject", suggested="propose_fetch", **args):
    return {"tool_name": "retrieve_evidence", "args": {"query": query, **args},
            "result_summary": {"chunks": [], "sufficient_evidence": False, "coverage_gap": {
                "searched_for": query, "best_score": -7.5, "threshold": -3.0,
                "scorer": "cross_encoder", "closest_topics": ["topic_a"],
                "suggested_next": suggested, "proposed_query": query,
                "filtered": bool(args.get("topic_filter") or args.get("chunk_types"))}}}


def passed(query="held subject", **args):
    return {"tool_name": "retrieve_evidence", "args": {"query": query, **args},
            "result_summary": {"chunks": [{"chunk_id": "pA__c0000"}],
                               "sufficient_evidence": True}}


SEARCH = {"tool_name": "search_literature", "args": {"query": "absent subject"},
          "result_summary": {"papers_added": []}}


def test_an_unfiltered_gap_survives_a_later_filtered_pass():
    proposal = derive_fetch_proposal([gap(), passed("a figure", chunk_types=["figure"])])
    assert proposal is not None and proposal["query"] == "absent subject"


def test_a_filtered_gap_after_an_unfiltered_pass_proposes_nothing():
    assert derive_fetch_proposal([passed(), gap(chunk_types=["figure"])]) is None
    assert derive_fetch_proposal([passed(), gap(topic_filter="topic_a")]) is None


def test_filtered_gaps_alone_propose_nothing():
    assert derive_fetch_proposal([gap(chunk_types=["table"]),
                                  gap("other", topic_filter="topic_b")]) is None


def test_a_search_after_the_last_unfiltered_gap_clears_it():
    assert derive_fetch_proposal([gap(), SEARCH]) is None
    # A filtered gap after the search does not bring it back.
    assert derive_fetch_proposal([gap(), SEARCH, gap(chunk_types=["figure"])]) is None


def test_the_last_unfiltered_call_decides_even_past_filtered_ones():
    records = [passed(), gap("x", chunk_types=["figure"]), gap("the real gap")]
    assert derive_fetch_proposal(records)["query"] == "the real gap"
    assert derive_fetch_proposal([gap(), gap("x", chunk_types=["figure"]), passed()]) is None


def test_an_empty_filter_counts_as_no_filter():
    assert is_filtered({"args": {"query": "q", "topic_filter": "", "chunk_types": []}}) is False
    assert is_filtered({"args": {"query": "q", "topic_filter": None}}) is False
    assert is_filtered({"args": {"query": "q", "chunk_types": ["text"]}}) is True
    assert derive_fetch_proposal([gap(topic_filter="", chunk_types=[])]) is not None


def test_an_empty_index_proposes_the_query_it_was_asked():
    cold = {"tool_name": "retrieve_evidence", "args": {"query": "anything at all"},
            "result_summary": {"error": "empty_index", "detail": "no chunks are indexed"}}
    assert derive_fetch_proposal([cold]) == {"query": "anything at all",
                                             "searched_for": "anything at all",
                                             "best_score": None, "closest_topics": []}
    assert derive_fetch_proposal([cold, SEARCH]) is None


def test_the_gap_says_whether_the_call_was_filtered(corpus):  # noqa: F811
    unfiltered = retrieve(corpus, reranker=TableReranker(default=-9.0))["coverage_gap"]
    by_topic = retrieve(corpus, reranker=TableReranker(default=-9.0),
                        topic_filter="alpha")["coverage_gap"]
    by_type = retrieve(corpus, reranker=TableReranker(default=-9.0),
                       chunk_types=["text"])["coverage_gap"]
    assert unfiltered["filtered"] is False
    assert by_topic["filtered"] is True and by_type["filtered"] is True


# ------------------------------------------------------------------ approval

def test_the_approval_wording_comes_from_the_prompt_files(cfg, tmp_path):  # noqa: F811
    built = approval_request("  sparse   attention\nkernels ", cfg)
    assert built["query"] == "sparse attention kernels"
    expected = cfg.prompt("fetch_approval").strip().replace("{query}", "sparse attention kernels")
    assert built["question"] == expected
    assert "search_literature" in built["question"]
    assert built["display"] == cfg.prompt("fetch_approval_title").strip().replace(
        "{query}", "sparse attention kernels")

    # Edit the file, and the wording follows: nothing is hardcoded.
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "fetch_approval.md").write_text("GO FETCH <{query}> {not_a_field}\n")
    (prompts / "fetch_approval_title.md").write_text("Fetch: {query}\n")
    cfg.paths.prompts = prompts
    built = approval_request("a {braced} query", cfg)
    assert built["question"] == "GO FETCH <a {braced} query> {not_a_field}"
    assert built["display"] == "Fetch: a {braced} query"


def test_the_approved_query_is_validated(cfg):  # noqa: F811
    limit = int(cfg.agent.fetch_query_max_chars)
    assert approval_request("", cfg)["error"] == "empty_fetch_query"
    assert approval_request("   \n ", cfg)["error"] == "empty_fetch_query"
    assert approval_request(None, cfg)["error"] == "bad_fetch_query"
    assert approval_request("x" * (limit + 1), cfg)["error"] == "fetch_query_too_long"
    assert "error" not in approval_request("x" * limit, cfg)


def test_a_missing_approval_prompt_is_an_error_not_a_raise(cfg, tmp_path):  # noqa: F811
    cfg.paths.prompts = tmp_path
    assert approval_request("q", cfg)["error"] == "fetch_prompt_missing"


# ------------------------------------------------------------------ turn stats

GAP_RESULT = gap()["result_summary"]


def test_a_thread_turn_stores_its_fetch_proposal(cfg, tmp_path):  # noqa: F811
    store = ThreadStore(tmp_path / "threads.sqlite3")
    tid = store.create_thread("t")["id"]
    client = ScriptedClient(tool_response([("c1", "retrieve_evidence", {"query": "absent subject"})]),
                            text_response("The corpus has no papers on this."),
                            text_response("absent subject keywords"))  # the rewrite (D28)
    loop = AgentLoop(registry_with(cfg, lambda **kw: dict(GAP_RESULT)), client=client,
                     config=cfg, store=store)
    result = loop.run("What about the absent subject?", thread_id=tid)
    assert result["fetch_proposal"]["query"] == "absent subject keywords"
    assert result["fetch_proposal"]["agent_query"] == "absent subject"
    assert result["fetch_proposal"]["query_source"] == "rewritten"
    stats = store.get_thread(tid)["turns"][0]["stats"]
    assert stats["fetch_proposal"] == result["fetch_proposal"]
    assert "approved_fetch" not in stats


def test_an_approval_run_is_stored_under_its_readable_question(cfg, tmp_path):  # noqa: F811
    store = ThreadStore(tmp_path / "threads.sqlite3")
    tid = store.create_thread("t")["id"]
    store.append_turn(tid, "What about the absent subject?", "The corpus lacks it.")
    approval = approval_request("absent subject", cfg)
    client = ScriptedClient(tool_response([("c1", "retrieve_evidence", {"query": "absent subject"})]),
                            text_response("Now answered [pA__c0000]."))
    loop = AgentLoop(registry_with(cfg, lambda **kw: ok_result()), client=client,
                     config=cfg, store=store)
    result = loop.run(approval["question"], thread_id=tid, approval=approval)
    assert result["approved_fetch"] == {"query": "absent subject"}

    # The agent was sent the prompt-file text, after the earlier turn as history.
    sent = client.requests[0]["messages"]
    assert sent[-1] == {"role": "user", "content": approval["question"]}
    assert any(m.get("content") == "What about the absent subject?" for m in sent)

    turn = store.get_thread(tid)["turns"][1]
    assert turn["question"] == approval["display"]
    assert turn["stats"]["approved_fetch"] == {"query": "absent subject"}
    assert turn["stats"]["fetch_proposal"] is None


# ------------------------------------------------------------------ the CLI

@pytest.fixture()
def cli(monkeypatch, cfg, tmp_path):  # noqa: F811
    """main.py with every model load stubbed and the store on tmp_path."""
    sys.path.insert(0, str(ROOT))
    import main as cli_main
    import agent.tool_registry as registry_module
    import config as config_module
    import memory.longterm as longterm_module
    import memory.threads as threads_module
    import retrieval.embedder as embedder_module
    import tools.retrieve_evidence as retrieve_module
    import analysis.nli as nli_module

    runs = []

    class FakeLoop:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, question, thread_id=None, approval=None):
            runs.append({"question": question, "thread_id": thread_id, "approval": approval})
            return {"answer": "The corpus has no papers on it.", "iterations": 1, "tool_calls": 1,
                    "context_tokens": 10, "stopped_because": "answered",
                    "tool_call_counts": {"retrieve_evidence": 1}, "trace": "t.jsonl",
                    "trace_records": [], "thread_id": thread_id, "turn_seq": 1,
                    "fetch_proposal": None if approval else {"query": "absent subject"}}

        def remember(self, result):
            return None

    store = ThreadStore(tmp_path / "threads.sqlite3")
    monkeypatch.setattr(config_module, "CFG", cfg)
    monkeypatch.setattr(cfg.paths, "ensure", lambda: None, raising=False)
    monkeypatch.setattr("agent.loop.AgentLoop", FakeLoop)
    monkeypatch.setattr(registry_module, "build_full_registry", lambda **kw: None)
    monkeypatch.setattr(embedder_module, "Embedder", lambda *a, **k: None)
    monkeypatch.setattr(retrieve_module, "EvidenceRetriever",
                        lambda *a, **k: SimpleNamespace(embedder=None))
    monkeypatch.setattr(nli_module, "NLIModel", lambda *a, **k: None)
    monkeypatch.setattr(longterm_module, "build_long_term_memory", lambda *a, **k: None)
    monkeypatch.setattr(threads_module, "ThreadStore", lambda *a, **k: store)
    return SimpleNamespace(main=cli_main, runs=runs, store=store)


def _ask(cli, argv):
    args = cli.main.build_parser().parse_args(["ask", *argv])
    return args, cli.main.cmd_ask(args)


def test_the_cli_prints_the_proposal_with_how_to_approve(cli, capsys):
    tid = cli.store.create_thread("t")["id"]
    _, code = _ask(cli, ["What about it?", "--thread", tid, "--quiet"])
    out = capsys.readouterr().out
    assert code == 0
    assert '--- fetch proposal: "absent subject"  (re-ask with --approve-fetch)' in out


def test_the_cli_approval_sends_the_prompt_file_text(cli, cfg, capsys):  # noqa: F811
    tid = cli.store.create_thread("t")["id"]
    args, code = _ask(cli, ["--thread", tid, "--approve-fetch", "absent subject", "--quiet"])
    assert code == 0 and args.approve_fetch == "absent subject"
    run = cli.runs[-1]
    assert run["question"] == approval_request("absent subject", cfg)["question"]
    assert run["approval"]["display"] == approval_request("absent subject", cfg)["display"]
    assert "fetch proposal" not in capsys.readouterr().out


def test_the_cli_approval_needs_a_thread_and_no_question(cli, capsys):
    _, code = _ask(cli, ["--approve-fetch", "absent subject", "--quiet"])
    assert code == 1 and "approval_needs_thread" in capsys.readouterr().err
    tid = cli.store.create_thread("t")["id"]
    _, code = _ask(cli, ["a question", "--thread", tid, "--approve-fetch", "q", "--quiet"])
    assert code == 1 and "question_and_approval" in capsys.readouterr().err
    _, code = _ask(cli, ["--thread", tid, "--approve-fetch", "  ", "--quiet"])
    assert code == 1 and "empty_fetch_query" in capsys.readouterr().err
    assert cli.runs == []


def test_the_cli_still_needs_a_question_without_an_approval(cli, capsys):
    _, code = _ask(cli, ["--quiet"])
    assert code == 1 and "empty_question" in capsys.readouterr().err


def test_cited_answer_gets_no_card_even_after_a_failed_search():
    """D34: the corpus answered, so a late failed search must not put a card under it."""
    assert derive_fetch_proposal([passed(), gap()], "It does [pA__c0001].") is None


def test_uncited_answer_gets_a_card_even_when_a_chunk_passed_the_gate():
    """D34: the agent could not answer from what passed (af01), so offer the fetch."""
    proposal = derive_fetch_proposal([passed("few-step sampling")],
                                     "The indexed corpus does not cover this.")
    assert proposal is not None and proposal["searched_for"] == "few-step sampling"


def test_uncited_answer_prefers_the_gap_subject():
    proposal = derive_fetch_proposal([gap("the real gap"), passed("later wording")],
                                     "Not covered.")
    assert proposal["searched_for"] == "the real gap"
