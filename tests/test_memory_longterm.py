"""
Offline tests for long-term memory (docs/DECISIONS.md D19, D20).

In:  a MemoryStore on tmp_path with a controllable clock, stub embedder / NLI / LLM, a
     hand-built chunk table, and the agent loop with a stubbed LLM. No network, no models.
Out: assertions on the writer (entailed only, never from abstentions or outside threads,
     duplicate -> refresh, contradiction -> supersede), forgetting (decay math, cap,
     min_strength, stale hashes, the use rule, purge), the reader (query prefix, budget,
     injected chunks visible / seen / released), and migration of an existing threads DB.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import yaml

from agent.conversation import Conversation
from agent.loop import AgentLoop
from config import load_config
from llm_client import LLMError
from memory import forgetting
from memory.longterm import LongTermMemory, cited_chunk_ids, gate_passed
from memory.memory_store import MemoryStore
from memory.threads import ThreadStore
from test_m5_agent import ScriptedClient, ok_result, registry_with, text_response, tool_response

DIM = 384
REPO = Path(__file__).resolve().parent.parent
DAY = forgetting.DAY_S

# ---------------------------------------------------------------- fixtures and stubs


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance_days(self, days: float) -> None:
        self.t += days * DAY


def _config(tmp_path, monkeypatch, memory=None, agent=None):
    """A Config rooted at tmp_path with memory/agent settings overridden in the YAML."""
    import config as config_module

    repo_root = REPO
    raw = yaml.safe_load((repo_root / "config.yaml").read_text(encoding="utf-8"))
    raw["memory"].update(memory or {})
    raw["agent"].update(agent or {})
    local = tmp_path / "config.yaml"
    local.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setattr(config_module, "REPO_ROOT", tmp_path)
    built = load_config(local)
    built.paths.ensure()
    built.paths.prompts = repo_root / "src" / "prompts"
    return built


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    return _config(tmp_path, monkeypatch)


def unit(*weights: float) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[: len(weights)] = weights
    return v / np.linalg.norm(v)


def at_cosine(c: float) -> np.ndarray:
    """A unit vector at cosine `c` from unit(1)."""
    return unit(c, math.sqrt(max(0.0, 1 - c * c)))


class StubEmbedder:
    """Explicit vectors for named texts, else a bag-of-words hash (identical text -> 1.0)."""

    query_prefix = "QUERY: "

    def __init__(self, vectors: Dict[str, np.ndarray] = None) -> None:
        self.vectors = dict(vectors or {})
        self.queries: List[str] = []
        self.passages: List[str] = []

    def _vec(self, text: str) -> np.ndarray:
        if text in self.vectors:
            return np.asarray(self.vectors[text], dtype=np.float32)
        v = np.zeros(DIM, dtype=np.float32)
        for word in text.lower().replace(".", " ").split():
            v[hash(word) % (DIM - 8) + 8] += 1.0
        return v / (np.linalg.norm(v) or 1.0)

    def encode_passages(self, texts, cache=None):
        self.passages.extend(texts)
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, DIM))

    def encode_query(self, query):
        self.queries.append(query)
        return self._vec(query)


class StubNLI:
    """entailed: hypotheses that are entailed by any premise. contradicts: (a, b) pairs, in
    the direction given, that are confident contradictions. Everything else is neutral."""

    def __init__(self, entailed=(), contradicts=()) -> None:
        self.entailed = set(entailed)
        self.contradicts = set(contradicts)
        self.pairs: List[tuple] = []

    def predict(self, pairs):
        out = []
        for premise, hypothesis in pairs:
            self.pairs.append((premise, hypothesis))
            if (premise, hypothesis) in self.contradicts:
                scores = {"contradiction": 0.95, "entailment": 0.02, "neutral": 0.03}
            elif hypothesis in self.entailed:
                scores = {"contradiction": 0.02, "entailment": 0.95, "neutral": 0.03}
            else:
                scores = {"contradiction": 0.05, "entailment": 0.05, "neutral": 0.9}
            out.append({"label": max(scores, key=scores.get), "scores": scores})
        return out


def chunk(chunk_id: str, text: str, digest: str) -> Dict[str, Any]:
    return {"chunk_id": chunk_id, "paper_id": chunk_id.split("__")[0], "paper_title": "Paper",
            "page": 2, "section": "Method", "chunk_type": "text", "text": text,
            "content_hash": digest}


CLAIM_A = "MethodA uses a balancing loss with weight 0.01."
CLAIM_B = "MethodA routes every token to two experts."
TEXT_A = f"{CLAIM_A} The loss is added to the task objective at every layer."
TEXT_B = f"{CLAIM_B} Routing is computed per layer."


def table_ab() -> Dict[str, Dict[str, Any]]:
    return {"pA__c0001": chunk("pA__c0001", TEXT_A, "hA"),
            "pB__c0002": chunk("pB__c0002", TEXT_B, "hB")}


def table_citing(claim: str) -> Dict[str, Dict[str, Any]]:
    """table_ab, with pB__c0002's text holding `claim` as a sentence, so the grounding
    step's premise selection finds it whatever vector the test gives the claim."""
    table = table_ab()
    table["pB__c0002"] = chunk("pB__c0002", f"{claim} Further detail follows in the text.", "hB")
    return table


def extraction(findings, preferences=()):
    return text_response(json.dumps({"findings": findings, "preferences": list(preferences)}))


def passing_trace(chunk_ids=("pA__c0001", "pB__c0002")):
    return [{"tool_name": "retrieve_evidence", "error": None,
             "result_summary": {"sufficient_evidence": True}, "chunk_ids_returned": list(chunk_ids)}]


def failing_trace():
    return [{"tool_name": "retrieve_evidence", "error": None,
             "result_summary": {"sufficient_evidence": False}, "chunk_ids_returned": []}]


def make(cfg, tmp_path, client=None, embedder=None, nli=None, table=None, clock=None):
    clock = clock or Clock()
    store = MemoryStore(tmp_path / "threads.sqlite3", dim=DIM, now=clock)
    tbl = table if table is not None else table_ab()
    memory = LongTermMemory(store, cfg, embedder=embedder or StubEmbedder(),
                            nli=nli or StubNLI(), client=client or ScriptedClient(),
                            chunk_table=lambda: tbl, now=clock)
    return memory, store, clock


def seed(store, text, chunk_ids=("pA__c0001",), hashes=None, vector=None, kind="finding"):
    hashes = hashes if hashes is not None else {c: table_ab()[c]["content_hash"]
                                                for c in chunk_ids if c in table_ab()}
    made = store.add(kind, text, list(chunk_ids), hashes,
                     vector if vector is not None else StubEmbedder()._vec(text), "t" * 32, "q")
    return made["memory"]["id"]


TID = "a" * 32
ANSWER = f"{CLAIM_A} [pA__c0001] And {CLAIM_B} [pB__c0002]"


# ================================================================ helpers

def test_cited_chunk_ids_reads_brackets_only():
    text = "x [pA__c0001][pB__c0002] y [pA__c0001, pC__c0003] mentions pD__c0004 in prose"
    assert cited_chunk_ids(text) == ["pA__c0001", "pB__c0002", "pC__c0003"]


def test_gate_passed_needs_a_passing_retrieval_with_chunks():
    assert gate_passed(passing_trace())
    assert not gate_passed(failing_trace())
    assert not gate_passed([{"tool_name": "analyze_corpus", "result_summary": {},
                             "chunk_ids_returned": ["x__c0001"]}])
    assert not gate_passed([])


# ================================================================ writer

def test_only_entailed_claims_are_written(cfg, tmp_path):
    client = ScriptedClient(extraction([
        {"text": CLAIM_A, "chunk_ids": ["pA__c0001"]},
        {"text": "MethodA beats every baseline by ten points.", "chunk_ids": ["pB__c0002"]},
    ]))
    memory, store, _ = make(cfg, tmp_path, client=client, nli=StubNLI(entailed={CLAIM_A}))
    report = memory.after_run("q?", ANSWER, TID, passing_trace())

    assert [w["text"] for w in report["written"]] == [CLAIM_A]
    assert report["dropped"] == {"uncited": 0, "ungrounded": 1}
    rows = store.list()["memories"]
    assert [r["text"] for r in rows] == [CLAIM_A]
    assert rows[0]["chunk_ids"] == ["pA__c0001"] and rows[0]["content_hashes"] == {"pA__c0001": "hA"}
    assert rows[0]["source_thread_id"] == TID and rows[0]["use_count"] == 1
    # One utility call, the extraction prompt, with the citations left in the answer.
    assert len(client.requests) == 1
    sent = client.requests[0]["messages"]
    assert sent[0]["content"] == cfg.prompt("memory_extract")
    assert "[pA__c0001]" in sent[1]["content"]


def test_a_claim_resting_on_chunks_not_cited_or_not_seen_is_dropped(cfg, tmp_path):
    client = ScriptedClient(extraction([
        {"text": CLAIM_A, "chunk_ids": ["pA__c0001"]},           # cited, but never retrieved
        {"text": CLAIM_B, "chunk_ids": ["pZ__c0009"]},           # not cited by the answer
    ]))
    memory, store, _ = make(cfg, tmp_path, client=client,
                            nli=StubNLI(entailed={CLAIM_A, CLAIM_B}))
    report = memory.after_run("q?", ANSWER, TID, passing_trace(["pB__c0002"]))
    assert report["written"] == [] and report["dropped"]["uncited"] == 2
    assert store.list()["memories"] == []


def test_an_injected_chunk_counts_as_seen_for_the_writer(cfg, tmp_path):
    client = ScriptedClient(extraction([{"text": CLAIM_A, "chunk_ids": ["pA__c0001"]}]))
    memory, store, _ = make(cfg, tmp_path, client=client, nli=StubNLI(entailed={CLAIM_A}))
    report = memory.after_run("q?", ANSWER, TID, passing_trace(["pB__c0002"]),
                              injected={"memories": [], "chunk_ids": ["pA__c0001"]})
    assert [w["text"] for w in report["written"]] == [CLAIM_A]


def test_abstentions_are_never_written(cfg, tmp_path):
    client = ScriptedClient(extraction([{"text": CLAIM_A, "chunk_ids": ["pA__c0001"]}]))
    memory, store, _ = make(cfg, tmp_path, client=client, nli=StubNLI(entailed={CLAIM_A}))
    report = memory.after_run("q?", "I cannot answer this from the evidence [pA__c0001].", TID,
                              failing_trace())
    assert report["skipped"] == "abstained" and report["written"] == []
    assert client.requests == [], "no extraction call for an abstention"
    assert store.list(include_inactive=True)["memories"] == []


def test_no_thread_means_no_memory_in_after_run(cfg, tmp_path):
    client = ScriptedClient(extraction([{"text": CLAIM_A, "chunk_ids": ["pA__c0001"]}]))
    memory, store, _ = make(cfg, tmp_path, client=client, nli=StubNLI(entailed={CLAIM_A}))
    assert memory.after_run("q?", ANSWER, None, passing_trace())["skipped"] == "no_thread"
    assert client.requests == [] and store.list(include_inactive=True)["memories"] == []


def test_duplicate_refreshes_instead_of_adding(cfg, tmp_path):
    reworded = "MethodA weights its balancing loss by 0.01."
    vectors = {CLAIM_A: unit(1), reworded: at_cosine(0.97)}
    client = ScriptedClient(extraction([{"text": reworded, "chunk_ids": ["pB__c0002"]}]))
    memory, store, clock = make(cfg, tmp_path, client=client, embedder=StubEmbedder(vectors),
                                nli=StubNLI(entailed={reworded}), table=table_citing(reworded))
    old = seed(store, CLAIM_A, vector=unit(1))
    clock.advance_days(10)
    report = memory.after_run("q?", f"{reworded} [pB__c0002]", TID, passing_trace())

    assert report["written"] == [] and [r["id"] for r in report["refreshed"]] == [old]
    assert report["refreshed"][0]["chunk_ids_added"] == ["pB__c0002"]
    rows = store.list()["memories"]
    assert len(rows) == 1
    assert rows[0]["use_count"] == 2 and rows[0]["last_used_at"] == clock.t
    assert rows[0]["chunk_ids"] == ["pA__c0001", "pB__c0002"]
    assert rows[0]["content_hashes"] == {"pA__c0001": "hA", "pB__c0002": "hB"}


def test_contradiction_invalidates_the_old_memory(cfg, tmp_path):
    newer = "MethodA uses a balancing loss with weight 0.1."
    vectors = {CLAIM_A: unit(1), newer: at_cosine(0.82)}          # related, not duplicate
    nli = StubNLI(entailed={newer}, contradicts={(CLAIM_A, newer), (newer, CLAIM_A)})
    client = ScriptedClient(extraction([{"text": newer, "chunk_ids": ["pB__c0002"]}]))
    memory, store, _ = make(cfg, tmp_path, client=client, embedder=StubEmbedder(vectors), nli=nli,
                            table=table_citing(newer))
    old = seed(store, CLAIM_A, vector=unit(1))
    report = memory.after_run("q?", f"{newer} [pB__c0002]", TID, passing_trace())

    new_id = report["written"][0]["id"]
    assert report["invalidated"] == [{**report["invalidated"][0], "id": old,
                                      "reason": "superseded", "superseded_by": new_id}]
    everything = {m["id"]: m for m in store.list(include_inactive=True)["memories"]}
    assert everything[old]["invalidated_at"] is not None and not everything[old]["active"]
    assert everything[old]["superseded_by"] == new_id and everything[old]["forget_reason"] == "superseded"
    # Kept for history, never shown again.
    assert [m["id"] for m in store.active()["memories"]] == [new_id]


def test_a_contradiction_wins_over_a_near_duplicate(cfg, tmp_path):
    """A number swap sits as close as a paraphrase; refreshing would merge it."""
    swapped = "MethodA uses a balancing loss with weight 0.02."
    vectors = {CLAIM_A: unit(1), swapped: at_cosine(0.99)}
    nli = StubNLI(entailed={swapped}, contradicts={(CLAIM_A, swapped), (swapped, CLAIM_A)})
    client = ScriptedClient(extraction([{"text": swapped, "chunk_ids": ["pB__c0002"]}]))
    memory, store, _ = make(cfg, tmp_path, client=client, embedder=StubEmbedder(vectors), nli=nli,
                            table=table_citing(swapped))
    old = seed(store, CLAIM_A, vector=unit(1))
    report = memory.after_run("q?", f"{swapped} [pB__c0002]", TID, passing_trace())
    assert report["refreshed"] == [] and [i["id"] for i in report["invalidated"]] == [old]


def test_a_one_way_contradiction_is_not_confident(cfg, tmp_path):
    newer = "MethodA uses a balancing loss with weight 0.1."
    vectors = {CLAIM_A: unit(1), newer: at_cosine(0.82)}
    nli = StubNLI(entailed={newer}, contradicts={(CLAIM_A, newer)})       # one direction only
    client = ScriptedClient(extraction([{"text": newer, "chunk_ids": ["pB__c0002"]}]))
    memory, store, _ = make(cfg, tmp_path, client=client, embedder=StubEmbedder(vectors), nli=nli,
                            table=table_citing(newer))
    seed(store, CLAIM_A, vector=unit(1))
    report = memory.after_run("q?", f"{newer} [pB__c0002]", TID, passing_trace())
    assert report["invalidated"] == [] and len(report["written"]) == 1
    assert len(store.active()["memories"]) == 2


def test_unrelated_claims_are_never_nli_checked_against_each_other(cfg, tmp_path):
    other = "MethodB prunes half of its attention heads."
    vectors = {CLAIM_A: unit(1), other: at_cosine(0.2)}
    nli = StubNLI(entailed={other})
    client = ScriptedClient(extraction([{"text": other, "chunk_ids": ["pB__c0002"]}]))
    memory, store, _ = make(cfg, tmp_path, client=client, embedder=StubEmbedder(vectors), nli=nli,
                            table=table_citing(other))
    seed(store, CLAIM_A, vector=unit(1))
    memory.after_run("q?", f"{other} [pB__c0002]", TID, passing_trace())
    assert not any(CLAIM_A in pair for pair in nli.pairs)


def test_preferences_are_written_without_chunks(cfg, tmp_path):
    client = ScriptedClient(extraction([], preferences=["Only consider papers from 2024 onward."]))
    memory, store, _ = make(cfg, tmp_path, client=client)
    report = memory.after_run("Only consider papers from 2024 onward. What losses?", ANSWER, TID,
                              passing_trace())
    assert [(w["kind"], w["chunk_ids"]) for w in report["written"]] == [("preference", [])]


def test_a_writer_failure_is_reported_never_raised(cfg, tmp_path):
    client = ScriptedClient(LLMError("provider_error", "upstream 502"))
    memory, store, _ = make(cfg, tmp_path, client=client)
    report = memory.after_run("q?", ANSWER, TID, passing_trace())
    assert report["error"] == "memory_write_failed" and "provider_error" in report["detail"]
    assert report["written"] == []


def test_a_garbled_extraction_is_an_error_not_a_crash(cfg, tmp_path):
    memory, _, _ = make(cfg, tmp_path, client=ScriptedClient(text_response("no json here")))
    assert memory.extract("q", "a")["error"] == "extraction_failed"
    memory, _, _ = make(cfg, tmp_path, client=ScriptedClient(text_response(
        '{"findings": [{"text": "x y z w", "chunk_ids": "pA__c0001"}], "preferences": [3]}')))
    got = memory.extract("q", "a")
    assert got["findings"][0]["chunk_ids"] == ["pA__c0001"]     # a bare string, wrapped
    assert got["preferences"] == []


# ================================================================ forgetting: math

def test_decay_math():
    assert forgetting.use_weight(1) == 1.0
    assert forgetting.use_weight(0) == 1.0
    assert forgetting.use_weight(3) == pytest.approx(1 + math.log(3))
    assert forgetting.recency(0, 0, 30) == 1.0
    assert forgetting.recency(0, 30 * DAY, 30) == pytest.approx(0.5)
    assert forgetting.recency(0, 90 * DAY, 30) == pytest.approx(0.125)
    m = {"use_count": 1, "last_used_at": 0}
    assert forgetting.strength(m, 60 * DAY, 30) == pytest.approx(0.25)
    assert forgetting.min_strength_for(90, 30) == pytest.approx(0.125)


def test_shipped_min_strength_forgets_a_single_use_after_about_90_days(cfg):
    half, floor = float(cfg.memory.half_life_days), float(cfg.memory.min_strength)
    once = {"use_count": 1, "last_used_at": 0}
    assert forgetting.strength(once, 89 * DAY, half) >= floor
    assert forgetting.strength(once, 91 * DAY, half) < floor
    assert floor == pytest.approx(forgetting.min_strength_for(90, half))


def test_cap_evicts_the_weakest_and_preferences_are_exempt():
    rows = [{"id": f"m{i}", "kind": "finding", "use_count": 1, "last_used_at": i * DAY}
            for i in range(5)]
    rows.append({"id": "pref", "kind": "preference", "use_count": 1, "last_used_at": -999 * DAY})
    victims = forgetting.decay_victims(rows, 5 * DAY, 30, 0.0, max_active=3)
    assert [v["id"] for v in victims] == ["m0", "m1"] and {v["why"] for v in victims} == {"over_cap"}


def test_is_stale_compares_hashes_not_ids():
    m = {"chunk_ids": ["a", "b"], "content_hashes": {"a": "h1", "b": "h2"}}
    assert not forgetting.is_stale(m, {"a": "h1", "b": "h2"})
    assert forgetting.is_stale(m, {"a": "h1", "b": "CHANGED"})      # same id, new text
    assert forgetting.is_stale(m, {"a": "h1"})                      # chunk gone
    assert forgetting.is_stale({"chunk_ids": ["a"], "content_hashes": {}}, {"a": "h1"})


# ================================================================ forgetting: applied

def test_sweep_forgets_below_min_strength(cfg, tmp_path):
    memory, store, clock = make(cfg, tmp_path)
    old = seed(store, CLAIM_A)
    clock.advance_days(2)
    fresh = seed(store, CLAIM_B, chunk_ids=("pB__c0002",))
    clock.advance_days(89)                    # old: 91 days unused, fresh: 89
    forgotten = memory.sweep()
    assert [f["id"] for f in forgotten] == [old] and forgotten[0]["reason"] == "decay"
    rows = {m["id"]: m for m in store.list(include_inactive=True)["memories"]}
    assert rows[old]["forgotten_at"] == clock.t and rows[old]["forget_reason"] == "decay"
    assert rows[fresh]["active"]


def test_sweep_caps_active_findings(tmp_path, monkeypatch):
    cfg = _config(tmp_path, monkeypatch, memory={"max_active": 2})
    memory, store, clock = make(cfg, tmp_path)
    ids = []
    for i in range(4):
        ids.append(seed(store, f"MethodA fact number {i} holds here."))
        clock.advance_days(1)
    forgotten = memory.sweep()
    assert [f["id"] for f in forgotten] == ids[:2]
    assert {f["why"] for f in forgotten} == {"over_cap"}
    assert {m["id"] for m in store.active()["memories"]} == set(ids[2:])


def test_sweep_runs_after_each_write(cfg, tmp_path):
    client = ScriptedClient(extraction([]))
    memory, store, clock = make(cfg, tmp_path, client=client)
    old = seed(store, CLAIM_A)
    clock.advance_days(100)
    report = memory.after_run("q?", ANSWER, TID, passing_trace())
    assert [f["id"] for f in report["forgotten"]] == [old]


def test_stale_hash_invalidates_at_read_time_and_in_the_sweep(cfg, tmp_path):
    question = "What balancing loss does MethodA use?"
    table = table_ab()
    vectors = {question: unit(1), CLAIM_A: unit(1), CLAIM_B: unit(1)}
    memory, store, _ = make(cfg, tmp_path, embedder=StubEmbedder(vectors), table=table)
    changed = seed(store, CLAIM_A, vector=unit(1))
    gone = seed(store, CLAIM_B, chunk_ids=("pB__c0002",), vector=unit(1))
    table["pA__c0001"] = {**table["pA__c0001"], "content_hash": "re-indexed"}   # same id, new text
    recalled = memory.recall(question)
    assert recalled["stale"] == [changed]
    assert [m["id"] for m in recalled["memories"]] == [gone]          # the valid one still goes in
    assert "pA__c0001" not in recalled["chunk_ids"]
    rows = {m["id"]: m for m in store.list(include_inactive=True)["memories"]}
    assert rows[changed]["forget_reason"] == "stale" and rows[changed]["invalidated_at"]
    assert rows[changed]["superseded_by"] is None
    # The sweep catches what no read touched: a chunk removed outright.
    del table["pB__c0002"]
    assert memory.sweep() == [{"id": gone, "reason": "stale"}]
    assert store.active()["memories"] == []


def test_injected_but_uncited_is_not_a_use(cfg, tmp_path):
    client = ScriptedClient(extraction([]), extraction([]))
    memory, store, clock = make(cfg, tmp_path, client=client)
    mid = seed(store, CLAIM_A)
    injected = {"memories": [{"id": mid, "text": CLAIM_A, "chunk_ids": ["pA__c0001"]}],
                "chunk_ids": ["pA__c0001"]}
    clock.advance_days(5)
    report = memory.after_run("q?", "Something else entirely [pB__c0002].", TID,
                              passing_trace(), injected=injected)
    row = store.list()["memories"][0]
    assert report["used"] == [] and row["use_count"] == 1 and row["last_used_at"] < clock.t

    report = memory.after_run("q?", f"{CLAIM_A} [pA__c0001]", TID, passing_trace(),
                              injected=injected)
    row = store.list()["memories"][0]
    assert [u["id"] for u in report["used"]] == [mid]
    assert row["use_count"] == 2 and row["last_used_at"] == clock.t


def test_hard_purge(cfg, tmp_path):
    memory, store, clock = make(cfg, tmp_path)
    long_ago = seed(store, CLAIM_A)
    recent = seed(store, CLAIM_B, chunk_ids=("pB__c0002",))
    store.forget([long_ago], "user")
    clock.advance_days(float(cfg.memory.purge_after_days) - 10)
    store.forget([recent], "user")
    clock.advance_days(11)
    memory.sweep()
    left = [m["id"] for m in store.list(include_inactive=True)["memories"]]
    assert left == [recent]


def test_manual_forget(cfg, tmp_path):
    _, store, _ = make(cfg, tmp_path)
    mid = seed(store, CLAIM_A)
    assert store.forget_one(mid) == {"forgotten": mid, "reason": "user"}
    assert store.forget_one(mid)["error"] == "memory_not_found"
    assert store.forget_one("../x")["error"] == "bad_memory_id"
    row = store.list(include_inactive=True)["memories"][0]
    assert row["forget_reason"] == "user" and not row["active"]


# ================================================================ reader

def test_recall_puts_the_bge_prefix_on_the_query_only(cfg, tmp_path):
    """With the real Embedder (fake model): the question carries the instruction, the
    memory text the writer embeds does not."""
    from retrieval.embedder import Embedder

    seen: List[str] = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            seen.extend(texts)
            return np.vstack([unit(1) for _ in texts])

    embedder = Embedder(cfg)
    embedder._model = FakeModel()
    client = ScriptedClient(extraction([{"text": CLAIM_A, "chunk_ids": ["pA__c0001"]}]))
    memory, store, _ = make(cfg, tmp_path, client=client, embedder=embedder,
                            nli=StubNLI(entailed={CLAIM_A}))
    prefix = str(cfg.embedding.query_prefix)
    assert prefix

    memory.after_run("q?", ANSWER, TID, passing_trace())
    assert CLAIM_A in seen and not any(s.startswith(prefix) for s in seen)

    seen.clear()
    recalled = memory.recall("What balancing loss is used?")
    assert seen == [prefix + "What balancing loss is used?"]
    assert [m["text"] for m in recalled["memories"]] == [CLAIM_A]


def test_recall_respects_k_cosine_floor_and_the_token_budget(tmp_path, monkeypatch):
    cfg = _config(tmp_path, monkeypatch, memory={"inject_k": 2, "inject_min_cosine": 0.7})
    question = "Which balancing loss?"
    vectors = {question: unit(1), "far": at_cosine(0.3)}
    memory, store, _ = make(cfg, tmp_path, embedder=StubEmbedder(vectors))
    near = [seed(store, f"MethodA fact {i} is stated here.", vector=at_cosine(0.95 - i * 0.05))
            for i in range(3)]
    seed(store, "far", vector=at_cosine(0.3))
    recalled = memory.recall(question)
    assert [m["id"] for m in recalled["memories"]] == near[:2]          # top k, best first
    assert recalled["chunk_ids"] == ["pA__c0001"]                       # passages de-duplicated
    assert "[pA__c0001]" in recalled["block"] and TEXT_A[:40] in recalled["block"]
    assert cfg.prompt("memory_block").splitlines()[0] in recalled["block"]
    assert recalled["tokens"] <= int(cfg.memory.inject_tokens)

    small = _config(tmp_path, monkeypatch, memory={"inject_tokens": 60})
    tight, _, _ = make(small, tmp_path, embedder=StubEmbedder(vectors))
    assert tight.recall(question)["block"] == ""                        # nothing fits: no block


def test_recall_with_nothing_relevant_injects_nothing(cfg, tmp_path):
    memory, store, _ = make(cfg, tmp_path, embedder=StubEmbedder({"q": unit(1)}))
    seed(store, CLAIM_A, vector=unit(0, 1))
    assert memory.recall("q")["block"] == ""


def test_preferences_are_always_injected(cfg, tmp_path):
    memory, store, _ = make(cfg, tmp_path, embedder=StubEmbedder({"q": unit(1)}))
    seed(store, "Only consider papers from 2024 onward.", chunk_ids=(), hashes={},
         vector=unit(0, 1), kind="preference")
    recalled = memory.recall("q")
    assert "Only consider papers from 2024 onward." in recalled["block"]
    assert recalled["chunk_ids"] == []


# ================================================================ reader, in the loop

class FakeMemory:
    """Stands in for LongTermMemory: a fixed recall, and remember() calls recorded."""

    def __init__(self, recalled):
        self.recalled = recalled
        self.questions: List[str] = []
        self.after: List[tuple] = []

    def recall(self, question):
        self.questions.append(question)
        return self.recalled

    def after_run(self, question, answer, thread_id, trace_records, injected=None):
        self.after.append((question, answer, thread_id, injected))
        return {"written": [], "refreshed": [], "invalidated": [], "used": [], "forgotten": [],
                "dropped": {}}


def recalled_block(chunk_ids=("pM__c0007",)):
    return {"memories": [{"id": "f" * 32, "label": "m1", "text": "Earlier finding.",
                          "chunk_ids": list(chunk_ids), "score": 0.9, "use_count": 1}],
            "preferences": [], "stale": [], "tokens": 40,
            "chunks": [{"chunk_id": c, "paper_id": c.split("__")[0], "text": "memory passage " * 30}
                       for c in chunk_ids],
            "chunk_ids": list(chunk_ids),
            "block": "## Known from earlier chats\n" + " ".join(f"[{c}] memory passage" for c in chunk_ids)}


class Retriever:
    def __init__(self):
        self.resets: List[List[str]] = []
        self.released: List[List[str]] = []

    def reset(self, visible=()):
        self.resets.append(list(visible))

    def release(self, ids):
        self.released.append(list(ids))


def test_injected_chunks_are_visible_and_seen(cfg, tmp_path):
    threads = ThreadStore(tmp_path / "threads.sqlite3")
    tid = threads.create_thread("t")["id"]
    fake = FakeMemory(recalled_block())
    retriever = Retriever()
    client = ScriptedClient(text_response("From memory [pM__c0007]."))
    loop = AgentLoop(registry_with(cfg, lambda **kw: ok_result()), client=client, config=cfg,
                     store=threads, memory=fake, retriever=retriever)
    result = loop.run("A new question", thread_id=tid)

    sent = client.requests[0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "user"]
    assert sent[1]["content"] == recalled_block()["block"] and sent[2]["content"] == "A new question"
    assert retriever.resets == [["pM__c0007"]], "injected chunks start out seen"
    assert result["memory_injected"]["chunk_ids"] == ["pM__c0007"]
    assert fake.questions == ["A new question"]

    # The memory step is stored with the turn, carrying the chunks as evidence in view.
    steps = threads.get_thread(tid)["turns"][0]["steps"]
    memory_step = next(s for s in steps if s["kind"] == "memory")
    assert memory_step["result"]["chunks"][0]["chunk_id"] == "pM__c0007"
    assert len(memory_step["result"]["chunks"][0]["text"]) <= int(cfg.agent.trace_text_chars)
    assert memory_step["memories"][0]["id"] == "f" * 32

    # remember(): after the answer, and its report lands on the stored turn.
    report = loop.remember(result)
    assert fake.after[0][2] == tid and fake.after[0][3]["chunk_ids"] == ["pM__c0007"]
    assert threads.get_thread(tid)["turns"][0]["stats"]["memory"] == report


def test_injected_chunks_are_released_when_the_block_is_elided(tmp_path, monkeypatch):
    cfg = _config(tmp_path, monkeypatch, agent={"max_context_tokens": 50})
    conversation = Conversation("sys", "question", cfg, memory_block="memory passage " * 40,
                                memory_chunk_ids=["pM__c0007"])
    assert conversation.visible_chunk_ids() == ["pM__c0007"]
    assert conversation.elide_if_needed() == 1
    assert conversation.last_released == ["pM__c0007"]
    assert conversation.visible_chunk_ids() == []
    assert conversation.messages()[-1] == {"role": "user", "content": "question"}

    # In the loop: the release reaches the retriever.
    threads = ThreadStore(tmp_path / "threads.sqlite3")
    tid = threads.create_thread("t")["id"]
    retriever = Retriever()
    client = ScriptedClient(tool_response([("c1", "retrieve_evidence", {"query": "x"})]),
                            text_response("done"))
    AgentLoop(registry_with(cfg, lambda **kw: ok_result(["pA__c0001"])), client=client,
              config=cfg, store=threads, memory=FakeMemory(recalled_block()),
              retriever=retriever).run("q", thread_id=tid)
    assert any("pM__c0007" in ids for ids in retriever.released)


def test_no_thread_means_memory_is_never_read_or_written(cfg):
    class Spy:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            def record(*a, **k):
                self.calls.append(name)
                return {}
            return record

    spy = Spy()
    client = ScriptedClient(text_response("answer"))
    loop = AgentLoop(registry_with(cfg, lambda **kw: ok_result()), client=client, config=cfg,
                     memory=spy)
    result = loop.run("a question")
    assert loop.remember(result) is None
    assert spy.calls == []
    assert [m["role"] for m in client.requests[0]["messages"]] == ["system", "user"]


def test_a_broken_recall_never_ends_the_run(cfg, tmp_path):
    class Broken(FakeMemory):
        def recall(self, question):
            raise RuntimeError("boom")

    threads = ThreadStore(tmp_path / "threads.sqlite3")
    tid = threads.create_thread("t")["id"]
    loop = AgentLoop(registry_with(cfg, lambda **kw: ok_result()), client=ScriptedClient(
        text_response("fine")), config=cfg, store=threads, memory=Broken(None))
    result = loop.run("q", thread_id=tid)
    assert result["answer"] == "fine" and result["memory_injected"]["error"] == "memory_recall_failed"
    assert "recall_error" in loop.remember(result)


# ================================================================ storage

OLD_SCHEMA = """
CREATE TABLE threads (id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at REAL NOT NULL,
                      updated_at REAL NOT NULL);
CREATE TABLE turns (thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL,
    stats_json TEXT NOT NULL, notes_json TEXT NOT NULL, steps_json TEXT NOT NULL,
    run_id TEXT, created_at REAL NOT NULL, PRIMARY KEY (thread_id, seq));
"""


def test_migration_adds_the_table_to_an_existing_threads_db(tmp_path):
    path = tmp_path / "threads.sqlite3"
    tid = "b" * 32
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(OLD_SCHEMA)
        conn.execute("INSERT INTO threads VALUES (?, 'old chat', 1, 1)", (tid,))
        conn.execute("INSERT INTO turns VALUES (?, 1, 'q', 'a', '{}', '[]', '[]', NULL, 1)", (tid,))

    store = MemoryStore(path, dim=DIM)
    assert store.list() == {"memories": []}
    made = store.add("finding", "x y z w", ["pA__c0001"], {"pA__c0001": "h"}, unit(1), tid, "q")
    assert made["memory"]["use_count"] == 1

    threads = ThreadStore(path)
    old = threads.get_thread(tid)
    assert old["title"] == "old chat" and old["turns"][0]["answer"] == "a"
    assert threads.append_turn(tid, "q2", "a2")["seq"] == 2
    assert threads.annotate_turn(tid, 2, {"memory": {"written": []}})["seq"] == 2
    assert threads.get_thread(tid)["turns"][1]["stats"] == {"memory": {"written": []}}
    assert threads.annotate_turn(tid, 9, {})["error"] == "thread_not_found"
    with sqlite3.connect(str(path)) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables >= {"threads", "turns", "memories"}


def test_store_errors_are_structured(tmp_path):
    folder = tmp_path / "a_directory"
    folder.mkdir()
    store = MemoryStore(folder, dim=DIM)
    assert store.list()["error"] == "memory_store_unavailable"
    ok = MemoryStore(tmp_path / "m.sqlite3", dim=DIM)
    assert ok.add("opinion", "x", [], {}, unit(1), None, "q")["error"] == "bad_memory_kind"
    assert ok.add("finding", "x", [], {}, np.zeros(3), None, "q")["error"] == "bad_embedding"
    assert ok.refresh("nope")["error"] == "bad_memory_id"
    assert ok.refresh("c" * 32)["error"] == "memory_not_found"


def test_a_wrong_length_embedding_is_listed_but_never_matched(tmp_path):
    store = MemoryStore(tmp_path / "m.sqlite3", dim=DIM)
    mid = store.add("finding", "x y z w", [], {}, unit(1), None, "q")["memory"]["id"]
    with sqlite3.connect(str(store.path)) as conn:
        conn.execute("UPDATE memories SET embedding = ? WHERE id = ?", (b"\x00" * 8, mid))
    assert store.active()["memories"][0]["embedding"] is None


def test_grounding_checks_the_unattributed_statement_and_stores_the_attributed_one(cfg, tmp_path):
    """Naming the paper in the hypothesis sinks NLI entailment (measured 0.99 -> 0.001), so
    the passage-worded `check` is what is grounded; the attributed `text` is remembered."""
    check = "The balancing loss weight is set to 0.01."
    attributed = "MethodA sets its balancing loss weight to 0.01."
    table = table_ab()
    table["pA__c0001"] = chunk("pA__c0001", f"{check} It is applied at every layer of the model.", "hA")
    nli = StubNLI(entailed={check})
    client = ScriptedClient(extraction([{"text": attributed, "check": check, "chunk_ids": ["pA__c0001"]}]))
    memory, store, _ = make(cfg, tmp_path, client=client, nli=nli, table=table)
    report = memory.after_run("q?", f"{attributed} [pA__c0001]", TID, passing_trace())
    assert [w["text"] for w in report["written"]] == [attributed]
    assert {h for _, h in nli.pairs} == {check}


def test_grounding_looks_past_the_tool_sentence_cap(cfg, tmp_path):
    """The supporting sentence is the tenth: nli.max_sentences_per_chunk (8) would miss it."""
    claim = "The router is trained with an auxiliary balancing objective."
    filler = " ".join(f"Sentence number {i} says something unrelated here." for i in range(9))
    table = table_ab()
    table["pA__c0001"] = chunk("pA__c0001", f"{filler} {claim}", "hA")
    client = ScriptedClient(extraction([{"text": claim, "chunk_ids": ["pA__c0001"]}]))
    memory, _, _ = make(cfg, tmp_path, client=client, nli=StubNLI(entailed={claim}), table=table)
    assert int(cfg.nli.max_sentences_per_chunk) < 10
    report = memory.after_run("q?", f"{claim} [pA__c0001]", TID, passing_trace())
    assert [w["text"] for w in report["written"]] == [claim]


def test_extraction_survives_latex_backslashes(cfg, tmp_path):
    reply = r'{"findings": [{"text": "L = \sum_i f_i \cdot \frac{a}{b}, \"quoted\" \\ kept", "chunk_ids": ["pA__c0001"]}], "preferences": []}'
    memory, _, _ = make(cfg, tmp_path, client=ScriptedClient(text_response(reply)))
    got = memory.extract("q", "a")
    assert got["findings"][0]["text"] == r'L = \sum_i f_i \cdot \frac{a}{b}, "quoted" \ kept'


def test_answer_from_memory_alone_is_not_labelled_abstention(cfg, tmp_path):
    client = ScriptedClient(extraction([{"text": CLAIM_A, "chunk_ids": ["pA__c0001"]}]))
    memory, store, _ = make(cfg, tmp_path, client=client, nli=StubNLI(entailed={CLAIM_A}))
    report = memory.after_run("q?", "Answer citing memory [pA__c0001].", TID, [])
    assert report["skipped"] == "no_new_retrieval" and report["written"] == []
    assert client.requests == []
