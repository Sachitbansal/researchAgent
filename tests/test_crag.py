"""
Offline tests for CRAG phase 1: coverage_gap, the admission gate, fetch proposals, and
the throwaway-copy fetch check's refusal.

In:  tmp_path corpora, stub embedder / reranker / arXiv client — no models, no network.
Out: assertions on the coverage_gap shape and its per-scorer margin, admission by score
     (including all-rejected and cross-encoder failure), proposal derivation, and that
     scripts/crag_fetch_check.py will not run against the real data dir.
"""

import importlib.util
import json
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from agent.fetch_proposal import derive_fetch_proposal
from agent.loop import AgentLoop
from agent.tool_registry import RETRIEVE_EVIDENCE_PARAMETERS, ToolRegistry
from common.records import chunk_record, paper_record
from config import load_config
from corpus import admission, arxiv_query, collect
from corpus.chunk_store import ChunkStore
from corpus.manifest import Manifest
from llm_client import _parse_response
from retrieval.embedder import Embedder
from retrieval.indexer import index_chunks
from retrieval.reranker import RerankError
from tools import search_literature as search_module
from tools.retrieve_evidence import EvidenceRetriever

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- fixtures

@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    import config as config_module

    repo_root = config_module.REPO_ROOT
    local = tmp_path / "config.yaml"
    local.write_text((repo_root / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(config_module, "REPO_ROOT", tmp_path)
    built = load_config(local)
    built.paths.ensure()
    built.paths.prompts = repo_root / "src" / "prompts"
    return built


def override(base, section, **values):
    """A copy of `base` with one section's keys overridden, written through the YAML."""
    raw = yaml.safe_load(base.source_path.read_text(encoding="utf-8"))
    raw[section].update(values)
    name = f"{section}_{abs(hash(tuple(sorted(values.items()))))}.yaml"
    scratch = base.paths.root / name
    scratch.write_text(yaml.safe_dump(raw), encoding="utf-8")
    built = load_config(scratch)
    built.paths.prompts = base.paths.prompts
    return built


class StubEmbedder(Embedder):
    def _encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            # crc32, not hash(): hash() is salted per process, so a query and a chunk
            # would land in the same bucket (cosine 1.0, gate passes) on random runs.
            out[row, zlib.crc32(text.encode("utf-8")) % self.dim] = 1.0
        return out


class TableReranker:
    """Scores chunks by a {chunk_id: score} table; everything else gets `default`."""

    last_latency_ms = 3

    def __init__(self, scores=None, default=-10.0):
        self.scores, self.default = scores or {}, default

    def rerank(self, query, candidates):
        scored = [{**c, "rerank_score": self.scores.get(c.get("chunk_id"), self.default)}
                  for c in candidates]
        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored


class DeadReranker:
    last_latency_ms = 0

    def rerank(self, query, candidates):
        raise RerankError("model checkpoint missing")


def build_corpus(cfg, papers):
    """papers: {paper_id: (topic_tags, n_chunks)}"""
    manifest = Manifest.load(cfg, strict_model_check=False)
    store = ChunkStore(config=cfg)
    for paper_id, (tags, n_chunks) in papers.items():
        manifest.add_paper(paper_record(
            arxiv_id=paper_id, title=f"Title {paper_id}", authors=[], abstract="",
            published="2024-01-01", year=2024, categories=[], pdf_url="", pdf_path="",
            topic_tags=tags))
        store.append([
            chunk_record(paper_id=paper_id, paper_title=f"Title {paper_id}", topic_tags=tags,
                         chunk_type="text", text=f"{paper_id} chunk {i}", page=1,
                         position=i, n_tokens=5, section="S")
            for i in range(n_chunks)
        ])
    manifest.save()
    index_chunks(cfg, embedder=StubEmbedder(cfg))


@pytest.fixture()
def corpus(cfg):
    build_corpus(cfg, {
        "pA": (["alpha"], 3), "pB": (["beta"], 3), "pC": (["gamma"], 2), "pD": (["delta"], 2),
    })
    return cfg


def retrieve(cfg, query="what is missing", reranker=None, **kwargs):
    agent = EvidenceRetriever(cfg, embedder=StubEmbedder(cfg),
                              reranker=reranker if reranker is not None else TableReranker())
    return agent.retrieve(query, **kwargs)


# ------------------------------------------------------------------- coverage_gap

GAP_KEYS = {"searched_for", "best_score", "threshold", "scorer", "closest_topics",
            "suggested_next", "proposed_query", "filtered"}


def test_a_gate_failure_carries_a_structured_coverage_gap(corpus):
    result = retrieve(corpus, "where is the evidence", reranker=TableReranker(default=-9.0))
    assert result["sufficient_evidence"] is False
    gap = result["coverage_gap"]
    assert set(gap) == GAP_KEYS
    assert gap["searched_for"] == gap["proposed_query"] == "where is the evidence"
    assert gap["best_score"] == -9.0
    assert gap["threshold"] == float(corpus.retrieval.relevance_threshold)
    assert gap["scorer"] == "cross_encoder"
    assert gap["suggested_next"] == "propose_fetch"
    assert gap["filtered"] is False
    json.dumps(result)  # tool results are JSON-serialisable, gap included


def test_closest_topics_follow_rank_order_and_stop_at_three(corpus):
    # Best chunk is delta's, then beta's, then alpha's, then gamma's: four topics ranked.
    scores = {"pD__c0000": -4.0, "pB__c0001": -5.0, "pA__c0000": -6.0, "pC__c0000": -7.0}
    gap = retrieve(corpus, reranker=TableReranker(scores, default=-10.0))["coverage_gap"]
    assert gap["closest_topics"] == ["delta", "beta", "alpha"]


@pytest.mark.parametrize("best, expected", [
    (-0.5, "rephrase"),        # inside the margin below the threshold
    (-1.0, "rephrase"),        # exactly on the margin's edge
    (-1.5, "propose_fetch"),   # beyond it
])
def test_suggested_next_flips_at_the_rephrase_margin(corpus, best, expected):
    tuned = override(corpus, "retrieval", relevance_threshold=0.0, rephrase_margin=1.0)
    gap = retrieve(tuned, reranker=TableReranker({"pA__c0000": best}, default=-10.0))["coverage_gap"]
    assert gap["threshold"] == 0.0
    assert gap["suggested_next"] == expected


def test_bi_encoder_fallback_uses_its_own_threshold_and_margin(corpus):
    """A reranker that fails at runtime must not leave the cross-encoder's numbers in the
    hint: a logit threshold or margin applied to cosines is the documented mistake."""
    # Cross-encoder margin enormous, bi-encoder margin zero: if the wrong one were read,
    # the fallback would say "rephrase".
    tuned = override(corpus, "retrieval", bi_encoder_relevance_threshold=0.99,
                     bi_encoder_rephrase_margin=0.0, rephrase_margin=100.0)
    result = retrieve(tuned, reranker=DeadReranker())
    gap = result["coverage_gap"]
    assert gap["scorer"] == "bi_encoder"
    assert gap["threshold"] == 0.99
    assert gap["suggested_next"] == "propose_fetch"
    assert gap["best_score"] < 0.99

    near = override(corpus, "retrieval", bi_encoder_relevance_threshold=0.99,
                    bi_encoder_rephrase_margin=5.0, rephrase_margin=0.0)
    assert retrieve(near, reranker=DeadReranker())["coverage_gap"]["suggested_next"] == "rephrase"


def test_rerank_disabled_reports_the_bi_encoder_as_scorer(corpus):
    tuned = override(corpus, "retrieval", rerank_enabled=False,
                     bi_encoder_relevance_threshold=0.99)
    gap = retrieve(tuned)["coverage_gap"]
    assert gap["scorer"] == "bi_encoder" and gap["threshold"] == 0.99


def test_no_coverage_gap_when_the_gate_passes_or_never_ran(corpus):
    passing = retrieve(corpus, reranker=TableReranker(default=5.0))
    assert passing["sufficient_evidence"] is True and "coverage_gap" not in passing
    # An empty filter is not a gate failure: no scorer ran, so there is no score to hint on.
    filtered = retrieve(corpus, topic_filter="no_such_topic")
    assert filtered["sufficient_evidence"] is False and "coverage_gap" not in filtered


# --------------------------------------------------------------- admission gate

class StubResult:
    def __init__(self, short_id, title, summary="An abstract."):
        self._short_id, self.title, self.summary = short_id, title, summary
        self.authors = [type("A", (), {"name": "Ada"})()]
        self.published = datetime(2024, 1, 2, tzinfo=timezone.utc)
        self.categories = ["cs.LG"]
        self.pdf_url = f"https://arxiv.org/pdf/{short_id}"
        self.downloads = 0

    def get_short_id(self):
        return self._short_id


class TitleReranker:
    """Admission scorer: score by a word in the passage (title + abstract)."""

    last_latency_ms = 1

    def __init__(self, by_word, default=-10.0):
        self.by_word, self.default, self.calls = by_word, default, 0

    def rerank(self, query, candidates):
        self.calls += 1
        out = []
        for c in candidates:
            score = next((s for w, s in self.by_word.items() if w in c["text"]), self.default)
            out.append({**c, "rerank_score": score})
        out.sort(key=lambda item: item["rerank_score"], reverse=True)
        return out


class BrokenReranker:
    last_latency_ms = 0

    def rerank(self, query, candidates):
        raise RuntimeError("CUDA error: device-side assert triggered")


@pytest.fixture()
def offline_search(monkeypatch):
    """Stub arXiv, planning, ingest and indexing. Returns the list ingest was called with."""
    planned = {"query": "abs:(x)", "categories": [], "reasoning": "", "cached": False,
               "fallback": False}
    monkeypatch.setattr(arxiv_query, "plan_query", lambda *a, **k: dict(planned))
    monkeypatch.setattr(collect, "plan_query", lambda *a, **k: dict(planned))
    ingested = []

    def fake_ingest(paper_id, client=None, config=None):
        ingested.append(paper_id)
        return {"chunks_added": 2, "figures_added": 0}

    monkeypatch.setattr(search_module, "ingest_paper", fake_ingest)
    monkeypatch.setattr(search_module, "index_chunks", lambda config: {"total_indexed": 0})
    return ingested


def serve(monkeypatch, results):
    class Client:
        def results(self, search):
            return list(results)

    by_url = {r.pdf_url: r for r in results}

    def fake_download(client, pdf_url, dest, timeout_s):
        by_url[pdf_url].downloads += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-1.4 stub")
        return dest

    monkeypatch.setattr(collect, "_build_client", lambda config: Client())
    monkeypatch.setattr(collect, "download_pdf", fake_download)


def test_gate_maps_scores_back_to_the_right_result_after_sorting():
    results = [StubResult("1", "low"), StubResult("2", "high"), StubResult("3", "mid")]
    gate = admission.gate_results("q", results, TitleReranker({"high": 5.0, "mid": 0.0}), -1.0)
    assert [r.get_short_id() for r in gate["admitted"]] == ["2", "3"]
    assert gate["rejected"] == [{"arxiv_id": "1", "title": "low", "score": -10.0}]
    assert gate["scores"] == {"1": -10.0, "2": 5.0, "3": 0.0}
    assert gate["status"] == "applied"


def test_search_literature_admits_and_rejects_by_abstract_score(cfg, monkeypatch, offline_search):
    good = StubResult("2401.00001v1", "On the relevant subject")
    bad = StubResult("2401.00002v1", "Something adjacent")
    serve(monkeypatch, [good, bad])
    gate_cfg = override(cfg, "collection", admission_threshold=0.0)
    searcher = search_module.LiteratureSearcher(
        gate_cfg, reranker=TitleReranker({"relevant": 3.0, "adjacent": -4.0}))

    result = searcher.search("the subject")
    assert "error" not in result
    assert result["admission_gate"] == "applied"
    assert result["admission_threshold"] == 0.0
    assert [p["arxiv_id"] for p in result["papers_added"]] == ["2401.00001v1"]
    assert result["papers_added"][0]["admission_score"] == 3.0
    assert result["papers_rejected"] == [
        {"arxiv_id": "2401.00002v1", "title": "Something adjacent", "score": -4.0}]
    assert bad.downloads == 0, "a rejected paper is never downloaded"
    assert good.downloads == 1
    assert offline_search == ["2401_00001v1"], "a rejected paper is never ingested"
    assert not Manifest.load(gate_cfg).has_paper("2401_00002v1")


def test_all_rejected_is_an_informative_result_not_an_error(cfg, monkeypatch, offline_search):
    results = [StubResult("2401.00001v1", "far away"), StubResult("2401.00002v1", "also far")]
    serve(monkeypatch, results)
    gate_cfg = override(cfg, "collection", admission_threshold=0.0)
    changed = []
    searcher = search_module.LiteratureSearcher(
        gate_cfg, reranker=TitleReranker({}, default=-8.0),
        on_corpus_change=lambda: changed.append(True))

    result = searcher.search("a topic arXiv barely covers")
    assert "error" not in result
    assert result["papers_added"] == [] and result["chunks_added"] == 0
    assert len(result["papers_rejected"]) == 2
    assert result["arxiv_coverage"] == "poor"
    assert "does not seem to cover this topic" in result["note"]
    assert sum(r.downloads for r in results) == 0
    assert offline_search == [] and changed == [], "nothing indexed, cache left valid"
    assert Manifest.load(gate_cfg).topics == {}, "no zero-paper topic is recorded"
    json.dumps(result)


def test_cross_encoder_failure_admits_everything_and_says_so(cfg, monkeypatch, offline_search):
    results = [StubResult("2401.00001v1", "one"), StubResult("2401.00002v1", "two")]
    serve(monkeypatch, results)
    result = search_module.LiteratureSearcher(cfg, reranker=BrokenReranker()).search("topic")
    assert "error" not in result
    assert result["admission_gate"].startswith("skipped: ")
    assert "CUDA error" in result["admission_gate"]
    assert len(result["papers_added"]) == 2 and result["papers_rejected"] == []
    assert "admission gate skipped" in result["note"]


def test_a_reranker_that_cannot_be_built_also_falls_back(cfg, monkeypatch, offline_search):
    import retrieval.reranker as reranker_module

    def explode(*args, **kwargs):
        raise OSError("no such checkpoint")

    monkeypatch.setattr(reranker_module, "Reranker", explode)
    serve(monkeypatch, [StubResult("2401.00001v1", "one")])
    result = search_module.LiteratureSearcher(cfg).search("topic")
    assert result["admission_gate"].startswith("skipped: ")
    assert len(result["papers_added"]) == 1


def test_a_rejected_dedup_hit_is_not_tagged_with_the_new_topic(cfg, monkeypatch, offline_search):
    held = StubResult("2401.00001v1", "held paper about routing")
    serve(monkeypatch, [held])
    search_module.LiteratureSearcher(cfg, reranker=TitleReranker({"routing": 5.0})).search(
        "routing")
    serve(monkeypatch, [held])
    result = search_module.LiteratureSearcher(
        cfg, reranker=TitleReranker({"routing": -9.0})).search("baking bread")
    assert result["papers_rejected"][0]["arxiv_id"] == "2401.00001v1"
    assert Manifest.load(cfg).get_paper("2401_00001v1")["topic_tags"] == ["routing"]


# --------------------------------------------------------------- fetch proposals

def gap_record(suggested="propose_fetch", query="q1"):
    return {"tool_name": "retrieve_evidence", "args": {"query": query},
            "result_summary": {"chunks": [], "sufficient_evidence": False, "coverage_gap": {
                "searched_for": query, "best_score": -6.0, "threshold": -3.0,
                "scorer": "cross_encoder", "closest_topics": ["alpha"],
                "suggested_next": suggested, "proposed_query": query}}}


PASS_RECORD = {"tool_name": "retrieve_evidence", "args": {"query": "q2"},
               "result_summary": {"chunks": [{"chunk_id": "pA__c0000"}],
                                  "sufficient_evidence": True}}
SEARCH_RECORD = {"tool_name": "search_literature", "args": {"query": "q1"},
                 "result_summary": {"papers_added": []}}


def test_proposal_present_when_the_last_retrieve_proposed_a_fetch():
    proposal = derive_fetch_proposal([PASS_RECORD, gap_record(query="missing thing")])
    assert proposal == {"query": "missing thing", "searched_for": "missing thing",
                        "best_score": -6.0, "closest_topics": ["alpha"]}


def test_no_proposal_after_search_literature_ran():
    assert derive_fetch_proposal([gap_record(), SEARCH_RECORD]) is None


def test_no_proposal_when_the_last_hint_was_rephrase():
    assert derive_fetch_proposal([gap_record("propose_fetch"), gap_record("rephrase")]) is None


def test_a_later_passing_retrieve_clears_an_earlier_gap():
    assert derive_fetch_proposal([gap_record(), PASS_RECORD]) is None


def test_no_proposal_without_any_retrieve():
    assert derive_fetch_proposal([]) is None
    assert derive_fetch_proposal([SEARCH_RECORD]) is None


def _tool_call(call_id, name, args):
    return _parse_response({"model": "stub", "choices": [{"message": {
        "role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]},
        "finish_reason": "tool_calls"}]})


def _text(text):
    return _parse_response({"model": "stub", "choices": [
        {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})


class Scripted:
    def __init__(self, *responses):
        self.responses = list(responses)

    def complete(self, messages, tools=None, role="agent", **kwargs):
        return self.responses.pop(0) if self.responses else _text("done")


def test_the_loop_result_carries_the_derived_proposal(cfg):
    from eval.metrics import fetch_proposal_detected

    gap = gap_record(query="an absent subject")["result_summary"]
    registry = ToolRegistry(cfg)
    registry.register("retrieve_evidence", RETRIEVE_EVIDENCE_PARAMETERS, lambda **kw: dict(gap))
    client = Scripted(_tool_call("c1", "retrieve_evidence", {"query": "an absent subject"}),
                      _text("The corpus cannot answer this."))
    result = AgentLoop(registry, client=client, config=cfg).run("question?")
    assert result["fetch_proposal"]["query"] == "an absent subject"
    assert fetch_proposal_detected(result) is True


def test_fetch_proposal_detected_on_synthetic_runs():
    from eval.metrics import fetch_proposal_correct, fetch_proposal_detected

    assert fetch_proposal_detected({"fetch_proposal": {"query": "x"}}) is True
    assert fetch_proposal_detected({"fetch_proposal": None}) is False
    assert fetch_proposal_detected({}) is False
    assert fetch_proposal_correct(True, True) is True
    assert fetch_proposal_correct(False, True) is False
    assert fetch_proposal_correct(True, None) is None, "unannotated questions are unscored"


def test_absent_fetchable_questions_respect_the_split():
    questions = json.loads((REPO / "eval" / "questions.json").read_text())["questions"]
    fetchable = [q for q in questions if q["category"] == "absent_fetchable"]
    assert {q["question_id"] for q in fetchable} >= {"ni02", "ni03"}
    assert all(q["expect_fetch_proposal"] is True and q["expect_abstention"] for q in fetchable)
    by_id = {q["question_id"]: q for q in questions}
    assert by_id["ni01"]["category"] == "not_in_corpus"
    assert "expect_fetch_proposal" not in by_id["ni01"]
    held = sum(q["held_out"] for q in questions)
    assert held * 3 == len(questions), "the held-out split stays one third"


# --------------------------------------------------------------- crag_fetch_check

def _load_script():
    spec = importlib.util.spec_from_file_location(
        "crag_fetch_check", REPO / "scripts" / "crag_fetch_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fetch_check_refuses_the_real_data_dir():
    script = _load_script()
    real = script.configured_paths(REPO / "config.yaml")
    reason = script.refusal_reason(real.values(), real["data_dir"])
    assert reason and "refusing" in reason
    # Inside it, and containing it, are refused too.
    assert script.refusal_reason([real["index_dir"]], real["data_dir"])
    assert script.refusal_reason([REPO], real["data_dir"])


def test_fetch_check_accepts_a_scratch_copy(tmp_path):
    script = _load_script()
    real = script.configured_paths(REPO / "config.yaml")
    scratch_config = script.write_scratch_config(tmp_path)
    paths = script.configured_paths(scratch_config)
    assert all(tmp_path in p.parents for p in paths.values())
    assert script.refusal_reason(paths.values(), real["data_dir"]) is None


def test_fetch_check_refuses_while_an_arxiv_cooldown_is_live(tmp_path, monkeypatch):
    script = _load_script()
    (tmp_path / "config.yaml").write_text((REPO / "config.yaml").read_text())
    cooldown = tmp_path / "data" / "cache" / "arxiv_cooldown.json"
    cooldown.parent.mkdir(parents=True)
    cooldown.write_text(json.dumps({"until": time.time() + 600}))
    monkeypatch.setattr(script, "ROOT", tmp_path)
    report = script.run_check("anything", 3)
    assert report["status"] == "refused" and "cooldown" in report["reason"]
