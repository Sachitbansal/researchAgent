"""
Retrieval ablation v2: baseline vs contextual headers vs hybrid BM25 vs multi-query.

In:  the tuning split only, and chunks.jsonl copied into scratch index dirs (one per
     embed format), so the real data/index is never touched. Local, no LLM, no arXiv.
Out: eval/results/retrieval_ablation_v2.json — per arm: recall@{1,3,5,10}, MRR, pool
     recall@40, gold-paper hit, abstention on absent questions, latency, gate sweep.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.metrics import any_gold_paper_hit, mean, recall_at_k, reciprocal_rank  # noqa: E402
from eval.retrieval_eval import config_with, load_questions  # noqa: E402
from eval.tune_threshold import CANDIDATES  # noqa: E402

RECALL_KS = (1, 3, 5, 10)
K = 10

# (label, contextual_headers, hybrid, multi_query, multi_query_rerank)
ARMS = (
    ("baseline", False, False, False, None),
    ("+headers", True, False, False, None),
    ("+hybrid", False, True, False, None),
    ("headers+hybrid", True, True, False, None),
    ("+multi(retrieving)", False, False, True, "retrieving"),
    ("hybrid+multi(all)", False, True, True, "all"),
    ("hybrid+multi(retrieving)", False, True, True, "retrieving"),
)

# The offline sub-queries. No LLM: the question itself, plus its clauses when it has
# several — split at sentence ends and at ", and <wh-word>" / ", and do(es)" joins. A
# question with one clause stays a single query. The second clause often leans on a
# pronoun ("what does it propose"), so this is a weak lower bound on what an agent
# writing its own sub-queries gets; the agent run measures that.
_SPLIT = re.compile(
    r"(?<=[.?])\s+|,\s+and\s+(?=(?:what|how|which|why|when|where|who|do|does|did|is|are)\b)",
    re.IGNORECASE,
)


def deterministic_subqueries(question: str, cap: int) -> List[str]:
    parts = [p.strip(" ,.?") for p in _SPLIT.split(question) if p and p.strip(" ,.?")]
    if len(parts) < 2:
        return [question]
    return ([question] + parts)[:cap]


def build_index(scratch: Path, headers: bool, embedder: Any) -> Dict[str, Any]:
    """A scratch copy of chunks + manifest, embedded in one format. The real index stays."""
    from retrieval.indexer import index_chunks

    index_dir = scratch / ("index_header" if headers else "index_plain")
    index_dir.mkdir(parents=True, exist_ok=True)
    for name in ("chunks.jsonl", "manifest.json"):
        shutil.copy2(ROOT / "data" / "index" / name, index_dir / name)
    for name in ("faiss.index", "embeddings.npy"):
        (index_dir / name).unlink(missing_ok=True)
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["faiss_id_map"], manifest["indexed_hashes"] = [], []
    (index_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    sections = {"paths": {"index_dir": str(index_dir), "cache_dir": str(scratch / "cache")}}
    config = config_with(sections=sections, contextual_headers=headers)
    result = index_chunks(config=config, embedder=embedder)
    if "error" in result:
        raise RuntimeError(f"scratch index failed: {result}")
    return {"sections": sections, "index": result}


def run_arm(questions, label, headers, hybrid, multi, mode, built, embedder, reranker,
            agent_calls: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """One arm over `questions`. With `agent_calls`, each question is asked as the agent
    first asked it in a past trace (its query, topic_filter, chunk_types) instead."""
    from tools.retrieve_evidence import EvidenceRetriever

    overrides = {"contextual_headers": headers, "hybrid": hybrid}
    if mode:
        overrides["multi_query_rerank"] = mode
    config = config_with(sections=built[headers]["sections"], **overrides)
    retriever = EvidenceRetriever(config, embedder=embedder, reranker=reranker)
    cap = int(config.retrieval.max_subqueries)

    rows = []
    for q in questions:
        retriever.reset()
        call = (agent_calls or {}).get(q["question_id"], {})
        text = call.get("query") or q["question"]
        filters = {"topic_filter": call.get("topic_filter"), "chunk_types": call.get("chunk_types")}
        subqueries = deterministic_subqueries(text, cap) if multi else [text]
        started = time.perf_counter()
        if len(subqueries) > 1:
            result = retriever.retrieve(queries=subqueries, k=K, **filters)
        else:
            result = retriever.retrieve(subqueries[0], k=K, **filters)
        latency_ms = int((time.perf_counter() - started) * 1000)
        if "error" in result:
            raise RuntimeError(f"{label} {q['question_id']}: {result}")
        chunks = result.get("chunks", [])
        retrieved = [c["chunk_id"] for c in chunks]
        gap = result.get("coverage_gap") or {}
        top = max((c["score"] for c in chunks), default=gap.get("best_score"))
        gold = q["gold_chunk_ids"]
        rows.append({
            "question_id": q["question_id"],
            "category": q["category"],
            "expect_abstention": q["expect_abstention"],
            "subqueries": subqueries,
            "sufficient_evidence": bool(result.get("sufficient_evidence")),
            "subquery_sufficient": result.get("subquery_sufficient"),
            "top_score": top,
            "recall": {str(k): recall_at_k(retrieved, gold, k) for k in RECALL_KS},
            "mrr": reciprocal_rank(retrieved[:5], gold),
            "pool_recall@40": recall_at_k(retriever.last_pool, gold, len(retriever.last_pool) or 1),
            "gold_paper_hit": any_gold_paper_hit([c["paper_id"] for c in chunks[:5]],
                                                 q["gold_paper_ids"]),
            "latency_ms": latency_ms,
            "rerank_ms": retriever.last_rerank_ms,
            "rerank_pairs": retriever.last_rerank_pairs,
            "retrieved": retrieved,
        })

    answerable = [r for r in rows if not r["expect_abstention"]]
    absent = [r for r in rows if r["expect_abstention"]]
    multi_rows = [r for r in rows if len(r["subqueries"]) > 1]
    sweep = []
    for t in CANDIDATES:
        a = mean([float(r["top_score"] is not None and r["top_score"] >= t) for r in answerable])
        b = mean([float(not (r["top_score"] is not None and r["top_score"] >= t)) for r in absent])
        sweep.append({"threshold": t, "gate_on_answerable": a, "gate_on_absent": b,
                      "balanced": 0.0 if a + b == 0 else round(2 * a * b / (a + b), 4)})
    return {
        "label": label,
        "recall": {str(k): mean([r["recall"][str(k)] for r in answerable]) for k in RECALL_KS},
        "mrr": mean([r["mrr"] for r in answerable]),
        "pool_recall@40": mean([r["pool_recall@40"] for r in answerable]),
        "gold_paper_hit_rate": mean([float(r["gold_paper_hit"]) for r in answerable]),
        "gate_correct_on_answerable": mean([float(r["sufficient_evidence"]) for r in answerable]),
        "abstention_on_absent": mean([float(not r["sufficient_evidence"]) for r in absent]),
        "median_latency_ms": int(statistics.median(r["latency_ms"] for r in rows)),
        "median_rerank_ms": int(statistics.median(r["rerank_ms"] for r in rows)),
        "median_rerank_ms_multi_questions": (
            int(statistics.median(r["rerank_ms"] for r in multi_rows)) if multi_rows else None),
        "median_rerank_pairs_multi_questions": (
            int(statistics.median(r["rerank_pairs"] for r in multi_rows)) if multi_rows else None),
        "n_multi_questions": len(multi_rows),
        "gate_sweep": sweep,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", type=Path,
                        default=ROOT / "eval" / "results" / "retrieval_ablation_v2.json")
    parser.add_argument("--traces", type=Path, default=None,
                        help="also replay each tuning question's first agent query from this "
                             "logs/traces directory (read only)")
    args = parser.parse_args()

    from retrieval.embedder import Embedder
    from retrieval.reranker import Reranker

    questions = load_questions(held_out=False)  # the held-out third is never read here
    base = config_with()
    embedder, reranker = Embedder(base), Reranker(base)
    out: Dict[str, Any] = {"n_questions": len(questions), "k": K, "arms": {}}
    with tempfile.TemporaryDirectory(prefix="retrieval_ablation_") as tmp:
        scratch = Path(tmp)
        built = {False: build_index(scratch, False, embedder),
                 True: build_index(scratch, True, embedder)}
        out["headers_truncated"] = built[True]["index"].get("headers_truncated")
        for label, headers, hybrid, multi, mode in ARMS:
            arm = run_arm(questions, label, headers, hybrid, multi, mode, built, embedder, reranker)
            out["arms"][label] = arm
            print(f"{label:<34} r@1 {arm['recall']['1']:.3f} r@3 {arm['recall']['3']:.3f} "
                  f"r@5 {arm['recall']['5']:.3f} r@10 {arm['recall']['10']:.3f} "
                  f"MRR {arm['mrr']:.3f} pool@40 {arm['pool_recall@40']:.3f} "
                  f"paper {arm['gold_paper_hit_rate']:.3f} ans {arm['gate_correct_on_answerable']:.3f} "
                  f"abst {arm['abstention_on_absent']:.3f} lat {arm['median_latency_ms']}ms "
                  f"rerank {arm['median_rerank_ms']}ms", flush=True)
        if args.traces is not None:
            from eval.diagnose_retrieval import first_agent_calls

            calls = first_agent_calls(args.traces)
            replayable = [q for q in questions if q["question_id"] in calls]
            out["agent_query"] = {"traces": str(args.traces),
                                  "question_ids": [q["question_id"] for q in replayable],
                                  "arms": {}}
            print(f"\nagent first query, replayed from traces ({len(replayable)} questions):")
            for label, headers, hybrid, multi, mode in ARMS:
                if multi:
                    continue  # keyword queries have no clauses to split: same as single
                arm = run_arm(replayable, label, headers, hybrid, False, None, built,
                              embedder, reranker, agent_calls=calls)
                arm["rows"] = [{**r, "agent_query": r["subqueries"][0]} for r in arm["rows"]]
                out["agent_query"]["arms"][label] = arm
                print(f"{label:<34} r@1 {arm['recall']['1']:.3f} r@3 {arm['recall']['3']:.3f} "
                      f"r@5 {arm['recall']['5']:.3f} r@10 {arm['recall']['10']:.3f} "
                      f"MRR {arm['mrr']:.3f} pool@40 {arm['pool_recall@40']:.3f} "
                      f"paper {arm['gold_paper_hit_rate']:.3f} "
                      f"ans {arm['gate_correct_on_answerable']:.3f} "
                      f"abst {arm['abstention_on_absent']:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
