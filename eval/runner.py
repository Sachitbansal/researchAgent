"""
Runs every eval question through the full agent and scores the result.

In:  a question split (tuning or held-out) and the live config.
Out: per-question rows and aggregates — retrieval recall from the traces, fact coverage,
     abstention correctness, groundedness, and the tool trace. Gold labels are used only
     here, in the scorer; the agent never sees them.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.metrics import (  # noqa: E402
    abstention_correct,
    any_gold_paper_hit,
    fact_coverage,
    looks_like_abstention,
    mean,
    recall_at_k,
    reciprocal_rank,
    tool_trace_match,
)
from eval.retrieval_eval import load_questions  # noqa: E402


def run_split(
    held_out: bool = False,
    limit: Optional[int] = None,
    check_groundedness: bool = True,
) -> Dict[str, Any]:
    """Run one split end to end. Returns rows plus aggregates."""
    from agent.loop import AgentLoop
    from agent.tool_registry import build_full_registry
    from config import CFG
    from retrieval.embedder import Embedder
    from tools.retrieve_evidence import EvidenceRetriever

    questions = load_questions(held_out=held_out)
    if limit:
        questions = questions[:limit]

    embedder = Embedder(CFG)
    retriever = EvidenceRetriever(CFG, embedder=embedder)
    registry = build_full_registry(retriever=retriever, config=CFG)
    loop = AgentLoop(registry, config=CFG, retriever=retriever)

    rows: List[Dict[str, Any]] = []
    for question in questions:
        started = time.perf_counter()
        result = loop.run(question["question"], question_id=question["question_id"])
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        if "error" in result:
            rows.append({
                "question_id": question["question_id"],
                "category": question["category"],
                "error": f"{result['error']}: {result['detail']}",
                "elapsed_ms": elapsed_ms,
            })
            print(f"  {question['question_id']}  ERROR {result['error']}", flush=True)
            continue

        # Retrieval is scored from the trace, which is why chunk_ids_returned is a
        # top-level trace field: no re-running and no parsing of result_summary.
        retrieved: List[str] = []
        for record in result["trace_records"]:
            for chunk_id in record["chunk_ids_returned"]:
                if chunk_id not in retrieved:
                    retrieved.append(chunk_id)
        papers = [c.rsplit("__c", 1)[0] for c in retrieved]
        gold = question["gold_chunk_ids"]
        answer = result["answer"]

        coverage = fact_coverage(answer, question["expected_facts"], embedder)
        abstained = looks_like_abstention(answer)
        row = {
            "question_id": question["question_id"],
            "category": question["category"],
            "expect_abstention": question["expect_abstention"],
            "abstained": abstained,
            "abstention_correct": abstention_correct(
                answer, bool(retrieved), question["expect_abstention"]
            ),
            "recall": {str(n): recall_at_k(retrieved, gold, n) for n in (1, 3, 5)},
            "mrr": reciprocal_rank(retrieved, gold),
            "gold_paper_hit": any_gold_paper_hit(papers, question["gold_paper_ids"]),
            "fact_coverage": coverage["ratio"],
            "facts_covered": f"{coverage['covered']}/{coverage['total']}",
            "citations": count_citations(answer, retrieved),
            "tools": tool_trace_match(
                [r["tool_name"] for r in result["trace_records"]], question["expected_tools"]
            ),
            "iterations": result["iterations"],
            "tool_calls": result["tool_calls"],
            "stopped_because": result["stopped_because"],
            "elapsed_ms": elapsed_ms,
            "answer": answer,
            "run_id": result["run_id"],
        }
        rows.append(row)
        print(f"  {question['question_id']:>5}  {question['category']:<15} "
              f"facts={row['facts_covered']:>5}  mrr={row['mrr']:.2f}  "
              f"abstain={'Y' if abstained else 'n'}{'' if row['abstention_correct'] else ' WRONG'}  "
              f"tools={len(row['tools']['used'])}  {elapsed_ms/1000:.1f}s", flush=True)

    return {"n_questions": len(rows), "rows": rows, **aggregate(rows)}


def count_citations(answer: str, retrieved: List[str]) -> Dict[str, Any]:
    """Inline citations in the answer, and how many resolve to a retrieved chunk.

    An unresolvable citation is worse than none: it reads as a source but points nowhere,
    so it scores as a fabricated reference rather than as evidence.
    """
    import re

    cited = re.findall(r"\[([A-Za-z0-9_]+__c\d+)\]", answer or "")
    unique = list(dict.fromkeys(cited))
    resolved = [c for c in unique if c in set(retrieved)]
    return {"total": len(unique), "resolved": len(resolved),
            "unresolved": [c for c in unique if c not in set(retrieved)]}


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics, split by the parts of the question set that differ."""
    scored = [r for r in rows if "error" not in r]
    answerable = [r for r in scored if not r["expect_abstention"]]
    abstaining = [r for r in scored if r["expect_abstention"]]

    return {
        "errors": len(rows) - len(scored),
        "recall": {n: mean([r["recall"][n] for r in answerable]) for n in ("1", "3", "5")},
        "mrr": mean([r["mrr"] for r in answerable]),
        "gold_paper_hit_rate": mean([float(r["gold_paper_hit"]) for r in answerable]),
        "fact_coverage": mean([r["fact_coverage"] for r in answerable]),
        "abstention_accuracy": mean([float(r["abstention_correct"]) for r in scored]),
        "abstention_on_absent": mean([float(r["abstained"]) for r in abstaining]),
        "false_abstention_on_answerable": mean([float(r["abstained"]) for r in answerable]),
        "citations_resolved": mean(
            [r["citations"]["resolved"] / r["citations"]["total"]
             for r in answerable if r["citations"]["total"]]
        ),
        "tool_trace_matched": mean([float(r["tools"]["matched"]) for r in scored]),
        "mean_iterations": mean([float(r["iterations"]) for r in scored]),
        "median_elapsed_ms": (
            sorted(r["elapsed_ms"] for r in scored)[len(scored) // 2] if scored else 0
        ),
    }


def save(result: Dict[str, Any], name: str) -> Path:
    out = ROOT / "eval" / "results" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
