"""
Sweeps the relevance-gate threshold and picks one from data, on the tuning split only.

In:  the non-held-out questions and a range of candidate thresholds.
Out: a table of gate behaviour per threshold and a recommendation. Cross-encoder scores
     are uncalibrated logits, so this number can only come from measurement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # so `eval.` resolves when run as a script

from eval.retrieval_eval import config_with, evaluate_retrieval, load_questions  # noqa: E402

CANDIDATES = [-8.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0]


def sweep(thresholds: List[float] = None) -> Dict[str, Any]:
    """Score each candidate threshold on the tuning split."""
    from retrieval.embedder import Embedder
    from retrieval.reranker import Reranker

    thresholds = thresholds or CANDIDATES
    questions = load_questions(held_out=False)

    # One model instance across the whole sweep. Reloading per threshold would dominate
    # the runtime and make the latency column meaningless.
    base = config_with()
    embedder = Embedder(base)
    reranker = Reranker(base)

    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        config = config_with(relevance_threshold=threshold)
        result = evaluate_retrieval(questions, config, embedder=embedder, reranker=reranker)

        answerable = result["gate_correct_on_answerable"]
        absent = result["gate_correct_on_absent"]
        # Both halves matter and they trade off: a permissive gate always answers and
        # never abstains, a strict one abstains on everything. The harmonic mean punishes
        # a configuration that wins one half by abandoning the other.
        balanced = (
            0.0 if (answerable + absent) == 0
            else round(2 * answerable * absent / (answerable + absent), 4)
        )
        rows.append({
            "threshold": threshold,
            "gate_on_answerable": answerable,
            "gate_on_absent": absent,
            "balanced": balanced,
            "recall@5": result["recall"]["5"],
            "mrr": result["mrr"],
            "gold_paper_hit_rate": result["gold_paper_hit_rate"],
        })

    best = max(rows, key=lambda r: (r["balanced"], r["mrr"]))
    return {"rows": rows, "recommended": best, "n_questions": len(questions)}


def main() -> int:
    result = sweep()
    print(f"tuning split: {result['n_questions']} questions "
          f"(held-out questions are not touched)\n", flush=True)
    header = f"{'thresh':>7}  {'answerable':>10}  {'absent':>7}  {'balanced':>8}  {'recall@5':>8}  {'MRR':>6}  {'paper':>6}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for row in result["rows"]:
        print(f"{row['threshold']:>7.1f}  {row['gate_on_answerable']:>10.3f}  "
              f"{row['gate_on_absent']:>7.3f}  {row['balanced']:>8.3f}  "
              f"{row['recall@5']:>8.3f}  {row['mrr']:>6.3f}  "
              f"{row['gold_paper_hit_rate']:>6.3f}", flush=True)

    best = result["recommended"]
    print(f"\nrecommended relevance_threshold: {best['threshold']}  "
          f"(balanced {best['balanced']}, answerable {best['gate_on_answerable']}, "
          f"absent {best['gate_on_absent']})", flush=True)

    out = ROOT / "eval" / "results" / "threshold_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"written: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
