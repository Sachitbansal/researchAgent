"""
Sweeps the bi-encoder relevance gate, for the configuration where rerank is disabled.

In:  the non-held-out questions and candidate cosine thresholds.
Out: gate behaviour per threshold plus a recommendation, scored exactly like the
     cross-encoder sweep so the two are comparable. Cosine is bounded 0..1, but where
     the useful cut sits is still an empirical question.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.retrieval_eval import config_with, evaluate_retrieval, load_questions  # noqa: E402

CANDIDATES = [0.0, 0.40, 0.50, 0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80]


def sweep(thresholds: List[float] = None) -> Dict[str, Any]:
    """Score each candidate on the tuning split, with reranking off."""
    from retrieval.embedder import Embedder

    thresholds = thresholds or CANDIDATES
    questions = load_questions(held_out=False)
    embedder = Embedder(config_with())

    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        config = config_with(rerank_enabled=False, bi_encoder_relevance_threshold=threshold)
        result = evaluate_retrieval(questions, config, embedder=embedder, reranker=None)
        answerable = result["gate_correct_on_answerable"]
        absent = result["gate_correct_on_absent"]
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
        print(f"  {threshold:>5.2f}  answerable={answerable:.3f}  absent={absent:.3f}  "
              f"balanced={balanced:.4f}  mrr={result['mrr']:.4f}", flush=True)

    best = max(rows, key=lambda r: (r["balanced"], r["mrr"]))
    return {"rows": rows, "recommended": best, "n_questions": len(questions)}


def main() -> int:
    print("Sweeping the bi-encoder relevance gate (rerank disabled), tuning split only:")
    result = sweep()
    out = ROOT / "eval/results/bi_threshold_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nrecommended: {result['recommended']}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
