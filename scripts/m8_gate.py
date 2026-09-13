"""
The M8 verification gate: a full eval run producing a results table.

In:  the tuned config and the annotated question set.
Out: a results table for the tuning split, then the held-out split run once and reported
     separately, plus the rerank ablation. Exits non-zero if the harness cannot produce
     a table — not if the numbers are disappointing, which is a finding, not a failure.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.ablation import run as run_ablation  # noqa: E402
from eval.runner import run_split, save  # noqa: E402

failures: List[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        failures.append(label)


def table(title: str, result: Dict[str, Any]) -> None:
    print(f"\n{'=' * 78}\n{title}  ({result['n_questions']} questions)\n{'=' * 78}", flush=True)
    header = (f"{'id':>5}  {'category':<15}  {'facts':>6}  {'recall@5':>8}  {'MRR':>5}  "
              f"{'abst':>5}  {'cites':>7}  {'iters':>5}")
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for row in result["rows"]:
        if "error" in row:
            print(f"{row['question_id']:>5}  {row['category']:<15}  ERROR {row['error'][:40]}",
                  flush=True)
            continue
        citations = row["citations"]
        mark = "ok" if row["abstention_correct"] else "WRONG"
        print(f"{row['question_id']:>5}  {row['category']:<15}  {row['facts_covered']:>6}  "
              f"{row['recall']['5']:>8.3f}  {row['mrr']:>5.2f}  {mark:>5}  "
              f"{citations['resolved']}/{citations['total']:<5}  {row['iterations']:>5}",
              flush=True)

    print(f"\n  recall@1/3/5            {result['recall']['1']:.3f} / "
          f"{result['recall']['3']:.3f} / {result['recall']['5']:.3f}", flush=True)
    print(f"  MRR                     {result['mrr']:.3f}", flush=True)
    print(f"  gold paper hit rate     {result['gold_paper_hit_rate']:.3f}", flush=True)
    print(f"  fact coverage           {result['fact_coverage']:.3f}", flush=True)
    print(f"  abstention accuracy     {result['abstention_accuracy']:.3f} "
          f"(abstained on absent {result['abstention_on_absent']:.3f}, "
          f"false abstention {result['false_abstention_on_answerable']:.3f})", flush=True)
    print(f"  citations resolved      {result['citations_resolved']:.3f}", flush=True)
    print(f"  tool trace matched      {result['tool_trace_matched']:.3f}", flush=True)
    print(f"  mean iterations         {result['mean_iterations']:.1f}", flush=True)
    print(f"  median wall clock       {result['median_elapsed_ms'] / 1000:.1f}s", flush=True)


def main() -> int:
    print("=== tuning split (threshold was fitted on these) ===\n", flush=True)
    tuning = run_split(held_out=False)
    table("TUNING SPLIT", tuning)
    save(tuning, "tuning_split")

    check("the harness produced a results table", tuning["n_questions"] > 0,
          f"{tuning['n_questions']} questions")
    check("no question crashed the harness", tuning["errors"] == 0,
          f"{tuning['errors']} errors")
    check("retrieval metrics are computed", tuning["mrr"] == tuning["mrr"],
          f"MRR={tuning['mrr']}")
    check("abstention is scored", tuning["abstention_accuracy"] == tuning["abstention_accuracy"],
          f"{tuning['abstention_accuracy']:.3f}")
    check("answers carry resolvable citations", tuning["citations_resolved"] > 0.5,
          f"{tuning['citations_resolved']:.3f} of cited ids resolve to retrieved chunks")

    print("\n\n=== rerank ablation ===\n", flush=True)
    ablation = run_ablation()
    on, off = ablation["arms"]["rerank_on"], ablation["arms"]["rerank_off"]
    header = f"{'arm':<12}  {'recall@5':>8}  {'MRR':>6}  {'paper':>6}  {'latency':>9}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for label, arm in (("rerank on", on), ("rerank off", off)):
        print(f"{label:<12}  {arm['recall']['5']:>8.3f}  {arm['mrr']:>6.3f}  "
              f"{arm['gold_paper_hit_rate']:>6.3f}  {arm['median_latency_ms']:>8}ms", flush=True)
    print(f"\n  rerank delta: MRR {on['mrr'] - off['mrr']:+.3f}, "
          f"latency {on['median_latency_ms'] - off['median_latency_ms']:+d}ms", flush=True)
    check("the ablation ran both arms", set(ablation["arms"]) == {"rerank_on", "rerank_off"})

    print("\n\n=== held-out split (run once, after tuning stopped) ===\n", flush=True)
    held = run_split(held_out=True)
    table("HELD-OUT SPLIT", held)
    save(held, "held_out_split")
    check("the held-out split ran", held["n_questions"] > 0, f"{held['n_questions']} questions")

    print(f"\n{'M8 GATE PASSED' if not failures else 'M8 GATE FAILED: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
