"""
The M6 verification gate: the four remaining tools, standalone then agent-selected.

In:  an index built by M3 and a funded .env key.
Out: pass/fail per tool — documented return shape, an error dict on bad input — then
     live questions whose traces show the model actually selecting each tool.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.loop import AgentLoop  # noqa: E402
from agent.tool_registry import build_full_registry  # noqa: E402
from config import CFG  # noqa: E402
from corpus.manifest import Manifest  # noqa: E402
from tools.analyze_corpus import CorpusAnalyzer  # noqa: E402
from tools.check_evidence_consistency import ConsistencyChecker  # noqa: E402
from tools.inspect_figure import FigureInspector  # noqa: E402
from tools.retrieve_evidence import EvidenceRetriever  # noqa: E402

failures: List[str] = []
_started = time.monotonic()


def _elapsed() -> str:
    return f"{time.monotonic() - _started:6.1f}s"


def step(message: str) -> None:
    print(f"[{_elapsed()}] .... {message}", flush=True)


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{_elapsed()}] [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""),
          flush=True)
    if not ok:
        failures.append(label)


def main() -> int:
    manifest = Manifest.load(CFG, strict_model_check=False)
    if not manifest.papers:
        print("Empty corpus. Run scripts/m3_gate.py first.")
        return 1

    # ---- analyze_corpus ----------------------------------------------------
    step("analyze_corpus, standalone")
    analyzer = CorpusAnalyzer(CFG)
    for operation in ("stats", "timeline", "compare_topics", "cluster"):
        result = analyzer.analyze(operation)
        ok = "error" not in result and set(result) >= {"operation", "result", "summary"}
        check(f"analyze_corpus({operation}) returns the documented shape", ok,
              result.get("error") or result["summary"][:88])
    check("analyze_corpus rejects an unknown operation",
          analyzer.analyze("nonsense").get("error") == "unknown_operation")

    cluster = analyzer.analyze("cluster")
    if "error" not in cluster:
        check("cluster is validated against known topic tags",
              {"adjusted_rand_index", "purity", "silhouette"} <= set(cluster["result"]),
              f"ARI={cluster['result']['adjusted_rand_index']} "
              f"purity={cluster['result']['purity']} k={cluster['result']['k']}")

    # ---- inspect_figure ----------------------------------------------------
    step("inspect_figure, standalone")
    inspector = FigureInspector(CFG)
    figure_ids = sorted(manifest.figures)
    if figure_ids:
        resolved = inspector.inspect(figure_id=figure_ids[0])
        ok = "error" not in resolved and Path(resolved.get("image_path", "")).is_file()
        check("inspect_figure resolves an indexed figure to an image on disk", ok,
              resolved.get("error") or f"{resolved['figure_id']} p{resolved['page']}")
        check("inspect_figure returns a path, never image bytes",
              not any(isinstance(v, (bytes, bytearray)) for v in resolved.values()))
    check("inspect_figure rejects a missing reference",
          inspector.inspect().get("error") == "missing_reference")
    check("inspect_figure rejects an unknown figure",
          inspector.inspect(figure_id="nope__f99").get("error") == "unknown_figure")

    # ---- check_evidence_consistency ---------------------------------------
    step("check_evidence_consistency, standalone (loads the NLI model)")
    retriever = EvidenceRetriever(CFG)
    evidence = retriever.retrieve("how are experts selected for each token", k=4)
    chunk_ids = [c["chunk_id"] for c in evidence.get("chunks", [])]
    checker = ConsistencyChecker(CFG)

    if len(chunk_ids) >= 2:
        contradiction = checker.check("contradiction", chunk_ids)
        ok = "error" not in contradiction and {"found", "conflicting_pairs"} <= set(contradiction)
        check("contradiction mode returns the documented shape", ok,
              contradiction.get("error")
              or f"found={contradiction['found']}, {contradiction['n_pairs_scored']} pairs scored")

        draft = ("The router selects experts for each token using a learned gating network. "
                 "The corpus reports that sourdough ferments best at 24 degrees.")
        grounded = checker.check("groundedness", chunk_ids, answer_text=draft)
        ok = "error" not in grounded and {"claims", "grounded_ratio",
                                          "unsupported_claims"} <= set(grounded)
        check("groundedness mode returns the documented shape", ok,
              grounded.get("error")
              or f"ratio={grounded.get('grounded_ratio')} "
                 f"unsupported={len(grounded.get('unsupported_claims', []))}")
        if "error" not in grounded:
            for claim in grounded["claims"]:
                print(f"        {claim['label']:14} {claim['confidence']:.3f}  "
                      f"{claim['claim'][:74]}", flush=True)
            check("the fabricated claim is not marked entailed",
                  any("sourdough" in c["claim"].lower() and c["label"] != "entailed"
                      for c in grounded["claims"]),
                  "the planted unsupported claim must not pass")

    check("check_evidence_consistency rejects an unknown mode",
          checker.check("vibes", chunk_ids or ["x"]).get("error") == "unknown_mode")

    # ---- the agent selecting tools -----------------------------------------
    registry = build_full_registry(retriever=retriever, config=CFG)
    check("all five tools are registered", len(registry.names) == 5, ", ".join(registry.names))

    loop = AgentLoop(registry, config=CFG, retriever=retriever)
    questions = {
        "analyze_corpus": "What topics does this corpus cover, and what years do the papers span?",
        "retrieve_evidence": "How is the load-balancing loss defined in these papers?",
    }
    for expected, question in questions.items():
        step(f"live question expected to use {expected}")
        result = loop.run(question, question_id=f"m6_{expected}")
        if "error" in result:
            check(f"the agent answers: {expected}", False, f"{result['error']}: {result['detail']}")
            continue
        used = [record["tool_name"] for record in result["trace_records"]]
        check(f"the agent chose {expected}", expected in used,
              f"called: {used or 'nothing'} in {result['iterations']} iteration(s)")
        print(f"        answer: {' '.join(result['answer'].split())[:150]}...", flush=True)

    print(f"\n{'M6 GATE PASSED' if not failures else 'M6 GATE FAILED: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
