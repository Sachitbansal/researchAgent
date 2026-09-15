"""
The command-line entry point: index a topic, ask a question, run the eval.

In:  argv. Out: printed results and a process exit code.
Lives at the repo root rather than in src/ because `eval` is a subcommand, and
`eval/` imports `src/` — never the reverse (.claude/CLAUDE.md, Testing).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent


def _bootstrap(config_path: Optional[str]) -> None:
    """Point the config loader at a file, then make src/ and the repo importable.

    Order matters: config.CFG is built at import time and is the default argument of
    nearly every function in src/, so $SCIAGENT_CONFIG has to be set before the first
    import of anything under src/. That is also what lets --config redirect the whole
    system at a different data directory.
    """
    if config_path:
        os.environ["SCIAGENT_CONFIG"] = str(Path(config_path).expanduser().resolve())
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))


def _failed(result: Dict[str, Any]) -> bool:
    return isinstance(result, dict) and "error" in result


def _report_error(stage: str, result: Dict[str, Any]) -> int:
    print(f"\n{stage} failed: {result.get('error')} — {result.get('detail')}", file=sys.stderr)
    return 1


# ---- index -----------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> int:
    """Collect papers for a topic, extract and chunk them, then embed and index."""
    from config import CFG
    from corpus.collect import search_and_fetch
    from extraction.ingest import ingest_all
    from retrieval.indexer import index_chunks

    CFG.paths.ensure()

    if args.clear_arxiv_cooldown:
        CFG.paths.arxiv_cooldown.unlink(missing_ok=True)
        print("      cleared the arXiv rate-limit cooldown", flush=True)

    print(f"[1/3] searching arXiv for {args.topic!r} ...", flush=True)
    collected = search_and_fetch(
        args.topic, max_results=args.max_results, categories=args.categories, config=CFG
    )
    rate_limited = False
    if _failed(collected):
        partial = collected.get("partial") or {}
        if collected["error"] != "arxiv_rate_limited" or not partial.get("papers_added"):
            return _report_error("collection", collected)
        # A 429 partway through still downloaded whole PDFs. Ingest and index them, or
        # dedup hides them from every later run and they are never chunked at all.
        print(f"      rate-limited partway: {collected['detail']}", file=sys.stderr)
        collected = partial
        rate_limited = True
    print(
        f"      {len(collected.get('papers_added', []))} added, "
        f"{collected.get('papers_skipped', 0)} already held, "
        f"{collected.get('papers_tagged', 0)} re-tagged, tag {collected.get('topic_tag')!r}"
    )

    print("[2/3] extracting text and figures ...", flush=True)
    ingested = ingest_all(config=CFG, describe=not args.no_describe)
    if _failed(ingested):
        return _report_error("ingestion", ingested)
    print(f"      {ingested['chunks_total']} chunks, {ingested['figures_total']} figures")

    print("[3/3] embedding and indexing ...", flush=True)
    indexed = index_chunks(config=CFG, rebuild=args.rebuild)
    if _failed(indexed):
        return _report_error("indexing", indexed)
    print(
        f"      {indexed.get('chunks_indexed', 0)} embedded, "
        f"{indexed.get('chunks_skipped', 0)} already indexed, "
        f"{indexed.get('cache_hits', 0)} cache hit(s), "
        f"{indexed.get('total_indexed', 0)} vectors in the index"
    )
    if rate_limited:
        # Non-zero: what did land is indexed, but the topic is short of what was asked
        # for, and a caller scripting this must not read that as a complete collection.
        print(
            "\ncollection was cut short by an arXiv rate limit; "
            "re-run this command once it lifts to fetch the rest",
            file=sys.stderr,
        )
        return 1
    return 0


# ---- ask -------------------------------------------------------------------------


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer one question with the agent loop, printing the answer and the trace path."""
    from agent.loop import AgentLoop
    from agent.tool_registry import build_full_registry
    from config import CFG
    from retrieval.embedder import Embedder
    from tools.retrieve_evidence import EvidenceRetriever

    CFG.paths.ensure()
    retriever = EvidenceRetriever(CFG, embedder=Embedder(CFG))
    loop = AgentLoop(build_full_registry(retriever=retriever, config=CFG),
                     config=CFG, retriever=retriever)

    result = loop.run(args.question)
    if _failed(result):
        partial = result.get("partial") or {}
        if partial.get("answer"):
            print(partial["answer"])
        return _report_error("the agent loop", result)

    print(result["answer"])
    print(
        f"\n--- {result['iterations']} iteration(s), {result['tool_calls']} tool call(s), "
        f"{result['context_tokens']} context tokens, stopped: {result['stopped_because']}"
    )
    print(f"--- trace: {result['trace']}")

    if args.show_tools:
        for record in result["trace_records"]:
            summary = f"{len(record.get('chunk_ids_returned') or [])} chunk(s)"
            print(f"    [{record['iteration']}] {record['tool_name']}"
                  f"({json.dumps(record.get('args'))[:80]}) "
                  f"-> {record.get('error') or summary}, {record['latency_ms']}ms")
    return 0


# ---- eval ------------------------------------------------------------------------


def _print_metrics(name: str, metrics: Dict[str, Any]) -> None:
    print(f"\n{name} — {metrics['n_questions']} question(s)")
    for label, key in (
        ("fact coverage", "fact_coverage"),
        ("abstention accuracy", "abstention_accuracy"),
        ("citations resolved", "citations_resolved"),
        ("gold-paper hit rate", "gold_paper_hit_rate"),
        ("MRR", "mrr"),
    ):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            print(f"  {label:<22} {value:.3f}")
    recall = metrics.get("recall") or {}
    if recall:
        print("  recall@k               " + "  ".join(f"@{k}={v:.3f}" for k, v in recall.items()))
    if metrics.get("errors"):
        print(f"  errors                 {metrics['errors']}")


def cmd_eval(args: argparse.Namespace) -> int:
    """Run one eval split and write the results JSON."""
    from eval.runner import run_split, save

    split = "held_out_split" if args.held_out else "tuning_split"
    # A truncated run must not overwrite the full one. docs/EVALUATION.md cites the
    # committed tuning_split.json by number, so a --limit 3 smoke run landing on that
    # filename would silently replace the evidence the write-up rests on.
    name = split if not args.limit else f"{split}_limit{args.limit}"
    result = run_split(
        held_out=args.held_out, limit=args.limit, check_groundedness=not args.no_groundedness
    )
    path = save(result, name)
    _print_metrics(split, result)
    print(f"\nwritten to {path}")
    return 0


# ---- parser ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The three things the system does, as three subcommands."""
    parser = argparse.ArgumentParser(
        prog="main.py", description="Agentic research assistant over scientific papers."
    )
    parser.add_argument(
        "--config", metavar="PATH",
        help="config.yaml to use (default: ./config.yaml, or $SCIAGENT_CONFIG). "
             "A config with different paths.* gives a separate corpus.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="collect, extract and index a topic")
    index.add_argument("topic", help="what to search arXiv for, in plain language")
    index.add_argument("--max-results", type=int, default=None,
                       help="papers to fetch (default: collection.max_results_default)")
    index.add_argument("--categories", nargs="*", default=None,
                       help="restrict to arXiv categories, e.g. cs.CL cs.LG")
    index.add_argument("--no-describe", action="store_true",
                       help="skip vision calls; figures index on caption text alone")
    index.add_argument("--clear-arxiv-cooldown", action="store_true",
                       help="forget a recorded arXiv rate limit and search anyway")
    index.add_argument("--rebuild", action="store_true",
                       help="rebuild the vector index from scratch")
    index.set_defaults(handler=cmd_index)

    ask = subparsers.add_parser("ask", help="answer one question against the index")
    ask.add_argument("question", help="the question, quoted")
    ask.add_argument("--show-tools", action="store_true", help="print each tool call made")
    ask.set_defaults(handler=cmd_ask)

    evaluate = subparsers.add_parser("eval", help="run the eval harness")
    evaluate.add_argument("--held-out", action="store_true",
                          help="run the held-out third instead of the tuning split")
    evaluate.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    evaluate.add_argument("--no-groundedness", action="store_true",
                          help="skip the NLI groundedness pass")
    evaluate.set_defaults(handler=cmd_eval)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _bootstrap(args.config)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
