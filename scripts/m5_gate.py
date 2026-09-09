"""
The M5 verification gate from docs/BUILD_PLAN.md: one real question, end to end.

In:  an index built by M3 and a funded .env key.
Out: the message array printed per iteration so the stateless re-send is visible, the
     answer, and the trace file for manual inspection. Exits non-zero on failure.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.loop import AgentLoop  # noqa: E402
from agent.tool_registry import build_default_registry  # noqa: E402
from common.storage import read_jsonl  # noqa: E402
from config import CFG  # noqa: E402
from corpus.chunk_store import ChunkStore  # noqa: E402
from tools.retrieve_evidence import EvidenceRetriever  # noqa: E402

QUESTION = "How do these papers decide which experts a token or patch gets routed to?"

TRACE_FIELDS = {
    "run_id", "question_id", "iteration", "tool_name", "args", "result_summary",
    "chunk_ids_returned", "error", "latency_ms", "timestamp",
}

failures: List[str] = []
_started = time.monotonic()


def _elapsed() -> str:
    return f"{time.monotonic() - _started:6.1f}s"


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{_elapsed()}] [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""),
          flush=True)
    if not ok:
        failures.append(label)


def describe(message: Dict[str, Any]) -> str:
    """One line summarising a message, for the conversation dump."""
    role = message.get("role")
    if role == "tool":
        return f"tool(id={message.get('tool_call_id')}) {len(str(message.get('content','')))} chars"
    calls = message.get("tool_calls") or []
    if calls:
        names = ", ".join(
            f"{(c.get('function') or {}).get('name')}(id={c.get('id')})" for c in calls
        )
        return f"assistant -> tool_calls: {names}"
    content = message.get("content")
    text = content if isinstance(content, str) else str(content)
    return f"{role}: {' '.join(text.split())[:80]}"


def main() -> int:
    if ChunkStore().count() == 0:
        print("No chunks indexed. Run scripts/m3_gate.py first.")
        return 1

    print(f"model: {CFG.llm.agent_model}   max_iterations: {CFG.agent.max_iterations}")
    print(f"question: {QUESTION}\n", flush=True)

    retriever = EvidenceRetriever(CFG)
    registry = build_default_registry(retriever, CFG)
    loop = AgentLoop(registry, config=CFG, retriever=retriever)

    result = loop.run(QUESTION, question_id="m5_gate")

    if "error" in result:
        check("the loop completes", False, f"{result['error']}: {result['detail']}")
        return 1
    check("the loop completes", True,
          f"{result['iterations']} iteration(s), stopped_because={result['stopped_because']}")

    # The conversation is the whole memory. Print it so the re-send is visible.
    conversation = result["conversation"]
    print(f"\n[{_elapsed()}] ---- final message array ({len(conversation)} messages) ----",
          flush=True)
    for position, message in enumerate(conversation.messages()):
        print(f"  [{position}] {describe(message)}", flush=True)

    messages = conversation.messages()
    roles = [m["role"] for m in messages]
    check("conversation starts with system then user", roles[:2] == ["system", "user"])

    # Every assistant tool_call must have exactly one matching tool message, or the API
    # would have rejected the request that carried them.
    requested = [c["id"] for m in messages for c in (m.get("tool_calls") or [])]
    answered = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    check("every tool_call has exactly one matching tool result",
          sorted(requested) == sorted(answered),
          f"{len(requested)} requested, {len(answered)} answered")
    check("tool results are strings, not dicts",
          all(isinstance(m["content"], str) for m in messages if m["role"] == "tool"))

    check("the model called at least one tool", result["tool_calls"] > 0,
          f"{result['tool_calls']} call(s)")
    check("an answer was produced", bool(result["answer"].strip()),
          f"{len(result['answer'])} chars")

    print(f"\n[{_elapsed()}] ---- answer ----\n", flush=True)
    print(result["answer"], flush=True)
    print(flush=True)

    # Citations are what make the answer checkable. Not all questions warrant one, so
    # this is reported rather than enforced.
    cited = [c for c in ChunkStore().all() if c["chunk_id"] in result["answer"]]
    print(f"[{_elapsed()}] .... answer cites {len(cited)} chunk_id(s) inline", flush=True)

    # Trace file, per DATA_SCHEMA section 7.
    records = list(read_jsonl(Path(result["trace"])))
    check("a trace record exists per tool call", len(records) == result["tool_calls"],
          f"{len(records)} records at {result['trace']}")
    if records:
        check("trace records match the documented schema",
              all(set(r) == TRACE_FIELDS for r in records),
              f"fields: {sorted(records[0])}")
        check("chunk_ids_returned is populated for retrievals",
              any(r["chunk_ids_returned"] for r in records))
        check("result_summary is truncated",
              all(len(str(chunk.get("text", ""))) <= int(CFG.agent.trace_text_chars)
                  for r in records
                  for chunk in (r["result_summary"].get("chunks") or [])))

        print(f"\n[{_elapsed()}] ---- trace ----", flush=True)
        for record in records:
            print(f"  iter={record['iteration']} {record['tool_name']} "
                  f"{record['latency_ms']}ms args={json.dumps(record['args'])[:70]}", flush=True)
            print(f"        -> {len(record['chunk_ids_returned'])} chunks: "
                  f"{record['chunk_ids_returned'][:4]}", flush=True)

    print(f"\n[{_elapsed()}] .... context: {result['context_tokens']} tokens across "
          f"{result['messages_in_context']} messages, {result['elided_results']} elided",
          flush=True)

    print(f"\n{'M5 GATE PASSED' if not failures else 'M5 GATE FAILED: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
