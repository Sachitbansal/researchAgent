"""
Short-term memory for one thread: earlier turns as plain messages, inside a token budget.

In:  a run's trace records (for the notes), or a thread's stored turns, a TokenCounter,
     the `agent.thread_history_tokens` budget and the notes header from src/prompts/.
Out: one-line tool-call notes per run, and the prior messages that go between the system
     prompt and the current question: user(question), assistant(note), assistant(answer).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from agent.progress import format_result


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def _args_phrase(args: Any, width: int) -> str:
    """`query="load balancing", k=5`, clipped. Values as JSON, so strings read as strings."""
    if not isinstance(args, dict) or not args:
        return ""
    parts = []
    for key, value in args.items():
        try:
            rendered = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            rendered = repr(value)
        parts.append(f"{key}={rendered}")
    return _clip(", ".join(parts), width)


def tool_note(record: Dict[str, Any], width: int) -> str:
    """One trace record as one line: the call, and what it came back with.

    Chunk ids are listed in full because they are what the stored answer cites: they let
    a later question tell which passages the earlier answer rested on, even though the
    passages' text is no longer in context.
    """
    name = str(record.get("tool_name") or "tool")
    call = f"{name}({_args_phrase(record.get('args'), width)})"
    summary = record.get("result_summary") or {}
    chunk_ids = [str(c) for c in (record.get("chunk_ids_returned") or [])]
    if record.get("error"):
        outcome = f"error: {record['error']}"
    elif chunk_ids:
        outcome = f"{len(chunk_ids)} chunk{'s' if len(chunk_ids) != 1 else ''} [{', '.join(chunk_ids)}]"
    else:
        outcome = format_result(summary, width)
    return f"{call} → {outcome}"


def build_notes(trace_records: Sequence[Dict[str, Any]], width: int) -> List[str]:
    """The notes stored with a turn: one line per dispatched tool call, in call order."""
    return [tool_note(record, width) for record in trace_records if isinstance(record, dict)]


def note_message(notes: Sequence[str], header: str) -> str:
    """The assistant message carrying a prior turn's notes. Empty when it made no calls."""
    if not notes:
        return ""
    return header.strip() + "\n" + "\n".join(f"- {note}" for note in notes)


def build_history(
    turns: Sequence[Dict[str, Any]], counter: Any, budget: int, header: str,
    partial_marker: str = "",
) -> Dict[str, Any]:
    """Prior turns as messages, newest kept first, whole turns only.

    Each turn costs its question + note + answer, counted with `counter` (the same ruler
    the context budget uses). Walking newest to oldest, the first turn that does not fit
    ends the walk, so what is dropped is always the oldest contiguous run, never a hole
    in the middle. A turn with no answer is skipped: a question followed by nothing
    tells the model nothing, and an empty assistant message is rejected by some
    providers.

    Never a tool message: prior turns become plain user/assistant text, so no replayed
    tool_call can be left without its result.

    A turn whose run failed (its stats carry an `error`) keeps its partial answer, but
    `partial_marker` is put in front of it: replayed bare, an answer cut short reads to
    the model like a complete one.
    """
    kept: List[List[Dict[str, Any]]] = []
    used = 0
    usable = [t for t in turns if str(t.get("answer") or "").strip()]
    dropped = 0
    for index in range(len(usable) - 1, -1, -1):
        turn = usable[index]
        question = str(turn.get("question") or "")
        answer = str(turn.get("answer") or "")
        error = (turn.get("stats") or {}).get("error")
        if error and partial_marker:
            answer = partial_marker.strip().replace("{error}", str(error)) + "\n\n" + answer
        note = note_message(turn.get("notes") or [], header)
        cost = counter.count(question) + counter.count(note) + counter.count(answer)
        if used + cost > budget:
            dropped = index + 1
            break
        used += cost
        messages = [{"role": "user", "content": question}]
        if note:
            messages.append({"role": "assistant", "content": note})
        messages.append({"role": "assistant", "content": answer})
        kept.append(messages)

    kept.reverse()
    return {
        "messages": [message for turn in kept for message in turn],
        "turns_used": len(kept),
        "turns_dropped": dropped,
        "turns_skipped_empty": len(turns) - len(usable),
        "tokens": used,
    }
