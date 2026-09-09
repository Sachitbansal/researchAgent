"""
The agent loop: call the model, run whatever tools it asks for, repeat until it answers.

In:  a question, an LLMClient, and a ToolRegistry.
Out: {"answer", "iterations", "tool_calls", "run_id", "stopped_because", ...}. Written
     directly against the provider SDK — no framework — because how this loop is shaped
     is itself part of what the project is evaluating (ARCHITECTURE section 14).
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from agent.conversation import Conversation
from agent.tool_registry import ToolRegistry
from agent.trace import TraceWriter
from config import CFG, Config
from llm_client import LLMClient, LLMError, LLMResponse

# What the model is told when it repeats a call it already made verbatim.
LOOP_NOTICE = (
    "You already made this exact call earlier in this conversation and its result is "
    "above. Repeating it returns nothing new. Either query with different wording, use a "
    "different tool, or answer from the evidence you already have."
)

CAP_NUDGE = (
    "You have used all available tool calls for this question. Answer now using only the "
    "evidence already retrieved above. State explicitly that the evidence is partial, and "
    "do not claim anything the retrieved chunks do not support."
)


def _canonical(name: str, arguments: Dict[str, Any]) -> str:
    """A stable key for one (tool, args) pair, for loop detection.

    Sorted keys so that the same call written with its arguments in a different order is
    still recognised as the same call.
    """
    try:
        payload = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        payload = repr(sorted(arguments.items())) if isinstance(arguments, dict) else repr(arguments)
    return f"{name}::{payload}"


class AgentLoop:
    """One question, start to finish.

    The provider is stateless, so every iteration re-sends the entire conversation. The
    Conversation object owns that array; this class decides what to append to it.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        client: Optional[LLMClient] = None,
        config: Config = CFG,
        retriever: Optional[Any] = None,
    ) -> None:
        self.registry = registry
        self.config = config
        self.client = client if client is not None else LLMClient(config=config)
        self.retriever = retriever

    def run(self, question: str, question_id: Optional[str] = None,
            run_id: Optional[str] = None) -> Dict[str, Any]:
        """Answer `question`, calling tools as the model chooses."""
        question = (question or "").strip()
        if not question:
            return {"error": "empty_question", "detail": "a non-empty question is required",
                    "partial": None}

        # Chunk dedup is per-conversation state, so a fresh question starts clean.
        if self.retriever is not None:
            self.retriever.reset()

        config = self.config
        conversation = Conversation(config.prompt("system"), question, config)
        tracer = TraceWriter(run_id=run_id, question_id=question_id, config=config)
        seen_calls: Set[str] = set()
        tool_calls_made = 0
        repeated_calls = 0
        stopped_because = "answered"
        max_iterations = int(config.agent.max_iterations)
        iteration = 0

        for iteration in range(max_iterations):
            try:
                response = self.client.complete(
                    conversation.messages(), tools=self.registry.schemas(), role="agent"
                )
            except LLMError as exc:
                return {
                    "error": exc.code,
                    "detail": exc.detail,
                    "partial": {
                        "answer": conversation.final_text(),
                        "iterations": iteration,
                        "run_id": tracer.run_id,
                        "trace": str(tracer.path),
                    },
                }

            if not response.has_tool_calls:
                conversation.add_assistant(response)
                stopped_because = "answered"
                break

            # The assistant turn must be appended before any of its results, and appended
            # verbatim, or the tool_call ids stop matching.
            conversation.add_assistant(response)

            for call in response.tool_calls:
                if not call.ok:
                    conversation.add_tool_result(call.id, {
                        "error": "bad_arguments",
                        "detail": f"arguments were not valid JSON: {call.parse_error}",
                        "partial": None,
                    })
                    continue

                key = _canonical(call.name, call.arguments)
                if bool(config.agent.loop_detection) and key in seen_calls:
                    # A synthetic result rather than a hard break: the situation is
                    # recoverable, and the model can change tack on the next turn.
                    repeated_calls += 1
                    conversation.add_tool_result(call.id, {
                        "error": "repeated_call", "detail": LOOP_NOTICE, "partial": None,
                    })
                    continue

                seen_calls.add(key)
                started = time.perf_counter()
                result = self.registry.dispatch(call.name, call.arguments)
                latency_ms = int((time.perf_counter() - started) * 1000)
                tool_calls_made += 1

                tracer.record(iteration, call.name, call.arguments, result, latency_ms)
                conversation.add_tool_result(call.id, result)

            conversation.elide_if_needed()
        else:
            # Cap reached with tools still being requested.
            stopped_because = "iteration_cap"
            if bool(config.agent.final_answer_nudge):
                conversation.add_user(CAP_NUDGE)
                try:
                    # No tools offered, so the model cannot ask for another call.
                    forced = self.client.complete(conversation.messages(), role="agent")
                    conversation.add_assistant(forced)
                except LLMError as exc:
                    return {
                        "error": exc.code,
                        "detail": exc.detail,
                        "partial": {
                            "answer": conversation.final_text(),
                            "iterations": max_iterations,
                            "run_id": tracer.run_id,
                            "trace": str(tracer.path),
                        },
                    }

        return {
            "answer": conversation.final_text(),
            "question": question,
            "iterations": iteration + 1,
            "tool_calls": tool_calls_made,
            "repeated_calls": repeated_calls,
            "stopped_because": stopped_because,
            "messages_in_context": len(conversation),
            "context_tokens": conversation.token_estimate(),
            "elided_results": conversation.elided,
            "run_id": tracer.run_id,
            "trace": str(tracer.path),
            "trace_records": tracer.records,
            "conversation": conversation,
        }
