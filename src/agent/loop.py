"""
The agent loop: call the model, run whatever tools it asks for, repeat until it answers.

In:  a question, an LLMClient, a ToolRegistry, and an optional on_event observer.
Out: {"answer", "iterations", "tool_calls", "tool_call_counts", "run_id",
     "stopped_because", ...}, plus a live event per step to on_event if one was given.
     Bounded by a per-tool call limit and two backstops, not an iteration cap
     (docs/DECISIONS.md D4). Written directly against the provider SDK — no framework.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from agent.conversation import Conversation
from agent.fetch_proposal import derive_fetch_proposal
from agent.tool_registry import ToolRegistry
from agent.trace import TraceWriter
from config import CFG, Config
from llm_client import LLMClient, LLMError, LLMResponse, attach_images

# What the model is told when it repeats a call it already made verbatim.
LOOP_NOTICE = (
    "You already made this exact call earlier in this conversation and its result is "
    "above. Repeating it returns nothing new. Either query with different wording, use a "
    "different tool, or answer from the evidence you already have."
)

# What the model is told when it calls a tool whose per-question budget is spent.
# Formatted with the tool name and its limit.
LIMIT_NOTICE = (
    "You have used all {limit} calls to {tool} allowed for this question, so this call was "
    "not run. {tool} is no longer offered. Use a different tool, or answer from the "
    "evidence you already have."
)

# The forced final answer, one message per reason the loop stopped asking for tools.
TOOLS_EXHAUSTED_NUDGE = (
    "All tool budgets for this question are used up, so no more tool calls are possible. "
    "Answer now using only the evidence already retrieved above. State explicitly that "
    "the evidence is partial, and do not claim anything the retrieved chunks do not support."
)

NO_PROGRESS_NUDGE = (
    "Several turns have passed without any new evidence: every recent tool call was "
    "refused. Stop calling tools and answer now using only the evidence already retrieved "
    "above. State explicitly that the evidence is partial, and do not claim anything the "
    "retrieved chunks do not support."
)

STOP_NUDGES = {"tools_exhausted": TOOLS_EXHAUSTED_NUDGE, "no_progress": NO_PROGRESS_NUDGE}

# Filled into the system prompt's {tool_call_budget} placeholder.
BUDGET_PLACEHOLDER = "{tool_call_budget}"


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
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.registry = registry
        self.config = config
        self.client = client if client is not None else LLMClient(config=config)
        self.retriever = retriever
        self.on_event = on_event

    def _emit(self, kind: str, **fields: Any) -> None:
        """Announce what the loop is about to do, or just did. Never raises.

        Same rule as TraceWriter.record: an observer is for watching a run, so a broken
        one must not be able to end it. No reporter means no work at all.
        """
        if self.on_event is None:
            return
        try:
            self.on_event({"kind": kind, **fields})
        except Exception:
            pass

    @staticmethod
    def _attach_image(conversation: Conversation, resolved: Dict[str, Any]) -> None:
        """Append the image on a user turn, which is the only place it is valid.

        docs/ARCHITECTURE.md section 9: the tool returns a path, the loop attaches the
        file. Keeping this in the loop rather than the tool is what lets inspect_figure
        stay a structured-dict tool like every other one.
        """
        prompt = resolved.get("question") or "Read this image and answer from what it shows."
        caption = resolved.get("caption")
        parts = [f"Image for {resolved.get('figure_id') or 'the file you asked to inspect'}."]
        if caption:
            parts.append(f"Caption: {caption}")
        parts.append(prompt)

        # attach_images appends image parts to the last user turn, so the text goes first.
        conversation.add_user(" ".join(parts))
        attached = attach_images(conversation.messages(), [resolved["image_path"]])
        conversation._messages[-1] = attached[-1]

    def _limits(self) -> Dict[str, int]:
        """Per-question dispatch limit for every registered tool.

        agent.max_calls_per_tool, unless agent.tool_call_limits names the tool.
        """
        agent = self.config.agent
        default = int(agent.max_calls_per_tool)
        overrides = agent.get("tool_call_limits") or {}
        return {name: int(overrides.get(name, default)) for name in self.registry.names}

    def _system_prompt(self, limits: Dict[str, int]) -> str:
        """The system prompt with the tool budget filled in from config, not hardcoded."""
        default = int(self.config.agent.max_calls_per_tool)
        phrase = f"{default} calls per tool"
        differing = {name: n for name, n in limits.items() if n != default}
        if differing:
            phrase += " (except " + ", ".join(
                f"{name}: {n}" for name, n in sorted(differing.items())) + ")"
        return self.config.prompt("system").replace(BUDGET_PLACEHOLDER, phrase)

    def run(self, question: str, question_id: Optional[str] = None,
            run_id: Optional[str] = None) -> Dict[str, Any]:
        """Answer `question`, calling tools as the model chooses.

        Termination, without an iteration cap: each tool is dispatched at most its limit,
        so at most sum(limits) turns make progress, and `max_no_progress_turns`
        consecutive turns that dispatch nothing end the run: at most sum(limits)*(N+1)+N
        turns. With the shipped config (4 tools x 10, search_literature 5, N=3), 183.
        """
        question = (question or "").strip()
        if not question:
            return {"error": "empty_question", "detail": "a non-empty question is required",
                    "partial": None}

        # Chunk dedup is per-conversation state, so a fresh question starts clean.
        if self.retriever is not None:
            self.retriever.reset()

        config = self.config
        limits = self._limits()
        counts: Dict[str, int] = {name: 0 for name in limits}
        max_no_progress = int(config.agent.max_no_progress_turns)
        conversation = Conversation(self._system_prompt(limits), question, config)
        tracer = TraceWriter(run_id=run_id, question_id=question_id, config=config)
        seen_calls: Set[str] = set()
        repeated_calls = 0
        stopped_because = "answered"
        no_progress_turns = 0
        turns = 0

        def failure(exc: LLMError) -> Dict[str, Any]:
            return {
                "error": exc.code,
                "detail": exc.detail,
                "partial": {
                    "answer": conversation.final_text(),
                    "iterations": turns,
                    "run_id": tracer.run_id,
                    "trace": str(tracer.path),
                },
            }

        def refuse(call_id: str, iteration: int, tool_name: str, error: Dict[str, Any],
                   reason: str) -> None:
            # Every tool_call id must get a result, refused or not, or the next request
            # is invalid. A refusal never consumes the tool's budget.
            conversation.add_tool_result(call_id, error)
            self._emit("skipped", iteration=iteration, tool_name=tool_name, reason=reason)

        while True:
            iteration = turns
            available = [name for name in limits if counts[name] < limits[name]]
            self._emit("thinking", iteration=iteration)
            try:
                response = self.client.complete(
                    conversation.messages(), tools=self.registry.schemas(available),
                    role="agent",
                )
            except LLMError as exc:
                return failure(exc)
            turns += 1

            if not response.has_tool_calls:
                conversation.add_assistant(response)
                self._emit("answering", iteration=iteration)
                stopped_because = "answered"
                break

            # The assistant turn must be appended before any of its results, and appended
            # verbatim, or the tool_call ids stop matching.
            conversation.add_assistant(response)

            pending_images: List[Dict[str, Any]] = []
            dispatched_this_turn = 0

            for call in response.tool_calls:
                if not call.ok:
                    refuse(call.id, iteration, call.name, {
                        "error": "bad_arguments",
                        "detail": f"arguments were not valid JSON: {call.parse_error}",
                        "partial": None,
                    }, "arguments were not valid JSON")
                    continue

                if call.name not in self.registry:
                    # Caught before dispatch so a hallucinated name is a refusal like any
                    # other: no budget, no loop-detection key, no trace record.
                    refuse(call.id, iteration, call.name,
                           self.registry.unknown_tool_error(call.name), "no such tool")
                    continue

                key = _canonical(call.name, call.arguments)
                if bool(config.agent.loop_detection) and key in seen_calls:
                    # A synthetic result rather than a hard break: the situation is
                    # recoverable, and the model can change tack on the next turn.
                    repeated_calls += 1
                    refuse(call.id, iteration, call.name, {
                        "error": "repeated_call", "detail": LOOP_NOTICE, "partial": None,
                    }, "already called with these arguments")
                    continue

                if counts[call.name] >= limits[call.name]:
                    # Checked per call, not per turn: one turn can ask for more parallel
                    # calls than the tool has budget left.
                    refuse(call.id, iteration, call.name, {
                        "error": "tool_limit_reached",
                        "detail": LIMIT_NOTICE.format(tool=call.name,
                                                      limit=limits[call.name]),
                        "partial": None,
                    }, f"{call.name} call limit ({limits[call.name]}) reached")
                    continue

                seen_calls.add(key)
                counts[call.name] += 1
                dispatched_this_turn += 1
                self._emit("tool_start", iteration=iteration, tool_name=call.name,
                           args=call.arguments)
                started = time.perf_counter()
                result = self.registry.dispatch(call.name, call.arguments)
                latency_ms = int((time.perf_counter() - started) * 1000)
                self._emit("tool_end", iteration=iteration, tool_name=call.name,
                           result=result, latency_ms=latency_ms)

                tracer.record(iteration, call.name, call.arguments, result, latency_ms)
                conversation.add_tool_result(call.id, result)

                # A tool that resolved an image cannot hand it back in its own result:
                # the OpenAI-compatible schema only accepts image parts on a user turn.
                # Collect it now and attach after every tool_call has been answered,
                # because an unanswered tool_call makes the whole request invalid.
                if isinstance(result, dict) and result.get("image_path") and "error" not in result:
                    pending_images.append(result)

            for image in pending_images:
                self._emit("image", iteration=iteration, image_path=image.get("image_path"))
                self._attach_image(conversation, image)

            conversation.elide_if_needed()

            # The two backstops. Together with the per-tool limits they are what bound
            # the loop; there is deliberately no absolute iteration ceiling.
            no_progress_turns = 0 if dispatched_this_turn else no_progress_turns + 1
            if all(counts[name] >= limits[name] for name in limits):
                stopped_because = "tools_exhausted"
            elif no_progress_turns >= max_no_progress:
                stopped_because = "no_progress"
            else:
                continue

            self._emit("cap", iteration=iteration, reason=stopped_because)
            if bool(config.agent.final_answer_nudge):
                conversation.add_user(STOP_NUDGES[stopped_because])
                try:
                    # No tools offered, so the model cannot ask for another call.
                    forced = self.client.complete(conversation.messages(), role="agent")
                    conversation.add_assistant(forced)
                except LLMError as exc:
                    return failure(exc)
            break

        return {
            "answer": conversation.final_text(),
            "question": question,
            "iterations": turns,
            "tool_calls": sum(counts.values()),
            "tool_call_counts": dict(counts),
            "repeated_calls": repeated_calls,
            "stopped_because": stopped_because,
            "messages_in_context": len(conversation),
            "context_tokens": conversation.token_estimate(),
            "elided_results": conversation.elided,
            "run_id": tracer.run_id,
            "trace": str(tracer.path),
            "trace_records": tracer.records,
            "fetch_proposal": derive_fetch_proposal(tracer.records),
            "conversation": conversation,
        }
