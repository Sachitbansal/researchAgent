"""
The agent loop: call the model, run whatever tools it asks for, repeat until it answers.

In:  a question, an LLMClient, a ToolRegistry, an optional on_event observer, and
     optionally a ThreadStore plus a thread_id (short-term memory, D12) and a
     LongTermMemory (memories across chats, D19), which only a thread run touches.
Out: {"answer", "iterations", "tool_calls", "tool_call_counts", "run_id",
     "stopped_because", "thread_id", "turn_seq", "memory_injected", "fetch_proposal",
     ...}, plus a live event per step to on_event; remember(result) writes memory after
     the answer. Bounded by per-tool limits and two backstops (D4). No thread: nothing
     read or persisted.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from agent.conversation import Conversation
from agent.fetch_proposal import derive_fetch_proposal
from agent.tool_registry import ToolRegistry
from agent.trace import TraceWriter, summarise_result
from common.tokenization import TokenCounter
from config import CFG, Config
from llm_client import LLMClient, LLMError, LLMResponse, attach_images
from memory.history import build_history, build_notes

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
        store: Optional[Any] = None,
        memory: Optional[Any] = None,
    ) -> None:
        self.registry = registry
        self.config = config
        self.client = client if client is not None else LLMClient(config=config)
        self.retriever = retriever
        self.on_event = on_event
        # A memory.threads.ThreadStore, or anything with get_thread/append_turn. Only
        # touched by a run that names a thread_id; the eval harness never does.
        self.store = store
        # A memory.longterm.LongTermMemory. Same rule as the store: read at the start of a
        # thread run and written by remember() after it; a run without a thread_id never
        # touches it, so the eval's runs are unaffected (D19).
        self.memory = memory
        # Summarised step events of the current thread run, stored with its turn so a
        # past answer's timeline can be redrawn. None outside a thread run: no work.
        self._steps: Optional[List[Dict[str, Any]]] = None
        self._started = 0.0

    def _emit(self, kind: str, **fields: Any) -> None:
        """Announce what the loop is about to do, or just did. Never raises.

        Same rule as TraceWriter.record: an observer is for watching a run, so a broken
        one must not be able to end it. No reporter and no thread means no work at all.
        """
        event = {"kind": kind, **fields}
        if self._steps is not None:
            try:
                step = dict(event)
                if "result" in step:
                    step["result"] = summarise_result(step["result"],
                                                      int(self.config.agent.trace_text_chars))
                step["t_ms"] = int((time.perf_counter() - self._started) * 1000)
                self._steps.append(step)
            except Exception:
                pass
        if self.on_event is None:
            return
        try:
            self.on_event(event)
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

    def _history(self, thread_id: str) -> Dict[str, Any]:
        """The thread's earlier turns as messages within budget, or an error dict."""
        if self.store is None:
            return {"error": "no_thread_store",
                    "detail": "a thread_id was given but this loop has no ThreadStore"}
        thread = self.store.get_thread(thread_id)
        if "error" in thread:
            return thread
        return build_history(
            thread.get("turns") or [], TokenCounter(self.config),
            int(self.config.agent.thread_history_tokens), self.config.prompt("thread_notes"),
            partial_marker=self.config.prompt("thread_partial"),
        )

    def _save_turn(self, thread_id: str, question: str, result: Dict[str, Any],
                   tracer: TraceWriter, history: Dict[str, Any],
                   approval: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Append this run as the thread's next turn; failed runs too, with their error.

        Returns the fields merged into the run's result: thread_id and turn_seq, plus
        thread_error if the store refused the write (the answer is still returned). An
        approval run is stored under its short readable question, and says in its stats
        which fetch it approved, so a reloaded chat knows that proposal was acted on (D26).
        """
        failed = "error" in result
        body = (result.get("partial") or {}) if failed else result
        stats: Dict[str, Any] = {
            "iterations": body.get("iterations"),
            "run_id": body.get("run_id"),
            "elapsed_ms": int((time.perf_counter() - self._started) * 1000),
            "history_turns": history.get("turns_used", 0),
            "history_turns_dropped": history.get("turns_dropped", 0),
            "history_tokens": history.get("tokens", 0),
        }
        if failed:
            stats.update(error=result.get("error"), detail=result.get("detail"))
        else:
            stats.update({key: result.get(key) for key in (
                "tool_calls", "tool_call_counts", "context_tokens", "stopped_because",
                "fetch_proposal")})
        if approval is not None:
            stats["approved_fetch"] = {"query": approval.get("query")}
            question = str(approval.get("display") or question)
        saved = self.store.append_turn(
            thread_id, question, str(body.get("answer") or ""), stats=stats,
            notes=build_notes(tracer.records, int(self.config.agent.thread_note_arg_chars)),
            steps=list(self._steps or []), run_id=body.get("run_id"),
        )
        if "error" in saved:
            return {"thread_id": thread_id, "turn_seq": None, "thread_error": saved}
        return {"thread_id": thread_id, "turn_seq": saved["seq"]}

    def _recall(self, question: str) -> Optional[Dict[str, Any]]:
        """Long-term memories to inject into this thread run, or None. Never raises.

        Emitted as a `memory` step whose `result.chunks` are the attached passages, so the
        stored steps (and the page's Sources and citation cards) treat them as evidence
        this run had in view.
        """
        if self.memory is None:
            return None
        try:
            recalled = self.memory.recall(question)
        except Exception as exc:  # recall is meant not to raise; a run must not end here
            recalled = {"error": "memory_recall_failed", "detail": f"{type(exc).__name__}: {exc}"}
        if "error" in recalled:
            return {"error": recalled["error"], "detail": recalled.get("detail", ""),
                    "memories": [], "preferences": [], "chunk_ids": [], "block": ""}
        if not recalled.get("block"):
            return None
        self._emit("memory", iteration=0,
                   memories=recalled.get("memories") or [],
                   preferences=recalled.get("preferences") or [],
                   stale=recalled.get("stale") or [],
                   tokens=recalled.get("tokens"),
                   result={"chunks": recalled.get("chunks") or []})
        return recalled

    def remember(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Write long-term memory from a finished thread run; None when there is nothing to do.

        Separate from run() so the caller sends the answer first: the web stream sends
        `answer`, then calls this, then sends the `memory` frame (D19). The report is also
        merged into the stored turn's stats, so a reloaded chat still shows it. A failure
        comes back in the report; it never fails the run that was already answered.
        """
        thread_id = result.get("thread_id")
        if thread_id is None or self.memory is None:
            return None
        injected = result.get("memory_injected") or {}
        if "error" in result:
            report: Dict[str, Any] = {"written": [], "refreshed": [], "invalidated": [],
                                      "used": [], "forgotten": [], "dropped": {},
                                      "skipped": "run_failed"}
        else:
            try:
                report = self.memory.after_run(
                    str(result.get("question") or ""), str(result.get("answer") or ""),
                    thread_id, result.get("trace_records") or [], injected)
            except Exception as exc:  # after_run is meant not to raise
                report = {"written": [], "refreshed": [], "invalidated": [], "used": [],
                          "forgotten": [], "dropped": {}, "error": "memory_write_failed",
                          "detail": f"{type(exc).__name__}: {exc}"}
        if injected.get("error"):
            report.setdefault("recall_error", f"{injected['error']}: {injected.get('detail', '')}")
        seq = result.get("turn_seq")
        if self.store is not None and seq is not None:
            saved = self.store.annotate_turn(thread_id, seq, {"memory": report})
            if "error" in saved:
                report["annotate_error"] = saved["error"]
        return report

    def run(self, question: str, question_id: Optional[str] = None,
            run_id: Optional[str] = None, thread_id: Optional[str] = None,
            approval: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Answer `question`, calling tools as the model chooses.

        With `thread_id`, the thread's earlier turns are prepended (memory.history) and
        this run is appended to it afterwards, answered or failed. Without one, exactly
        the single-question behaviour: a fresh conversation, nothing read or written.
        `approval` (agent.fetch_proposal.approval_request) marks a thread run as the
        user's approved fetch: `question` is then its prompt-file text, and the turn is
        stored under approval["display"] with `approved_fetch` in its stats.

        Termination, without an iteration cap: each tool is dispatched at most its limit,
        so at most sum(limits) turns make progress, and `max_no_progress_turns`
        consecutive turns that dispatch nothing end the run: at most sum(limits)*(N+1)+N
        turns. With the shipped config (4 tools x 10, search_literature 5, N=3), 183.
        """
        question = (question or "").strip()
        if not question:
            return {"error": "empty_question", "detail": "a non-empty question is required",
                    "partial": None, "thread_id": thread_id, "turn_seq": None}

        config = self.config
        limits = self._limits()
        system_prompt = self._system_prompt(limits)
        history: Dict[str, Any] = {}
        if thread_id is not None:
            history = self._history(thread_id)
            if "error" in history:
                return {"error": history["error"], "detail": history.get("detail", ""),
                        "partial": None, "thread_id": thread_id, "turn_seq": None}
            if history["messages"]:
                # Only when there is history: a run outside a thread sends exactly the
                # system prompt it always did, so the eval's prompt is unchanged.
                system_prompt += "\n\n" + config.prompt("thread_history").strip()

        tracer = TraceWriter(run_id=run_id, question_id=question_id, config=config)
        self._started = time.perf_counter()
        self._steps = [] if thread_id is not None else None
        try:
            # Only inside a thread, and only after the steps list exists, so the `memory`
            # step is stored with the turn.
            injected = self._recall(question) if thread_id is not None else None
            result = self._run(question, tracer, limits, system_prompt,
                               history.get("messages") or [], injected or {})
            if injected is not None and "error" not in result:
                result["memory_injected"] = {
                    key: injected.get(key) for key in
                    ("memories", "preferences", "chunk_ids", "tokens", "error", "detail")
                    if injected.get(key) is not None}
            elif "error" not in result:
                result["memory_injected"] = None
            if thread_id is not None:
                result.update(self._save_turn(thread_id, question, result, tracer, history,
                                              approval=approval))
            else:
                result.update(thread_id=None, turn_seq=None)
            if approval is not None and "error" not in result:
                result["approved_fetch"] = {"query": approval.get("query")}
        finally:
            self._steps = None
        return result

    def _run(self, question: str, tracer: TraceWriter, limits: Dict[str, int],
             system_prompt: str, history: List[Dict[str, Any]],
             injected: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """One question's tool loop over a conversation that may open with history and a
        long-term memory block."""
        config = self.config
        counts: Dict[str, int] = {name: 0 for name in limits}
        max_no_progress = int(config.agent.max_no_progress_turns)
        injected = injected or {}
        conversation = Conversation(system_prompt, question, config, history=history,
                                    memory_block=str(injected.get("block") or ""),
                                    memory_chunk_ids=injected.get("chunk_ids") or [])

        # Chunk dedup follows "seen = still visible" (docs/DECISIONS.md D13): the seen set
        # starts as the chunks whose text this conversation already shows. History is
        # notes only, never chunk text; the memory block's passages are text, so they
        # start out seen, and elision releases them like any result (D19).
        if self.retriever is not None:
            visible = conversation.visible_chunk_ids()
            if visible:
                self.retriever.reset(visible)
            else:
                self.retriever.reset()
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
            # Elided text is out of view, so its chunks must be fetchable again (D13).
            release = getattr(self.retriever, "release", None)
            if conversation.last_released and callable(release):
                release(conversation.last_released)

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
