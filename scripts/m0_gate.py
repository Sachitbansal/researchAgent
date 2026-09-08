"""
The M0 verification gate from docs/BUILD_PLAN.md, run as a script.

In:  a populated .env (OPENROUTER_API_KEY) and config.yaml; nothing else.
Out: prints a pass/fail line per check — imports, directory scaffolding, one live text
     completion, and one live tool-calling round trip — and exits non-zero on failure.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402  (path bootstrap must run first)
import llm_client  # noqa: E402

from config import CFG, ConfigError  # noqa: E402
from llm_client import LLMClient, LLMError  # noqa: E402

PROBE_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "lookup_year",
        "description": "Look up the publication year of a paper by its exact title.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string", "description": "The paper title."}},
            "required": ["title"],
        },
    },
}

failures: List[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        failures.append(label)


def main() -> int:
    print(f"config:  {CFG.source_path}")
    print(f"models:  agent={CFG.llm.agent_model}  vision={CFG.llm.vision_model}  "
          f"utility={CFG.llm.utility_model}\n")

    check("import config, llm_client", bool(config.CFG and llm_client.LLMClient))

    CFG.paths.ensure()
    missing = [str(p) for p in CFG.paths.directories() if not p.is_dir()]
    check("directory scaffolding exists", not missing, ", ".join(missing))

    check("system prompt loads from file", "retrieve_evidence" in CFG.prompt("system"))

    try:
        CFG.require_api_key()
    except ConfigError as exc:
        check("OPENROUTER_API_KEY is set", False, str(exc))
        print("\nSkipping the two live checks: no API key.")
        return 1

    client = LLMClient()

    try:
        response = client.complete(
            [{"role": "user", "content": "Reply with exactly: gate ok"}], role="utility"
        )
        check("live text completion", bool(response.text.strip()), repr(response.text.strip()[:60]))
    except LLMError as exc:
        check("live text completion", False, f"{exc.code}: {exc.detail}")

    try:
        response = client.complete(
            [
                {
                    "role": "user",
                    "content": "What year was 'Attention Is All You Need' published? "
                    "Use the lookup_year tool.",
                }
            ],
            tools=[PROBE_TOOL],
            role="agent",
        )
        calls = response.tool_calls
        well_formed = (
            len(calls) == 1
            and calls[0].name == "lookup_year"
            and calls[0].ok
            and isinstance(calls[0].arguments.get("title"), str)
        )
        detail = (
            f"{calls[0].name}({calls[0].arguments})" if calls else f"no tool call; text={response.text[:60]!r}"
        )
        check("live tool-calling round trip", well_formed, detail)
        check(
            "assistant turn round-trips tool_call ids",
            bool(calls) and response.assistant_message().get("tool_calls"),
        )
    except LLMError as exc:
        check("live tool-calling round trip", False, f"{exc.code}: {exc.detail}")

    print(f"\n{'M0 GATE PASSED' if not failures else 'M0 GATE FAILED: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
