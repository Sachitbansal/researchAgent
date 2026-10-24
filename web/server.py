"""
Entry point for the browser UI: `python web/server.py [--host H] [--port N]`.

In:  command-line flags; SCIAGENT_CONFIG and .env as the CLI reads them. Out: one uvicorn
     worker serving web/api.py, with the agent, its models and the index loaded once at
     startup. Lives outside src/ and imports it, as eval/ does — nothing in src/ knows
     this exists, and the system under evaluation is still the CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Tuple

ROOT = Path(__file__).resolve().parent.parent
# src/ for the agent's own flat imports (`from config import CFG`), the repo root for
# `web.*`. Run as a script, Python also puts web/ itself first on the path, which is why
# no module in web/ may share a name with a top-level package in src/.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def build_agent() -> Tuple[Any, Any]:
    """Load the models and the index once, the same way `main.py ask` builds its agent."""
    from agent.loop import AgentLoop
    from agent.tool_registry import build_full_registry
    from config import CFG
    from retrieval.embedder import Embedder
    from tools.retrieve_evidence import EvidenceRetriever

    print("loading models and the index ...", flush=True)
    CFG.paths.ensure()
    retriever = EvidenceRetriever(CFG, embedder=Embedder(CFG))
    loop = AgentLoop(build_full_registry(retriever=retriever, config=CFG), config=CFG, retriever=retriever)

    from web.corpus_view import read_corpus

    print(f"ready — {len(read_corpus(CFG)['papers'])} paper(s) indexed", flush=True)
    return loop, CFG


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Load the models, then answer requests until interrupted."""
    import uvicorn

    from web.api import DEFAULT_DIST, create_app

    if not (DEFAULT_DIST / "index.html").is_file():
        print("note: frontend not built — the API will run, but / shows build instructions.\n"
              "      run: cd frontend && npm install && npm run build", flush=True)

    def builder() -> Tuple[Any, Any]:
        built = build_agent()
        # Printed only once the models are in memory: before that, the page would load
        # but the first question would wait on startup.
        print(f"open http://{host}:{port}", flush=True)
        return built

    # One worker and no reload, deliberately: the models live in this process, and a
    # second worker would load them twice and hold a second, unsynchronised agent.
    uvicorn.run(create_app(builder), host=host, port=port, workers=1, reload=False,
                log_level="warning")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Browser UI for the research assistant.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.host, args.port)
