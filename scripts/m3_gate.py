"""
The M3 verification gate from docs/BUILD_PLAN.md, including a real process restart.

In:  a data/ tree with chunks from M2 and a .env key (only for fetching a 4th paper).
Out: pass/fail per check — lockstep persistence, faiss_id_map surviving a restart in a
     genuinely separate process, incremental append, and model-mismatch detection.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import collect  # noqa: E402
from chunk_store import ChunkStore  # noqa: E402
from config import CFG  # noqa: E402
from indexer import index_chunks  # noqa: E402
from ingest import ingest_paper  # noqa: E402
from manifest import Manifest  # noqa: E402
from vector_index import VectorIndex  # noqa: E402

QUERY = "how are experts selected for each token by the routing network"
NEW_TOPIC = "vision transformer patch embedding design"

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


RESTART_PROBE = """
import json, sys
sys.path.insert(0, {src!r})
from config import CFG
from indexer import search
hits = search({query!r}, k=5)
print("__RESULT__" + json.dumps([
    {{"chunk_id": h["chunk_id"], "paper_id": h["paper_id"], "score": round(h["score"], 4),
      "chunk_type": h["chunk_type"], "text": " ".join(h["text"].split())[:150]}}
    for h in hits
]))
"""


def restart_and_search() -> List[Dict[str, Any]]:
    """Load the index and search in a brand-new interpreter.

    Doing this in-process would prove nothing: the mapping is already in memory. The
    whole point is that faiss_id_map survived to disk, so it has to be read back cold.
    """
    code = RESTART_PROBE.format(src=str(ROOT / "src"), query=QUERY)
    completed = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), "-c", code],
        capture_output=True, text=True, cwd=str(ROOT),
        env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
    )
    for line in completed.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    raise RuntimeError(f"probe produced no result\nstdout={completed.stdout[-800:]}\n"
                       f"stderr={completed.stderr[-800:]}")


def main() -> int:
    store = ChunkStore()
    if store.count() == 0:
        print("No chunks. Run scripts/m2_gate.py first.")
        return 1

    step("indexing any chunks not yet embedded")
    built = index_chunks()
    if "error" in built:
        check("index builds", False, f"{built['error']}: {built['detail']}")
        return 1
    check("index builds", True, f"{built['total_indexed']} vectors")

    # 1. The three artefacts must agree exactly. FAISS ids are positional, so a mismatch
    #    means lookups silently resolve to the wrong chunk.
    manifest = Manifest.load(CFG)
    index = VectorIndex.load(manifest, CFG)
    stored = np.load(CFG.paths.embeddings)
    n = index.index.ntotal
    check("faiss.index, embeddings.npy and faiss_id_map agree",
          n == len(manifest.data["faiss_id_map"]) == stored.shape[0] == store.count(),
          f"index={n} map={len(manifest.data['faiss_id_map'])} npy={stored.shape[0]} chunks={store.count()}")
    check("embedding model and dim are recorded",
          manifest.data["embedding_model"] == CFG.embedding.model
          and manifest.data["embedding_dim"] == CFG.embedding.dim,
          f"{manifest.data['embedding_model']} dim={manifest.data['embedding_dim']}")

    # 2. Restart, cold-load, search. This is the faiss_id_map persistence test.
    step("restarting in a separate process and searching cold")
    try:
        hits = restart_and_search()
    except RuntimeError as exc:
        check("index loads and searches after a restart", False, str(exc)[:200])
        return 1

    check("index loads and searches after a restart", bool(hits), f"{len(hits)} hits")
    if hits:
        by_id = {c["chunk_id"]: c for c in store}
        resolved = [h for h in hits if h["chunk_id"] in by_id]
        check("returned chunk_ids resolve to real chunks", len(resolved) == len(hits),
              f"{len(resolved)}/{len(hits)}")
        matched = [h for h in hits if by_id[h["chunk_id"]]["text"].startswith(h["text"][:60])]
        check("resolved text matches what the index returned", len(matched) == len(hits),
              f"{len(matched)}/{len(hits)} — proves the map did not shift")
        descending = all(a["score"] >= b["score"] for a, b in zip(hits, hits[1:]))
        check("scores are ranked descending", descending)

        print(f"\n[{_elapsed()}] ---- top hits for {QUERY!r} ----", flush=True)
        for hit in hits:
            print(f"  {hit['score']:.4f}  {hit['chunk_id']}  [{hit['chunk_type']}]", flush=True)
            print(f"          {hit['text'][:120]}...", flush=True)
        print(flush=True)

    # 3. A new paper must add only its own vectors.
    before_ids = set(manifest.data["faiss_id_map"])
    step(f"fetching a 4th paper on a new topic to test incremental append")
    fetched = collect.search_and_fetch(NEW_TOPIC, topic_tag="gate_m3", max_results=1)
    if "error" in fetched or not fetched.get("papers_added"):
        detail = fetched.get("detail", "no new paper returned")
        print(f"[{_elapsed()}] [SKIP] incremental append — could not fetch a 4th paper: {detail}",
              flush=True)
    else:
        new_paper = fetched["papers_added"][0]["paper_id"]
        ingested = ingest_paper(new_paper, describe=False)
        if "error" in ingested:
            check("4th paper ingests", False, f"{ingested['error']}: {ingested['detail']}")
        else:
            check("4th paper ingests", True, f"{ingested['chunks_added']} new chunks")
            again = index_chunks()
            check("only the new paper was embedded",
                  again["chunks_indexed"] == ingested["chunks_added"]
                  and again["chunks_skipped"] == len(before_ids),
                  f"indexed={again['chunks_indexed']} skipped={again['chunks_skipped']} "
                  f"(existing={len(before_ids)})")
            after = Manifest.load(CFG)
            kept = list(after.data["faiss_id_map"])[: len(before_ids)]
            check("existing map entries kept their positions",
                  kept == list(manifest.data["faiss_id_map"]),
                  "appending must not reorder what is already indexed")

    # 4. A changed embedding model must force a rebuild, not an append.
    step("checking embedding-model mismatch detection")
    live = Manifest.load(CFG)
    original = live.data["embedding_model"]
    live.data["embedding_model"] = "some/other-model"
    live.save()
    try:
        from indexer import model_changed
        check("a changed embedding model is detected", model_changed(live, CFG),
              "appending vectors from a different model would corrupt retrieval silently")
    finally:
        live.data["embedding_model"] = original
        live.save()

    print(f"\n{'M3 GATE PASSED' if not failures else 'M3 GATE FAILED: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
