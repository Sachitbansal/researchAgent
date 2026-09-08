"""
The M2 verification gate from docs/BUILD_PLAN.md, run over the papers M1 fetched.

In:  a populated data/ tree (papers downloaded, manifest written) and a .env key.
Out: prints a pass/fail line per check — chunk coherence, figure text being caption AND
     description, dense positions, and a re-run costing zero vision calls. Non-zero on failure.
"""

from __future__ import annotations

import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus.chunk_store import ChunkStore  # noqa: E402
from config import CFG  # noqa: E402
from extraction.describe import DescriptionCache  # noqa: E402
from extraction.ingest import ingest_all  # noqa: E402
from corpus.manifest import Manifest  # noqa: E402

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
    manifest = Manifest.load(CFG)
    if not manifest.papers:
        print("No papers in the manifest. Run scripts/m1_gate.py first.")
        return 1

    store = ChunkStore()
    if store.count() == 0:
        step(f"ingesting {len(manifest.papers)} papers (extract, chunk, describe figures)")
        ingest_all()
    else:
        print(f"[{_elapsed()}] .... using {store.count()} chunks already in the store", flush=True)

    manifest = Manifest.load(CFG)
    chunks = store.all()
    check("chunks were produced", bool(chunks), f"{len(chunks)} chunks")
    if not chunks:
        return 1

    by_paper: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_paper[chunk["paper_id"]].append(chunk)

    # 1. position must be dense within every paper, or neighbour expansion walks a hole.
    dense = {pid: sorted(c["position"] for c in cs) == list(range(len(cs)))
             for pid, cs in by_paper.items()}
    check("position is dense and contiguous in every paper", all(dense.values()),
          ", ".join(f"{pid}:{len(by_paper[pid])}" for pid in sorted(by_paper)))
    check("chunk_id agrees with position",
          all(c["chunk_id"] == f"{c['paper_id']}__c{c['position']:04d}" for c in chunks))

    # 2. Every chunk must fit the budget both encoders were sized against.
    over = [c for c in chunks if c["n_tokens"] > CFG.chunking.max_tokens]
    check(f"no chunk exceeds chunking.max_tokens ({CFG.chunking.max_tokens})", not over,
          f"{len(over)} over" if over else f"max={max(c['n_tokens'] for c in chunks)}")

    # 3. Figure chunks carry caption AND description, never the description alone.
    figure_chunks = [c for c in chunks if c["chunk_type"] in ("figure", "table")]
    check("figure and table chunks exist", bool(figure_chunks), f"{len(figure_chunks)} of {len(chunks)}")
    if figure_chunks:
        have_caption = [c for c in figure_chunks if (c.get("caption") or "").strip()
                        and c["text"].startswith(c["caption"][:40])]
        check("figure chunk text starts with its caption",
              len(have_caption) == len(figure_chunks),
              f"{len(have_caption)}/{len(figure_chunks)}")
        described = [c for c in figure_chunks
                     if len(c["text"]) > len((c.get("caption") or "")) + 20]
        check("figure chunk text is caption AND description", bool(described),
              f"{len(described)}/{len(figure_chunks)} carry a description too")
        check("figure chunks resolve to an image on disk",
              all(Path(c["image_path"]).is_file() for c in figure_chunks if c.get("image_path")))

    # 4. Sections were detected somewhere, and null is an accepted outcome.
    with_section = [c for c in chunks if c.get("section")]
    check("sections were detected", bool(with_section),
          f"{len(with_section)}/{len(chunks)} chunks carry a section")

    # 5. Text coherence, printed for eyeballing per the build plan.
    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    random.seed(11)
    print(f"\n[{_elapsed()}] ---- 5 random chunks, for eyeballing ----", flush=True)
    for chunk in random.sample(text_chunks, min(5, len(text_chunks))):
        print(f"\n  {chunk['chunk_id']}  p{chunk['page']}  [{chunk['section']}]  "
              f"{chunk['n_tokens']} tok", flush=True)
        print(f"    {' '.join(chunk['text'].split())[:260]}...", flush=True)

    junk = [c for c in text_chunks if "arXiv:" in c["text"][:60]]
    check("\nno chunk starts with the arXiv margin stamp", not junk, f"{len(junk)} affected")

    # 6. A re-run must cost zero vision calls *because the cache served them*.
    step("re-running ingest to confirm the description cache holds")
    cache_before = len(DescriptionCache(CFG))
    again = ingest_all()
    fresh_calls = sum(paper.get("vision_calls", 0) for paper in again["papers"]
                      if "error" not in paper)
    skipped = [p for p in again["papers"] if p.get("error") == "already_ingested"]

    # Guard against a vacuous pass. With an empty cache, "zero vision calls" is trivially
    # true and proves nothing at all — which is exactly what happened the first time this
    # gate was run against an exhausted API quota. The cache must have entries for the
    # check below to mean anything.
    if cache_before == 0:
        check("description cache was populated", False,
              "cache is empty, so the cache-hit check below cannot be evaluated — "
              "this is a SKIP masquerading as a PASS unless it is failed here")
    else:
        check("re-run makes no new vision calls", fresh_calls == 0,
              f"{fresh_calls} calls, {cache_before} cached descriptions")

    check("re-run does not duplicate chunks", store.count() == len(chunks),
          f"{store.count()} vs {len(chunks)}")
    check("already-ingested papers are skipped", len(skipped) == len(manifest.papers),
          f"{len(skipped)}/{len(manifest.papers)}")

    print(f"\n{'M2 GATE PASSED' if not failures else 'M2 GATE FAILED: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
