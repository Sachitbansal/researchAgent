"""
The M1 verification gate from docs/BUILD_PLAN.md, run against live arXiv and the LLM.

In:  a populated .env and an empty (or existing) data/ tree.
Out: prints a pass/fail line per check — fetch 3, refetch with 0 downloads, a second
     overlapping topic that backfills tags onto chunk records, and a network failure
     that returns an error dict rather than raising. Exits non-zero on failure.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus import collect  # noqa: E402
from corpus.chunk_store import ChunkStore  # noqa: E402
from config import CFG  # noqa: E402
from corpus.manifest import Manifest  # noqa: E402
from common.records import chunk_record  # noqa: E402

TOPIC_A = "sparse mixture-of-experts routing in transformers"
# The backfill check re-runs TOPIC_A under a second tag rather than picking a different
# topic and hoping the two overlap. Overlap is the thing under test, so it has to be
# guaranteed, not left to whatever arXiv ranks that day.
TAG_A, TAG_A_ALIAS = "gate_a", "gate_a_alias"

failures: List[str] = []
_started = time.monotonic()


def _elapsed() -> str:
    return f"{time.monotonic() - _started:6.1f}s"


def step(message: str) -> None:
    """Announce a network-bound step before it starts, so a stall is visible."""
    print(f"[{_elapsed()}] .... {message}", flush=True)


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{_elapsed()}] [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        failures.append(label)


def main() -> int:
    CFG.paths.ensure()
    print(f"papers dir: {CFG.paths.papers}\n")

    # 1. Fetch three papers on a fresh topic.
    step("fetching 3 papers from arXiv (downloads PDFs, ~30-60s)")
    first = collect.search_and_fetch(TOPIC_A, topic_tag=TAG_A, max_results=3)
    if "error" in first:
        check("fetch 3 papers", False, f"{first['error']}: {first['detail']}")
        return 1
    added = first["papers_added"]
    detail = f"{len(added)} added, query={first['arxiv_query'][:60]!r}"
    if first.get("failures"):
        detail += f", failures={first['failures']}"
    check("fetch 3 papers", len(added) == 3, detail)
    check("query was LLM-planned, not the fallback", not first["query_was_fallback"])

    manifest = Manifest.load(CFG)
    fields = {"paper_id", "arxiv_id", "title", "authors", "abstract", "year", "published",
              "categories", "pdf_url", "pdf_path", "topic_tags", "n_chunks", "n_figures",
              "indexed_at", "parse_status", "parse_note"}
    records = [manifest.get_paper(p["paper_id"]) for p in added]
    check("paper records are well formed", all(r and set(r) == fields for r in records))
    check("topic_tags is a list on every record", all(isinstance(r["topic_tags"], list) for r in records))
    on_disk = [Path(r["pdf_path"]).is_file() for r in records]
    check("PDFs downloaded", all(on_disk), f"{sum(on_disk)}/{len(on_disk)}")

    # 2. Same topic again: nothing new should be downloaded.
    sizes_before = {r["paper_id"]: Path(r["pdf_path"]).stat().st_mtime for r in records}
    step("refetching the same topic (should download nothing)")
    second = collect.search_and_fetch(TOPIC_A, topic_tag=TAG_A, max_results=3)
    unchanged = all(Path(r["pdf_path"]).stat().st_mtime == sizes_before[r["paper_id"]] for r in records)
    check("refetch adds 0 papers", second.get("papers_added") == [], f"skipped={second.get('papers_skipped')}")
    check("refetch re-downloads nothing", unchanged)
    check("planned query came from cache", second.get("query_was_cached") is True)

    # 3. The same papers under a second tag must be tagged, not re-added, and the tag
    #    must reach their chunk records as well as the manifest.
    store = ChunkStore()
    # Pick the target from what is indexed under TAG_A *now*, not from the first fetch:
    # if planning ever changes between calls, the first fetch's papers may not be the
    # ones the next search re-encounters.
    indexed = Manifest.load(CFG).papers_for_topic(TAG_A)
    target = indexed[0] if indexed else added[0]["paper_id"]
    if not store.for_paper(target):
        # Chunking is M2; stand-in records let the backfill path be exercised now.
        store.append([
            chunk_record(paper_id=target, paper_title="stand-in", topic_tags=[TAG_A],
                         chunk_type="text", text=f"stand-in chunk {i}", page=1, position=i, n_tokens=4)
            for i in range(3)
        ])

    step("same topic under a second tag (should tag, not add)")
    third = collect.search_and_fetch(TOPIC_A, topic_tag=TAG_A_ALIAS, max_results=3)
    if "error" in third:
        check("same topic under a second tag", False, f"{third['error']}: {third['detail']}")
    else:
        check("same papers are tagged, not re-added", third["papers_added"] == [],
              f"added={len(third['papers_added'])}, skipped={third['papers_skipped']}, "
              f"tagged={third['papers_tagged']}")
        reloaded = Manifest.load(CFG)
        both_tags = {TAG_A, TAG_A_ALIAS}
        overlapped = [p for p in reloaded.papers.values() if both_tags <= set(p["topic_tags"])]
        check("papers carry both tags in the manifest", len(overlapped) == len(added),
              f"{len(overlapped)}/{len(added)}")
        chunks = store.for_paper(target)
        check("their chunk records were backfilled too",
              bool(chunks) and all(both_tags <= set(c["topic_tags"]) for c in chunks),
              f"{len(chunks)} chunks for {target}")
        check("no phantom ids in topics.paper_ids",
              all(reloaded.get_paper(pid) is not None
                  for tag in (TAG_A, TAG_A_ALIAS) for pid in reloaded.papers_for_topic(tag)))

    # 4. A dead network must return an error dict, not raise.
    class DeadClient:
        def results(self, search):
            raise ConnectionError("network is unreachable")

    original = collect._build_client
    collect._build_client = lambda config: DeadClient()
    try:
        step("simulating a dead network")
        dead = collect.search_and_fetch("anything at all", topic_tag="gate_dead")
    except Exception as exc:  # noqa: BLE001 — the point of the check
        check("network failure returns an error dict", False, f"it raised {type(exc).__name__}: {exc}")
    else:
        check("network failure returns an error dict", dead.get("error") == "arxiv_unavailable", str(dead.get("error")))
    finally:
        collect._build_client = original

    print(f"\n{'M1 GATE PASSED' if not failures else 'M1 GATE FAILED: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
