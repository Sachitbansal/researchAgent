#!/usr/bin/env python3
"""
One live end-to-end check against arXiv: search, download one PDF, verify it is a PDF.

In:  nothing (optional --topic). Out: printed timings and exit code 0 on success.
Deliberately fetches a single paper — the point is to prove the path works without
spending requests on a host that may still be rate-limiting this IP.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import CFG  # noqa: E402
from corpus import collect  # noqa: E402
from corpus.arxiv_fetch import read_cooldown  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="graph neural networks for molecules")
    parser.add_argument("--clear-cooldown", action="store_true")
    args = parser.parse_args()

    CFG.paths.ensure()
    if args.clear_cooldown:
        CFG.paths.arxiv_cooldown.unlink(missing_ok=True)

    remaining = read_cooldown(CFG.paths.arxiv_cooldown)
    if remaining > 0:
        print(f"cooldown active: {remaining:.0f}s left. Re-run later, "
              f"or pass --clear-cooldown to try anyway.")
        return 2

    started = time.monotonic()
    result = collect.search_and_fetch(args.topic, topic_tag="_arxiv_check",
                                      max_results=1, config=CFG)
    elapsed = time.monotonic() - started

    if "error" in result:
        print(f"FAIL  {result['error']}: {result['detail']}  ({elapsed:.1f}s)")
        return 1

    added = result.get("papers_added", [])
    if not added and result.get("papers_skipped"):
        print(f"already held that paper; nothing downloaded. Search reached arXiv fine "
              f"({elapsed:.1f}s). Delete data/papers to force a real download.")
        return 0
    if not added:
        print(f"FAIL  no papers and no skips: {result}")
        return 1

    pdf = CFG.paths.paper_pdf(added[0]["paper_id"])
    magic = pdf.read_bytes()[:5] if pdf.is_file() else b""
    ok = magic.startswith(b"%PDF")
    print(f"{'OK  ' if ok else 'FAIL'}  {added[0]['paper_id']}  "
          f"{pdf.stat().st_size if pdf.is_file() else 0:,d} B  magic={magic!r}  "
          f"({elapsed:.1f}s for 1 search + 1 download)")
    print(f"      query: {result['arxiv_query']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
