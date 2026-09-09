"""
Fills gold_chunk_ids by matching hand-written keywords against chunk text.

In:  eval/questions.json with gold_paper_ids, and the indexed chunk store.
Out: the same file with gold_chunk_ids filled, plus a printed report to eyeball.

Deliberately does NOT call retrieve_evidence. Labelling with the same component the
eval scores would make recall@k trivially 1.0 and measure nothing; plain keyword
matching over the gold papers is an independent mechanism.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus.chunk_store import ChunkStore  # noqa: E402

# Terms a chunk must contain to count as gold for that question. Written by hand from the
# abstracts, using the papers' own vocabulary — the questions themselves are paraphrased,
# which is what keeps retrieval from succeeding by string matching.
# Terms a chunk must contain to count as gold. Conjunctive on purpose: single common
# terms ("calibration", "retraining") match a paper's front matter and reference list as
# readily as its claims, and a gold set containing the title page makes recall@k
# meaningless — retrieving an affiliation block would score as a hit.
KEYWORDS: Dict[str, List[List[str]]] = {
    "sp01": [["sparsemixer", "gradient"], ["backpropagation", "sparse"],
             ["gradient", "estimator"], ["st estimator"]],
    "sp02": [["milora", "expert"], ["multi-tenant", "latency"], ["prompt-aware", "rout"],
             ["lora", "router"]],
    "sp03": [["qpruner", "quantiz"], ["quantization", "pruning", "memory"],
             ["fine-tuning", "memory", "prun"]],
    "sp04": [["adapruner", "prun"], ["calibration", "random"], ["calibration", "sample"],
             ["importance", "estimation", "prun"]],
    "sp05": [["routing collapse", "entropy"], ["entropy", "final layers"],
             ["hebrew", "rout"], ["deep-layer", "collapse"]],
    "mp01": [["trsp", "regulariz"], ["retraining", "prun"], ["knowledge loss", "prun"],
             ["calibration", "importance"], ["quantization", "memory", "prun"]],
    "mp02": [["context incompleteness"], ["unstable", "rout"], ["inconsistent", "rout"],
             ["entropy", "collapse"], ["specialis", "rout"], ["specializ", "rout"]],
    "mp03": [["subset of experts", "rout"], ["structured pruning", "remov"],
             ["redundant", "parameters", "remov"], ["conditional", "comput"]],
    "mp04": [["edge", "memory"], ["speculative decoding"], ["binariz", "prun"],
             ["binarization", "quantization"]],
    "cf01": [["necessitat", "fine-tun"], ["retraining", "avoid"], ["extensive retraining"],
             ["knowledge loss", "retrain"], ["accuracy degradation", "prun"]],
    "cf02": [["unstable", "inconsistent"], ["semantically inconsistent"],
             ["linguistically structured"], ["mutual information", "rout"],
             ["specialisation", "categor"]],
    "cf03": [["routing collapse"], ["context fusion", "rout"], ["bottleneck", "specializ"],
             ["bottleneck", "specialis"]],
}

# Chunks that match a keyword but cannot answer anything: title pages, affiliations,
# copyright blocks, bare author lists.
NOISE = re.compile(
    r"(copyright ©|all rights reserved|@[a-z0-9.-]+\.(edu|com|org|cn)|"
    r"school of |department of |university\b.{0,40}\b(china|usa|uk)\b)",
    re.IGNORECASE,
)


def is_noise(text: str) -> bool:
    """True for front matter — matched a term, but carries no claim to retrieve."""
    head = " ".join(text.split())[:400]
    return bool(NOISE.search(head))


def matches(text: str, groups: List[List[str]]) -> bool:
    lowered = text.lower()
    return any(all(term in lowered for term in group) for group in groups)


def main() -> int:
    path = ROOT / "eval" / "questions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = ChunkStore().all()

    for question in payload["questions"]:
        qid = question["question_id"]
        if question["expect_abstention"]:
            question["gold_chunk_ids"] = []
            print(f"{qid}  abstention — no gold chunks by design", flush=True)
            continue

        groups = KEYWORDS.get(qid, [])
        papers = set(question["gold_paper_ids"])
        hits = [c for c in chunks
                if c["paper_id"] in papers
                and matches(c["text"], groups)
                and not is_noise(c["text"])]
        # Prefer text chunks; a figure caption rarely carries the claim being scored.
        hits.sort(key=lambda c: (c["chunk_type"] != "text", c["position"]))
        question["gold_chunk_ids"] = [c["chunk_id"] for c in hits[:6]]

        print(f"\n{qid}  {len(question['gold_chunk_ids'])} gold chunks "
              f"from {sorted(papers)}", flush=True)
        for chunk in hits[:3]:
            print(f"    {chunk['chunk_id']} [{chunk['chunk_type']}] "
                  f"{' '.join(chunk['text'].split())[:110]}", flush=True)

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    unlabelled = [q["question_id"] for q in payload["questions"]
                  if not q["gold_chunk_ids"] and not q["expect_abstention"]]
    print(f"\nunlabelled non-abstention questions: {unlabelled or 'none'}", flush=True)
    return 1 if unlabelled else 0


if __name__ == "__main__":
    raise SystemExit(main())
