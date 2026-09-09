"""
The M7 verification gate: the eval corpus and the question set are sound.

In:  data/ built by eval/build_corpus.py and eval/questions.json annotated.
Out: pass/fail per requirement — gold labels present, all four categories covered, and
     no question copying paper wording, which is checked by n-gram overlap not by eye.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus.chunk_store import ChunkStore  # noqa: E402
from corpus.manifest import Manifest  # noqa: E402

CATEGORIES = {"single_paper", "multi_paper", "not_in_corpus", "conflicting"}
NGRAM = 5

failures: List[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        failures.append(label)


def ngrams(text: str, n: int = NGRAM) -> Set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def main() -> int:
    payload = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    questions = payload["questions"]
    manifest = Manifest.load(ROOT and __import__("config").CFG, strict_model_check=False)
    chunks = ChunkStore().all()

    print(f"corpus: {len(manifest.papers)} papers, {len(chunks)} chunks, "
          f"{len(manifest.figures)} figures\n", flush=True)

    # 1. Two topics, one index.
    tags = sorted({tag for paper in manifest.papers.values() for tag in paper["topic_tags"]})
    check("two contrasting topics are indexed", len(tags) == 2, ", ".join(t[:38] for t in tags))
    check("both topics have papers",
          all(sum(1 for p in manifest.papers.values() if tag in p["topic_tags"]) >= 4
              for tag in tags),
          ", ".join(f"{tag[:26]}={sum(1 for p in manifest.papers.values() if tag in p['topic_tags'])}"
                    for tag in tags))
    check("both topics share one index",
          len({c["chunk_id"] for c in chunks}) == len(chunks) and len(chunks) > 0,
          f"{len(chunks)} chunks in a single store")

    # 2. Question set shape.
    counts = Counter(q["category"] for q in questions)
    check("12-15 questions", 12 <= len(questions) <= 15, str(len(questions)))
    check("all four categories are represented", set(counts) == CATEGORIES, dict(counts))
    check("at least two not_in_corpus", counts["not_in_corpus"] >= 2, str(counts["not_in_corpus"]))
    check("at least two conflicting", counts["conflicting"] >= 2, str(counts["conflicting"]))
    held = sum(1 for q in questions if q["held_out"])
    check("roughly a third is held out", 0.2 <= held / len(questions) <= 0.45,
          f"{held}/{len(questions)}")
    check("held-out set covers more than one category",
          len({q["category"] for q in questions if q["held_out"]}) >= 2,
          str(sorted({q["category"] for q in questions if q["held_out"]})))

    # 3. Gold labels.
    known_papers = set(manifest.papers)
    known_chunks = {c["chunk_id"] for c in chunks}
    missing_gold = [q["question_id"] for q in questions
                    if not q["expect_abstention"] and not q["gold_chunk_ids"]]
    check("every answerable question has gold chunks", not missing_gold, str(missing_gold))
    check("abstention questions have no gold chunks",
          all(not q["gold_chunk_ids"] for q in questions if q["expect_abstention"]))
    bad_papers = [(q["question_id"], p) for q in questions
                  for p in q["gold_paper_ids"] if p not in known_papers]
    check("every gold paper_id resolves", not bad_papers, str(bad_papers[:4]))
    bad_chunks = [(q["question_id"], c) for q in questions
                  for c in q["gold_chunk_ids"] if c not in known_chunks]
    check("every gold chunk_id resolves", not bad_chunks, str(bad_chunks[:4]))
    mismatched = [q["question_id"] for q in questions
                  for c in q["gold_chunk_ids"]
                  if c.rsplit("__c", 1)[0] not in set(q["gold_paper_ids"])]
    check("gold chunks belong to their gold papers", not mismatched, str(sorted(set(mismatched))))
    check("every question has expected_facts or expects abstention",
          all(q["expected_facts"] or q["expect_abstention"] for q in questions))

    # 4. No question copies paper wording. Checked, not asserted.
    corpus_ngrams: Set[str] = set()
    for chunk in chunks:
        corpus_ngrams |= ngrams(chunk["text"])
    copied = []
    for question in questions:
        overlap = ngrams(question["question"]) & corpus_ngrams
        if overlap:
            copied.append((question["question_id"], sorted(overlap)[:2]))
    check(f"no question copies a {NGRAM}-gram from the corpus", not copied, str(copied[:3]))

    # 5. Abstention questions must genuinely be absent, not merely assumed absent.
    #
    # Checked on declared subject markers, not on every word. An earlier version flagged
    # any shared vocabulary, which fails exactly the questions that are supposed to share
    # it: ni02 deliberately reuses "quantization", "accuracy" and "transformers" so that
    # the relevance gate has to react to meaning rather than to surface overlap. What
    # must be absent is the subject, so the subject is what gets named and tested.
    corpus_text = " ".join(c["text"].lower() for c in chunks)
    for question in questions:
        if not question["expect_abstention"]:
            continue
        markers = question.get("absent_markers") or []
        check(f"{question['question_id']} declares what makes it absent", bool(markers))
        present = [m for m in markers if m.lower() in corpus_text]
        check(f"{question['question_id']} subject really is absent from the corpus",
              not present, f"found in corpus: {present}" if present else f"markers: {markers}")

    # A shared-vocabulary abstention case is what makes the gate meaningful, so confirm at
    # least one exists rather than only trivially absurd questions.
    overlapping = [q["question_id"] for q in questions
                   if q["expect_abstention"]
                   and len(ngrams(q["question"], 1) & {w for c in chunks
                                                       for w in ngrams(c["text"], 1)}) >= 6]
    check("at least one abstention question shares vocabulary with the corpus",
          bool(overlapping), f"hard cases: {overlapping}")

    print(f"\n{'M7 GATE PASSED' if not failures else 'M7 GATE FAILED: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
