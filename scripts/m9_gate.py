"""
The M9 verification gate: are the deliverables actually there, and do they say anything?

In:  the repo.
Out: pass/fail per deliverable. Checks content, not just existence — a README that never
     mentions the eval, or a write-up that omits the ablation, is the failure mode here.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]

failures: List[str] = []


def check(label: str, ok: bool, detail: str = "", why: str = "") -> None:
    """`detail` explains a failure and prints only on FAIL; `why` is rationale, always shown."""
    note = why or (detail if not ok else "")
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {note}" if note else ""), flush=True)
    if not ok:
        failures.append(label)


def read(relative: str) -> str:
    path = ROOT / relative
    return path.read_text(encoding="utf-8") if path.exists() else ""


print("=== M9: deliverables ===\n")

# ---- the CLI ---------------------------------------------------------------------

main_py = read("main.py")
check("main.py exists at the repo root", bool(main_py),
      why="eval/ imports src/, so the CLI cannot live in src/")

completed = subprocess.run(
    [sys.executable, str(ROOT / "main.py"), "--help"],
    capture_output=True, text=True, cwd=ROOT,
)
help_text = completed.stdout
check("main.py --help runs", completed.returncode == 0, completed.stderr.strip()[:160])
for command in ("index", "ask", "eval"):
    check(f"CLI exposes `{command}`", command in help_text)

# ---- README ----------------------------------------------------------------------

readme = read("README.md")
check("README.md exists", bool(readme))
check("README documents setup", "pip install -r requirements.txt" in readme)
check("README documents the env vars", "OPENROUTER_API_KEY" in readme)
for command, phrase in (("indexing", "main.py index"), ("a question", "main.py ask"),
                        ("the eval", "main.py eval")):
    check(f"README documents how to run {command}", phrase in readme)

# ---- cold-start demo -------------------------------------------------------------

cold = read("examples/cold_start.md")
check("cold-start transcript exists", bool(cold),
      "run scripts/m9_cold_start.py")
check("cold-start transcript shows indexing from empty", "no papers, no chunks" in cold)
check("cold-start transcript shows the tool calls", "retrieve_evidence" in cold)
check("cold-start transcript is substantial", len(cold) > 2000, f"{len(cold)} chars")

# ---- worked examples -------------------------------------------------------------

examples = read("examples/worked_examples.md")
check("worked examples exist", bool(examples))
check("worked examples include an abstention case", "Abstention" in examples)
check("worked examples include a conflicting-evidence case", "Conflicting evidence" in examples)
check("worked examples show resolved citations",
      bool(re.search(r"citations \d+/\d+ resolved", examples)))

# ---- eval write-up ---------------------------------------------------------------

evaluation = read("docs/EVALUATION.md")
check("docs/EVALUATION.md exists", bool(evaluation),
      why="ARCHITECTURE section 13 has always pointed at this file")
check("write-up names the corpus that produced the numbers", "406 chunks" in evaluation)
check("write-up separates tuning from held-out", "held out" in evaluation.lower())
check("write-up reports the rerank ablation", "rerank" in evaluation.lower())
check("write-up does not bury the ablation's cost",
      "+0.027" in evaluation and "1527" in evaluation,
      why="the marginal gain and its latency cost must both be stated")
check("write-up states what the numbers do not say",
      "do not say" in evaluation.lower() or "limitation" in evaluation.lower())
check("results JSON backing the write-up is present",
      all((ROOT / "eval/results" / name).exists()
          for name in ("tuning_split.json", "held_out_split.json", "ablation.json",
                       "threshold_sweep.json")))

# ---- the architecture section ----------------------------------------------------

architecture = read("docs/ARCHITECTURE.md")
check("ARCHITECTURE has the 'how AI agents were used' section",
      "How AI agents were used" in architecture)
check("that section gives concrete examples, not narrative",
      architecture.count("**") > 20 and "256" in architecture,
      why="expected specific measurements, e.g. the bi-encoder's real token limit")
check("ARCHITECTURE's reference to EVALUATION.md now resolves",
      "docs/EVALUATION.md" in architecture and (ROOT / "docs/EVALUATION.md").exists())

# ---- standing rules --------------------------------------------------------------

# Derived from eval/topics.yaml rather than hardcoded, so a topic added later is
# covered without editing this gate. Matched on the first few words of each query:
# a whole query string never appears verbatim, but its distinctive opening would.
phrases = []
for line in read("eval/topics.yaml").splitlines():
    stripped = line.strip()
    if stripped.startswith("query:"):
        query = stripped.split(":", 1)[1].strip().strip('"\'')
        phrases.append(" ".join(query.lower().split()[:4]))
check("eval topics are readable", bool(phrases),
      "eval/topics.yaml has no query: lines",
      why=f"{len(phrases)} topic(s) checked for")

leaked = []
for path in (ROOT / "src").rglob("*"):
    if not path.is_file() or path.suffix not in {".py", ".md"}:
        continue
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    for phrase in phrases:
        if phrase in text:
            leaked.append(f"{path.relative_to(ROOT)}: {phrase}")
check("no eval topic name leaked into src/", not leaked, "; ".join(leaked))

print()
if failures:
    print(f"M9 GATE: {len(failures)} FAILURE(S) — " + "; ".join(failures))
    sys.exit(1)
print("M9 GATE: all checks passed")
