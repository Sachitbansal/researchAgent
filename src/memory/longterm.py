"""
Long-term research memory (docs/DECISIONS.md D19, D20): grounded findings across chats.

In:  a MemoryStore, the bi-encoder, the NLI model, an LLM client (role `utility`) and the
     chunk table. recall() takes the question at a thread run's start; after_run() takes
     the finished run (question, answer, trace records, what recall injected).
Out: recall -> the memory block to inject plus the memories and chunks behind it;
     after_run -> {written, refreshed, invalidated, used, forgotten, dropped}. Not a tool,
     calls no tool, and never raises: failures come back as {"error", "detail"}.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from common.tokenization import TokenCounter, split_sentences
from config import CFG, Config
from memory import forgetting

log = logging.getLogger("sciagent.memory")

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_BRACKETS = re.compile(r"\[([^\[\]]+)\]")
_CHUNK_ID = re.compile(r"[A-Za-z0-9._+-]+__c\d+")
# On the fallback path only: \f, \b, \t, \n, \r are LaTeX (\frac, \beta, \theta) far more
# often than control characters in a findings list, so they are doubled too.
_ESCAPE = re.compile(r"\\(u[0-9a-fA-F]{4}|.)", re.DOTALL)


def _repair_escapes(text: str) -> str:
    r"""Keep \\, \", \/ and \uXXXX; double every other backslash. Pairs are consumed
    left to right, so an escaped backslash is never split."""
    return _ESCAPE.sub(lambda m: m.group(0) if m.group(1) in ('"', "\\", "/")
                       or len(m.group(1)) == 5 else "\\\\" + m.group(1), text)


def _error(code: str, detail: str) -> Dict[str, Any]:
    return {"error": code, "detail": detail}


def cited_chunk_ids(answer: str) -> List[str]:
    """chunk_ids the answer cites in square brackets, in first-cited order.

    Accepts `[a__c1][b__c2]` and `[a__c1, b__c2]` alike; only text inside brackets counts,
    so a chunk id mentioned in prose is not a citation.
    """
    out: List[str] = []
    for group in _BRACKETS.findall(answer or ""):
        for chunk_id in _CHUNK_ID.findall(group):
            if chunk_id not in out:
                out.append(chunk_id)
    return out


def gate_passed(trace_records: Sequence[Dict[str, Any]]) -> bool:
    """True when some retrieve_evidence call cleared the relevance gate with chunks.

    The write trigger's abstention test (D19): an answer written without any passing
    retrieval is an abstention, or built on nothing, and nothing from it is remembered.
    """
    for record in trace_records or []:
        if not isinstance(record, dict) or record.get("tool_name") != "retrieve_evidence":
            continue
        summary = record.get("result_summary") or {}
        if record.get("error") or summary.get("sufficient_evidence") is not True:
            continue
        if record.get("chunk_ids_returned"):
            return True
    return False


def _sentences(text: str, cap: int) -> List[str]:
    """Same premise sentences as the groundedness check: 4+ words, capped per chunk."""
    return [s for s in split_sentences(text) if len(s.split()) >= 4][:cap]


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(0, width - 1)].rstrip() + "…"


class LongTermMemory:
    """Reads memories into a thread run and writes grounded findings out of it."""

    def __init__(
        self,
        store: Any,
        config: Config = CFG,
        embedder: Any = None,
        nli: Any = None,
        client: Any = None,
        chunk_table: Optional[Callable[[], Dict[str, Dict[str, Any]]]] = None,
        counter: Any = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.config = config
        self.memory = config.memory
        self._embedder = embedder
        self._nli = nli
        self._client = client
        self._chunk_table = chunk_table
        self._counter = counter
        self.now = now

    # ---- lazily built collaborators --------------------------------------------

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from retrieval.embedder import Embedder
            self._embedder = Embedder(self.config)
        return self._embedder

    @property
    def nli(self) -> Any:
        if self._nli is None:
            from analysis.nli import NLIModel
            self._nli = NLIModel(self.config)
        return self._nli

    @property
    def client(self) -> Any:
        if self._client is None:
            from llm_client import LLMClient
            self._client = LLMClient(config=self.config)
        return self._client

    @property
    def counter(self) -> Any:
        if self._counter is None:
            self._counter = TokenCounter(self.config)
        return self._counter

    def chunks(self) -> Dict[str, Dict[str, Any]]:
        """chunk_id -> record, read fresh: a re-index between runs must be seen."""
        if self._chunk_table is not None:
            return self._chunk_table()
        from corpus.chunk_store import ChunkStore
        return {c["chunk_id"]: c for c in ChunkStore(config=self.config)}

    # ================================================================ reading

    def recall(self, question: str) -> Dict[str, Any]:
        """The memories to inject at the start of a thread run, as one block of text.

        The question is embedded WITH the BGE query instruction (it is a query); memories
        were embedded without it (they are passages). Top `inject_k` findings above
        `inject_min_cosine`, each re-validated against the current chunk hashes, so a
        stale memory is invalidated here rather than shown. Active preferences are always
        included: they are few, and a cosine gate would rarely surface "only 2024 onward"
        for a question about something else. Everything fits `inject_tokens`.
        """
        try:
            return self._recall(question)
        except Exception as exc:  # noqa: BLE001 - recall must never end a run
            log.warning("memory recall failed: %s: %s", type(exc).__name__, exc)
            return _error("memory_recall_failed", f"{type(exc).__name__}: {exc}")

    def _recall(self, question: str) -> Dict[str, Any]:
        active = self.store.active()
        if "error" in active:
            return active
        rows = active["memories"]
        preferences = [m for m in rows if m["kind"] == "preference"]
        preferences.sort(key=lambda m: m["last_used_at"], reverse=True)
        preferences = preferences[: int(self.memory.max_preferences)]
        findings = [m for m in rows if m["kind"] == "finding" and m.get("embedding") is not None]

        empty = {"memories": [], "preferences": [], "chunks": [], "chunk_ids": [], "block": "",
                 "tokens": 0, "stale": []}
        ranked: List[Tuple[float, Dict[str, Any]]] = []
        if findings:
            query = np.asarray(self.embedder.encode_query(question), dtype=np.float32)
            matrix = np.vstack([m["embedding"] for m in findings])
            scores = matrix @ query
            floor = float(self.memory.inject_min_cosine)
            ranked = sorted(((float(s), m) for s, m in zip(scores, findings) if s >= floor),
                            key=lambda pair: pair[0], reverse=True)
        if not ranked and not preferences:
            return empty

        table = self.chunks() if ranked else {}
        current = {cid: str(c.get("content_hash")) for cid, c in table.items()}
        chosen: List[Tuple[float, Dict[str, Any]]] = []
        stale: List[str] = []
        for score, memory in ranked:
            if len(chosen) >= int(self.memory.inject_k):
                break
            if forgetting.is_stale(memory, current):
                stale.append(memory["id"])
                continue
            chosen.append((score, memory))
        if stale:
            result = self.store.invalidate(stale, "stale")
            if "error" in result:
                log.warning("could not invalidate stale memories: %s", result)

        block, picked, passages, tokens = self._render(chosen, preferences, table)
        if not block:
            return {**empty, "stale": stale}
        return {
            "memories": [{"id": m["id"], "label": label, "text": m["text"],
                          "chunk_ids": list(m["chunk_ids"]), "score": round(score, 4),
                          "use_count": m["use_count"]}
                         for label, score, m in picked],
            "preferences": [{"id": m["id"], "text": m["text"]} for m in preferences],
            "chunks": passages,
            "chunk_ids": [p["chunk_id"] for p in passages],
            "block": block,
            "tokens": tokens,
            "stale": stale,
        }

    def _render(self, chosen: Sequence[Tuple[float, Dict[str, Any]]],
                preferences: Sequence[Dict[str, Any]], table: Dict[str, Dict[str, Any]],
                ) -> Tuple[str, List[Tuple[str, float, Dict[str, Any]]], List[Dict[str, Any]], int]:
        """Fill memory_block.md, adding findings best-first while the block fits the budget.

        A finding goes in whole (its text and every passage it rests on) or not at all: a
        finding without its passages cannot be cited, which is the point of attaching them.
        """
        template = self.config.prompt("memory_block")
        budget = int(self.memory.inject_tokens)
        width = int(self.memory.passage_chars)
        prefs = "\n".join(f"- {m['text']}" for m in preferences) or "(none)"

        def fill(lines: List[str], blocks: List[str]) -> str:
            return (template.replace("{preferences}", prefs)
                    .replace("{findings}", "\n".join(lines) or "(none)")
                    .replace("{passages}", "\n\n".join(blocks) or "(none)")).strip()

        lines: List[str] = []
        blocks: List[str] = []
        passages: List[Dict[str, Any]] = []
        picked: List[Tuple[str, float, Dict[str, Any]]] = []
        for score, memory in chosen:
            label = f"m{len(picked) + 1}"
            ids = [c for c in memory["chunk_ids"] if c in table]
            new_blocks, new_passages = [], []
            for chunk_id in ids:
                if any(p["chunk_id"] == chunk_id for p in passages + new_passages):
                    continue
                chunk = table[chunk_id]
                text = _clip(str(chunk.get("text") or ""), width)
                new_passages.append({
                    "chunk_id": chunk_id, "paper_id": chunk.get("paper_id"),
                    "paper_title": chunk.get("paper_title"), "page": chunk.get("page"),
                    "section": chunk.get("section"), "chunk_type": chunk.get("chunk_type"),
                    "text": text,
                })
                new_blocks.append(f"[{chunk_id}] {chunk.get('paper_title') or ''}, "
                                  f"p. {chunk.get('page')}:\n{text}")
            cites = "".join(f"[{c}]" for c in ids)
            line = f"- {label}: {memory['text']} (rests on {cites})"
            candidate = fill(lines + [line], blocks + new_blocks)
            if self.counter.count(candidate) > budget:
                continue
            lines.append(line)
            blocks.extend(new_blocks)
            passages.extend(new_passages)
            picked.append((label, score, memory))

        if not picked and not preferences:
            return "", [], [], 0
        block = fill(lines, blocks)
        tokens = self.counter.count(block)
        if tokens > budget:  # preferences alone over budget: show none rather than overrun
            return "", [], [], 0
        return block, picked, passages, tokens

    # ================================================================ writing

    def after_run(self, question: str, answer: str, thread_id: Optional[str],
                  trace_records: Sequence[Dict[str, Any]],
                  injected: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Everything that follows an answered thread run. Never raises.

        1. Injected memories whose chunks the answer cites are marked used.
        2. If the run passed the relevance gate: extract findings and preferences (one
           utility call), keep only NLI-entailed findings, reconcile each against the
           active memories (duplicate -> refresh, contradiction -> supersede, else add).
        3. Sweep: stale, decay, cap, purge.
        """
        report: Dict[str, Any] = {"written": [], "refreshed": [], "invalidated": [],
                                  "used": [], "forgotten": [], "dropped": {}}
        if thread_id is None:
            return {**report, "skipped": "no_thread"}
        if not (answer or "").strip():
            return {**report, "skipped": "empty_answer"}
        try:
            self._mark_used(answer, injected or {}, report)
            if gate_passed(trace_records):
                self._write(question, answer, thread_id, trace_records, injected or {}, report)
            elif any(isinstance(r, dict) and r.get("tool_name") == "retrieve_evidence"
                     for r in trace_records or []):
                report["skipped"] = "abstained"
            else:
                # Answered from injected memory (or no evidence at all) without a fresh
                # retrieval: nothing new was found, so nothing new is written.
                report["skipped"] = "no_new_retrieval"
        except Exception as exc:  # noqa: BLE001 - a writer failure must never fail a run
            log.warning("memory write failed: %s: %s", type(exc).__name__, exc)
            report["error"] = "memory_write_failed"
            report["detail"] = f"{type(exc).__name__}: {exc}"
        try:
            # Its own guard: a failed extraction must not also stop old memories decaying.
            report["forgotten"].extend(self.sweep())
        except Exception as exc:  # noqa: BLE001
            log.warning("memory sweep failed: %s: %s", type(exc).__name__, exc)
            report.setdefault("error", "memory_sweep_failed")
            report.setdefault("detail", f"{type(exc).__name__}: {exc}")
        return report

    def _mark_used(self, answer: str, injected: Dict[str, Any], report: Dict[str, Any]) -> None:
        """Injected AND cited is a use. Injected but uncited is not (D20)."""
        cited = set(cited_chunk_ids(answer))
        used = [m for m in injected.get("memories") or []
                if cited.intersection(m.get("chunk_ids") or [])]
        if not used:
            return
        result = self.store.mark_used([m["id"] for m in used])
        if "error" in result:
            raise RuntimeError(f"mark_used: {result['error']}: {result.get('detail')}")
        done = set(result["used"])
        report["used"] = [{"id": m["id"], "text": m["text"]} for m in used if m["id"] in done]

    def extract(self, question: str, answer: str) -> Dict[str, Any]:
        """{"findings": [{"text", "check", "chunk_ids"}], "preferences": [str]} or an error.

        `text` is the finding as remembered: it names its paper or method, so two papers'
        findings neither merge nor read as a contradiction. `check` is the same assertion
        as the passage states it, without the attribution, and is what NLI grounds:
        measured, naming the paper in the hypothesis drops this model's entailment for
        a supported claim from 0.99 to 0.001. The attribution itself is checked by the
        citation: the finding may only rest on chunks the answer cited for it.
        """
        from llm_client import LLMError

        messages = [
            {"role": "system", "content": self.config.prompt("memory_extract")},
            {"role": "user", "content": f"Question:\n{question}\n\nAnswer:\n{answer}"},
        ]
        try:
            response = self.client.complete(
                messages, role="utility",
                timeout=float(self.config.llm.planning_timeout_s),
                max_retries=int(self.config.llm.planning_retries),
            )
        except LLMError as exc:
            return _error("extraction_failed", f"{exc.code}: {exc.detail}")
        match = _JSON_OBJECT.search(getattr(response, "text", "") or "")
        if not match:
            return _error("extraction_failed", "the reply contained no JSON object")
        try:
            payload = json.loads(match.group(0))
        except ValueError:
            # Answers carry LaTeX, and a model copying "\sum" into JSON leaves an invalid
            # escape. Double every backslash that is not an escaped quote, backslash,
            # slash or \uXXXX.
            try:
                payload = json.loads(_repair_escapes(match.group(0)))
            except ValueError as exc:
                return _error("extraction_failed", f"the reply was not valid JSON: {exc}")
        if not isinstance(payload, dict):
            return _error("extraction_failed", "the reply was not a JSON object")

        findings = []
        for item in payload.get("findings") or []:
            if not isinstance(item, dict):
                continue
            text = " ".join(str(item.get("text") or "").split())
            check = " ".join(str(item.get("check") or "").split()) or text
            ids = item.get("chunk_ids") or []
            if isinstance(ids, str):  # a bare string would iterate character by character
                ids = [ids]
            ids = [str(c).strip().strip("[]") for c in ids if str(c).strip()]
            if text:
                findings.append({"text": text, "check": check,
                                 "chunk_ids": list(dict.fromkeys(ids))})
        preferences = [" ".join(str(p).split()) for p in payload.get("preferences") or []
                       if isinstance(p, str) and p.strip()]
        return {"findings": findings[: int(self.memory.max_findings)],
                "preferences": preferences[: int(self.memory.max_preferences)]}

    def ground(self, findings: Sequence[Dict[str, Any]], table: Dict[str, Dict[str, Any]],
               min_similarity: Optional[float] = None) -> List[Dict[str, Any]]:
        """Each finding labelled against its own cited chunks, the groundedness way.

        Mirrors check_evidence_consistency's groundedness mode (a tool, so not called
        from here): the `premises_per_claim` sentences of its chunks most similar to the
        claim are scored, and a pair below `min_pair_similarity` is not scored at all.
        Entailed = best entailment >= entailment_threshold and >= its contradiction score.
        The hypothesis is the finding's `check` (its text if absent). Unlike the tool, the
        candidate pool is `memory.premise_sentences_per_chunk`, not the first
        nli.max_sentences_per_chunk: a finding cites one to three chunks, so the wider pool
        costs embeddings only, never NLI pairs, and the supporting sentence is often past
        the eighth. Returns the findings with "label", "confidence", "premise".
        """
        nli_cfg = self.config.nli
        cap = int(self.memory.premise_sentences_per_chunk)
        top_n = int(nli_cfg.premises_per_claim)
        # min_similarity overrides the floor only for measurement (scripts/tune_memory.py).
        floor = float(nli_cfg.min_pair_similarity if min_similarity is None else min_similarity)
        entail_t = float(nli_cfg.entailment_threshold)

        plans: List[Tuple[Dict[str, Any], List[str], Optional[float]]] = []
        for finding in findings:
            premises = [s for cid in finding["chunk_ids"]
                        for s in _sentences(str(table[cid].get("text") or ""), cap)]
            if not premises:
                plans.append((finding, [], None))
                continue
            hypothesis = finding.get("check") or finding["text"]
            vectors = self.embedder.encode_passages(premises + [hypothesis])
            sims = np.asarray(vectors[:-1]) @ np.asarray(vectors[-1])
            order = [int(i) for i in sims.argsort()[::-1][:top_n] if sims[int(i)] >= floor]
            best_sim = float(sims.max())
            plans.append((finding, [premises[i] for i in order], best_sim))

        pairs = [(premise, finding.get("check") or finding["text"])
                 for finding, chosen, _ in plans for premise in chosen]
        scored = self.nli.predict(pairs) if pairs else []
        out: List[Dict[str, Any]] = []
        cursor = 0
        for finding, chosen, best_sim in plans:
            window = scored[cursor:cursor + len(chosen)]
            cursor += len(chosen)
            label, confidence, premise = "neutral", 0.0, None
            if not chosen:
                label = "no_premise" if best_sim is None or best_sim < floor else "neutral"
            else:
                best = max(range(len(window)),
                           key=lambda i: window[i]["scores"].get("entailment", 0.0))
                entail = window[best]["scores"].get("entailment", 0.0)
                contra = window[best]["scores"].get("contradiction", 0.0)
                premise = chosen[best]
                confidence = entail
                if entail >= entail_t and entail >= contra:
                    label = "entailed"
            out.append({**finding, "label": label, "confidence": round(float(confidence), 4),
                        "premise": premise, "best_premise_similarity":
                            None if best_sim is None else round(best_sim, 4)})
        return out

    def _write(self, question: str, answer: str, thread_id: str,
               trace_records: Sequence[Dict[str, Any]], injected: Dict[str, Any],
               report: Dict[str, Any]) -> None:
        cited = cited_chunk_ids(answer)
        seen: Set[str] = {str(c) for r in trace_records for c in (r.get("chunk_ids_returned") or [])}
        seen.update(str(c) for c in injected.get("chunk_ids") or [])
        table = self.chunks()
        # A finding may rest only on chunks the answer cites, that this run actually had
        # in view (retrieved, or injected from memory), and that still exist.
        allowed = {c for c in cited if c in seen and c in table}

        extracted = self.extract(question, answer)
        if "error" in extracted:
            raise RuntimeError(f"{extracted['error']}: {extracted['detail']}")

        dropped = {"uncited": 0, "ungrounded": 0}
        candidates = []
        for finding in extracted["findings"]:
            ids = [c for c in finding["chunk_ids"] if c in allowed]
            if not ids:
                dropped["uncited"] += 1
                continue
            candidates.append({"text": finding["text"], "check": finding.get("check") or finding["text"],
                               "chunk_ids": ids})
        grounded = [f for f in self.ground(candidates, table) if f["label"] == "entailed"]
        dropped["ungrounded"] = len(candidates) - len(grounded)
        report["dropped"] = dropped

        items = [{"kind": "finding", "text": f["text"], "chunk_ids": f["chunk_ids"],
                  "hashes": {c: str(table[c].get("content_hash")) for c in f["chunk_ids"]}}
                 for f in grounded]
        items += [{"kind": "preference", "text": p, "chunk_ids": [], "hashes": {}}
                  for p in extracted["preferences"]]
        if not items:
            return
        vectors = self.embedder.encode_passages([item["text"] for item in items])
        for item, vector in zip(items, vectors):
            self._reconcile(item, np.asarray(vector, dtype=np.float32), thread_id, question,
                            report)

    def conflicts(self, old: str, new: str) -> Tuple[bool, float]:
        """NLI both ways between two claims. A confident contradiction, per config:
        contradiction >= memory.contradiction_threshold and above entailment, in both
        directions (or either, if memory.contradiction_both_directions is false)."""
        threshold = float(self.memory.contradiction_threshold)
        scored = self.nli.predict([(old, new), (new, old)])
        verdicts = []
        for result in scored:
            contra = result["scores"].get("contradiction", 0.0)
            entail = result["scores"].get("entailment", 0.0)
            verdicts.append((contra >= threshold and contra > entail, contra))
        hits = [hit for hit, _ in verdicts]
        confident = all(hits) if bool(self.memory.contradiction_both_directions) else any(hits)
        return confident, min(c for _, c in verdicts)

    def _reconcile(self, item: Dict[str, Any], vector: np.ndarray, thread_id: str,
                   question: str, report: Dict[str, Any]) -> None:
        """Duplicate -> refresh; contradiction -> supersede the old; otherwise add.

        Every active memory of the same kind at or above related_cosine is NLI-checked,
        best first, before the duplicate test. A contradiction wins over a refresh: a
        number-swapped claim sits as close to the original as a paraphrase does, and
        refreshing it would merge contradicting evidence into the old memory.
        """
        active = self.store.active(item["kind"])
        if "error" in active:
            raise RuntimeError(f"active: {active['error']}: {active.get('detail')}")
        rows = [m for m in active["memories"] if m.get("embedding") is not None]
        neighbours: List[Tuple[float, Dict[str, Any]]] = []
        if rows:
            sims = np.vstack([m["embedding"] for m in rows]) @ vector
            neighbours = sorted(((float(s), m) for s, m in zip(sims, rows)
                                 if s >= float(self.memory.related_cosine)),
                                key=lambda pair: pair[0], reverse=True)
            neighbours = neighbours[: int(self.memory.max_conflict_checks)]

        contradicted = []
        for similarity, old in neighbours:
            confident, score = self.conflicts(old["text"], item["text"])
            if confident:
                contradicted.append((similarity, score, old))

        if not contradicted and neighbours and neighbours[0][0] >= float(self.memory.duplicate_cosine):
            similarity, old = neighbours[0]
            result = self.store.refresh(old["id"], item["chunk_ids"], item["hashes"])
            if "error" in result:
                raise RuntimeError(f"refresh: {result['error']}: {result.get('detail')}")
            report["refreshed"].append({"id": old["id"], "kind": old["kind"], "text": old["text"],
                                        "similarity": round(similarity, 4),
                                        "chunk_ids_added": result["chunk_ids_added"],
                                        "matched": item["text"]})
            return

        added = self.store.add(item["kind"], item["text"], item["chunk_ids"], item["hashes"],
                               vector, thread_id, question)
        if "error" in added:
            raise RuntimeError(f"add: {added['error']}: {added.get('detail')}")
        memory = added["memory"]
        report["written"].append({"id": memory["id"], "kind": memory["kind"],
                                  "text": memory["text"], "chunk_ids": memory["chunk_ids"]})
        if contradicted:
            ids = [old["id"] for _, _, old in contradicted]
            result = self.store.invalidate(ids, "superseded", superseded_by=memory["id"])
            if "error" in result:
                raise RuntimeError(f"invalidate: {result['error']}: {result.get('detail')}")
            for similarity, score, old in contradicted:
                if old["id"] in result["invalidated"]:
                    report["invalidated"].append({
                        "id": old["id"], "kind": old["kind"], "text": old["text"],
                        "reason": "superseded", "superseded_by": memory["id"],
                        "similarity": round(similarity, 4), "contradiction": round(score, 4)})

    # ================================================================ forgetting

    def sweep(self) -> List[Dict[str, Any]]:
        """Stale, then decay and cap, then the hard purge. Returns what it forgot."""
        forgotten: List[Dict[str, Any]] = []
        active = self.store.active()
        if "error" in active:
            raise RuntimeError(f"active: {active['error']}: {active.get('detail')}")
        rows = active["memories"]
        if any(m["kind"] == "finding" for m in rows):
            current = {cid: str(c.get("content_hash")) for cid, c in self.chunks().items()}
            stale = forgetting.stale_ids(rows, current)
            if stale:
                result = self.store.invalidate(stale, "stale")
                if "error" in result:
                    raise RuntimeError(f"invalidate: {result['error']}")
                forgotten += [{"id": i, "reason": "stale"} for i in result["invalidated"]]
                rows = [m for m in rows if m["id"] not in set(stale)]

        now = float(self.now())
        victims = forgetting.decay_victims(
            rows, now, float(self.memory.half_life_days), float(self.memory.min_strength),
            int(self.memory.max_active))
        if victims:
            result = self.store.forget([v["id"] for v in victims], "decay")
            if "error" in result:
                raise RuntimeError(f"forget: {result['error']}")
            done = set(result["forgotten"])
            forgotten += [{"id": v["id"], "reason": "decay", "why": v["why"],
                           "strength": round(v["strength"], 4)} for v in victims if v["id"] in done]

        purged = self.store.purge(now - float(self.memory.purge_after_days) * forgetting.DAY_S)
        if "error" in purged:
            raise RuntimeError(f"purge: {purged['error']}")
        return forgotten

    def strengths(self, memories: Iterable[Dict[str, Any]]) -> Dict[str, float]:
        """id -> current strength, for display."""
        now = float(self.now())
        half = float(self.memory.half_life_days)
        return {m["id"]: round(forgetting.strength(m, now, half), 4) for m in memories}


def build_long_term_memory(config: Config = CFG, embedder: Any = None,
                           nli: Any = None) -> Optional[LongTermMemory]:
    """The LongTermMemory the CLI and the web server hand their AgentLoop, or None when
    `memory.enabled` is false. Shares the threads file, the retriever's embedder and the
    consistency tool's NLI model, so nothing is loaded twice."""
    if not bool(config.memory.enabled):
        return None
    from memory.memory_store import MemoryStore

    store = MemoryStore(config.paths.threads_db, dim=int(config.embedding.dim))
    return LongTermMemory(store, config, embedder=embedder, nli=nli)
