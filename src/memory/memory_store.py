"""
MemoryStore: long-term memories (docs/DECISIONS.md D19) in the threads' SQLite file.

In:  the database path (config `paths.threads_db`, shared with ThreadStore), and per
     memory its kind, text, supporting chunk_ids with their content_hashes, a float32
     passage embedding and where it came from. Tables: docs/DATA_SCHEMA.md §10.
Out: plain dicts. Never raises: errors come back as {"error": ..., "detail": ...}.
     Rows are soft-deleted (invalidated_at / forgotten_at plus a reason) and purged later.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

MEMORY_ID = re.compile(r"^[0-9a-f]{32}$")
KINDS = ("finding", "preference")
# Why a memory stopped being active. superseded/stale set invalidated_at; decay/user set
# forgotten_at. Either way it is never injected again (D20).
REASONS = ("superseded", "stale", "decay", "user")

# CREATE ... IF NOT EXISTS is the whole migration: a threads file written before this
# table existed gains it on first open, and its threads and turns are untouched.
SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id                  TEXT PRIMARY KEY,
    kind                TEXT NOT NULL,
    text                TEXT NOT NULL,
    chunk_ids_json      TEXT NOT NULL,
    content_hashes_json TEXT NOT NULL,
    embedding           BLOB NOT NULL,
    source_thread_id    TEXT,
    source_question     TEXT NOT NULL,
    created_at          REAL NOT NULL,
    last_used_at        REAL NOT NULL,
    use_count           INTEGER NOT NULL,
    invalidated_at      REAL,
    superseded_by       TEXT,
    forgotten_at        REAL,
    forget_reason       TEXT
);
CREATE INDEX IF NOT EXISTS memories_by_state ON memories (invalidated_at, forgotten_at);
"""

ACTIVE = "invalidated_at IS NULL AND forgotten_at IS NULL"
COLUMNS = ("id, kind, text, chunk_ids_json, content_hashes_json, embedding, source_thread_id, "
           "source_question, created_at, last_used_at, use_count, invalidated_at, "
           "superseded_by, forgotten_at, forget_reason")


def _error(code: str, detail: str) -> Dict[str, Any]:
    return {"error": code, "detail": detail}


def valid_memory_id(memory_id: Any) -> bool:
    return isinstance(memory_id, str) and bool(MEMORY_ID.match(memory_id))


def _load(text: Optional[str], fallback: Any) -> Any:
    try:
        value = json.loads(text) if text else fallback
    except (TypeError, ValueError):
        return fallback
    return value if isinstance(value, type(fallback)) else fallback


class _Missing(Exception):
    """Internal: the memory a write names does not exist. Rolls the transaction back."""


class MemoryStore:
    """CRUD over the memories table. One short-lived connection per call, as ThreadStore.

    The embedding is a float32 BLOB of `dim` values. A row whose blob has the wrong
    length (a model swap) is listed but never matched against: comparing vectors from two
    models is meaningless, and failing loudly on read would hide every other memory.
    """

    def __init__(self, path: Path, dim: int, now: Callable[[], float] = time.time,
                 timeout_s: float = 10.0) -> None:
        self.path = Path(path)
        self.dim = int(dim)
        self.now = now
        self.timeout_s = timeout_s
        self._ready = False

    # ---- plumbing --------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=self.timeout_s, isolation_level=None)
        conn.row_factory = sqlite3.Row
        if not self._ready:
            conn.executescript(SCHEMA)
            self._ready = True
        return conn

    def _run(self, work: Callable[[sqlite3.Connection], Dict[str, Any]],
             write: bool) -> Dict[str, Any]:
        try:
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                try:
                    result = work(conn)
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
                conn.execute("COMMIT")
                return result
        except _Missing as exc:
            return _error("memory_not_found", str(exc))
        except (sqlite3.Error, OSError, ValueError) as exc:
            return _error("memory_store_unavailable", f"{self.path}: {type(exc).__name__}: {exc}")

    def _row(self, row: sqlite3.Row, with_embedding: bool) -> Dict[str, Any]:
        out = {
            "id": row["id"],
            "kind": row["kind"],
            "text": row["text"],
            "chunk_ids": [str(c) for c in _load(row["chunk_ids_json"], [])],
            "content_hashes": {str(k): str(v) for k, v in
                               _load(row["content_hashes_json"], {}).items()},
            "source_thread_id": row["source_thread_id"],
            "source_question": row["source_question"],
            "created_at": row["created_at"],
            "last_used_at": row["last_used_at"],
            "use_count": int(row["use_count"]),
            "invalidated_at": row["invalidated_at"],
            "superseded_by": row["superseded_by"],
            "forgotten_at": row["forgotten_at"],
            "forget_reason": row["forget_reason"],
            "active": row["invalidated_at"] is None and row["forgotten_at"] is None,
        }
        if with_embedding:
            blob = row["embedding"] or b""
            out["embedding"] = (np.frombuffer(blob, dtype=np.float32).copy()
                                if len(blob) == 4 * self.dim else None)
        return out

    # ---- reads -----------------------------------------------------------------

    def active(self, kind: Optional[str] = None) -> Dict[str, Any]:
        """{"memories": [...]}: active rows with their embedding (None if unusable)."""
        def work(conn: sqlite3.Connection) -> Dict[str, Any]:
            sql = f"SELECT {COLUMNS} FROM memories WHERE {ACTIVE}"
            args: List[Any] = []
            if kind is not None:
                sql += " AND kind = ?"
                args.append(kind)
            rows = conn.execute(sql + " ORDER BY created_at, rowid", args).fetchall()
            return {"memories": [self._row(r, with_embedding=True) for r in rows]}
        return self._run(work, write=False)

    def list(self, include_inactive: bool = False) -> Dict[str, Any]:
        """{"memories": [...]} without embeddings, most recently used first."""
        def work(conn: sqlite3.Connection) -> Dict[str, Any]:
            where = "" if include_inactive else f"WHERE {ACTIVE}"
            rows = conn.execute(f"SELECT {COLUMNS} FROM memories {where} "
                                "ORDER BY last_used_at DESC, rowid DESC").fetchall()
            return {"memories": [self._row(r, with_embedding=False) for r in rows]}
        return self._run(work, write=False)

    def get(self, ids: Sequence[str]) -> Dict[str, Any]:
        """{"memories": [...]} for the valid ids that exist, in the order asked."""
        wanted = [i for i in dict.fromkeys(ids) if valid_memory_id(i)]
        if not wanted:
            return {"memories": []}

        def work(conn: sqlite3.Connection) -> Dict[str, Any]:
            marks = ",".join("?" * len(wanted))
            rows = conn.execute(f"SELECT {COLUMNS} FROM memories WHERE id IN ({marks})",
                                wanted).fetchall()
            by_id = {r["id"]: self._row(r, with_embedding=False) for r in rows}
            return {"memories": [by_id[i] for i in wanted if i in by_id]}
        return self._run(work, write=False)

    # ---- writes ----------------------------------------------------------------

    def add(self, kind: str, text: str, chunk_ids: Sequence[str],
            content_hashes: Mapping[str, str], embedding: Any,
            source_thread_id: Optional[str], source_question: str) -> Dict[str, Any]:
        """A new active memory with use_count 1 (being found counts as its first use)."""
        if kind not in KINDS:
            return _error("bad_memory_kind", f"kind must be one of {list(KINDS)}, got {kind!r}")
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dim:
            return _error("bad_embedding", f"expected {self.dim} values, got {vector.shape[0]}")
        memory_id = uuid.uuid4().hex
        stamp = float(self.now())
        ids = [str(c) for c in chunk_ids]
        hashes = {str(c): str(content_hashes[c]) for c in ids if c in content_hashes}

        def work(conn: sqlite3.Connection) -> Dict[str, Any]:
            conn.execute(
                "INSERT INTO memories (id, kind, text, chunk_ids_json, content_hashes_json, "
                "embedding, source_thread_id, source_question, created_at, last_used_at, "
                "use_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                (memory_id, kind, str(text), json.dumps(ids), json.dumps(hashes),
                 vector.tobytes(), source_thread_id, str(source_question or ""), stamp, stamp))
            row = conn.execute(f"SELECT {COLUMNS} FROM memories WHERE id = ?",
                               (memory_id,)).fetchone()
            return {"memory": self._row(row, with_embedding=False)}
        return self._run(work, write=True)

    def refresh(self, memory_id: str, chunk_ids: Sequence[str] = (),
                content_hashes: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        """A near-duplicate re-finding: recency resets, use_count + 1, new chunks merged."""
        if not valid_memory_id(memory_id):
            return _error("bad_memory_id", f"expected a 32-character hex id, got {memory_id!r}")
        stamp = float(self.now())
        hashes = dict(content_hashes or {})

        def work(conn: sqlite3.Connection) -> Dict[str, Any]:
            row = conn.execute(f"SELECT {COLUMNS} FROM memories WHERE id = ? AND {ACTIVE}",
                               (memory_id,)).fetchone()
            if row is None:
                raise _Missing(f"no active memory {memory_id}")
            current = self._row(row, with_embedding=False)
            ids = list(current["chunk_ids"])
            stored = dict(current["content_hashes"])
            added = [str(c) for c in chunk_ids if str(c) not in ids]
            for chunk_id in added:
                ids.append(chunk_id)
                if chunk_id in hashes:
                    stored[chunk_id] = str(hashes[chunk_id])
            conn.execute("UPDATE memories SET last_used_at = ?, use_count = use_count + 1, "
                         "chunk_ids_json = ?, content_hashes_json = ? WHERE id = ?",
                         (stamp, json.dumps(ids), json.dumps(stored), memory_id))
            return {"id": memory_id, "chunk_ids_added": added,
                    "use_count": current["use_count"] + 1}
        return self._run(work, write=True)

    def mark_used(self, ids: Iterable[str]) -> Dict[str, Any]:
        """Injected into a run AND cited by its answer: recency resets, use_count + 1."""
        wanted = [i for i in dict.fromkeys(ids) if valid_memory_id(i)]
        stamp = float(self.now())

        def work(conn: sqlite3.Connection) -> Dict[str, Any]:
            done = [i for i in wanted if conn.execute(
                f"UPDATE memories SET last_used_at = ?, use_count = use_count + 1 "
                f"WHERE id = ? AND {ACTIVE}", (stamp, i)).rowcount]
            return {"used": done}
        return self._run(work, write=True)

    def invalidate(self, ids: Iterable[str], reason: str,
                   superseded_by: Optional[str] = None) -> Dict[str, Any]:
        """superseded (a newer contradicting finding) or stale (its chunk text changed)."""
        if reason not in ("superseded", "stale"):
            return _error("bad_reason", f"invalidation reason must be superseded or stale, got {reason!r}")
        wanted = [i for i in dict.fromkeys(ids) if valid_memory_id(i)]
        stamp = float(self.now())

        def work(conn: sqlite3.Connection) -> Dict[str, Any]:
            done = [i for i in wanted if conn.execute(
                f"UPDATE memories SET invalidated_at = ?, superseded_by = ?, forget_reason = ? "
                f"WHERE id = ? AND {ACTIVE}", (stamp, superseded_by, reason, i)).rowcount]
            return {"invalidated": done, "reason": reason}
        return self._run(work, write=True)

    def forget(self, ids: Iterable[str], reason: str) -> Dict[str, Any]:
        """decay (weak or over the cap) or user (deleted by hand)."""
        if reason not in ("decay", "user"):
            return _error("bad_reason", f"forget reason must be decay or user, got {reason!r}")
        wanted = [i for i in dict.fromkeys(ids) if valid_memory_id(i)]
        stamp = float(self.now())

        def work(conn: sqlite3.Connection) -> Dict[str, Any]:
            done = [i for i in wanted if conn.execute(
                f"UPDATE memories SET forgotten_at = ?, forget_reason = ? WHERE id = ? AND {ACTIVE}",
                (stamp, reason, i)).rowcount]
            return {"forgotten": done, "reason": reason}
        return self._run(work, write=True)

    def forget_one(self, memory_id: str, reason: str = "user") -> Dict[str, Any]:
        """The manual delete: an error for a bad, unknown or already inactive id."""
        if not valid_memory_id(memory_id):
            return _error("bad_memory_id", f"expected a 32-character hex id, got {memory_id!r}")
        result = self.forget([memory_id], reason)
        if "error" in result:
            return result
        if not result["forgotten"]:
            return _error("memory_not_found", f"no active memory {memory_id}")
        return {"forgotten": memory_id, "reason": reason}

    def purge(self, before: float) -> Dict[str, Any]:
        """Hard-delete rows that stopped being active before `before` (epoch seconds)."""
        def work(conn: sqlite3.Connection) -> Dict[str, Any]:
            n = conn.execute("DELETE FROM memories WHERE COALESCE(forgotten_at, invalidated_at) "
                             "< ?", (float(before),)).rowcount
            return {"purged": int(n)}
        return self._run(work, write=True)
