"""
ThreadStore: conversation threads and their turns, persisted in one SQLite file.

In:  the database path (config `paths.threads_db`), thread ids (uuid4 hex), and per turn
     the question, answer, stats, tool-call notes and summarised step events. The same
     file holds the long-term `memories` table (memory_store.MemoryStore).
Out: plain dicts. Never raises: a bad id, a missing thread or a corrupt file comes back as
     {"error": ..., "detail": ...}, the tools' envelope. Tables: docs/DATA_SCHEMA.md §9.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

THREAD_ID = re.compile(r"^[0-9a-f]{32}$")
UNTITLED = "Untitled chat"

SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    thread_id   TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    stats_json  TEXT NOT NULL,
    notes_json  TEXT NOT NULL,
    steps_json  TEXT NOT NULL,
    run_id      TEXT,
    created_at  REAL NOT NULL,
    PRIMARY KEY (thread_id, seq)
);
CREATE INDEX IF NOT EXISTS threads_by_updated ON threads (updated_at);
"""


def _error(code: str, detail: str) -> Dict[str, Any]:
    return {"error": code, "detail": detail}


def valid_thread_id(thread_id: Any) -> bool:
    """True for a uuid4 hex id. Checked before any query, so nothing else reaches SQL."""
    return isinstance(thread_id, str) and bool(THREAD_ID.match(thread_id))


def new_thread_id() -> str:
    return uuid.uuid4().hex


def title_from(question: str, limit: int) -> str:
    """A thread's name from its first question: one line, cut at a word boundary."""
    flat = " ".join((question or "").split())
    if not flat:
        return UNTITLED
    if len(flat) <= limit:
        return flat
    cut = flat[:limit].rsplit(" ", 1)[0] or flat[:limit]
    return cut.rstrip(",.;:") + "…"


def _dump(value: Any) -> str:
    # default=str: stats and steps come from the loop, and a stray non-JSON value must
    # degrade to text rather than lose the whole turn.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _load(text: Optional[str], fallback: Any) -> Any:
    """One unreadable column must not hide the rest of the turn."""
    try:
        value = json.loads(text) if text else fallback
    except (TypeError, ValueError):
        return fallback
    return value if isinstance(value, type(fallback)) else fallback


class _Missing(Exception):
    """Internal: the thread a write names does not exist. Rolls the transaction back."""


class ThreadStore:
    """CRUD over threads and turns. One short-lived connection per call.

    Short-lived because the web server reads from request threads while the agent's worker
    thread writes, and the CLI may write the same file from another process. SQLite
    serialises those; a shared connection would not be safe across threads.
    """

    def __init__(self, path: Path, now: Callable[[], float] = time.time,
                 timeout_s: float = 10.0) -> None:
        self.path = Path(path)
        self.now = now
        self.timeout_s = timeout_s
        self._ready = False

    # ---- plumbing --------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Autocommit mode: every write below opens its own BEGIN IMMEDIATE, so the
        # transaction boundaries are the ones written here, not the module's implicit ones.
        conn = sqlite3.connect(str(self.path), timeout=self.timeout_s, isolation_level=None)
        conn.row_factory = sqlite3.Row
        # Off by default and per connection: without this the cascade silently does nothing.
        conn.execute("PRAGMA foreign_keys = ON")
        if not self._ready:
            conn.executescript(SCHEMA)
            self._ready = True
        return conn

    def _read(self, work: Callable[[sqlite3.Connection], Dict[str, Any]]) -> Dict[str, Any]:
        try:
            with closing(self._connect()) as conn:
                conn.execute("BEGIN")  # one snapshot across the queries in `work`
                try:
                    return work(conn)
                finally:
                    conn.execute("COMMIT")
        except _Missing as exc:
            return _error("thread_not_found", str(exc))
        except (sqlite3.Error, OSError) as exc:
            return _error("thread_store_unavailable", f"{self.path}: {type(exc).__name__}: {exc}")

    def _write(self, work: Callable[[sqlite3.Connection], Dict[str, Any]]) -> Dict[str, Any]:
        try:
            with closing(self._connect()) as conn:
                # IMMEDIATE takes the write lock up front, so a seq read inside `work`
                # cannot race another writer between the read and the insert.
                conn.execute("BEGIN IMMEDIATE")
                try:
                    result = work(conn)
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
                conn.execute("COMMIT")
                return result
        except _Missing as exc:
            return _error("thread_not_found", str(exc))
        except (sqlite3.Error, OSError) as exc:
            return _error("thread_store_unavailable", f"{self.path}: {type(exc).__name__}: {exc}")

    @staticmethod
    def _bad_id(thread_id: Any) -> Dict[str, Any]:
        return _error("bad_thread_id", f"expected a 32-character hex thread id, got {thread_id!r}")

    @staticmethod
    def _summary(row: sqlite3.Row) -> Dict[str, Any]:
        return {"id": row["id"], "title": row["title"], "created_at": row["created_at"],
                "updated_at": row["updated_at"], "n_turns": int(row["n_turns"])}

    # ---- threads ---------------------------------------------------------------

    def create_thread(self, title: str = "") -> Dict[str, Any]:
        """A new, empty thread. Returns its summary ({"id", "title", ...})."""
        thread_id = new_thread_id()
        name = " ".join(str(title or "").split()) or UNTITLED
        stamp = float(self.now())

        def work(conn: sqlite3.Connection) -> Dict[str, Any]:
            conn.execute("INSERT INTO threads (id, title, created_at, updated_at) "
                         "VALUES (?, ?, ?, ?)", (thread_id, name, stamp, stamp))
            return {"id": thread_id, "title": name, "created_at": stamp,
                    "updated_at": stamp, "n_turns": 0}

        return self._write(work)

    def list_threads(self) -> Dict[str, Any]:
        """{"threads": [summary, ...]}, most recently updated first."""
        def work(conn: sqlite3.Connection) -> Dict[str, Any]:
            rows = conn.execute(
                "SELECT t.id, t.title, t.created_at, t.updated_at, "
                "       (SELECT COUNT(*) FROM turns u WHERE u.thread_id = t.id) AS n_turns "
                "FROM threads t ORDER BY t.updated_at DESC, t.rowid DESC"
            ).fetchall()
            return {"threads": [self._summary(row) for row in rows]}

        return self._read(work)

    def get_thread(self, thread_id: str) -> Dict[str, Any]:
        """The thread's summary plus "turns", oldest first, JSON columns decoded."""
        if not valid_thread_id(thread_id):
            return self._bad_id(thread_id)

        def work(conn: sqlite3.Connection) -> Dict[str, Any]:
            row = conn.execute(
                "SELECT t.id, t.title, t.created_at, t.updated_at, "
                "       (SELECT COUNT(*) FROM turns u WHERE u.thread_id = t.id) AS n_turns "
                "FROM threads t WHERE t.id = ?", (thread_id,)
            ).fetchone()
            if row is None:
                raise _Missing(f"no thread {thread_id}")
            turns = conn.execute(
                "SELECT seq, question, answer, stats_json, notes_json, steps_json, run_id, "
                "created_at FROM turns WHERE thread_id = ? ORDER BY seq", (thread_id,)
            ).fetchall()
            return {**self._summary(row), "turns": [{
                "seq": int(t["seq"]),
                "question": t["question"],
                "answer": t["answer"],
                "stats": _load(t["stats_json"], {}),
                "notes": _load(t["notes_json"], []),
                "steps": _load(t["steps_json"], []),
                "run_id": t["run_id"],
                "created_at": t["created_at"],
            } for t in turns]}

        return self._read(work)

    def rename_thread(self, thread_id: str, title: str) -> Dict[str, Any]:
        if not valid_thread_id(thread_id):
            return self._bad_id(thread_id)
        name = " ".join(str(title or "").split()) or UNTITLED

        def work(conn: sqlite3.Connection) -> Dict[str, Any]:
            if conn.execute("UPDATE threads SET title = ? WHERE id = ?",
                            (name, thread_id)).rowcount == 0:
                raise _Missing(f"no thread {thread_id}")
            return {"id": thread_id, "title": name}

        return self._write(work)

    def delete_thread(self, thread_id: str) -> Dict[str, Any]:
        """Delete a thread and every turn in it, in one transaction."""
        if not valid_thread_id(thread_id):
            return self._bad_id(thread_id)

        def work(conn: sqlite3.Connection) -> Dict[str, Any]:
            # Explicit as well as ON DELETE CASCADE, so a file created without the
            # foreign key (or opened by a tool that leaves the pragma off) still cascades.
            turns = conn.execute("DELETE FROM turns WHERE thread_id = ?", (thread_id,)).rowcount
            if conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,)).rowcount == 0:
                raise _Missing(f"no thread {thread_id}")
            return {"deleted": thread_id, "turns_deleted": int(turns)}

        return self._write(work)

    # ---- turns -----------------------------------------------------------------

    def append_turn(
        self,
        thread_id: str,
        question: str,
        answer: str,
        stats: Optional[Dict[str, Any]] = None,
        notes: Optional[Sequence[str]] = None,
        steps: Optional[Sequence[Dict[str, Any]]] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add the next turn. Returns {"thread_id", "seq"}; seq starts at 1.

        seq is assigned inside the INSERT and the thread's updated_at moves in the same
        transaction, so two writers can neither share a seq nor leave the list unsorted.
        """
        if not valid_thread_id(thread_id):
            return self._bad_id(thread_id)
        stamp = float(self.now())

        def work(conn: sqlite3.Connection) -> Dict[str, Any]:
            if conn.execute("UPDATE threads SET updated_at = ? WHERE id = ?",
                            (stamp, thread_id)).rowcount == 0:
                raise _Missing(f"no thread {thread_id}")
            cursor = conn.execute(
                "INSERT INTO turns (thread_id, seq, question, answer, stats_json, notes_json, "
                "                   steps_json, run_id, created_at) "
                "SELECT ?, COALESCE(MAX(seq), 0) + 1, ?, ?, ?, ?, ?, ?, ? "
                "FROM turns WHERE thread_id = ?",
                (thread_id, str(question or ""), str(answer or ""), _dump(dict(stats or {})),
                 _dump([str(n) for n in (notes or [])]), _dump(list(steps or [])),
                 run_id, stamp, thread_id),
            )
            seq = conn.execute("SELECT seq FROM turns WHERE rowid = ?",
                               (cursor.lastrowid,)).fetchone()["seq"]
            return {"thread_id": thread_id, "seq": int(seq)}

        return self._write(work)

    def annotate_turn(self, thread_id: str, seq: int, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Merge `fields` into a stored turn's stats, read and written in one transaction.

        For what is only known after the turn was saved: the long-term memory write runs
        after the answer (D19), and its report must be on the turn the page reloads.
        """
        if not valid_thread_id(thread_id):
            return self._bad_id(thread_id)

        def work(conn: sqlite3.Connection) -> Dict[str, Any]:
            row = conn.execute("SELECT stats_json FROM turns WHERE thread_id = ? AND seq = ?",
                               (thread_id, int(seq))).fetchone()
            if row is None:
                raise _Missing(f"no turn {seq} in thread {thread_id}")
            stats = {**_load(row["stats_json"], {}), **dict(fields)}
            conn.execute("UPDATE turns SET stats_json = ? WHERE thread_id = ? AND seq = ?",
                         (_dump(stats), thread_id, int(seq)))
            return {"thread_id": thread_id, "seq": int(seq)}

        return self._write(work)
