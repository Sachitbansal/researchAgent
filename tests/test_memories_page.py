"""
Offline tests for the Memories page (docs/DECISIONS.md D23): edit, add, restore, migrate.

In:  a real MemoryStore on tmp_path with a controllable clock, LongTermMemory with the stub
     embedder / NLI / chunk table from test_memory_longterm, and the FastAPI app over a
     stub loop carrying that memory. No network, no models.
Out: assertions on the store (edit, restore, origin, over-cap detail), the migration of an
     old-shape memories table, NLI re-verification on edit, the memory block's unverified
     mark, the catalogue the page reads, and the /api/memories routes with their errors.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from memory.longterm import UNVERIFIED_MARK  # noqa: E402
from memory.memory_store import MemoryStore  # noqa: E402
from memory.threads import ThreadStore  # noqa: E402
from test_memory_longterm import (  # noqa: E402
    CLAIM_A, CLAIM_B, DAY, DIM, StubEmbedder, StubNLI, _config, make, seed, table_ab, unit,
)
from web.api import create_app  # noqa: E402

EDIT_OK = "MethodA adds a balancing loss weighted 0.01."   # the stub NLI will entail this
EDIT_BAD = "MethodA uses no balancing loss at all."         # and not this


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    return _config(tmp_path, monkeypatch)


# ================================================================ store


def test_update_text_sets_the_flags_and_is_not_a_use(cfg, tmp_path):
    _, store, clock = make(cfg, tmp_path)
    mid = seed(store, CLAIM_A)
    before = store.get([mid])["memories"][0]
    clock.advance_days(3)
    saved = store.update_text(mid, "  new   text  here ", unit(1), verified=False)["memory"]
    assert saved["text"] == "new text here"
    assert saved["user_edited"] is True and saved["verified"] is False
    assert saved["edited_at"] == clock.t
    assert saved["last_used_at"] == before["last_used_at"] and saved["use_count"] == before["use_count"]
    vector = store.active()["memories"][0]["embedding"]
    assert np.allclose(vector, unit(1))

    assert store.update_text("nope", "x", unit(1))["error"] == "bad_memory_id"
    assert store.update_text(mid, "   ", unit(1))["error"] == "empty_text"
    assert store.update_text(mid, "x", np.zeros(3))["error"] == "bad_embedding"
    store.forget_one(mid)
    assert store.update_text(mid, "x", unit(1))["error"] == "memory_not_found"


def test_restore_clears_every_reason_and_resets_recency(cfg, tmp_path):
    _, store, clock = make(cfg, tmp_path)
    old = seed(store, CLAIM_A)
    new = seed(store, CLAIM_B)
    store.invalidate([old], "superseded", superseded_by=new)
    clock.advance_days(10)
    row = store.restore(old)["memory"]
    assert row["active"] and row["last_used_at"] == clock.t
    assert row["superseded_by"] is None and row["forget_reason"] is None
    assert row["invalidated_at"] is None and row["forgotten_at"] is None
    assert store.restore(old)["error"] == "memory_active"
    assert store.restore("c" * 32)["error"] == "memory_not_found"
    assert store.restore("../x")["error"] == "bad_memory_id"


def test_add_records_its_origin(tmp_path):
    store = MemoryStore(tmp_path / "m.sqlite3", dim=DIM)
    made = store.add("preference", "Only 2024 onward.", [], {}, unit(1), None, "", origin="user")
    assert made["memory"]["origin"] == "user" and made["memory"]["verified"] is None
    assert store.add("finding", "x", [], {}, unit(1), None, "q")["memory"]["origin"] == "agent"
    assert store.add("finding", "x", [], {}, unit(1), None, "q", origin="bot")["error"] == "bad_origin"


def test_the_sweep_records_why_a_finding_decayed(tmp_path, monkeypatch):
    cfg = _config(tmp_path, monkeypatch, memory={"max_active": 1})
    memory, store, clock = make(cfg, tmp_path)
    weak = seed(store, CLAIM_A)
    clock.advance_days(1)
    seed(store, CLAIM_B)
    memory.sweep()
    row = {m["id"]: m for m in store.list(include_inactive=True)["memories"]}[weak]
    assert row["forget_reason"] == "decay" and row["forget_detail"] == "over_cap"


# ================================================================ migration

PRE_D23 = """
CREATE TABLE memories (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, text TEXT NOT NULL,
    chunk_ids_json TEXT NOT NULL, content_hashes_json TEXT NOT NULL, embedding BLOB NOT NULL,
    source_thread_id TEXT, source_question TEXT NOT NULL, created_at REAL NOT NULL,
    last_used_at REAL NOT NULL, use_count INTEGER NOT NULL, invalidated_at REAL,
    superseded_by TEXT, forgotten_at REAL, forget_reason TEXT);
"""


def test_migration_adds_the_new_columns_to_an_existing_memories_table(tmp_path):
    path = tmp_path / "threads.sqlite3"
    mid = "d" * 32
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(PRE_D23)
        conn.execute("INSERT INTO memories VALUES (?, 'finding', 'old claim here', '[\"pA__c0001\"]', "
                     "'{\"pA__c0001\": \"hA\"}', ?, NULL, 'q', 1, 1, 1, NULL, NULL, NULL, NULL)",
                     (mid, unit(1).tobytes()))

    store = MemoryStore(path, dim=DIM)
    row = store.list()["memories"][0]
    # The old row reads as what it was: written by the agent, unedited, NLI-verified.
    assert row["id"] == mid and row["origin"] == "agent"
    assert row["user_edited"] is False and row["verified"] is True and row["edited_at"] is None
    assert store.update_text(mid, "edited claim", unit(1), verified=False)["memory"]["user_edited"]
    with sqlite3.connect(str(path)) as conn:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
    assert columns >= {"forget_detail", "origin", "user_edited", "verified", "edited_at"}

    # A second process opening the same file finds the columns there and changes nothing.
    again = MemoryStore(path, dim=DIM)
    assert again.list()["memories"][0]["text"] == "edited claim"


# ================================================================ edit and verify


def test_an_entailed_edit_is_verified(cfg, tmp_path):
    table = table_ab()
    table["pA__c0001"]["text"] = f"{EDIT_OK} The loss is added at every layer."
    table["pA__c0001"]["content_hash"] = "hA2"
    nli = StubNLI(entailed={EDIT_OK})
    emb = StubEmbedder()
    memory, store, _ = make(cfg, tmp_path, nli=nli, embedder=emb, table=table)
    mid = seed(store, CLAIM_A)

    out = memory.edit(mid, EDIT_OK)
    assert out["changed"] and out["verification"]["verified"] is True
    assert out["memory"]["verified"] is True and out["memory"]["user_edited"] is True
    # The edited text was the hypothesis, and it was embedded passage-side (no prefix).
    assert any(h == EDIT_OK for _, h in nli.pairs)
    assert EDIT_OK in emb.passages and emb.queries == []
    # Verified against the chunk as it is now, so its current hash is what is stored.
    assert store.get([mid])["memories"][0]["content_hashes"] == {"pA__c0001": "hA2"}


def test_an_unsupported_edit_is_kept_but_marked_unverified(cfg, tmp_path):
    memory, store, _ = make(cfg, tmp_path, nli=StubNLI(entailed=set()))
    mid = seed(store, CLAIM_A)
    out = memory.edit(mid, EDIT_BAD)
    assert out["changed"] and out["verification"]["verified"] is False
    row = store.get([mid])["memories"][0]
    assert row["text"] == EDIT_BAD and row["active"]
    assert row["user_edited"] is True and row["verified"] is False
    assert row["content_hashes"] == {"pA__c0001": "hA"}   # unchanged: nothing re-checked


def test_an_edit_whose_chunks_are_gone_is_unverified_not_a_crash(cfg, tmp_path):
    memory, store, _ = make(cfg, tmp_path, table={})
    mid = seed(store, CLAIM_A)
    out = memory.edit(mid, EDIT_OK)
    assert out["verification"]["label"] == "no_premise" and out["memory"]["verified"] is False


def test_editing_a_preference_skips_nli(cfg, tmp_path):
    nli = StubNLI()
    memory, store, _ = make(cfg, tmp_path, nli=nli)
    mid = seed(store, "Only papers from 2023 onward.", chunk_ids=(), kind="preference")
    out = memory.edit(mid, "Only papers from 2024 onward.")
    assert out["verification"] is None and nli.pairs == []
    assert out["memory"]["user_edited"] is True and out["memory"]["verified"] is None


def test_edit_validation(cfg, tmp_path):
    memory, store, _ = make(cfg, tmp_path)
    mid = seed(store, CLAIM_A)
    assert memory.edit("x", "y")["error"] == "bad_memory_id"
    assert memory.edit(mid, "   ")["error"] == "empty_text"
    assert memory.edit(mid, "w" * (int(cfg.memory.max_text_chars) + 1))["error"] == "text_too_long"
    assert memory.edit("c" * 32, "y")["error"] == "memory_not_found"
    assert memory.edit(mid, f"  {CLAIM_A} ") == {"memory": store.get([mid])["memories"][0],
                                                 "changed": False, "verification": None}
    store.forget_one(mid)
    assert memory.edit(mid, "y")["error"] == "memory_inactive"


def test_the_block_marks_unverified_edits_only(cfg, tmp_path):
    memory, store, _ = make(cfg, tmp_path, nli=StubNLI(entailed=set()))
    edited = seed(store, CLAIM_A, vector=unit(1))
    plain = seed(store, CLAIM_B, chunk_ids=("pB__c0002",), vector=unit(1))
    memory.edit(edited, EDIT_BAD)
    # The edit re-embeds the text; put it back on the question's axis so both are recalled.
    with sqlite3.connect(str(store.path)) as conn:
        conn.execute("UPDATE memories SET embedding = ? WHERE id = ?", (unit(1).tobytes(), edited))
    memory._embedder.vectors["question"] = unit(1)

    recalled = memory.recall("question")
    lines = [line for line in recalled["block"].splitlines() if line.startswith("- m")]
    marked = [line for line in lines if UNVERIFIED_MARK in line]
    assert len(lines) == 2 and len(marked) == 1 and EDIT_BAD in marked[0]
    assert CLAIM_B in next(line for line in lines if line not in marked)
    flags = {m["id"]: m["verified"] for m in recalled["memories"]}
    assert flags == {edited: False, plain: True}
    assert "edited by user, unverified" in cfg.prompt("memory_block")


# ================================================================ add and restore


def test_add_preference_adds_then_refreshes_a_restatement(cfg, tmp_path):
    memory, store, _ = make(cfg, tmp_path)
    first = memory.add_preference("Only consider papers from 2024 onward")
    assert first["outcome"] == "added" and first["memory"]["origin"] == "user"
    again = memory.add_preference("  Only consider papers from 2024 onward ")
    assert again["outcome"] == "refreshed" and again["memory"]["id"] == first["memory"]["id"]
    assert again["memory"]["use_count"] == 2
    assert len(store.active("preference")["memories"]) == 1
    assert memory.add_preference(" ")["error"] == "empty_text"


def test_restore_refuses_a_finding_whose_evidence_changed(cfg, tmp_path):
    table = table_ab()
    memory, store, _ = make(cfg, tmp_path, table=table)
    mid = seed(store, CLAIM_A)
    store.forget_one(mid)
    table["pA__c0001"]["content_hash"] = "changed"
    assert memory.restore(mid)["error"] == "evidence_changed"
    table["pA__c0001"]["content_hash"] = "hA"
    assert memory.restore(mid)["memory"]["active"]
    assert memory.restore(mid)["error"] == "memory_active"


def test_a_hard_purged_memory_cannot_be_restored(cfg, tmp_path):
    memory, store, clock = make(cfg, tmp_path)
    mid = seed(store, CLAIM_A)
    store.forget_one(mid)
    clock.advance_days(float(cfg.memory.purge_after_days) + 1)
    memory.sweep()
    out = memory.restore(mid)
    assert out["error"] == "memory_not_found" and "for good" in out["detail"]


def test_catalogue_carries_what_the_page_shows(cfg, tmp_path):
    table = table_ab()
    memory, store, clock = make(cfg, tmp_path, table=table)
    live = seed(store, CLAIM_A)
    gone = seed(store, CLAIM_B, chunk_ids=("pB__c0002",))
    store.forget_one(gone)
    table["pB__c0002"]["content_hash"] = "moved"
    clock.advance_days(30)

    out = memory.catalogue({"t" * 32})
    rows = {m["id"]: m for m in out["memories"]}
    s = out["settings"]
    assert s["min_strength"] == cfg.memory.min_strength and round(s["fresh_days"]) == 90
    a = rows[live]
    # Used once, unused for one half-life: strength 0.5, so 60 of 90 days are left.
    assert a["strength"] == 0.5 and round(a["fades_in_days"]) == 60
    assert round(a["strength_fraction"], 3) == round(60 / 90, 3)
    assert a["passages"][0]["present"] and not a["passages"][0]["changed"]
    assert a["source_thread_exists"] is True and "content_hashes" not in a
    b = rows[gone]
    assert b["strength"] is None and b["passages"][0]["changed"]
    assert b["restorable"] is False and "changed" in b["restore_blocked"]
    assert round(b["purge_in_days"]) == round(cfg.memory.purge_after_days - 30)


# ================================================================ routes


@pytest.fixture()
def page(cfg, tmp_path):
    table = table_ab()
    table["pA__c0001"]["text"] = f"{EDIT_OK} The loss is added at every layer."
    memory, store, clock = make(cfg, tmp_path, nli=StubNLI(entailed={EDIT_OK}), table=table)
    threads = ThreadStore(tmp_path / "threads.sqlite3")
    loop = SimpleNamespace(store=threads, memory=memory, on_event=None)
    config = SimpleNamespace(paths=SimpleNamespace(figures=tmp_path / "figures"),
                             agent=SimpleNamespace(trace_text_chars=50, thread_title_chars=24))
    app = create_app(lambda: (loop, config), dist_dir=tmp_path / "no_dist")
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, store=store, threads=threads, clock=clock, app=app)


def test_patch_verifies_findings_and_flags_unsupported_edits(page):
    c = page.client
    mid = seed(page.store, CLAIM_A)
    ok = c.patch(f"/api/memories/{mid}", json={"text": EDIT_OK})
    assert ok.status_code == 200 and ok.json()["memory"]["verified"] is True
    bad = c.patch(f"/api/memories/{mid}", json={"text": EDIT_BAD}).json()
    assert bad["verification"]["verified"] is False
    listed = {m["id"]: m for m in c.get("/api/memories").json()["memories"]}
    assert listed[mid]["user_edited"] and listed[mid]["verified"] is False
    assert c.patch("/api/memories/nope", json={"text": "x"}).status_code == 400
    assert c.patch(f"/api/memories/{'c' * 32}", json={"text": "x"}).status_code == 404
    assert c.patch(f"/api/memories/{mid}", json={"text": " "}).json()["error"] == "empty_text"
    missing = c.patch(f"/api/memories/{mid}", json={})
    assert missing.status_code == 422 and missing.json()["error"] == "bad_request"


def test_post_adds_preferences_only(page):
    c = page.client
    made = c.post("/api/memories", json={"kind": "preference", "text": "Only 2024 onward."})
    assert made.status_code == 201 and made.json()["outcome"] == "added"
    again = c.post("/api/memories", json={"kind": "preference", "text": "Only 2024 onward."})
    assert again.status_code == 200 and again.json()["outcome"] == "refreshed"
    finding = c.post("/api/memories", json={"kind": "finding", "text": "x"})
    assert finding.status_code == 400 and finding.json()["error"] == "findings_need_evidence"
    assert c.post("/api/memories", json={"kind": "opinion", "text": "x"}).status_code == 400
    assert c.post("/api/memories", json={"text": "x"}).json()["error"] == "bad_request"


def test_delete_then_restore(page):
    c = page.client
    tid = page.threads.create_thread("a chat")["id"]
    mid = page.store.add("finding", CLAIM_A, ["pA__c0001"], {"pA__c0001": "hA"}, unit(1),
                         tid, "q")["memory"]["id"]
    assert c.delete(f"/api/memories/{mid}").json() == {"forgotten": mid, "reason": "user"}
    assert c.get("/api/memories").json()["memories"] == []
    everything = c.get("/api/memories?all=1").json()
    row = everything["memories"][0]
    assert row["forget_reason"] == "user" and row["restorable"] and row["source_thread_exists"]
    assert set(everything["settings"]) >= {"min_strength", "half_life_days", "purge_after_days"}

    page.clock.advance_days(5)
    back = c.post(f"/api/memories/{mid}/restore")
    assert back.status_code == 200 and back.json()["memory"]["last_used_at"] == page.clock.t
    assert c.post(f"/api/memories/{mid}/restore").status_code == 409
    assert c.post(f"/api/memories/{'c' * 32}/restore").status_code == 404
    assert c.post("/api/memories/zz/restore").status_code == 400

    page.threads.delete_thread(tid)
    assert c.get("/api/memories").json()["memories"][0]["source_thread_exists"] is False


def test_new_routes_need_long_term_memory(tmp_path):
    loop = SimpleNamespace(store=None, memory=None, on_event=None)
    config = SimpleNamespace(paths=SimpleNamespace(figures=tmp_path), agent=SimpleNamespace())
    with TestClient(create_app(lambda: (loop, config), dist_dir=tmp_path / "x")) as c:
        assert c.patch("/api/memories/" + "a" * 32, json={"text": "x"}).status_code == 503
        assert c.post("/api/memories", json={"kind": "preference", "text": "x"}).status_code == 503
        assert c.post("/api/memories/" + "a" * 32 + "/restore").status_code == 503


def test_edits_and_adds_wait_for_a_running_question(page):
    """They share the run's embedder and NLI model, so they are refused mid-run; delete
    and restore use no model and stay open."""
    c = page.client
    mid = seed(page.store, CLAIM_A)
    lock = page.app.state.lock
    lock.acquire()
    try:
        busy = c.patch(f"/api/memories/{mid}", json={"text": EDIT_OK})
        assert busy.status_code == 409 and busy.json()["error"] == "agent_busy"
        assert c.post("/api/memories", json={"kind": "preference", "text": "x y"}).status_code == 409
        assert c.delete(f"/api/memories/{mid}").status_code == 200
        assert c.post(f"/api/memories/{mid}/restore").status_code == 200
    finally:
        lock.release()
    assert c.patch(f"/api/memories/{mid}", json={"text": EDIT_OK}).status_code == 200
