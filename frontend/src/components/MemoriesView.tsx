/* The Memories page (#/memories, docs/DECISIONS.md D23): every long-term memory, editable.
   In:  GET /api/memories?all=1 (re-read when `version` changes), the corpus papers (for the
        citation card) and the chat list (for source-chat titles). Out: preferences you can
        add, edit and delete; findings you can edit and delete, with their passages, uses
        and how soon they fade; and the forgotten and replaced ones, which can be restored. */

import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent, type MouseEvent } from "react";
import * as api from "../api";
import type { MemoryRecord, MemorySettings, Paper, ThreadSummary } from "../types";
import { clip, humanise, plural, relTime } from "../format";
import { Icon } from "../icons";
import { CitationCard, type CiteTarget } from "./CitationCard";

interface Props {
  papersById: Map<string, Paper>;
  threads: ThreadSummary[] | null;
  version: number;
}

interface Status {
  text: string;
  tone?: "warn" | "error";
  undo?: string; // a memory id that Undo restores
}

const errText = (e: unknown, what: string) => (e instanceof api.ApiError ? `${what}: ${e.detail || humanise(e.code)}` : `${what}.`);
const days = (n: number) => (n < 1 ? "less than a day" : plural(Math.round(n), "day"));

export function MemoriesView({ papersById, threads, version }: Props) {
  const [rows, setRows] = useState<MemoryRecord[] | null>(null);
  const [settings, setSettings] = useState<MemorySettings | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<Status | null>(null);
  const [cite, setCite] = useState<CiteTarget | null>(null);

  const load = useCallback(async () => {
    try {
      const got = await api.memoryCatalogue();
      setRows(got.memories);
      setSettings(got.settings ?? null);
      setLoadError(null);
    } catch (e) {
      setLoadError(e instanceof api.ApiError ? e.detail || humanise(e.code) : "The server is not answering.");
    }
  }, []);

  useEffect(() => {
    load();
  }, [load, version]);

  useEffect(() => {
    if (!cite) return;
    const onKey = (e: globalThis.KeyboardEvent) => e.key === "Escape" && setCite(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cite]);

  const titles = useMemo(() => new Map((threads ?? []).map((t) => [t.id, t.title])), [threads]);

  const q = query.trim().toLowerCase();
  const shown = (rows ?? []).filter((m) => !q || m.text.toLowerCase().includes(q));
  const prefs = shown.filter((m) => m.active && m.kind === "preference");
  const findings = shown.filter((m) => m.active && m.kind === "finding");
  const gone = shown.filter((m) => !m.active);
  const nActive = (rows ?? []).filter((m) => m.active).length;

  // ---------------------------------------------------------------- actions

  const save = async (m: MemoryRecord, text: string): Promise<string | null> => {
    try {
      const r = await api.editMemory(m.id, text);
      if (!r.changed) setStatus({ text: "Nothing changed." });
      else if (r.verification && !r.verification.verified) {
        setStatus({ tone: "warn", text: "Saved. Its passages don't support the new wording, so it is marked unverified and the agent is told to treat it with caution." });
      } else if (r.verification) setStatus({ text: "Saved. Its passages still support it." });
      else setStatus({ text: "Saved." });
      await load();
      return null;
    } catch (e) {
      return errText(e, "Not saved");
    }
  };

  const remove = async (m: MemoryRecord) => {
    try {
      await api.deleteMemory(m.id);
      setStatus({ text: "Memory deleted. It is no longer given to the agent.", undo: m.id });
    } catch (e) {
      setStatus({ tone: "error", text: errText(e, "Not deleted") });
    }
    load();
  };

  const restore = async (id: string) => {
    try {
      await api.restoreMemory(id);
      setStatus({ text: "Restored. It counts as just used, so it will not fade again soon." });
    } catch (e) {
      setStatus({ tone: "error", text: errText(e, "Not restored") });
    }
    load();
  };

  const add = async (text: string): Promise<string | null> => {
    try {
      const r = await api.addPreference(text);
      const replaced = r.replaced.length ? ` It replaces “${clip(r.replaced[0].text, 80)}”.` : "";
      setStatus({ text: r.outcome === "refreshed" ? "You already have that preference, so it was kept as it is." : `Preference added.${replaced}` });
      await load();
      return null;
    } catch (e) {
      return errText(e, "Not added");
    }
  };

  const openChunk = (e: MouseEvent<HTMLButtonElement>, chunk: string, paper: string, text: string | null) => {
    setCite({ chunk, paper, rect: e.currentTarget.getBoundingClientRect(), messageIndex: -1, excerpt: text ? clip(text, 420) : undefined });
  };

  // ---------------------------------------------------------------- layout

  if (loadError && rows === null) return <div className="page"><p className="quiet">The memories could not be loaded: {loadError}</p></div>;
  if (rows === null) return <div className="page"><p className="quiet">Loading memories…</p></div>;

  const item = (m: MemoryRecord) => (
    <MemoryItem
      key={m.id}
      memory={m}
      settings={settings}
      threadTitle={m.source_thread_id ? titles.get(m.source_thread_id) : undefined}
      onSave={(text) => save(m, text)}
      onDelete={() => remove(m)}
      onRestore={() => restore(m.id)}
      onChunk={openChunk}
    />
  );

  return (
    <div className="page memories-page">
      <h2 className="page-title">Memories</h2>
      <p className="page-lede">
        {plural(nActive, "active memory", "active memories")}. The agent keeps findings from answers that their own passages
        support, and preferences you state. Findings fade when they go unused. Memories are leads the agent checks against their
        passages before it relies on them, never sources it cites.
      </p>

      <div className="mx-toolbar">
        <label className="mx-search">
          <Icon name="search" size={16} />
          <span className="sr-only">Filter memories</span>
          <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Filter by text" />
        </label>
        {q && <span className="quiet small">{plural(shown.length, "match", "matches")}</span>}
      </div>

      <div className="mx-status" role="status" aria-live="polite">
        {status && (
          <p className={`mx-note${status.tone ? ` is-${status.tone}` : ""}`}>
            <span>{status.text}</span>
            {status.undo && <button className="link-btn" onClick={() => { const id = status.undo!; setStatus(null); restore(id); }}>Undo</button>}
            <button className="icon-btn mx-note-close" onClick={() => setStatus(null)} aria-label="Dismiss"><Icon name="close" size={14} /></button>
          </p>
        )}
      </div>
      {loadError && <p className="small mem-error">Could not refresh: {loadError}</p>}

      <section className="mx-section" aria-labelledby="mx-prefs">
        <h3 className="mx-heading" id="mx-prefs">Preferences <span className="topic-count">{prefs.length}</span></h3>
        <p className="mx-explain">Followed in every chat unless a question says otherwise. They never fade; they change only when you edit them or state a new one.</p>
        <AddPreference onAdd={add} maxChars={settings?.max_text_chars} />
        {prefs.length ? <ul className="mx-list">{prefs.map(item)}</ul> : (
          <p className="mx-empty">{q ? "No preference matches the filter." : "No preferences yet. Add one above, or state one in a chat (“only consider papers from 2024 onward”)."}</p>
        )}
      </section>

      <section className="mx-section" aria-labelledby="mx-findings">
        <h3 className="mx-heading" id="mx-findings">Findings <span className="topic-count">{findings.length}</span></h3>
        <p className="mx-explain">
          Given to the agent, with their passages, when a question is close to one.
          {settings && ` A finding used once and then left alone is forgotten after about ${days(settings.fresh_days)}; each time an answer cites it again, it lasts longer.`}
          {" "}Findings can't be added by hand: each one has to rest on passages an answer cited.
        </p>
        {findings.length ? <ul className="mx-list">{findings.map(item)}</ul> : (
          <p className="mx-empty">{q ? "No finding matches the filter." : "No findings yet. They are kept after answers in a chat whose passages support them."}</p>
        )}
      </section>

      <details className="mx-section mx-gone">
        <summary>
          <Icon name="chevron" size={16} />
          Forgotten and replaced <span className="topic-count">{gone.length}</span>
        </summary>
        <p className="mx-explain">
          No longer given to the agent.{" "}
          {settings ? `Each is kept for ${days(settings.purge_after_days)} after it stopped being active, then deleted for good.` : "Each is kept for a while, then deleted for good."}
          {" "}A memory deleted for good is gone and can't be restored.
        </p>
        {gone.length ? <ul className="mx-list">{gone.map(item)}</ul> : (
          <p className="mx-empty">{q ? "Nothing here matches the filter." : "Nothing forgotten yet."}</p>
        )}
      </details>

      {cite && (
        <CitationCard target={cite} paper={papersById.get(cite.paper)} excerpt={cite.excerpt} onClose={() => setCite(null)} />
      )}
    </div>
  );
}

// ------------------------------------------------------------------ add a preference

function AddPreference({ onAdd, maxChars }: { onAdd: (text: string) => Promise<string | null>; maxChars?: number }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!text.trim() || busy) return;
    setBusy(true);
    const problem = await onAdd(text);
    setBusy(false);
    setError(problem);
    if (!problem) setText("");
  };

  return (
    <form className="mx-add" onSubmit={submit}>
      <label className="sr-only" htmlFor="mx-add-input">Add a preference</label>
      <input
        id="mx-add-input"
        value={text}
        maxLength={maxChars}
        onChange={(e) => { setText(e.target.value); setError(null); }}
        placeholder="Add a preference, e.g. Only consider papers from 2024 onward"
        aria-invalid={!!error}
        aria-describedby={error ? "mx-add-error" : undefined}
        disabled={busy}
      />
      <button type="submit" className="btn btn-primary" disabled={!text.trim() || busy}>{busy ? "Adding…" : "Add"}</button>
      {error && <p className="mx-field-error" id="mx-add-error">{error}</p>}
    </form>
  );
}

// ------------------------------------------------------------------ one memory

function reasonText(m: MemoryRecord, s: MemorySettings | null): string {
  switch (m.forget_reason) {
    case "superseded":
      return m.superseded_by_text
        ? `Replaced by a newer finding that contradicts it: “${clip(m.superseded_by_text, 140)}”`
        : m.superseded_by ? "Replaced by a newer finding that contradicts it (that one has since been deleted for good)" : "Replaced by a newer finding";
    case "stale":
      return "Its passage changed or left the index";
    case "decay":
      return m.forget_detail === "over_cap"
        ? `Over the cap of ${s ? s.max_active : "the most"} findings: the weakest go first`
        : "Faded: not used for a long time";
    case "user":
      return "Deleted by you";
    default:
      return humanise(m.forget_reason) || "No longer active";
  }
}

interface ItemProps {
  memory: MemoryRecord;
  settings: MemorySettings | null;
  threadTitle?: string;
  onSave: (text: string) => Promise<string | null>;
  onDelete: () => void;
  onRestore: () => void;
  onChunk: (e: MouseEvent<HTMLButtonElement>, chunk: string, paper: string, text: string | null) => void;
}

function MemoryItem({ memory: m, settings, threadTitle, onSave, onDelete, onRestore, onChunk }: ItemProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(m.text);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const editBtn = useRef<HTMLButtonElement>(null);
  const finding = m.kind === "finding";
  const errId = `mx-err-${m.id}`;

  const cancel = () => {
    setEditing(false);
    setDraft(m.text);
    setError(null);
    // Back to where the edit started, so keyboard users keep their place.
    requestAnimationFrame(() => editBtn.current?.focus());
  };

  const submit = async (e?: FormEvent) => {
    e?.preventDefault();
    if (!draft.trim() || busy) return;
    setBusy(true);
    const problem = await onSave(draft);
    setBusy(false);
    if (problem) setError(problem);
    else {
      setEditing(false);
      setError(null);
      requestAnimationFrame(() => editBtn.current?.focus());
    }
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      cancel();
    } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      submit();
    }
  };

  const ended = m.forgotten_at ?? m.invalidated_at;
  const fading = m.fades_in_days != null && settings != null && m.fades_in_days <= settings.fading_soon_days;

  const source = m.origin === "user" ? <span>Added by you</span>
    : m.source_thread_id && m.source_thread_exists !== false ? (
      <span>From <a href={`#/chat/${m.source_thread_id}`}>{threadTitle ? clip(threadTitle, 60) : "the chat it came from"}</a></span>
    ) : m.source_thread_id ? <span>From a chat since deleted</span> : null;

  return (
    <li className={`mx-item${m.active ? "" : " is-gone"}`}>
      {editing ? (
        <form className="mx-edit" onSubmit={submit}>
          <label className="sr-only" htmlFor={`mx-edit-${m.id}`}>Edit this {m.kind}</label>
          <textarea
            id={`mx-edit-${m.id}`}
            value={draft}
            autoFocus
            rows={Math.min(8, Math.max(2, Math.ceil(draft.length / 80)))}
            maxLength={settings?.max_text_chars}
            onChange={(e) => { setDraft(e.target.value); setError(null); }}
            onKeyDown={onKey}
            aria-invalid={!!error}
            aria-describedby={error ? errId : undefined}
            disabled={busy}
          />
          {error && <p className="mx-field-error" id={errId}>{error}</p>}
          <div className="mx-edit-actions">
            <button type="submit" className="btn btn-primary" disabled={!draft.trim() || busy}>{busy ? (finding ? "Checking…" : "Saving…") : "Save"}</button>
            <button type="button" className="btn" onClick={cancel} disabled={busy}>Cancel</button>
            <span className="mx-hint">
              {finding ? "Saving re-checks the new wording against its passages. " : ""}Ctrl+Enter saves, Esc cancels.
            </span>
          </div>
        </form>
      ) : (
        <p className="mx-text">{m.text}</p>
      )}

      {finding && m.active && m.user_edited && !editing && (
        m.verified === false ? (
          <p className="mx-flag is-warn">Edited by you; not supported by its passages. The agent is told to treat it with extra caution.</p>
        ) : (
          <p className="mx-flag">Edited by you; still supported by its passages.</p>
        )
      )}

      {finding && (m.passages?.length || m.chunk_ids.length) > 0 && (
        <div className="mx-chips" aria-label="Passages it rests on">
          {(m.passages ?? m.chunk_ids.map((c) => ({ chunk_id: c, paper_id: c.split("__")[0], text: null, present: true, changed: false }))).map((p) => (
            <button
              key={p.chunk_id}
              className={`mx-chip${!p.present ? " is-missing" : p.changed ? " is-changed" : ""}`}
              onClick={(e) => onChunk(e, p.chunk_id, p.paper_id, p.text)}
              aria-label={`Passage ${p.chunk_id}${!p.present ? ", no longer in the index" : p.changed ? ", changed since checked" : ""}`}
              title={!p.present ? "No longer in the index" : p.changed ? "Its text changed since this was checked" : "Show the passage"}
            >
              {p.chunk_id}
              {!p.present ? <span className="mx-chip-note">gone</span> : p.changed ? <span className="mx-chip-note">changed</span> : null}
            </button>
          ))}
        </div>
      )}

      <div className="mx-meta">
        {m.active ? (
          <>
            {finding ? <span>Used {m.use_count}×, last {relTime(m.last_used_at)}</span> : <span>Added {relTime(m.created_at)}</span>}
            {!finding && m.user_edited && <span>Edited by you</span>}
            {source}
          </>
        ) : (
          <>
            <span>{reasonText(m, settings)}{ended != null ? `, ${relTime(ended)}` : ""}</span>
            {m.purge_in_days != null && <span>Deleted for good in {days(m.purge_in_days)}</span>}
          </>
        )}
      </div>

      {finding && m.active && m.strength_fraction != null && (
        <div className={`mx-strength${fading ? " is-fading" : ""}`}>
          <span
            className="meter"
            role="meter"
            aria-label="Strength"
            aria-valuemin={0}
            aria-valuemax={1}
            aria-valuenow={m.strength_fraction}
            aria-valuetext={`${m.strength?.toFixed(2)}, forgotten below ${settings?.min_strength}`}
            title={`Strength ${m.strength?.toFixed(2)}: uses × recency. Forgotten below ${settings?.min_strength}.`}
          >
            <span style={{ width: `${Math.max(3, Math.round(m.strength_fraction * 100))}%` }} />
          </span>
          <span>
            {fading ? "Fading: " : ""}forgotten in about {days(m.fades_in_days ?? 0)} unless an answer uses it
            <span className="mx-strength-num"> · strength {m.strength?.toFixed(2)}, forgotten below {settings?.min_strength}</span>
          </span>
        </div>
      )}

      {!editing && (
        <div className="mx-actions">
          {m.active ? (
            confirming ? (
              <span className="mx-confirm" role="group" aria-label="Confirm delete">
                <span>Delete this {m.kind}?</span>
                <button className="btn btn-danger" autoFocus onClick={() => { setConfirming(false); onDelete(); }}>Delete</button>
                <button className="btn" onClick={() => setConfirming(false)}>Keep</button>
              </span>
            ) : (
              <>
                <button ref={editBtn} className="btn btn-quiet" onClick={() => { setDraft(m.text); setEditing(true); }}>Edit</button>
                <button className="btn btn-quiet" onClick={() => setConfirming(true)}>Delete</button>
              </>
            )
          ) : (
            <>
              <button className="btn" onClick={onRestore} disabled={m.restorable === false} aria-describedby={m.restore_blocked ? `mx-rb-${m.id}` : undefined}>Restore</button>
              {m.restore_blocked && <span className="mx-hint" id={`mx-rb-${m.id}`}>Can't be restored: {m.restore_blocked}.</span>}
            </>
          )}
        </div>
      )}
    </li>
  );
}
