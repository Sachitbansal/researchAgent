/* The three-pane shell: chats on the left, the conversation in the centre, the agent's
   working on the right (docs/DECISIONS.md, D3).
   In:  the API in api.ts, the URL hash (#/chat/<id>, #/corpus), and the sample fixtures.
   Out: the page. Chats are threads stored by the server (D12): the list and each chat's
        turns are fetched, and the first question of a new chat creates its thread. One
        run at a time, mirroring the server's single agent. */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "./api";
import type {
  AnswerEvent, AssistantMessage, Chat, Corpus, FailedEvent, MemoryReport, Message, Paper, Step, StoredTurn, ThreadEvent,
  ThreadSummary,
} from "./types";
import { Sidebar } from "./components/Sidebar";
import { ChatView } from "./components/ChatView";
import { Inspector, type InspectorTab } from "./components/Inspector";
import { CorpusView } from "./components/CorpusView";
import { CitationCard, type CiteTarget } from "./components/CitationCard";
import { SAMPLE_CHATS } from "./fixtures/sampleChats";
import { Icon } from "./icons";

type Route = { view: "chat"; id: string | null } | { view: "corpus" };
type Theme = "system" | "light" | "dark";

interface Run {
  chatId: string | null; // null until the server's `thread` frame names the new chat
  question: string;
  steps: Step[];
  startedAt: number;
}

const THREAD_ID = /^[0-9a-f]{32}$/;

/* Samples are rebased onto "now" so the sidebar reads as recent, not as a fixed date. */
function seedSamples(): Chat[] {
  const now = Date.now() / 1000;
  const newest = Math.max(...SAMPLE_CHATS.map((c) => c.updated_at));
  const shift = now - 3600 - newest;
  return SAMPLE_CHATS.map((c) => ({
    ...c,
    created_at: c.created_at + shift,
    updated_at: c.updated_at + shift,
    messages: c.messages.map((m) => ({ ...m, ts: m.ts + shift })),
  }));
}

/* A stored turn as the two messages the conversation shows. */
function turnMessages(turn: StoredTurn): Message[] {
  const s = turn.stats ?? {};
  return [
    { role: "user", content: turn.question, ts: turn.created_at },
    {
      role: "assistant",
      status: s.error ? "failed" : "answered",
      content: turn.answer,
      steps: turn.steps ?? [],
      stats: s,
      error: s.error ? { error: s.error, detail: s.detail ?? "" } : undefined,
      elapsed_ms: s.elapsed_ms,
      notes: turn.notes ?? [],
      memory: s.memory,
      ts: turn.created_at,
    },
  ];
}

function parseHash(): Route {
  const h = window.location.hash.replace(/^#\/?/, "");
  if (h === "corpus") return { view: "corpus" };
  const m = h.match(/^chat\/([a-z0-9-]{1,64})$/);
  return { view: "chat", id: m ? m[1] : null };
}

const go = (hash: string) => {
  if (window.location.hash !== hash) window.location.hash = hash;
};

const THEME_KEY = "sciagent.theme";

function readTheme(): Theme {
  try {
    const t = localStorage.getItem(THEME_KEY);
    return t === "light" || t === "dark" ? t : "system";
  } catch {
    return "system"; // storage blocked (private window, disabled site data)
  }
}

export function App() {
  const [route, setRoute] = useState<Route>(parseHash);
  const [corpus, setCorpus] = useState<Corpus | null>(null);
  const [corpusError, setCorpusError] = useState<string | null>(null);
  const samples = useMemo(seedSamples, []);
  const [threads, setThreads] = useState<ThreadSummary[] | null>(null);
  const [threadsError, setThreadsError] = useState<string | null>(null);
  // Turns of the threads opened so far, keyed by thread id. Refetched after each run.
  const [loaded, setLoaded] = useState<Record<string, Message[]>>({});
  // A thread id from the URL that the server does not have (deleted, or a stale link).
  const [missing, setMissing] = useState<string | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [selected, setSelected] = useState<number | null>(null);
  const [tab, setTab] = useState<InspectorTab>("calls");
  const [highlight, setHighlight] = useState<string | null>(null);
  const [cite, setCite] = useState<CiteTarget | null>(null);
  // Bumped after every run, so the Memory tab re-reads the long-term list (D19).
  const [memoryVersion, setMemoryVersion] = useState(0);
  // Only the theme choice is kept in the browser (docs/DECISIONS.md, D8). Chats are kept
  // by the server.
  const [theme, setTheme] = useState<Theme>(readTheme);
  const [inspectorOpen, setInspectorOpen] = useState(() => window.innerWidth >= 1180);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const runRef = useRef<Run | null>(null);
  runRef.current = run;

  // ------------------------------------------------------------------ data

  const loadCorpus = useCallback(async () => {
    try {
      setCorpus(await api.getCorpus());
      setCorpusError(null);
    } catch {
      setCorpusError("The server is not answering. Is python web/server.py still running?");
    }
  }, []);

  const loadThreads = useCallback(async () => {
    try {
      setThreads(await api.listThreads());
      setThreadsError(null);
    } catch (e) {
      setThreadsError(e instanceof api.ApiError ? `Chats could not be loaded: ${e.detail}` : "Chats could not be loaded.");
    }
  }, []);

  const loadThread = useCallback(async (id: string): Promise<boolean> => {
    try {
      const t = await api.getThread(id);
      setLoaded((all) => ({ ...all, [id]: t.turns.flatMap(turnMessages) }));
      setMissing((m) => (m === id ? null : m));
      return true;
    } catch (e) {
      if (e instanceof api.ApiError && (e.code === "thread_not_found" || e.code === "bad_thread_id")) setMissing(id);
      return false;
    }
  }, []);

  useEffect(() => {
    loadCorpus();
    loadThreads();
    const onHash = () => setRoute(parseHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [loadCorpus, loadThreads]);

  // The view is keyed off the route, not off the loaded list: right after the first
  // question names a new thread, the route points at it before its turns are fetched.
  const routeId = route.view === "chat" ? route.id : null;
  const sample = samples.find((c) => c.id === routeId) ?? null;
  const threadId = routeId && THREAD_ID.test(routeId) && missing !== routeId ? routeId : null;
  const summary = threads?.find((t) => t.id === threadId) ?? null;
  const chatId = sample?.id ?? threadId;

  useEffect(() => {
    if (threadId && !loaded[threadId] && runRef.current?.chatId !== threadId) loadThread(threadId);
  }, [threadId, loaded, loadThread]);

  useEffect(() => {
    setSelected(null);
    setCite(null);
    setSidebarOpen(false);
  }, [chatId, route.view]);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") delete root.dataset.theme;
    else root.dataset.theme = theme;
    try {
      if (theme === "system") localStorage.removeItem(THEME_KEY);
      else localStorage.setItem(THEME_KEY, theme);
    } catch {
      /* not remembered, which is harmless */
    }
  }, [theme]);

  const papersById = useMemo(() => {
    const m = new Map<string, Paper>();
    (corpus?.papers ?? []).forEach((p) => m.set(p.paper_id, p));
    return m;
  }, [corpus]);

  // ------------------------------------------------------------------ messages

  const messages: Message[] = useMemo(() => {
    const base = sample ? [...sample.messages] : threadId ? [...(loaded[threadId] ?? [])] : [];
    if (run && run.chatId === chatId) {
      base.push({ role: "user", content: run.question, ts: run.startedAt });
      base.push({ role: "assistant", status: "running", content: "", steps: run.steps, ts: run.startedAt });
    }
    return base;
  }, [sample, threadId, loaded, run, chatId]);

  const lastAssistant = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) if (messages[i].role === "assistant") return i;
    return null;
  }, [messages]);
  const inspected = selected ?? lastAssistant;
  const inspectedMessage =
    inspected !== null && messages[inspected]?.role === "assistant"
      ? (messages[inspected] as AssistantMessage)
      : null;

  // ------------------------------------------------------------------ actions

  const send = useCallback(async (raw: string) => {
    const question = raw.trim();
    if (!question || runRef.current) return;
    setNotice(null);
    setCite(null);
    setDraft("");

    // A sample is a fixture, not a thread, so asking from one starts a new chat.
    if (!threadId) go("#/");
    let target: string | null = threadId;
    const startedAt = Date.now() / 1000;
    const steps: Step[] = [];
    setRun({ chatId: target, question, steps: [], startedAt });
    setSelected(null);
    setTab("calls");

    let busy = false;
    let papersAdded = false;
    let final = null as AssistantMessage | null;
    const finish = (m: Omit<AssistantMessage, "role" | "steps" | "ts" | "elapsed_ms">) => {
      final = { role: "assistant", steps: [...steps], ts: Date.now() / 1000, elapsed_ms: Math.round((Date.now() / 1000 - startedAt) * 1000), ...m };
    };

    try {
      await api.ask(question, target, (event, data) => {
        if (event === "thread") {
          // Always the first frame of a run: which thread this question belongs to.
          const t = data as ThreadEvent;
          target = t.thread_id;
          if (t.created) {
            const now = Date.now() / 1000;
            const made: ThreadSummary = { id: t.thread_id, title: t.title, created_at: now, updated_at: now, n_turns: 0 };
            setThreads((ts) => [made, ...(ts ?? []).filter((x) => x.id !== made.id)]);
            setLoaded((all) => ({ ...all, [made.id]: [] }));
            go(`#/chat/${made.id}`);
          }
          setRun((r) => (r ? { ...r, chatId: t.thread_id } : r));
        } else if (event === "step") {
          const step = data as Step;
          steps.push(step);
          const added = (step.result as any)?.papers_added;
          if (Array.isArray(added) && added.length) {
            papersAdded = true;
            loadCorpus(); // the answer may cite a paper the page has not heard of yet
          }
          setRun((r) => (r ? { ...r, steps: [...steps] } : r));
        } else if (event === "busy") {
          busy = true;
        } else if (event === "answer") {
          const a = data as AnswerEvent;
          finish({
            status: "answered",
            content: a.answer ?? "",
            stats: { iterations: a.iterations, tool_calls: a.tool_calls, context_tokens: a.context_tokens, stopped_because: a.stopped_because, run_id: a.run_id },
          });
        } else if (event === "failed") {
          const f = data as FailedEvent;
          finish({ status: "failed", content: f.answer ?? "", error: { error: f.error, detail: f.detail }, stats: { run_id: f.run_id } });
        } else if (event === "memory") {
          // After the answer: what long-term memory wrote from it. Also stored on the turn.
          if (final) final = { ...final, memory: data as MemoryReport };
        }
      });
    } catch (e) {
      if (!final) {
        finish({ status: "failed", content: "", error: { error: "connection_lost", detail: (e as Error).message } });
      }
    }

    if (busy) {
      setDraft(question);
      setNotice("The agent is answering another question, in another tab or window. Try again once it finishes.");
    } else if (target) {
      // The server's copy is the record: it carries the stored notes and stats. If it
      // cannot be read back, keep what the stream showed so the answer is not lost.
      const id = target;
      if (!(await loadThread(id))) {
        const shown: AssistantMessage = final ?? {
          role: "assistant", status: "failed", content: "", steps, ts: Date.now() / 1000,
          error: { error: "no_answer", detail: "the stream ended without an answer" },
        };
        setLoaded((all) => ({ ...all, [id]: [...(all[id] ?? []), { role: "user", content: question, ts: startedAt }, shown] }));
      }
      loadThreads();
    } else {
      // No thread was named, so the server refused before the run (its store failed).
      setDraft(question);
      setNotice(final ? `The question was not run: ${final.error?.detail || final.error?.error}` : "The question was not run.");
    }
    setRun(null);
    setMemoryVersion((v) => v + 1);
    if (papersAdded) loadCorpus();
  }, [threadId, loadCorpus, loadThread, loadThreads]);

  const removeThread = useCallback(async (id: string) => {
    try {
      await api.deleteThread(id);
    } catch (e) {
      setNotice(e instanceof api.ApiError ? `The chat was not deleted: ${e.detail}` : "The chat was not deleted.");
      return;
    }
    setThreads((ts) => (ts ?? []).filter((t) => t.id !== id));
    setLoaded((all) => {
      const rest = { ...all };
      delete rest[id];
      return rest;
    });
    if (threadId === id) go("#/");
  }, [threadId]);

  const openCitation = useCallback((target: CiteTarget) => {
    setCite(target);
    setSelected(target.messageIndex);
    setHighlight(target.chunk);
  }, []);

  const showInSources = useCallback((chunk: string) => {
    setTab("sources");
    setHighlight(chunk);
    setInspectorOpen(true);
    setCite(null);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setCite(null);
        setSidebarOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ------------------------------------------------------------------ layout

  const title = route.view === "corpus" ? "Corpus" : sample ? sample.title : summary ? summary.title : "New chat";
  const missingNotice = routeId && missing === routeId ? "That chat no longer exists. Ask a question to start a new one." : null;

  return (
    <div className={`shell${inspectorOpen && route.view === "chat" ? " with-inspector" : ""}${sidebarOpen ? " sidebar-open" : ""}`}>
      <Sidebar
        threads={threads}
        threadsError={threadsError}
        samples={samples}
        activeId={route.view === "chat" ? chatId : null}
        corpusActive={route.view === "corpus"}
        corpus={corpus}
        runningChatId={run?.chatId ?? null}
        theme={theme}
        onTheme={setTheme}
        onNew={() => {
          setDraft("");
          go("#/");
        }}
        onDelete={removeThread}
        onClose={() => setSidebarOpen(false)}
      />
      <div className="scrim" onClick={() => { setSidebarOpen(false); if (window.innerWidth < 1180) setInspectorOpen(false); }} />

      <main className="centre">
        <header className="topbar">
          <button className="icon-btn only-narrow" onClick={() => setSidebarOpen(true)} aria-label="Open the chat list">
            <Icon name="menu" />
          </button>
          <h1 className="topbar-title" title={title}>{title}</h1>
          {sample && <span className="sample-tag" title="Written by hand to show the interface, not produced by the agent">Sample</span>}
          {route.view === "chat" && (
            <button
              className={`icon-btn${inspectorOpen ? " is-on" : ""}`}
              onClick={() => setInspectorOpen((o) => !o)}
              aria-label={inspectorOpen ? "Hide the agent's working" : "Show the agent's working"}
              aria-pressed={inspectorOpen}
              title="Tool calls, sources and memory"
            >
              <Icon name="panel" />
            </button>
          )}
        </header>

        {route.view === "corpus" ? (
          <CorpusView corpus={corpus} error={corpusError} />
        ) : (
          <ChatView
            messages={messages}
            sample={!!sample}
            loading={!!threadId && !loaded[threadId] && run?.chatId !== threadId}
            papersById={papersById}
            corpus={corpus}
            corpusError={corpusError}
            running={!!run}
            draft={draft}
            setDraft={setDraft}
            onSend={send}
            notice={notice ?? missingNotice}
            onDismissNotice={() => {
              setNotice(null);
              if (missingNotice) go("#/");
            }}
            selected={inspected}
            onSelect={(i) => {
              setSelected(i);
              setTab("calls");
              setInspectorOpen(true);
            }}
            onCite={openCitation}
          />
        )}
      </main>

      {route.view === "chat" && (
        <Inspector
          open={inspectorOpen}
          message={inspectedMessage}
          isLatest={inspected === lastAssistant}
          sample={!!sample}
          papersById={papersById}
          tab={tab}
          setTab={setTab}
          highlight={highlight}
          memoryVersion={memoryVersion}
          onShowSource={showInSources}
          onClose={() => setInspectorOpen(false)}
        />
      )}

      {cite && (
        <CitationCard
          target={cite}
          paper={papersById.get(cite.paper)}
          excerpt={cite.excerpt}
          onClose={() => setCite(null)}
          onShowInSources={showInSources}
        />
      )}
    </div>
  );
}
