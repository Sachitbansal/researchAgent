/* The three-pane shell: chats on the left, the conversation in the centre, the agent's
   working on the right (docs/DECISIONS.md, D3).
   In:  the API in api.ts, the URL hash (#/chat/<id>, #/corpus), and the sample fixtures.
   Out: the page. Owns all state. Chats live only in memory — storage is undecided — and
        one run at a time, mirroring the server's single agent. */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "./api";
import type { AnswerEvent, AssistantMessage, Chat, Corpus, FailedEvent, Message, Paper, Step } from "./types";
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
  chatId: string;
  question: string;
  steps: Step[];
  startedAt: number;
}

const TITLE_CHARS = 72;

/* The first question, on one line and cut at a word boundary, as the chat's name. */
function titleFrom(question: string): string {
  const flat = question.split(/\s+/).join(" ").trim();
  if (flat.length <= TITLE_CHARS) return flat;
  const cut = flat.slice(0, TITLE_CHARS).replace(/\s+\S*$/, "") || flat.slice(0, TITLE_CHARS);
  return `${cut.replace(/[,.;:]$/, "")}…`;
}

const newId = () => Math.random().toString(16).slice(2, 10) + Date.now().toString(16);

/* Samples are rebased onto "now" so the sidebar reads as recent, not as a fixed date. */
function seedChats(): Chat[] {
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
  const [chats, setChats] = useState<Chat[]>(seedChats);
  const [run, setRun] = useState<Run | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [selected, setSelected] = useState<number | null>(null);
  const [tab, setTab] = useState<InspectorTab>("calls");
  const [highlight, setHighlight] = useState<string | null>(null);
  const [cite, setCite] = useState<CiteTarget | null>(null);
  // Only the theme choice is stored in the browser (docs/DECISIONS.md, D8). Chats are not.
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

  useEffect(() => {
    loadCorpus();
    const onHash = () => setRoute(parseHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [loadCorpus]);

  const routeId = route.view === "chat" ? route.id : null;
  const chat = chats.find((c) => c.id === routeId) ?? null;
  // A hash naming a chat this page doesn't hold (say, after a reload) is a new chat.
  const chatId = chat?.id ?? null;

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
    const base = chat ? [...chat.messages] : [];
    if (run && run.chatId === chatId) {
      base.push({ role: "user", content: run.question, ts: run.startedAt });
      base.push({ role: "assistant", status: "running", content: "", steps: run.steps, ts: run.startedAt });
    }
    return base;
  }, [chat, run, chatId]);

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

    let id = chatId;
    if (!id) {
      const now = Date.now() / 1000;
      id = newId();
      const created: Chat = { id, title: titleFrom(question), created_at: now, updated_at: now, messages: [] };
      setChats((cs) => [created, ...cs]);
      go(`#/chat/${id}`);
    }
    const target = id;
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
      await api.ask(question, (event, data) => {
        if (event === "step") {
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
      // A chat created just for this question holds nothing; don't leave it in the list.
      setChats((cs) => cs.filter((c) => c.id !== target || c.messages.length));
      if (!chatId) go("#/");
    } else {
      const done: AssistantMessage = final ?? {
        role: "assistant", status: "failed", content: "", steps, ts: Date.now() / 1000,
        error: { error: "no_answer", detail: "the stream ended without an answer" },
      };
      setChats((cs) => cs.map((c) => (c.id === target
        ? { ...c, updated_at: Date.now() / 1000, messages: [...c.messages, { role: "user", content: question, ts: startedAt }, done] }
        : c)));
    }
    setRun(null);
    if (papersAdded) loadCorpus();
  }, [chatId, loadCorpus]);

  const removeChat = useCallback((id: string) => {
    setChats((cs) => cs.filter((c) => c.id !== id));
    if (chatId === id) go("#/");
  }, [chatId]);

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

  const sorted = useMemo(() => [...chats].sort((a, b) => b.updated_at - a.updated_at), [chats]);
  const title = route.view === "corpus" ? "Corpus" : chat ? chat.title : "New chat";

  return (
    <div className={`shell${inspectorOpen && route.view === "chat" ? " with-inspector" : ""}${sidebarOpen ? " sidebar-open" : ""}`}>
      <Sidebar
        chats={sorted}
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
        onDelete={removeChat}
        onClose={() => setSidebarOpen(false)}
      />
      <div className="scrim" onClick={() => { setSidebarOpen(false); if (window.innerWidth < 1180) setInspectorOpen(false); }} />

      <main className="centre">
        <header className="topbar">
          <button className="icon-btn only-narrow" onClick={() => setSidebarOpen(true)} aria-label="Open the chat list">
            <Icon name="menu" />
          </button>
          <h1 className="topbar-title" title={title}>{title}</h1>
          {chat?.sample && <span className="sample-tag" title="Written by hand to show the interface, not produced by the agent">Sample</span>}
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
            sample={!!chat?.sample}
            papersById={papersById}
            corpus={corpus}
            corpusError={corpusError}
            running={!!run}
            draft={draft}
            setDraft={setDraft}
            onSend={send}
            notice={notice}
            onDismissNotice={() => setNotice(null)}
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
          papersById={papersById}
          tab={tab}
          setTab={setTab}
          highlight={highlight}
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
