/* Left pane: new chat, the stored chats (newest first), the samples, the corpus entry,
   the theme switch.
   In:  the server's thread list, the sample fixtures, the active route, the corpus (for
        its paper count), callbacks. Out: navigation only; it never fetches. Delete asks
        once, inline, before it deletes. Samples sit in their own group, tagged as such. */

import { useState } from "react";
import type { Chat, Corpus, ThreadSummary } from "../types";
import { plural, relTime } from "../format";
import { Icon } from "../icons";

type Theme = "system" | "light" | "dark";
const NEXT: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };
const THEME_LABEL: Record<Theme, string> = { system: "Theme follows the system", light: "Light theme", dark: "Dark theme" };
const THEME_ICON: Record<Theme, string> = { system: "auto", light: "sun", dark: "moon" };

interface Props {
  threads: ThreadSummary[] | null;
  threadsError: string | null;
  samples: Chat[];
  activeId: string | null;
  corpusActive: boolean;
  corpus: Corpus | null;
  runningChatId: string | null;
  theme: Theme;
  onTheme: (t: Theme) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onClose: () => void;
}

export function Sidebar(p: Props) {
  const [confirming, setConfirming] = useState<string | null>(null);
  const n = p.corpus?.papers.length ?? 0;

  return (
    <nav className="sidebar" aria-label="Chats">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true" />
        <span className="brand-name">sciagent</span>
        <button className="icon-btn only-narrow" onClick={p.onClose} aria-label="Close the chat list">
          <Icon name="close" />
        </button>
      </div>

      <button className="new-chat" onClick={p.onNew}>
        <Icon name="plus" size={16} />
        New chat
      </button>

      <a className={`nav-item${p.corpusActive ? " is-on" : ""}`} href="#/corpus" aria-current={p.corpusActive ? "page" : undefined}>
        <Icon name="books" size={17} />
        <span className="nav-label">Corpus</span>
        {p.corpus && <span className="nav-count">{n}</span>}
      </a>

      <div className="chat-groups">
        <div className="chat-list-head">Chats</div>
        <ul className="chat-list">
          {p.threadsError && <li className="chat-empty">{p.threadsError}</li>}
          {!p.threadsError && p.threads === null && <li className="chat-empty">Loading…</li>}
          {!p.threadsError && p.threads?.length === 0 && <li className="chat-empty">No chats yet. Ask a question to start one.</li>}
          {(p.threads ?? []).map((t) => (
            <li key={t.id} className={`chat-row${t.id === p.activeId ? " is-on" : ""}`}>
              {confirming === t.id ? (
                <div className="confirm">
                  <span>Delete this chat?</span>
                  <button className="confirm-yes" onClick={() => { setConfirming(null); p.onDelete(t.id); }}>Delete</button>
                  <button className="confirm-no" onClick={() => setConfirming(null)}>Keep</button>
                </div>
              ) : (
                <>
                  <a href={`#/chat/${t.id}`} className="chat-link" title={t.title} aria-current={t.id === p.activeId ? "page" : undefined}>
                    <span className="chat-title">
                      {t.id === p.runningChatId && <span className="live-dot" aria-label="answering" />}
                      {t.title}
                    </span>
                    <span className="chat-meta">{relTime(t.updated_at)}, {plural(t.n_turns, "question")}</span>
                  </a>
                  <button
                    className="icon-btn chat-delete"
                    onClick={() => setConfirming(t.id)}
                    aria-label={`Delete chat: ${t.title}`}
                    disabled={t.id === p.runningChatId}
                  >
                    <Icon name="trash" size={16} />
                  </button>
                </>
              )}
            </li>
          ))}
        </ul>

        {p.samples.length > 0 && (
          <>
            <div className="chat-list-head">Samples</div>
            <ul className="chat-list">
              {p.samples.map((c) => (
                <li key={c.id} className={`chat-row${c.id === p.activeId ? " is-on" : ""}`}>
                  <a href={`#/chat/${c.id}`} className="chat-link" title={c.title} aria-current={c.id === p.activeId ? "page" : undefined}>
                    <span className="chat-title">{c.title}</span>
                    <span className="chat-meta">
                      <span className="sample-tag">Sample</span>
                      {plural(Math.ceil(c.messages.length / 2), "question")}
                    </span>
                  </a>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>

      <div className="sidebar-foot">
        <button className="theme-btn" onClick={() => p.onTheme(NEXT[p.theme])} title="Change theme">
          <Icon name={THEME_ICON[p.theme]} size={16} />
          {THEME_LABEL[p.theme]}
        </button>
      </div>
    </nav>
  );
}
