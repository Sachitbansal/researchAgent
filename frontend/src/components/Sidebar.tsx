/* Left pane: new chat, the stored chats (newest first), the samples, the corpus entry,
   and at the foot the user menu (Memories, theme).
   In:  the server's thread list, the sample fixtures, the active route, the corpus (for
        its paper count), callbacks. Out: navigation only; it never fetches. Delete asks
        once, inline, before it deletes. Samples sit in their own group, tagged as such. */

import { useState } from "react";
import type { Chat, Corpus, ThreadSummary } from "../types";
import { plural, relTime } from "../format";
import { Icon } from "../icons";
import { UserMenu, type Theme } from "./UserMenu";

interface Props {
  threads: ThreadSummary[] | null;
  threadsError: string | null;
  samples: Chat[];
  activeId: string | null;
  corpusActive: boolean;
  memoriesActive: boolean;
  corpus: Corpus | null;
  runningChatId: string | null;
  theme: Theme;
  onTheme: (t: Theme) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onClose: () => void;
  open: boolean; // the drawer, below the md breakpoint
}

const ROW = "relative flex items-center rounded-md transition-colors";
const HEAD = "px-3 pt-4 pb-1 text-12 font-semibold text-faint";
const EMPTY = "px-3 py-1 text-13 text-faint";
const TITLE = "block truncate text-14";
const META = "flex items-center gap-1.5 text-12 text-faint";
const LINK = "min-w-0 flex-1 rounded-md py-1.5 pr-1 pl-3 text-ink no-underline focus-visible:-outline-offset-2";

export function Sidebar(p: Props) {
  const [confirming, setConfirming] = useState<string | null>(null);
  const n = p.corpus?.papers.length ?? 0;

  return (
    <nav
      className={`flex min-h-0 flex-col gap-0.5 border-r border-rule bg-side px-2 pt-3 pb-2
        max-md:fixed max-md:inset-y-0 max-md:left-0 max-md:z-25 max-md:w-[min(var(--width-side),86vw)]
        max-md:transition-[translate,visibility] max-md:duration-200 ${
        p.open ? "max-md:visible max-md:translate-x-0 max-md:shadow-md" : "max-md:invisible max-md:-translate-x-full"}`}
      aria-label="Chats"
    >
      <div className="flex min-h-9 items-center gap-2 px-2 pt-1 pb-3">
        <span className="brand-mark" aria-hidden="true" />
        <span className="flex-1 text-15 font-semibold">sciagent</span>
        <button className="icon-btn hidden max-md:inline-grid" onClick={p.onClose} aria-label="Close the chat list">
          <Icon name="close" />
        </button>
      </div>

      <button
        className="mb-2 flex min-h-9 items-center gap-2 rounded-md border border-rule-strong bg-raise px-3 py-2 font-medium shadow-sm transition-colors hover:border-faint active:bg-hover"
        onClick={p.onNew}
      >
        <Icon name="plus" size={16} />
        New chat
      </button>

      <a
        className={`flex min-h-9 items-center gap-2 rounded-md px-3 py-2 text-ink no-underline transition-colors focus-visible:-outline-offset-2 ${
          p.corpusActive ? "bg-selected font-medium" : "hover:bg-hover"}`}
        href="#/corpus"
        aria-current={p.corpusActive ? "page" : undefined}
      >
        <Icon name="books" size={17} />
        <span className="flex-1">Corpus</span>
        {p.corpus && <span className="text-12 text-faint tabular-nums">{n}</span>}
      </a>

      {/* Stored chats, then the samples: one scroll area, so neither group can starve the other. */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className={HEAD}>Chats</div>
        <ul>
          {p.threadsError && <li className={EMPTY}>{p.threadsError}</li>}
          {!p.threadsError && p.threads === null && <li className={EMPTY}>Loading…</li>}
          {!p.threadsError && p.threads?.length === 0 && <li className={EMPTY}>No chats yet. Ask a question to start one.</li>}
          {(p.threads ?? []).map((t) => (
            <li key={t.id} className={`group ${ROW} ${t.id === p.activeId ? "bg-selected" : "hover:bg-hover"}`}>
              {confirming === t.id ? (
                <div className="confirm">
                  <span>Delete this chat?</span>
                  <button className="confirm-yes" onClick={() => { setConfirming(null); p.onDelete(t.id); }}>Delete</button>
                  <button className="confirm-no" onClick={() => setConfirming(null)}>Keep</button>
                </div>
              ) : (
                <>
                  <a href={`#/chat/${t.id}`} className={LINK} title={t.title} aria-current={t.id === p.activeId ? "page" : undefined}>
                    <span className={`${TITLE}${t.id === p.activeId ? " font-medium" : ""}`}>
                      {t.id === p.runningChatId && <span className="live-dot" aria-label="answering" />}
                      {t.title}
                    </span>
                    <span className={META}>{relTime(t.updated_at)}, {plural(t.n_turns, "question")}</span>
                  </a>
                  <button
                    className="icon-btn mr-1 size-[30px] opacity-0 group-focus-within:opacity-100 group-hover:opacity-100 disabled:opacity-40 max-md:size-11 max-md:opacity-100 pointer-coarse:opacity-100"
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
            <div className={HEAD}>Samples</div>
            <ul>
              {p.samples.map((c) => (
                <li key={c.id} className={`${ROW} ${c.id === p.activeId ? "bg-selected" : "hover:bg-hover"}`}>
                  <a href={`#/chat/${c.id}`} className={LINK} title={c.title} aria-current={c.id === p.activeId ? "page" : undefined}>
                    <span className={`${TITLE}${c.id === p.activeId ? " font-medium" : ""}`}>{c.title}</span>
                    <span className={META}>
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

      <div className="mt-1 border-t border-rule pt-2">
        <UserMenu theme={p.theme} onTheme={p.onTheme} memoriesActive={p.memoriesActive} />
      </div>
    </nav>
  );
}
