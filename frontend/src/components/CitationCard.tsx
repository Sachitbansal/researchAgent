/* The card that opens on a citation: which paper, which passage, where to read it.
   In:  the clicked citation (chunk id, paper id, its screen rectangle), the paper record
        if the corpus knows it, and the excerpt retrieval returned. Out: a positioned card.
        "Show in sources" only where there is a Sources pane to show it in. */

import { useLayoutEffect, useRef, useState } from "react";
import type { Paper } from "../types";
import { who } from "../format";

export interface CiteTarget {
  chunk: string;
  paper: string;
  rect: DOMRect;
  messageIndex: number;
  excerpt?: string;
}

interface Props {
  target: CiteTarget;
  paper?: Paper;
  excerpt?: string;
  onClose: () => void;
  onShowInSources?: (chunk: string) => void;
}

export function CitationCard({ target, paper, excerpt, onClose, onShowInSources }: Props) {
  const card = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: target.rect.left, top: target.rect.bottom + 8 });

  useLayoutEffect(() => {
    const el = card.current;
    if (!el) return;
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    const left = Math.max(12, Math.min(target.rect.left - 16, window.innerWidth - w - 12));
    const below = target.rect.bottom + 8;
    const top = below + h > window.innerHeight - 12 ? Math.max(12, target.rect.top - h - 8) : below;
    setPos({ left, top });
  }, [target]);

  return (
    <>
      <div className="fixed inset-0 z-30" onClick={onClose} />
      <div
        ref={card}
        className="fixed z-31 w-[min(var(--container-card),calc(100vw-24px))] rounded-lg border border-rule-strong bg-raise px-4 py-3 shadow-md"
        role="dialog" aria-label="Cited passage" style={pos}>
        <div className="text-14.5 leading-1.4 font-semibold">{paper ? paper.title : "Not in the corpus as loaded"}</div>
        {paper && <div className="mt-0.5 text-13 text-soft">{who(paper)}</div>}
        {excerpt && <p className="mt-2.5 max-h-[9.6em] overflow-auto text-13.5 leading-1.6"><mark>{excerpt}…</mark></p>}
        <div className="mt-3 flex flex-wrap items-center justify-between gap-x-3 gap-y-1.5 text-13">
          <code className="text-11.5 text-faint">{target.chunk}</code>
          <span className="flex gap-4">
            {onShowInSources && (
              <button className="link-btn" autoFocus onClick={() => onShowInSources(target.chunk)}>Show in sources</button>
            )}
            {paper?.pdf_url && (
              <a href={paper.pdf_url} target="_blank" rel="noopener noreferrer" autoFocus={!onShowInSources}>Open on arXiv</a>
            )}
          </span>
        </div>
      </div>
    </>
  );
}
