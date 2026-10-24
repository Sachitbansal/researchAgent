/* The card that opens on a citation: which paper, which passage, where to read it.
   In:  the clicked citation (chunk id, paper id, its screen rectangle), the paper record
        if the corpus knows it, and the excerpt retrieval returned. Out: a positioned card. */

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
  onShowInSources: (chunk: string) => void;
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
      <div className="card-catcher" onClick={onClose} />
      <div ref={card} className="cite-card" role="dialog" aria-label="Cited passage" style={pos}>
        <div className="card-title">{paper ? paper.title : "Not in the corpus as loaded"}</div>
        {paper && <div className="card-who">{who(paper)}</div>}
        {excerpt && <p className="card-excerpt"><mark>{excerpt}…</mark></p>}
        <div className="card-foot">
          <code>{target.chunk}</code>
          <span className="card-actions">
            <button className="link-btn" autoFocus onClick={() => onShowInSources(target.chunk)}>Show in sources</button>
            {paper?.pdf_url && <a href={paper.pdf_url} target="_blank" rel="noopener noreferrer">Open on arXiv</a>}
          </span>
        </div>
      </div>
    </>
  );
}
