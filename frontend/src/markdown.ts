/* The answer renderer, ported from the old web/static/app.js — still no library.
   In:  the agent's markdown answer. Out: escaped HTML with every citation turned into a
        numbered button, plus the reference list those numbers point to. Deliberately
        small: it handles what the agent actually emits, not general markdown. */

export interface Ref {
  n: number;
  paper: string;
  chunks: string[]; // every chunk of this paper the answer cites, in first-seen order
}

export const esc = (s: string) =>
  s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] as string);

/* A chunk id is "<paper_id>__c0007". The agent writes them inline, several in a row and
   often several from one paper, which is unreadable as raw ids. They are numbered per
   answer by paper, in order of first appearance, the way a paper numbers its references.
   Handled before any other inline rule so an id with underscores never reaches the
   emphasis pass. */
const CITE = /\[([A-Za-z0-9._+-]+__[cf]\d+)\]/g;
const CITE_RUN = /(?:\[[A-Za-z0-9._+-]+__[cf]\d+\]\s*){2,}/g;

export const paperOf = (chunkId: string) => chunkId.split("__")[0];

class RefTable {
  private byPaper = new Map<string, Ref>();

  number(chunkId: string): number {
    const paper = paperOf(chunkId);
    let ref = this.byPaper.get(paper);
    if (!ref) {
      ref = { n: this.byPaper.size + 1, paper, chunks: [] };
      this.byPaper.set(paper, ref);
    }
    if (!ref.chunks.includes(chunkId)) ref.chunks.push(chunkId);
    return ref.n;
  }

  list(): Ref[] {
    return [...this.byPaper.values()];
  }
}

function inline(text: string, refs: RefTable): string {
  // "[a__c1][a__c9][b__c2]" cites two papers, not three: one marker per paper per run.
  // The collapsed ids still count, so the Sources view lists every chunk cited.
  const src = String(text).replace(CITE_RUN, (run) => {
    const seen = new Set<string>();
    const kept: string[] = [];
    for (const token of run.match(CITE) ?? []) {
      const id = token.slice(1, -1);
      const paper = paperOf(id);
      if (!seen.has(paper)) {
        seen.add(paper);
        kept.push(token);
      } else {
        refs.number(id);
      }
    }
    return kept.join("");
  });

  let out = esc(src);
  out = out.replace(/`([^`]+)`/g, (_, code) => `<code>${code}</code>`);
  out = out.replace(CITE, (_, id: string) =>
    `<button type="button" class="cite" data-paper="${esc(paperOf(id))}" data-chunk="${esc(id)}" ` +
    `aria-label="Reference ${refs.number(id)}">${refs.number(id)}</button>`);
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return out;
}

const cells = (row: string) => row.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|");

export function renderMarkdown(source: string): { html: string; refs: Ref[] } {
  const refs = new RefTable(); // fresh per answer: numbering never leaks between answers
  const lines = String(source || "").replace(/\r\n/g, "\n").split("\n");
  const html: string[] = [];
  let list: "ul" | "ol" | null = null;
  let para: string[] = [];

  const flushPara = () => {
    if (para.length) {
      html.push(`<p>${inline(para.join(" "), refs)}</p>`);
      para = [];
    }
  };
  const flushList = () => {
    if (list) {
      html.push(`</${list}>`);
      list = null;
    }
  };
  const flush = () => {
    flushPara();
    flushList();
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (/^\s*```/.test(line)) {
      flush();
      const body: string[] = [];
      while (++i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i]);
      html.push(`<pre><code>${esc(body.join("\n"))}</code></pre>`);
      continue;
    }

    if (!line.trim()) {
      flush();
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flush();
      const level = Math.min(heading[1].length + 1, 4);
      html.push(`<h${level}>${inline(heading[2], refs)}</h${level}>`);
      continue;
    }

    if (/^\s*([-*_])\1{2,}\s*$/.test(line)) {
      flush();
      html.push("<hr>");
      continue;
    }

    if (/^\s*\|.*\|\s*$/.test(line) && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1] || "")) {
      flush();
      const head = cells(line).map((c) => `<th>${inline(c.trim(), refs)}</th>`).join("");
      const body: string[] = [];
      i += 1;
      while (/^\s*\|.*\|\s*$/.test(lines[i + 1] || "")) {
        i += 1;
        body.push("<tr>" + cells(lines[i]).map((c) => `<td>${inline(c.trim(), refs)}</td>`).join("") + "</tr>");
      }
      html.push(`<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body.join("")}</tbody></table></div>`);
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      flush();
      html.push(`<blockquote>${inline(line.replace(/^\s*>\s?/, ""), refs)}</blockquote>`);
      continue;
    }

    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    const number = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || number) {
      flushPara();
      const want = bullet ? "ul" : "ol";
      if (list !== want) {
        flushList();
        html.push(`<${want}>`);
        list = want;
      }
      html.push(`<li>${inline((bullet || number)![1], refs)}</li>`);
      continue;
    }

    flushList();
    para.push(line.trim());
  }

  flush();
  return { html: html.join("\n"), refs: refs.list() };
}
