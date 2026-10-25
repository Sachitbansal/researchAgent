/* The answer renderer: hand-written markdown plus KaTeX math, no library (DECISIONS D7).
   In:  the agent's markdown answer, or a memory's plain text. Out: renderMarkdown gives
        escaped HTML with numbered citation buttons, typeset math and the reference list;
        renderMathText gives escaped text with math only; renderMath typesets one span. */

import katex from "katex";

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
// A code span holding nothing but citations (the agent sometimes backticks its ids) is
// read as those citations, not as code.
const CITES_ONLY = /^\s*(?:\[[A-Za-z0-9._+-]+__[cf]\d+\]\s*)+$/;

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

/* KaTeX's output is the only HTML that enters an answer unescaped. KaTeX escapes the TeX
   source itself, `trust` stays off (no \href, \htmlClass, \includegraphics), and a parse
   error renders as the escaped source in a .katex-error span instead of throwing. Errors
   take the theme's danger colour: KaTeX's default #cc0000 fails AA on the dark surface. */
export function renderMath(tex: string, display: boolean): string {
  return katex.renderToString(tex, {
    displayMode: display, throwOnError: false, strict: "ignore", trust: false, errorColor: "var(--color-danger)",
  });
}

/* Code spans and math are cut out before any other inline rule and put back last, so a
   `_`, `*`, `[` or `$` inside them never reaches the citation, emphasis or link passes
   (except a code span of citations only, which is unwrapped into those citations).
   One left-to-right scan, so whichever starts first wins: `$x$` in backticks stays code.
   Inline $…$ follows the usual (pandoc) rule so prices are not math: no space just inside
   either dollar, and no digit right after the closing one ("$5 and $10", "$5-$10"). */
const PROTECTED = new RegExp(
  [
    "`([^`]+)`", // 1: code span
    "\\$\\$([\\s\\S]+?)\\$\\$", // 2: $$display$$
    "\\\\\\[([\\s\\S]+?)\\\\\\]", // 3: \[display\]
    "\\\\\\(([\\s\\S]+?)\\\\\\)", // 4: \(inline\)
    "(?<![\\\\$])\\$(?=[^\\s$])((?:\\\\.|[^$\\\\\\n])+?)(?<=\\S)\\$(?!\\d)", // 5: $inline$
  ].join("|"),
  "g",
);
// Placeholders are private-use characters, stripped from the input first, so the source
// text can never forge one.
const SLOT_OPEN = "";
const SLOT_CLOSE = "";
const SLOT_CHARS = /[]/g;
const SLOT = /(\d+)/g;

function inline(text: string, refs: RefTable): string {
  const slots: string[] = [];
  const keep = (html: string) => `${SLOT_OPEN}${slots.push(html) - 1}${SLOT_CLOSE}`;
  let src = String(text)
    .replace(SLOT_CHARS, "")
    .replace(PROTECTED, (_, code?: string, dd?: string, bd?: string, bi?: string, di?: string) =>
      code !== undefined ? (CITES_ONLY.test(code) ? code.trim() : keep(`<code>${esc(code)}</code>`))
        : dd !== undefined || bd !== undefined ? keep(renderMath((dd ?? bd)!.trim(), true))
          : keep(renderMath((bi ?? di)!.trim(), false)));

  // "[a__c1][a__c9][b__c2]" cites two papers, not three: one marker per paper per run.
  // The collapsed ids still count, so the Sources view lists every chunk cited.
  src = src.replace(CITE_RUN, (run) => {
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
  out = out.replace(CITE, (_, id: string) =>
    `<button type="button" class="cite" data-paper="${esc(paperOf(id))}" data-chunk="${esc(id)}" ` +
    `aria-label="Reference ${refs.number(id)}">${refs.number(id)}</button>`);
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  // A URL holding a protected span (math, code) is not linked: a slot must never land in an href.
  out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s\uE000\uE001]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return out.replace(SLOT, (_, i: string) => slots[Number(i)]);
}

const cells = (row: string) => row.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|");

/* A display-math block that spans lines: "$$" or "\[" opens it, the matching closer ends
   it. Returns the TeX and the index of the closing line, or null if it never closes (then
   the lines are ordinary text; a blank line ends the search, so a stray "$$" cannot
   swallow the paragraphs, and citations, after it). A block that opens and closes on one line is left to the
   inline pass, which renders it in display mode too. */
function mathBlock(lines: string[], i: number): { tex: string; end: number } | null {
  const open = lines[i].match(/^\s*(\$\$|\\\[)(.*)$/);
  if (!open) return null;
  const close = open[1] === "$$" ? "$$" : "\\]";
  if (open[2].includes(close)) return null;
  const body = [open[2]];
  for (let j = i + 1; j < lines.length; j++) {
    if (!lines[j].trim()) return null; // display math has no blank lines: an unclosed opener
    const at = lines[j].indexOf(close);
    if (at >= 0) {
      if (lines[j].slice(at + close.length).trim()) return null; // text after the closer
      body.push(lines[j].slice(0, at));
      return { tex: body.join("\n").trim(), end: j };
    }
    body.push(lines[j]);
  }
  return null;
}

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

    const block = mathBlock(lines, i);
    if (block) {
      flush();
      html.push(`<div class="math-display">${renderMath(block.tex, true)}</div>`);
      i = block.end;
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

/* Plain text with math only: escaped, and $…$ / $$…$$ / \(…\) / \[…\] typeset. For text
   that is answer-like but not an answer (a memory's wording): no citations, no emphasis. */
export function renderMathText(text: string): string {
  const slots: string[] = [];
  const src = String(text || "")
    .replace(SLOT_CHARS, "")
    .replace(PROTECTED, (whole: string, code?: string, dd?: string, bd?: string, bi?: string, di?: string) => {
      const html = code !== undefined ? esc(whole)
        : dd !== undefined || bd !== undefined ? renderMath((dd ?? bd)!.trim(), true)
          : renderMath((bi ?? di)!.trim(), false);
      return `${SLOT_OPEN}${slots.push(html) - 1}${SLOT_CLOSE}`;
    });
  return esc(src).replace(SLOT, (_, i: string) => slots[Number(i)]);
}
