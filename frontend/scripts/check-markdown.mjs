/* Unit checks for the answer renderer (src/markdown.ts): math, citations, escaping.
   In:  nothing; run `npm run check:markdown` from frontend/. It compiles markdown.ts with
        tsc into a throwaway folder inside frontend/ (so `katex` resolves from node_modules).
   Out: one line per check and exit code 0, or the first failure and exit code 1. */

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(dirname(fileURLToPath(import.meta.url)));
const out = mkdtempSync(join(here, ".md-check-"));

try {
  execFileSync(join(here, "node_modules", ".bin", "tsc"), [
    join(here, "src", "markdown.ts"), "--ignoreConfig", "--outDir", out, "--module", "esnext",
    "--target", "es2022", "--moduleResolution", "bundler", "--skipLibCheck", "--strict",
  ], { stdio: "inherit" });
  const md = await import(pathToFileURL(join(out, "markdown.js")).href);
  let n = 0;
  const check = (name, fn) => {
    fn();
    n += 1;
    console.log(`ok  ${name}`);
  };
  const html = (s) => md.renderMarkdown(s).html;
  const count = (s, sub) => s.split(sub).length - 1;
  // KaTeX keeps the TeX it was given in a MathML annotation: proof nothing upstream changed it.
  const tex = (s) => [...s.matchAll(/<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>/g)].map((m) => m[1]);

  check("inline math", () => {
    const h = html("The loss is $L = \\alpha x_i$ per token.");
    assert.equal(count(h, 'class="katex"'), 1);
    assert.ok(!h.includes("katex-display"));
    assert.deepEqual(tex(h), ["L = \\alpha x_i"]);
  });

  check("display math: a $$ block over several lines", () => {
    const h = html("Before.\n\n$$\n\\mathcal{L} = \\sum_{i=1}^{N} f_i P_i\n$$\n\nAfter.");
    assert.ok(h.includes('<div class="math-display">'));
    assert.ok(h.includes("katex-display"));
    assert.deepEqual(tex(h), ["\\mathcal{L} = \\sum_{i=1}^{N} f_i P_i"]);
    assert.ok(h.startsWith("<p>Before.</p>") && h.endsWith("<p>After.</p>"));
  });

  check("display math: $$…$$ and \\[…\\] on one line, \\(…\\) inline", () => {
    const h = html("Here $$a^2 + b^2$$ and \\(c_1\\) then \\[d_{2}\\] end.");
    assert.equal(count(h, "katex-display"), 2);
    assert.deepEqual(tex(h), ["a^2 + b^2", "c_1", "d_{2}"]);
  });

  check("display math: a \\[ block over several lines", () => {
    const h = html("\\[\n\\frac{a}{b}\n\\]");
    assert.ok(h.startsWith('<div class="math-display">'));
    assert.deepEqual(tex(h), ["\\frac{a}{b}"]);
  });

  check("an unclosed $$ does not swallow later paragraphs or their citations", () => {
    const { html: h, refs } = md.renderMarkdown("$$\nbroken\n\nNext claim [p1__c0001].\n\n$$x$$");
    assert.equal(refs.length, 1);
    assert.equal(count(h, 'class="cite"'), 1);
  });

  check("underscores, asterisks and brackets inside math are not mangled", () => {
    const h = html("Take $a_{[i]} * b_j * c_k$ and $x_1 + x_2$ with *emphasis* outside.");
    assert.deepEqual(tex(h), ["a_{[i]} * b_j * c_k", "x_1 + x_2"]);
    assert.equal(count(h, "<em>"), 1);
    assert.ok(h.includes("<em>emphasis</em>"));
  });

  check("a citation-like id inside math is not a citation", () => {
    const { html: h, refs } = md.renderMarkdown("See $p[a__c1]$ here.");
    assert.equal(count(h, 'class="cite"'), 0);
    assert.equal(refs.length, 0);
  });

  check("currency is not math", () => {
    for (const s of ["It costs $5 and $10 a month.", "Between $5-$10.", "From $5 to $10, or $20."]) {
      const h = html(s);
      assert.ok(!h.includes("katex"), s);
      assert.ok(h.includes("$5"), s);
    }
  });

  check("citations next to math are still numbered, runs still collapse", () => {
    const { html: h, refs } = md.renderMarkdown(
      "Gain $\\alpha$ [p1__c0001] and $\\beta$[p2__c0003][p1__c0002] then [p1__c0004]$\\gamma$.");
    assert.equal(count(h, 'class="katex"'), 3);
    assert.deepEqual(refs.map((r) => r.n), [1, 2]);
    assert.deepEqual(refs[0].chunks, ["p1__c0001", "p1__c0002", "p1__c0004"]);
    assert.deepEqual(refs[1].chunks, ["p2__c0003"]);
    // three markers: p1, the run [p2][p1] keeps one per paper, then p1 again
    assert.equal(count(h, 'class="cite"'), 4);
    assert.ok(h.includes('aria-label="Reference 2">2</button>'));
  });

  check("$ inside backticks and fences stays code", () => {
    const h = html("Use `$x$` literally.\n\n```\nprice = $5 and $x$\n```");
    assert.ok(!h.includes("katex"));
    assert.ok(h.includes("<code>$x$</code>"));
    assert.ok(h.includes("<pre><code>price = $5 and $x$</code></pre>"));
  });

  check("backticked citations are plain citations, not chips inside <code>", () => {
    const { html: h, refs } = md.renderMarkdown("Shown here `[p1__c0001]` and `[p2__c0002] [p1__c0003]`.");
    assert.ok(!h.includes("<code>"));
    assert.equal(count(h, 'class="cite"'), 3);
    assert.deepEqual(refs.map((r) => r.paper), ["p1", "p2"]);
    assert.deepEqual(refs[0].chunks, ["p1__c0001", "p1__c0003"]);
  });

  check("a code span with a citation among other text stays code", () => {
    const { html: h, refs } = md.renderMarkdown("Run `grep [p1__c0001] chunks.jsonl` to see it.");
    assert.ok(h.includes("<code>grep [p1__c0001] chunks.jsonl</code>"));
    assert.equal(refs.length, 0);
  });

  check("radicals, fractions, accents and matrices render without errors", () => {
    const h = html("$\\sqrt{d_k}$, $\\frac{QK^T}{\\sqrt{d}}$, $\\hat{y}$ and $$\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}$$");
    assert.equal(count(h, "katex-error"), 0);
    assert.equal(count(h, 'class="katex"'), 4);
  });

  check("escaping: raw HTML never passes, in or out of math", () => {
    const h = html("<script>alert(1)</script> and $<img src=x onerror=alert(1)>$ and $x$ <b>b</b>");
    assert.ok(!/<script|<img|<b>/i.test(h));
    assert.ok(h.includes("&lt;script&gt;"));
  });

  check("escaping: a TeX error shows the escaped source, never throws", () => {
    const h = html("Broken $\\frac{a}{$ and $\\nosuchmacro{<i>}$ here.");
    assert.ok(h.includes("katex-error"));
    assert.ok(!h.includes("<i>"));
  });

  check("escaping: \\href stays untrusted", () => {
    const h = html("$\\href{javascript:alert(1)}{x}$");
    assert.ok(!/href="javascript/i.test(h));
  });

  check("escaping: placeholder characters in the source cannot forge a slot", () => {
    const h = html("$x$ then \uE0000\uE001 and `c`");
    assert.equal(count(h, 'class="katex"'), 1);
    assert.ok(!h.includes("\uE000") && !h.includes("\uE001"));
    assert.equal(count(h, "<code>c</code>"), 1);
  });

  check("escaping: math or code inside a link URL never lands in an href", () => {
    for (const s of ["See [paper](https://x.org/a$b$c) now.", "See [paper](https://x.org/a`b`c) now."]) {
      const h = html(s);
      assert.ok(!/href="[^"]*[<\uE000\uE001]/.test(h), s);
    }
    assert.ok(html("A [plain link](https://x.org/p_1) works.").includes('<a href="https://x.org/p_1"'));
  });

  check("renderMathText: escapes, typesets math, nothing else", () => {
    const h = md.renderMathText("Uses $k_i$ of <b>N</b> [p1__c0001] *x* for $5 and $10");
    assert.deepEqual(tex(h), ["k_i"]);
    assert.ok(h.includes("&lt;b&gt;N&lt;/b&gt;"));
    assert.ok(h.includes("[p1__c0001] *x* for $5 and $10"));
  });

  console.log(`\n${n} checks passed`);
} finally {
  rmSync(out, { recursive: true, force: true });
}
