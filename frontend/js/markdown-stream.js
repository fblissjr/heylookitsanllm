// Incremental markdown rendering for a message that is still streaming.
//
// WHY THIS EXISTS. The painter used to re-parse the WHOLE accumulated
// response every animation frame and assign the result to innerHTML,
// destroying and rebuilding the message's entire subtree each time. marked's
// parse is SUPERLINEAR in document length -- measured on a non-repeating
// prose/list/code document, doubling the length costs ~3.2x, not 2x -- so the
// per-frame cost grew faster than the response did, and past roughly 16k
// characters a single parse alone exceeded a 120Hz frame budget before
// DOMPurify, the DOM rebuild and layout were counted. On a phone that reads
// as a saturated main thread for the length of a long generation.
//
// THE FIX. Split the accumulated text at the latest SAFE boundary, render
// each new segment ONCE into a committed prefix, and re-render only the tail
// on subsequent paints. Per-paint cost becomes proportional to the tail --
// bounded by the largest single block (one paragraph, one table, one fenced
// code block), not by the response. Prefix upkeep is linear overall because a
// segment is rendered exactly once and its NODES are left alone afterwards,
// so the committed part of the message is never re-laid-out either.
//
// WHAT MAKES A BOUNDARY SAFE. Only that re-parsing [0,b) and [b,end)
// separately yields the same HTML as parsing the whole -- i.e. no markdown
// construct can span it. The rules in `isHardBlockStart` and the fence
// tracking below are the conditions under which that holds; they are REASONED
// here and PROVEN in tests/e2e/render.mjs, which grows generated documents one
// chunk at a time through this class and diffs the result against a
// whole-document render. If you touch the boundary rules, make that check fail
// first -- the seam bugs this replaces are invisible to any check that only
// renders a finished document.
//
// Everything still goes through markdown.js: this module chooses WHERE to cut,
// never how to render, and never sanitizes anything itself.

import { renderMarkdown } from './markdown.js';

// An opening fence: up to 3 leading spaces, then 3+ backticks or tildes.
const FENCE_OPEN = /^ {0,3}(`{3,}|~{3,})/;
// A list marker at column 0 (indentation is already excluded by the caller).
const LIST_ITEM = /^(?:[-*+]|\d{1,9}[.)])(?:[ \t]|$)/;
// A link-reference definition or a GFM footnote definition. Both can be USED
// by text arbitrarily far below them, so their presence makes every split
// unsafe -- see the fallback in render().
const LINK_DEF = /^ {0,3}\[[^\]]*\]:/;
// CommonMark HTML blocks of types 1-5 END ON A CLOSING CONDITION, NOT ON A
// BLANK LINE -- so, exactly like a link-reference definition, they reach
// forward past the only thing this scanner treats as a block break:
//
//   type 1  <pre> <script> <style> <textarea>   until the matching close tag
//   type 2  <!--                                until -->
//   type 3  <?                                  until ?>
//   type 4  <!LETTER                            until >
//   type 5  <![CDATA[                           until ]]>
//
// `<pre>\nfoo\n\nbar\n</pre>` renders as ONE html token whole, but split at
// `bar` -- unindented, at column 0, after a blank line, so isHardBlockStart
// says yes -- it renders as two, and the committed prefix is never revisited,
// so the seam is permanent. Type 6 (<div> and friends) DOES end at a blank
// line and is deliberately not here.
//
// Treated as `unsafe` rather than tracked open-to-close, which is the
// conservative half of the choice: the cost is that one message loses
// incremental rendering, the alternative is five open/close pairs interacting
// with fence state inside the module whose whole claim is that its cut is
// provably safe. Per-block tracking is the refinement if a real case ever
// makes the fallback expensive.
const HTML_BLOCK_OPEN = /^ {0,3}(?:<(?:pre|script|style|textarea)[\s>/]|<!--|<\?|<![A-Za-z]|<!\[CDATA\[)/i;

// One reusable inert parser. <template> content is never scripted or
// connected, and the HTML reaching it is already sanitized by renderMarkdown.
const parser = document.createElement('template');
function parseNodes(html) {
  parser.innerHTML = html;
  // Snapshot before appending: append() MOVES the nodes out of the fragment.
  return Array.from(parser.content.childNodes);
}

export class MarkdownStream {
  // `el` is the container the message's rendered content lives in. This class
  // owns every child of it for as long as the stream runs.
  constructor(el) {
    this.el = el;
    this.reset();
  }

  reset() {
    this.el.replaceChildren();
    this.text = '';
    // Line scanner state. The scan is incremental: text only ever grows, so
    // each line is classified exactly once.
    this.scanPos = 0;
    this.inFence = false;
    this.fenceChar = '';
    this.fenceLen = 0;
    // The start of the document behaves like the position after a blank line.
    this.prevBlank = true;
    this.boundary = 0;
    this.unsafe = false;
    this.prefixNodes = 0;
  }

  // Render `text` (the full accumulated message) into the container.
  render(text) {
    if (text === this.text) return;
    const prevBoundary = this.boundary;
    this.text = text;
    if (!this.unsafe) this.scan(text);

    if (this.unsafe) {
      // A reference definition resolves against the WHOLE document, so no
      // split can be shown safe. Fall back to the whole-document render --
      // correctness over speed, and rare enough in model output to be worth
      // paying for outright rather than approximating.
      this.el.innerHTML = renderMarkdown(text);
      this.prefixNodes = 0;
      this.boundary = 0;
      return;
    }

    if (this.boundary > prevBoundary) {
      // Commit [prevBoundary, boundary) as prefix. Both ends are safe splits,
      // so appending this segment's HTML equals re-rendering the whole prefix
      // -- which is what keeps prefix upkeep linear instead of quadratic.
      const nodes = parseNodes(renderMarkdown(text.slice(prevBoundary, this.boundary)));
      this.replaceTail(nodes);
      this.prefixNodes += nodes.length;
    }
    this.replaceTail(parseNodes(renderMarkdown(text.slice(this.boundary))));
  }

  // Swap everything after the committed prefix for `nodes`. Prefix nodes are
  // never touched, so the browser never re-lays-out the settled part of the
  // message.
  replaceTail(nodes) {
    const el = this.el;
    while (el.childNodes.length > this.prefixNodes) el.lastChild.remove();
    el.append(...nodes);
  }

  // Advance over every line COMPLETED since the last call. An unterminated
  // trailing line is deliberately not classified: "``" is not a fence until
  // its newline arrives, and a line's meaning can still change while it grows.
  scan(text) {
    let i = this.scanPos;
    for (;;) {
      const nl = text.indexOf('\n', i);
      if (nl === -1) break;
      let end = nl;
      if (end > i && text[end - 1] === '\r') end--;
      this.consumeLine(text.slice(i, end), i);
      i = nl + 1;
    }
    this.scanPos = i;
  }

  consumeLine(line, start) {
    if (this.inFence) {
      if (this.closesFence(line)) this.inFence = false;
      // A blank line INSIDE a fence is content, not a block break.
      this.prevBlank = false;
      return;
    }
    if (line.trim() === '') {
      this.prevBlank = true;
      return;
    }
    if (LINK_DEF.test(line) || HTML_BLOCK_OPEN.test(line)) this.unsafe = true;
    if (this.prevBlank && this.isHardBlockStart(line)) this.boundary = start;
    const fence = FENCE_OPEN.exec(line);
    if (fence) {
      this.inFence = true;
      this.fenceChar = fence[1][0];
      this.fenceLen = fence[1].length;
    }
    this.prevBlank = false;
  }

  // Can this line start a block that nothing above it can reach into?
  isHardBlockStart(line) {
    // Indented: an indented code block, or a continuation of a list item
    // above -- either way the construct spans the split.
    if (/^[ \t]/.test(line)) return false;
    // A list marker after a blank line CONTINUES a list above it as one loose
    // list; split, it would render as two lists instead.
    if (LIST_ITEM.test(line)) return false;
    // Same shape for block quotes.
    if (line[0] === '>') return false;
    return true;
  }

  // CommonMark: a closing fence matches the opening character, is at least as
  // long, and carries no info string.
  closesFence(line) {
    const m = /^ {0,3}(`{3,}|~{3,})[ \t]*$/.exec(line);
    return Boolean(m) && m[1][0] === this.fenceChar && m[1].length >= this.fenceLen;
  }
}

// The thinking box is plain text in a single node, but rewriting the whole
// string every paint is the same O(response) write the markdown path had.
// Thinking is append-only, so append the delta instead. Returns the new
// written length, which the caller carries for the next paint.
export function appendPlainText(el, full, writtenLen) {
  const node = el.firstChild;
  if (node && node.nodeType === Node.TEXT_NODE && full.length >= writtenLen
      && node.length === writtenLen) {
    if (full.length > writtenLen) node.appendData(full.slice(writtenLen));
  } else {
    el.textContent = full;
  }
  return full.length;
}
