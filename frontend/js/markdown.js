// The only sanctioned path for model/user text -> HTML. Never bypass,
// never double-sanitize.
//
// RAW HTML IN MODEL TEXT IS SHOWN LITERALLY, NEVER RENDERED (v1.79.5).
// marked passes raw HTML through, and DOMPurify then DELETES any tag outside
// its allowlist while keeping the tag's content -- so a model writing
// `<d>tag</d>` rendered as "tag" and the tags vanished silently, while Copy
// (which reads the stored text) still showed them. Same root cause mangled
// plain prose: "a <b and c> d" parsed as an inline <b>. Escaping at the
// renderer makes the rendered message match the stored text for EVERY tag
// rather than only for the ones DOMPurify happens to drop. Model output is
// markdown, not HTML; a model that wants HTML shown as HTML fences it, and
// fenced/inline code was never affected. DOMPurify stays as the backstop for
// everything marked's other renderers emit (link hrefs, image srcs).

import { marked } from './vendor/marked.esm.js';
import DOMPurify from './vendor/purify.es.mjs';

marked.use({ gfm: true, breaks: true });

// One override covers BOTH token levels: marked's block parser and its inline
// parser dispatch `case "html"` to this same renderer method.
marked.use({ renderer: { html: ({ text }) => escapeHtml(text) } });

// LINK AND IMAGE URLS ARE SCHEME-CHECKED HERE, and this is the PRIMARY guard,
// not a nicety. marked does not filter URL schemes -- verified on 18.0.11, it
// emits `<a href="javascript:alert(1)">` for FOUR different markdown spellings
// (inline link, image, autolink, reference link). Before this, the comment
// above calling DOMPurify "the backstop for link hrefs, image srcs" was
// aspirational: DOMPurify was the SOLE guard. It stays as the second layer --
// it has been attacked by professionals for a decade and this function has
// not -- but the decision about what a rendered link may point at belongs in
// the file that owns text->HTML, where it can be read and tested.
//
// A renderer returning `false` falls back to marked's own implementation
// (verified on 18.0.11; returning `''` does NOT fall back -- it drops the
// content silently, which is how the accept path could have been written
// wrong without any test noticing).
const SAFE_LINK_SCHEMES = new Set(['http:', 'https:', 'mailto:', 'tel:']);
const SAFE_IMAGE_SCHEMES = new Set(['http:', 'https:']);
const SCHEME_RE = /^([A-Za-z][A-Za-z0-9+.-]*):/;

// DECODE BEFORE CHECKING, and this is the whole correctness argument. The
// browser resolves the DECODED attribute value, so a check that runs on the
// raw text is checking a different string than the one that will be navigated.
// The first version of this guard tested for entities only in the part BEFORE
// a literal colon, which missed the case where the COLON ITSELF is an entity:
// `javascript&colon;alert(1)` has no literal colon at all, took the
// "no colon, therefore relative" early return, and was emitted verbatim --
// the HTML parser then supplied the colon and it executed. Verified in real
// Chrome by review, 2026-09-06. DOMPurify caught it, which is exactly what a
// second layer is for, but this one claimed to be the primary and was not.
//
// A detached <textarea> is the standard inert decoder: its content model is
// text (RCDATA), so assigning innerHTML parses entities and executes nothing.
// It decodes ONCE, which is what the HTML parser does -- so `&amp;#58;`
// correctly stays the literal text `&#58;` and reads as relative, matching
// what the browser will conclude.
let _decoder = null;
function decodeEntities(raw) {
  if (!_decoder) {
    if (typeof document === 'undefined') return null;   // fail closed
    _decoder = document.createElement('textarea');
  }
  _decoder.innerHTML = String(raw ?? '');
  return _decoder.value;
}

function safeUrl(raw, allowed) {
  const decoded = decodeEntities(raw);
  if (decoded === null) return false;
  // Strip C0 controls and spaces: a SUPERSET of what the HTML URL parser
  // removes (tab, LF, CR), so this can never read "relative" where the browser
  // reads a scheme. Only the decision uses this form; a refusal re-renders the
  // original text, and an approval hands marked the untouched href.
  const url = decoded.replace(/[\u0000-\u0020\u007f]/g, '');
  if (url === '') return false;
  const scheme = SCHEME_RE.exec(url);
  // No scheme means relative or a bare fragment, which can only resolve
  // against our own origin -- the same conclusion the browser reaches from the
  // same grammar. This is why the old `[&%\\]` prefix heuristic is gone: it
  // was guarding against a decode that now happens explicitly, and it dropped
  // ordinary relative URLs like `/search?q=a&t=1:30` on the way.
  if (!scheme) return true;
  return allowed.has(`${scheme[1].toLowerCase()}:`);
}

marked.use({
  renderer: {
    link(token) {
      if (safeUrl(token.href, SAFE_LINK_SCHEMES)) return false;
      // Refused: keep the link TEXT, drop the anchor. parseInline keeps any
      // emphasis inside the label rather than flattening it to source.
      return this.parser.parseInline(token.tokens);
    },
    image(token) {
      if (safeUrl(token.href, SAFE_IMAGE_SCHEMES)) return false;
      return escapeHtml(token.text ?? '');
    },
  },
});

const ESCAPE_MAP = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
const ESCAPE_REGEX = /[&<>"']/g;

export function escapeHtml(text) {
  if (text == null) return '';
  return String(text).replace(ESCAPE_REGEX, (m) => ESCAPE_MAP[m]);
}

export function renderMarkdown(text) {
  try {
    return DOMPurify.sanitize(marked.parse(text ?? ''));
  } catch {
    return escapeHtml(text);
  }
}
