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
const SAFE_LINK_SCHEMES = new Set(['http:', 'https:', 'mailto:']);
const SAFE_IMAGE_SCHEMES = new Set(['http:', 'https:']);
const SCHEME_RE = /^([A-Za-z][A-Za-z0-9+.-]*):/;

function safeUrl(raw, allowed) {
  // The HTML parser STRIPS tab, LF and CR from a URL attribute before
  // resolving it, so a literal `jav<TAB>ascript:` is a live vector that
  // reads as harmless here. Normalize the way the browser will.
  const url = String(raw ?? '').replace(/[\t\n\r]/g, '').trim();
  if (url === '') return false;
  const colon = url.indexOf(':');
  if (colon === -1) return true;              // relative path or #anchor
  const scheme = SCHEME_RE.exec(url);
  if (scheme) return allowed.has(`${scheme[1].toLowerCase()}:`);
  // A colon with no parseable scheme is only safe if it cannot BECOME one
  // once the browser entity-decodes the attribute: `java&#115;cript:` does
  // exactly that, and reads as a relative path until it is decoded. Refuse
  // on any escape marker rather than reimplement HTML entity decoding.
  return !/[&%\\]/.test(url.slice(0, colon));
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
