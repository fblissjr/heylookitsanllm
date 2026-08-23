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

export function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text ?? '';
  return div.innerHTML;
}

export function renderMarkdown(text) {
  try {
    return DOMPurify.sanitize(marked.parse(text ?? ''));
  } catch {
    return escapeHtml(text);
  }
}
