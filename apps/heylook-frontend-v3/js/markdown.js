// The only sanctioned path for model/user text -> HTML. Never bypass,
// never double-sanitize.

import { marked } from './vendor/marked.esm.js';
import DOMPurify from './vendor/purify.es.mjs';

marked.use({ gfm: true, breaks: true });

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
