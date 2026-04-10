// Markdown renderer using marked + DOMPurify for sanitization
// Libraries vendored locally in js/vendor/ (no CDN dependency)

import { marked } from '../vendor/marked.esm.js'
import DOMPurify from '../vendor/purify.es.mjs'

marked.use({ gfm: true, breaks: false })

export function ensureMarked() {
  // No-op -- libraries are now loaded synchronously via static imports.
  // Kept for API compatibility with pages that call ensureMarked() in init().
}

export function renderMarkdown(text) {
  if (!text) return ''
  try {
    const raw = marked.parse(text)
    return DOMPurify.sanitize(raw)
  } catch {
    return escapeHtml(text)
  }
}

export function sanitize(html) {
  return DOMPurify.sanitize(html)
}

export function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}
