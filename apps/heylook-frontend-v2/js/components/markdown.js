// Markdown renderer using marked + DOMPurify for sanitization

let _marked = null
let _purify = null

export async function ensureMarked() {
  if (_marked && _purify) return
  try {
    const [markedMod, purifyMod] = await Promise.all([
      import('https://esm.run/marked@17'),
      import('https://esm.run/dompurify@3'),
    ])
    markedMod.marked.use({ gfm: true, breaks: false })
    _marked = markedMod.marked
    _purify = purifyMod.default
  } catch {
    // CDN unreachable -- renderMarkdown falls back to escapeHtml
    console.warn('Failed to load marked/DOMPurify from CDN, markdown disabled')
  }
}

export function renderMarkdown(text) {
  if (!_marked || !_purify || !text) return escapeHtml(text || '')
  try {
    const raw = _marked.parse(text)
    return _purify.sanitize(raw)
  } catch {
    return escapeHtml(text)
  }
}

export function sanitize(html) {
  if (!_purify) return escapeHtml(html)
  return _purify.sanitize(html)
}

export function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}
