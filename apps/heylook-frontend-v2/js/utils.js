// Shared utilities

export function createEl(tag, attrs, text) {
  const el = document.createElement(tag)
  if (attrs) for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v)
  if (text) el.textContent = text
  return el
}
