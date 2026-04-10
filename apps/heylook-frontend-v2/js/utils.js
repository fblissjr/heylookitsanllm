// Shared utilities

export function createEl(tag, attrs, text) {
  const el = document.createElement(tag)
  if (attrs) for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v)
  if (text) el.textContent = text
  return el
}

export function statCard(label, value) {
  const card = createEl('div', { class: 'stat-card' })
  card.append(
    createEl('div', { class: 'stat-value' }, String(value)),
    createEl('div', { class: 'stat-label' }, label),
  )
  return card
}
