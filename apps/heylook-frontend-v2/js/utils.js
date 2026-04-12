// Shared utilities

export function createEl(tag, attrs, text) {
  const el = document.createElement(tag)
  if (attrs) for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v)
  if (text) el.textContent = text
  return el
}

export function beforeUnloadGuard(e) {
  e.preventDefault()
}

/**
 * Throttle a function to run at most once per animation frame.
 * Calls are coalesced -- only the latest args are used when the frame fires.
 * Call throttled.reset() in teardown to prevent stale pending frames.
 */
export function throttleToFrame(fn) {
  let pending = false
  let latestArgs = []
  const throttled = (...args) => {
    latestArgs = args
    if (pending) return
    pending = true
    requestAnimationFrame(() => { pending = false; fn(...latestArgs) })
  }
  throttled.reset = () => { pending = false }
  return throttled
}

export function statCard(label, value) {
  const card = createEl('div', { class: 'stat-card' })
  card.append(
    createEl('div', { class: 'stat-value' }, String(value)),
    createEl('div', { class: 'stat-label' }, label),
  )
  return card
}
