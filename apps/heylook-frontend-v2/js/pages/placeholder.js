// Placeholder page for applets not yet built

export function mount(el, name) {
  const div = document.createElement('div')
  div.className = 'page-placeholder'
  div.textContent = `${name || 'Page'} -- coming soon`
  el.append(div)
  return { teardown() {} }
}
