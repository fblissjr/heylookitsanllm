// App shell -- hash router, theme, init

const routes = {
  '#/chat': () => import('./pages/chat.js'),
  '#/batch': () => import('./pages/batch.js'),
  '#/models': () => import('./pages/models.js'),
  '#/perf': () => import('./pages/perf.js'),
  '#/notebook': () => import('./pages/notebook.js'),
  '#/explore': () => import('./pages/explore.js'),
}

let currentPage = null
const main = document.getElementById('main')

async function router() {
  const hash = location.hash || '#/chat'

  if (currentPage?.teardown) currentPage.teardown()
  main.replaceChildren()

  const loader = routes[hash] || routes['#/chat']
  const mod = await loader()

  const name = hash.replace('#/', '')
  currentPage = mod.mount(main, name)

  // Update nav active state + sidebar visibility
  const app = document.getElementById('app')
  app.dataset.page = name
  document.querySelectorAll('.nav-item').forEach(item => {
    item.classList.toggle('nav-item--active', item.dataset.page === name)
  })
}

window.addEventListener('hashchange', router)
router()
