// Hash router. Routes -> dynamic import of page modules (code-splitting
// without a bundler). Nav (desktop sidebar + mobile bottom bar) is generated
// from ROUTES, so nav-active bookkeeping lives here, not in pages.

import { createEl } from './utils.js';
import { mountSettingsDrawer } from './settings-drawer.js';

const ROUTES = {
  chat:     { title: 'Chat', short: 'Chat', load: () => import('./pages/chat.js') },
  notebook: { title: 'Notebook', short: 'Notes', load: () => import('./pages/notebook.js') },
  explore:  { title: 'Token Explorer', short: 'Explore', load: () => import('./pages/explore.js') },
  jspace:   { title: 'J-Space', short: 'J-Space', load: () => import('./pages/jspace.js') },
  models:   { title: 'Models', short: 'Models', load: () => import('./pages/models.js'), admin: true },
  perf:     { title: 'Performance', short: 'Perf', load: () => import('./pages/perf.js'), admin: true },
};

const main = document.getElementById('main');
const navDesktop = document.getElementById('nav-desktop');
const navBottom = document.getElementById('bottom-nav');
const navLinks = [];

let sepAdded = false;
for (const [name, route] of Object.entries(ROUTES)) {
  if (route.admin && !sepAdded) {
    navDesktop.append(createEl('div', { class: 'nav-sep' }));
    sepAdded = true;
  }
  const desktop = createEl('a', { class: 'nav-item', href: `#/${name}`, dataset: { route: name } }, [route.title]);
  const bottom = createEl('a', { class: 'nav-item', href: `#/${name}`, dataset: { route: name } }, [route.short]);
  navDesktop.append(desktop);
  navBottom.append(bottom);
  navLinks.push(desktop, bottom);
}

// App-shell singleton: a persistent gear + right slide-over shared by every
// page. Pages contribute to it in setup() and clear on teardown.
mountSettingsDrawer(navDesktop, navBottom);

let currentPage = null;
let navToken = 0;

async function navigate() {
  const name = (location.hash.replace(/^#\/?/, '') || 'chat').split('/')[0];
  const route = ROUTES[name] || ROUTES.chat;
  const routeName = ROUTES[name] ? name : 'chat';
  const token = ++navToken;

  await currentPage?.unmount();
  if (token !== navToken) return; // superseded by a faster navigation
  currentPage = null;
  main.replaceChildren();

  document.title = `${route.title} · heylook`;
  document.body.dataset.page = routeName;
  for (const link of navLinks) {
    const active = link.dataset.route === routeName;
    link.classList.toggle('nav-item--active', active);
    if (active) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }

  try {
    const mod = await route.load();
    if (token !== navToken) return;
    currentPage = mod.default;
    await currentPage.mount(main);
  } catch (err) {
    // A page that fails to load or mount must not brick the router: show the
    // failure in place and leave nav/hashchange fully working.
    if (token !== navToken) return;
    console.error(`page "${routeName}" failed to mount`, err);
    currentPage = null;
    main.replaceChildren(createEl('div', { class: 'error-note', role: 'alert' }, [
      `This page failed to load (${err.message}). `,
      'Navigation still works -- pick another page or reload.',
    ]));
  }
}

window.addEventListener('hashchange', navigate);
navigate();
