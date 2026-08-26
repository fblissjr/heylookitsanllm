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

// v3 ships unbundled modules with no content hashes, so the ONE failure a
// mount error is most likely to be is a cache artifact, not a code bug: a
// module cached under an older server (before the no-cache headers of
// v1.62.2) keeps its HEURISTIC freshness -- roughly a tenth of the file's
// age, which for a long-lived file is days -- and is never re-requested, so
// a new caller runs against an old module. That is what "X is not a
// function" means here, and it took a bisect to say so out loud the first
// time it happened (new chat.js + pre-v1.62.3 preset-bar.js, whose export
// list had no promptState -> the whole chat page, top bar included, replaced
// by this note). The distinction the reader needs is that a plain reload
// does NOT fix it: reload revalidates the document, then serves the stale
// SUBRESOURCE straight back out of cache. Only a hard reload evicts it.
function staleModuleHint(err) {
  const msg = String(err?.message ?? '');
  const mixedVersions = /is not a (function|constructor)|undefined is not an object/.test(msg);
  const moduleFetch = /dynamically imported module|Importing a module script failed|Failed to load module/.test(msg);
  if (!mixedVersions && !moduleFetch) return null;
  return 'This usually means the browser is serving a stale cached copy of one of the '
    + 'frontend modules. A normal reload will not clear it -- hard-reload the page '
    + '(Cmd-Shift-R / Ctrl-Shift-R). If it survives that, it is a real bug.';
}

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
      staleModuleHint(err) ?? 'Navigation still works -- pick another page or reload.',
    ]));
  }
}

window.addEventListener('hashchange', navigate);
navigate();
