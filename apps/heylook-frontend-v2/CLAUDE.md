# Frontend v2

Vanilla JS frontend for heylookitsanllm. No framework, no bundler, no node_modules.

## Running

Start the backend server, then visit `http://localhost:8080/v2`. No build step required.

## File Structure

```
index.html              -- app shell, sidebar, bottom nav
css/app.css             -- all styles, responsive (mobile @768px)
js/
  app.js                -- hash router, sets data-page on #app
  api.js                -- fetch wrappers for all backend endpoints
  bus.js                -- EventTarget event bus
  settings.js           -- localStorage sampler settings, samplerParams()
  streaming.js          -- SSE client (fetch + ReadableStream)
  utils.js              -- shared: createEl, statCard, beforeUnloadGuard
  components/
    markdown.js          -- marked + DOMPurify (vendored), renderMarkdown()
    settings_panel.js    -- collapsible sampler controls (Core + Advanced sections)
  pages/
    chat.js              -- conversations, streaming, edit+regenerate
    batch.js             -- multi-prompt batch completions
    models.js            -- admin model list, load/unload, scan+import
    perf.js              -- system metrics polling, performance profile
    notebook.js          -- text scratchpad with LLM generation
    placeholder.js       -- stub for unbuilt pages
  vendor/
    marked.esm.js        -- marked v17 (vendored, no CDN)
    purify.es.mjs        -- DOMPurify v3 (vendored, no CDN)
```

## Page Pattern

Every page module exports `mount(el)` returning `{ teardown() }`.

```js
let container = null
let state = null

function freshState() {
  return { /* ... */ }
}

export function mount(el) {
  container = el
  state = freshState()    // always reset on mount
  // build DOM, attach listeners, fetch data
  return { teardown }
}

function teardown() {
  // stop streams, clear intervals, reset RAF flags
  // null handlers on persistent DOM (sidebar, nav)
  state = null
  container = null
}
```

## Rules

- **State**: reset via `freshState()` on every `mount()`. Null `state` in `teardown()`. Guard async callbacks with `if (!state) return`.
- **Streaming**: RAF-throttle DOM writes. Never update DOM per-token -- accumulate in state, flush in `requestAnimationFrame`. Use `_rafPending` flag pattern.
- **Sanitization**: all user-generated HTML goes through `renderMarkdown()` (DOMPurify). Never set `innerHTML` with raw user content.
- **Settings**: `samplerParams()` from `settings.js` builds request params. Null values mean "use backend default" -- only sends explicitly set params. `save()` is debounced (300ms) -- cache updates immediately, localStorage persistence deferred.
- **Settings cascade**: Backend applies global defaults -> thinking mode -> models.toml per-model -> request params. Send null to respect model defaults.
- **Polling**: use recursive `setTimeout` (not `setInterval`) to prevent overlapping requests.
- **Imports**: use `createEl`, `statCard`, `beforeUnloadGuard` from `utils.js`. Don't redefine locally.
- **beforeunload**: add listener during streaming, remove on complete/error/stop/teardown.
- **Delete**: always `confirm()` before destructive actions.
- **Sidebar**: only visible on `#/chat`. Router sets `data-page` on `#app`, CSS hides sidebar on other pages.

## Backend Dependencies

- `/v1/conversations/*` -- conversation + message CRUD (SQLite)
- `/v1/notebooks/*` -- notebook CRUD (SQLite)
- `/v1/chat/completions` -- streaming generation
- `/v1/admin/models/*` -- model management
- `/v1/system/metrics` -- system resource metrics
- `/v1/performance/profile/{range}` -- performance analytics
- `/v1/batch/chat/completions` -- batch inference
- `/v1/models` -- list available models

Route handler DB access: `get_db()` from `db.py` (shared helper, not per-module).

## Gotchas

- `marked.use()` not `marked.setOptions()` (removed in marked v5+)
- `renderMarkdown()` already sanitizes -- never double-wrap with `sanitize()`
- `<base href="/v2/">` in HTML is required so relative paths resolve under `/v2`
- Static file handler uses `resolve()` + `is_relative_to()` for path traversal prevention
- `Intl.Segmenter` required (Safari 16.4+, Chrome 111+, Firefox 125+)
- Elements created inside `buildShell()` are function-scoped. Query by ID in `mount()` if needed for event listeners.
