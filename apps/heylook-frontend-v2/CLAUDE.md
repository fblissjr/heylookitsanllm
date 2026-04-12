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
  utils.js              -- shared: createEl, statCard, beforeUnloadGuard, throttleToFrame
  components/
    markdown.js          -- marked + DOMPurify (vendored), renderMarkdown()
    settings_panel.js    -- collapsible sampler controls (Core + Advanced sections)
    pretext_chat_model.js   -- Pretext layout math (translated from markdown-chat.model.ts)
    pretext_chat_renderer.js -- Pretext virtualized DOM rendering (translated from markdown-chat.ts)
  pages/
    chat.js              -- conversations, streaming, edit+regenerate
    batch.js             -- multi-prompt batch completions
    models.js            -- admin model list, load/unload, scan+import
    perf.js              -- system metrics polling, performance profile
    notebook.js          -- text scratchpad with LLM generation
    explore.js           -- token explorer with logprobs visualization
    placeholder.js       -- stub for unbuilt pages
  vendor/
    marked.esm.js        -- marked v17 (vendored, no CDN)
    purify.es.mjs        -- DOMPurify v3 (vendored, no CDN)
    pretext/             -- @chenglou/pretext v0.0.5 (built from ~/workspace/pretext)
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
- **Streaming**: RAF-throttle DOM writes via `throttleToFrame()` from `utils.js`. Never update DOM per-token -- accumulate in state, the throttle flushes at frame rate. Call `throttled.reset()` in teardown.
- **Sanitization**: all user-generated HTML goes through `renderMarkdown()` (DOMPurify). Never set `innerHTML` with raw user content.
- **Settings**: `samplerParams()` from `settings.js` builds request params. Null values mean "use backend default" -- only sends explicitly set params. `save()` is debounced (300ms) -- cache updates immediately, localStorage persistence deferred.
- **Settings cascade**: Backend applies global defaults -> thinking mode -> models.toml per-model -> request params. Send null to respect model defaults.
- **Polling**: use recursive `setTimeout` (not `setInterval`) to prevent overlapping requests.
- **Imports**: use `createEl`, `statCard`, `beforeUnloadGuard`, `throttleToFrame` from `utils.js`. Don't redefine locally.
- **Throttle cleanup**: null module-level throttle variables in `teardown()` (not just `.reset()`) to prevent stale closures across mounts.
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

## Data Storage

- **Server-side (SQLite)**: Conversations, messages, notebooks stored in `data/conversations.db`. This is the source of truth. `POST /v1/data/clear` deletes all conversations and notebooks.
- **Browser (localStorage)**: Only sampler settings (temperature, top_p, etc.). No conversation data in the browser.
- Users can clear all server data from the Models page ("Clear all conversations & notebooks" button).

## Gotchas

- `marked.use()` not `marked.setOptions()` (removed in marked v5+)
- `renderMarkdown()` already sanitizes -- never double-wrap with `sanitize()`
- `<base href="/v2/">` in HTML is required so relative paths resolve under `/v2`
- Static file handler uses `resolve()` + `is_relative_to()` for path traversal prevention
- `Intl.Segmenter` required (Safari 16.4+, Chrome 111+, Firefox 125+)
- Elements created inside `buildShell()` are function-scoped. Query by ID in `mount()` if needed for event listeners.
