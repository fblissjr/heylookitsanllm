---
paths:
  - "frontend/**"
  - "src/heylook_llm/frontend_static.py"
---

# Frontend (`frontend/`)

Map and backend coupling: [docs/frontend_v3.md](../../docs/frontend_v3.md). User-facing behaviour: [docs/frontend_v3_user_guide.md](../../docs/frontend_v3_user_guide.md); read it before changing chat UX, and update its rough-edges section when you close one.

## Structure

- Vanilla JS, no build, served at `/`; pages are chat, notebook, models and perf. Read `js/page.js` (the `createPage` lifecycle) before touching any page. New UI honours the a11y and mobile-parity rules in [frontend/DESIGN.md](../../frontend/DESIGN.md) §7: touch-reveal fallbacks (`@media (hover:none)`), the settings drawer as a modal (seals `#app` with `inert`, closes on `hashchange`), aria-live states, label association.
- No SPA fallback and no catch-all route. The app routes on the hash. `mount_frontend` registers only the tree's real shape (`/`, `/index.html`, `/js/*`, `/css/*`); a new top-level asset needs a route added there. Gzip is per request with no cache (the handlers are sync, so it runs off the event loop); `_serve` must catch `ValueError` from `resolve()` on a NUL byte. Revisit the fallback only if the app moves to the History API.
- The system-prompt editor and preset bar are shared sections (`prompt-section.js`, `preset-bar.js`) used by chat and notebook; fix bugs in the shared factory, never in one page's copy.
- The models page edits per-model config schema-driven off `/v1/admin/model-options` (`js/model-config.js`); `ui:"hidden"` keeps a field out.
- `app.css` carries a global `[hidden]{display:none!important}`; keep it, because an author `display` otherwise beats the `hidden` attribute. The sampler panel's reset hides by visibility so rows do not jog.

## Generation and the store mirror

- Chat generates over `POST /v1/conversations/{id}/generate`: the server builds the request from the store and owns persistence (including abort and disconnect), ending with a `heylook_saved` event carrying the authoritative rows. The client adopts those rows; it never does position arithmetic. Notebook speaks `/v1/messages`.
- A terminal path that awaits must re-check stream identity (`s.stream`), not just conversation identity. `handleStreamError` is safe only because it does not await between `releaseStream` and its writes; keep it that way. The composer being unbarred during the post-stream resync is correct: the server 409s writes while a generation claim is held (`conversation_api._refuse_while_generating`), and chat restores the typed text and staged attachments on that 409. Do not add a client-side bar.
- The page is a mirror of the store with exactly two invalidation points: document select and resume (`ctx.onResume` -> `refreshAfterResume`). Nothing polls; re-clicking the active conversation does not refetch.
- Use `createPage`'s `ctx.onHide` / `ctx.onResume`; never hand-wire `visibilitychange`/`pagehide`/`pageshow` in a page. Every debounced writer owns its hide flush. Hide flushes use `keepalive` and go ahead of the PUT chain. A prompt section's hide hook lives as long as the section; chat `release()`s the one it replaces.
- Resume refetches only when the list's `updated_at` moved, commits the new stamp only after everything it covers is adopted, adopts via `adoptConversationMeta`, and never touches a live stream's rows, the prompt box while it is being typed in, or the sidebar during a rename.
- A document's `params` is the sampler bag, and everything in it reaches the model. Never stash non-sampler state there; display prefs and preset provenance live elsewhere. `samplerParams(caps)` filters capability-gated keys at the wire for the same reason.

## Attachments and model switching

- All attach inputs (picker, paste, drop) go through `addFiles` -> `addPendingFiles`, where the cap gate, count cap and aria-live announcement live. A new input calls `addFiles`; that routine is also the backstop for inputs without an `accept` list. Paste listens on `document`, not the page root; verify the real event target, not the listener. The other-editable guard is load-bearing, because the drawer's system-prompt box sits outside `#app`. Call `preventDefault` only once something will really stage.
- Refresh capability-gated chrome after `modelSelect.value` moves, never before (it reads `currentCaps()`). Drag/drop is desktop-only on purpose and is not a §7 violation.
- On a model switch, history media the model cannot take is dropped at the wire with a per-message disclosure, while staged attachments still block. This asymmetry is deliberate and commented at both sites; do not unify it. Media is also cap-checked at staging time.
- Chat reads residency from `/v1/admin/models` and its Load button calls `load?warm=true`. Only loss gates a model switch: load cost is disclosed (residency dots, the Load button, a live pre-first-token status), never confirmed (owner call).

## Rendering

- Markdown URL schemes are allowlisted in `markdown.js`'s `link`/`image` renderer overrides after decoding HTML entities once; DOMPurify is the second layer. A check for this must assert on the protocol the browser resolves, never on the rendered HTML string. A renderer returning `''` drops content silently; return `false` to fall back. Vendored libs are pinned by `js/vendor/vendor.json` + `scripts/vendor_frontend.py`.
- A streaming message renders incrementally (`markdown-stream.js`): segments cut at boundaries no markdown construct spans are rendered once and never touched again; only the tail re-renders; a link-reference or footnote definition disables splitting. `tests/e2e/render.mjs` checks this as a property. Painters whose cost scales with the document use `ctx.throttleTime`, never `ctx.throttle`. Measure scroll-follow at the top of the painter, before it mutates; never use a cached flag fed by scroll events. Do not re-add `content-visibility`.

## Presets and the system prompt

- The system prompt is an override box (owner rule): a preset owns and carries a prompt, but a preset with an empty prompt makes no claim and leaves the document's prompt alone. Empty never means "set to empty". Only a carrying preset can arm "Replace prompt?" or count as drift.
- Preset Save's overwrite guard is the ordered logic in `wouldOverwritePresetPrompt`; read it there rather than restating it. Only a save onto a preset the document is not running arms, and blanking always arms. Enter in the name box goes to Save as new; Update is reachable only by its own button.
- `armedConfirm` takes a `target()` and re-arms instead of firing if the target moved; this check lives in the primitive, not in consumer wiring. Only the select disarms. The preset section previews the selected preset's own prompt; the select follows `applied_preset_id` until an explicit pick.
- A prompt typed before any conversation exists is parked in localStorage until a conversation adopts it. `.chat__sysprompt-chip` states what is in force, including "No system prompt".
- A new document made from an open one starts as `presetForNewDoc()` or blank, never from the open document's panel or prompt. With no document open the drawer is the draft and the next document is created from exactly what it shows; only Apply puts a preset in (it stamps the draft). Entering the no-document state clears the panel (`hydrateDocParams(null)`) and the stamp: the sampler cache is shared by chat and notebook.
- The thinking control offers `on` only when the template has no default level; `thinking_budget_tokens` is its sub-row (`sub` in PARAM_META), hidden while thinking is off and not built when `budget.enforced` is false.

Why and history: [sharp_edges.md](../../docs/architecture/sharp_edges.md) "Frontend".
