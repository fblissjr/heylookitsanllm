# Frontend: open bugs

last updated: 2026-11-24

Defects found by the 2026-11-24 review of `frontend/` that are still open. What
that review fixed is in `CHANGELOG.md` (2.0.279, 2.0.281, 2.0.282) and is not
repeated here.

Every entry outside the last table was re-checked against the tree at commit
`acf17724`. Entries name a file and a symbol rather than a line, because lines
move. Close an entry by deleting its row in the commit that fixes it; this file
holds open bugs only.

Status means:

- confirmed: the behaviour follows from the code as it stands.
- needs a device: the code allows it, and whether it shows depends on what
  Safari does on the phone. Nothing in `tests/e2e/` can see it.
- latent: cannot happen with today's data, and will the day the data changes.

Not in this file: the battery suspects, the cleanup and duplication list, the
layout ideas and the checks still to port. Those are in `docs/project/TODO.md`
("From the 2026-11-24 frontend review") and the local review report.

## Chat page

| ID | What happens | Where | Status | Smallest fix |
|---|---|---|---|---|
| FB-01 | On an iPhone, in a very long conversation, focusing an in-place message editor makes the view scroll up without stopping until Send is tapped. No app code can drive it. v2.0.279 shipped a trial (scroll anchoring off for every scroller in the chat chain). | `docs/architecture/sharp_edges.md`, "Scroll anchoring" | needs a device: the trial is unconfirmed | Reproduce on the phone. If it persists, the capture snippet and the next candidates are in the local report, section 1. |
| FB-02 | Messages from the settings drawer are written to the chat status line, which the drawer covers on a phone and which sits inside the page the drawer makes inert. Apply's "removed settings" note, "name already taken" and "Overwrite refused" are all unreadable there. | `frontend/js/pages/chat.js`, `onStatus` passed to `createPresetBar`; `frontend/css/app.css`, `.drawer` width | confirmed | Give the preset section a `role="status"` line of its own and send `onStatus` there while the drawer is open. |
| FB-03 | "End thinking" can stay on "Ending…". The click disables the button and discards the result of `endReasoningGenerate`, so a refused request leaves it disabled until the stream ends. | `chat.js`, the `buildThinkingEl` callback in `startStream` | confirmed | Restore the button when the call rejects, and say why on the status line. |
| FB-04 | While editing a long message with the keyboard up, Save and Cancel sit below the keyboard. The editor's height cap follows the layout viewport, which the keyboard does not shrink. | `app.css`, `.message-edit textarea` max-height; `chat.js`, `growCap` in `buildEditEl` | needs a device | Cap from the visual viewport. The two caps must move together (`frontend/js/utils.js`, `autoGrow`). |
| FB-05 | A wide table or a large image in a reply has no style rule, so a table wider than the phone can make the whole thread scroll sideways. | `app.css`: no rule for `.message-content table` or `.message-content img` | confirmed that no rule exists; the overflow was not rendered | `display: block; overflow-x: auto` on tables and `max-width: 100%` on images, with cell padding and borders from tokens. |
| FB-06 | Switching conversations in the moment before a reply launches says the conversation "keeps generating" when no run was started. `selectConversation` reads `s.stream`, which is set before the pre-launch flush and before a busy retry's wait. | `chat.js`, `leaving` in `selectConversation`; the flush at the end of `startStream` | confirmed; the window is short unless a settings write is pending | Mark the stream when `launch()` has run and only claim a run is going when it has. |
| FB-07 | Leave Chat for another page mid-reply and the nav "generating" dot stays on after the run ends, until Chat is opened again. The page is unmounted, so nothing sends the event that clears it. | `chat.js`, `releaseStream` (the `!ctx.alive` branch) and `notifyChatActivity`; `frontend/js/app.js`, the `heylook:chat-activity` listener | confirmed | The lightest honest fix is to clear the dot on unmount and let Chat restate it on mount. |
| FB-08 | A failed save of a conversation's first title is swallowed. The list shows the new title and a reload shows the old one. | `chat.js`, `send`, the `updateConversation` call with `{ title }` | confirmed | Report it on the status line, as the `model_id` save now does. |
| FB-09 | With an editor open on a message that is then changed on another device, the rebuilt editor carries the local draft over the newer text without saying so. | `chat.js`, `carryEditorDraft` and `msgSignature` | confirmed | Say on the status line that the message changed elsewhere and the draft was kept. |
| FB-10 | The reason Save & Continue is disabled exists only as the button's `title`, which a touch screen cannot show. | `chat.js`, `saveContinue` in `buildEditEl` | confirmed | Put the reason in a line of text under the buttons. |
| FB-11 | Copy gives no sign that it worked, and does nothing where the clipboard API is unavailable (a page not served over HTTPS or loopback). | `chat.js`, `buildActions`, the Copy button | confirmed | Briefly relabel the button on success, and say so when the API is missing. |
| FB-12 | Nothing limits overscroll on the page or its scrollers, so a pull down on the phone can reload the page in the middle of a generation. | `app.css`: `overscroll-behavior` appears on three inner panes only | needs a device | `overscroll-behavior-y: none` on `html`, `contain` on the message list, the drawer and the list panes. |

## Phone layout and styles

| ID | What happens | Where | Status | Smallest fix |
|---|---|---|---|---|
| FB-13 | Models, Perf, Presets and Embeddings add the top safe area twice on a phone: once on `#phone-nav` and again in the page's own padding. | `app.css`, the `(max-width: 767px)` rules for `.models`, `.perf`, `.embeddings`, `.presets` | confirmed in the rules; the gap shows only where the inset is not zero (the home-screen app) | Drop `env(safe-area-inset-top)` from those four paddings. |
| FB-14 | Two height caps use `vh` where the rule this file's own comments state is `dvh`, so they do not follow the mobile toolbar. | `app.css`, `.prompt-preview__text` and `.chat__engine-panel` | confirmed | `dvh`. |
| FB-15 | The active Embeddings tab uses the brand colour as a fill with the on-accent text colour. `frontend/DESIGN.md` reserves brand for selection glints. | `app.css`, `.embeddings__tab[aria-pressed="true"]` | confirmed against the design rule; the contrast was not computed | Use `--brand-tint` with ink, as other chosen items do. |
| FB-16 | While a reply generates, the Chat entry in the desktop sidebar becomes an inline box, so its highlight shrinks to the width of its text. | `app.css`, `.nav-item--busy` sets `display: inline-flex` | confirmed that the rule changes the box; the result was not rendered | `display: flex`. |
| FB-17 | The home-screen manifest hard-codes the dark theme colours, while `index.html` carries a light and a dark pair that a unit test checks. | `frontend/manifest.json` | confirmed; what the launch screen shows in light mode needs a device | Derive it, or accept one colour and say so in the manifest's neighbour comment. |
| FB-18 | The system prompt box has no accessible name: no label and no `aria-label`. A `<summary>` above it is not a label. | `frontend/js/prompt-section.js`, the `sysprompt-input` textarea | confirmed | `aria-label`. |
| FB-19 | A phone held sideways is wider than the phone breakpoint and gets the desktop layout: sidebar, list pane and a thread narrower than in portrait. | `app.css`, the `(max-width: 767px)` breakpoint | needs a device | Add a height condition to the phone breakpoint. The value is repeated across the file, so this is several edits. |
| FB-20 | After a tap, hover styles stay applied on touch. The readout line keeps its "expanded" look after its panel closes, because hover and expanded share one rule. | `app.css`, `.chat__readout:where(:hover)` | needs a device | Put hover rules behind `@media (hover: hover)`. |

## Other pages and shared modules

| ID | What happens | Where | Status | Smallest fix |
|---|---|---|---|---|
| FB-21 | Any action on the Models page rebuilds every row and the open config editor. With a panel open that re-posts the fit check, forces a fresh system-metrics read, and drops focus from the field being typed in, which closes the phone keyboard. | `frontend/js/pages/models.js`, the callers of `renderModelList`; `frontend/js/model-config.js`, `fitMeter.refresh` at the end of `createModelConfigEditor` | confirmed | Update the busy row in place, as `onFitGate` already does, and rebuild a panel only when its own model's row changed. |
| FB-22 | A notebook system prompt typed before any notebook exists is held only in page state and is lost on reload. Chat parks its equivalent in browser storage. | `frontend/js/pages/notebook.js`, the prompt section's `set` | confirmed; it needs zero notebooks to bite | Park it the way chat does, through `frontend/js/storage.js`. |
| FB-23 | Typing in a notebook's title re-renders the whole notebook list on every keystroke, which also resets a Delete button that was armed. | `notebook.js`, the title input's `input` listener calling `renderList` | confirmed | Update the one row's text. |
| FB-24 | A long notebook's hide-time save is sent with `keepalive` and the whole document as its body. Browsers cap a keepalive body, and `frontend/js/api.js` says so beside `request`. Past the cap the last edits before backgrounding are not saved. | `notebook.js`, `ctx.onHide` calling `scheduleSave.flush({ keepalive: true })` | confirmed that nothing bounds the body; the failure itself was not observed | Fall back to a plain request when the body is over the cap. |
| FB-25 | The Embeddings video preview dialog relies on `closedby="any"` to close on an outside tap. `frontend/js/hotkeys.js` adds a fallback for browsers without it; this dialog has none. Its Close button works. | `frontend/js/pages/embeddings.js`, `previewDialog` | needs a device | Reuse the fallback from `hotkeys.js`. |
| FB-26 | Embeddings thumbnails are drawn without honouring a photo's EXIF orientation. `frontend/js/image-prep.js` passes `imageOrientation: 'from-image'` and explains why; `embed-kit.js` does not. | `frontend/js/embed-kit.js`, `thumbnail` | confirmed in code; a rotated thumbnail was not observed | Pass the same option. |
| FB-27 | The model config editor sorts fields into four named groups and shows nothing for a field whose effect class is none of them. | `model-config.js`, the `fixed`, `advanced`, `live` and `reload` filters | latent: today the only unnamed class is `identity` | A catch-all group, so a new class is visible the day it is added. |
| FB-28 | Saving observability settings on the Perf page passes the page's abort signal, so leaving the page mid-save cancels the write. Other pages leave writes unaborted on purpose. | `frontend/js/pages/perf.js`, the `adminSetConfig` call | confirmed | Do not pass the signal on the write. |
| FB-29 | The fit meter's request and its debounce are not tied to the page, so a fit check can still be sent after leaving the Models page. | `model-config.js`, the meter's own `AbortController` and `scheduleRefresh` | confirmed | Cancel both on teardown. |

## Docs and comments that state something false

| ID | What it says | Where |
|---|---|---|
| FB-30 | Describes a fixed `#bottom-nav`, removed in v2.0.212. | `tests/e2e/ios-sim.mjs`, header comment |
| FB-31 | Says the e2e harness shells out to it. The multi-engine arms that did were removed in v2.0.281. | `tests/helpers/engines.py`, docstring |
| FB-32 | Lists the pages as chat, notebook, models and perf. Presets and embeddings exist. | `.claude/rules/frontend.md`, Structure |
| FB-33 | Describes "Display preferences" and a "Show special tokens" toggle, removed in v2.0.38. | `docs/frontend_v3_user_guide.md`, section 7 |

## Reported by a reviewer, not re-checked

These came from the review and were not confirmed against the current tree,
either because the file has changed since or because they depend on timing.

| ID | Claim | Where to look |
|---|---|---|
| FB-34 | On the Presets page, the Escape hotkey clicks the first Cancel on the page, which discards an edited prompt with no confirmation. Every other destructive action there arms first. | `frontend/js/pages/presets.js`, the `onEscape` passed to `registerPageHotkeys` |
| FB-35 | A hide-time save is dispatched ahead of the save chain, so an older save already in flight can land after it and win. | `frontend/js/document-writer.js`, `putSystemPrompt` |
| FB-36 | Stopping the e2e server with `SIGKILL` signals `uv`, which may leave its Python child running. | `tests/e2e/lib/server.mjs`, `stop` |
