# Frontend: open bugs

last updated: 2026-11-24

Defects found by the 2026-11-24 review of `frontend/` that are still open. What
that review fixed is in `CHANGELOG.md` (2.0.279, 2.0.281, 2.0.282) and is not
repeated here.

The entries in the first four tables (FB-01 to FB-33) were re-checked against
the tree at commit `acf17724`. The two sections after them say how far each
was checked. Entries name a file and a symbol rather than a line, because
lines move. Close an entry by deleting its row in the commit that fixes it; this file
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

## Found while mapping the flows

Writing the flow maps in `docs/wiki/frontend/` turned up these. Each was read
from the code by the page's writer and none was run. Only FB-37 and FB-38 were
re-checked by a second reader; treat the rest as leads. The page named in the
last column maps the flow each one sits in.

Security:

| ID | Claim | Where | Page |
|---|---|---|---|
| FB-37 | An SVG attached as an image is stored and served back as `image/svg+xml` from the app's own origin. Shown in an `img` it is inert; opened directly as a document, a script inside it would run with the app's origin. Re-checked in code (the type test accepts anything starting with `image/`); not tried in a browser. | `src/heylook_llm/db.py`, `_externalize_media`; the media route in `src/heylook_llm/conversation_api.py` | chat-attachments.md |

Chat:

| ID | Claim | Where | Page |
|---|---|---|---|
| FB-38 | "End thinking" posts to the conversation open at click time, not the one the stream belongs to. Re-checked in code. | `chat.js`, the `buildThinkingEl` callback in `startStream` | chat-generation.md |
| FB-39 | After the first send creates a conversation, the page takes it as the open one without checking that nothing else was opened during the create. A conversation opened in that moment is overridden, and its prompt can be written onto the new one. | `chat.js`, `send`, after `createConversation` | chat-generation.md |
| FB-40 | After a failed conversation-list load at boot, the sampler panel is not cleared and the parked draft prompt is not read. The first Send then deletes that parked prompt. | `chat.js`, `setup` | chat-documents.md |
| FB-41 | The resume sync applies the `generating` flag it read for one conversation to whichever conversation is open when its awaits finish. | `chat.js`, `refreshAfterResume` | chat-documents.md |
| FB-42 | While a conversation loads, the previous one's Stop button and remote-run flag stay. Stop pressed then targets the new conversation and reports "Stopped.". | `chat.js`, `selectConversation` | chat-documents.md |
| FB-43 | Deleting the last conversation clears the composer text but leaves staged attachments. | `chat.js`, `deleteConversation` | chat-attachments.md |
| FB-44 | New from the no-conversation state carries the typed text and drops staged attachments without saying so. | `chat.js`, `newConversation` | chat-documents.md |
| FB-45 | "Conversation cloned." overwrites the line saying the conversation being left keeps generating. | `chat.js`, `cloneConversation` | chat-documents.md |
| FB-46 | If the open conversation was deleted on another device, the resume sync keeps showing it and says nothing. | `chat.js`, `refreshAfterResume` | chat-documents.md |
| FB-47 | A local stream never marks its list row as generating, so after switching away from a live run the nav dot can go out while the run continues. | `chat.js`, `startStream`, `notifyChatActivity` | chat-generation.md |
| FB-48 | The recovery retry ignores a server answer of "still generating", so its "recovering…" line can replace the abandon note and outlive the reply. Latent. | `chat.js`, `finishGenerate` | chat-generation.md |
| FB-49 | Stopping a run the page is not subscribed to ends with "Stopped." even when the server answered that no run was active. | `chat.js`, `stopRemote` | chat-generation.md |
| FB-50 | Stop's local abort on a "no active run" answer assumes the generate request never reached the server. If it did, the run continues detached. | `chat.js`, `stopStream` | chat-generation.md |
| FB-51 | Cancel on an editor whose row was deleted elsewhere leaves that row on screen as an ordinary row. The comment in `mergeServerRows` says Cancel drops it. | `chat.js`, `cancel` in `buildEditEl` | chat-messages.md |
| FB-52 | Opening a second editor drops the first editor's draft without a word and never revokes its staged object URLs. | `chat.js`, the Edit handler in `buildActions` | chat-messages.md |
| FB-53 | The thinking edit box's own height cap and floor are outranked by the content box's rule, so they do not apply. Read from the selectors, not rendered. | `frontend/css/app.css`, `.message-edit__thinking` | chat-messages.md |
| FB-54 | In Save and in the editor's preview, the block assembly is awaited outside the `try`, so a failed encode shows nothing. | `chat.js`, `save` in `buildEditEl` | chat-messages.md |
| FB-55 | After the edit's save lands, the page re-checks the conversation but not whether a stream has started. A composer send in that moment makes the follow-on generation return silently, and the rows after the edited message stay hidden. | `chat.js`, `save` in `buildEditEl` | chat-messages.md |
| FB-56 | A successful save or delete never clears an earlier "Save failed" or "Delete failed" line, and the editor's buttons have no in-flight guard. | `chat.js`, `save`, `deleteMessage` | chat-messages.md |
| FB-57 | The composer's prompt preview leaves staged attachments out without saying so, and does not flush pending settings or prompt writes first. | `chat.js`, `previewNextPrompt` | chat-model-and-settings.md |
| FB-58 | Applying a preset rebuilds the drawer, which releases the old prompt section, whose flush can send a still-pending typed prompt after Apply's own prompt write. | `frontend/js/preset-bar.js`, `apply` | chat-model-and-settings.md |
| FB-59 | With thinking off and a depth level that means off, the first press of the thinking button leaves thinking off. Latent. | `chat.js`, the thinking button's click handler | chat-model-and-settings.md |
| FB-60 | A failed model-list load at boot is silent. | `chat.js`, `setup` | chat-model-and-settings.md |
| FB-61 | A send that fails restores its text and attachments over anything typed or staged during the send, and leaks those object URLs. | `chat.js`, `send`, the catch | chat-attachments.md |
| FB-62 | Object URLs of media newly attached in an editor are revoked only by Cancel, not by Save. | `chat.js`, `save` in `buildEditEl` | chat-attachments.md |
| FB-63 | Image pricing: its failure path has no staleness check, and one over-size source makes the whole request a 400. | `chat.js`, `priceStagedImages` | chat-attachments.md |
| FB-64 | Fit does not re-check that the image is still staged after its await. | `chat.js`, `fitStagedImage` | chat-attachments.md |
| FB-65 | A refusal message reads "a assistant message". | `chat.js`, `addPendingFiles` | chat-attachments.md |

Shell and shared modules:

| ID | Claim | Where | Page |
|---|---|---|---|
| FB-66 | A per-operation controller made after teardown never aborts, so a hook registered late stays live. Latent. | `frontend/js/page.js`, `linkedController` | shell-and-lifecycle.md |
| FB-67 | Sign out revokes the session before the page's hide-time saves run, so a write still pending at that moment is refused. Inferred from the order of the code. | `frontend/js/api.js`, `logoutSession` | shell-and-lifecycle.md |
| FB-68 | The "stale cached module" hint is shown even when the page's module failed to load because of a sign-in refusal or a server that is down. | `frontend/js/app.js`, `staleModuleHint` | shell-and-lifecycle.md |
| FB-69 | The shortcuts sheet says Enter sends, which is not true on a touch device since v2.0.282. | `frontend/js/hotkeys.js`, `SHORTCUT_ROWS` | shell-and-lifecycle.md |
| FB-70 | The sign-in page's "misconfigured key" branches cannot be reached, because the gate answers first. | `src/heylook_llm/auth_api.py`, `login_page`, `login_submit` | shell-and-lifecycle.md |

Notebook:

| ID | Claim | Where | Page |
|---|---|---|---|
| FB-71 | While a notebook loads, the editor still shows the previous one and accepts typing. A save in that window writes the old title and body to the new notebook. | `frontend/js/pages/notebook.js`, `selectNotebook` | notebook.md |
| FB-72 | When a notebook fails to load, the sampler panel and the prompt box are not cleared, and New creates from those values. | `notebook.js`, `selectNotebook` | notebook.md |
| FB-73 | A failed save is never retried by a later flush, and a failure of the save flushed on a switch shows nothing. | `notebook.js`, `doSave` | notebook.md |
| FB-74 | After a generation error the partial text stays on screen but is not in the page's own copy of the document. | `notebook.js`, `handleGenerateError` | notebook.md |
| FB-75 | The "Server busy" line stays after a run that then succeeds. | `notebook.js`, `finishGenerate` | notebook.md |
| FB-76 | A notebook deleted on another device stays open, and every save then fails. | `notebook.js`, `refreshAfterResume` | notebook.md |

Models page:

| ID | Claim | Where | Page |
|---|---|---|---|
| FB-77 | If the option schema arrives before the model list, the page shows "No models yet…" until the list loads. | `frontend/js/pages/models.js`, `setup` | models-and-config.md |
| FB-78 | A row reads "Unloading…" for the whole of a Reload. | `models.js`, `buildModelRow` | models-and-config.md |
| FB-79 | Saving a discovered model's config answers with `source: "config"`, so its row loses "no entry" until the list is fetched again. | `src/heylook_llm/admin_api.py`, `update_model_config` | models-and-config.md |
| FB-80 | Saving a chat template never tells the page, so the stale marker and "Reload now" do not appear until the list is fetched again. | `frontend/js/model-config.js`, `buildChatTemplatePanel` | models-and-config.md |
| FB-81 | The fit gate is judged on unsaved drafts and stays after the panel closes. | `models.js`, `onFitGate` | models-and-config.md |
| FB-82 | A value typed while a config save is in flight is lost. | `model-config.js`, `save` | models-and-config.md |
| FB-83 | Embedding metadata in the engine view would be labelled "live" and its arrays dropped. | `frontend/js/engine.js`, `slotRows` | models-and-config.md |

Presets and Perf:

| ID | Claim | Where | Page |
|---|---|---|---|
| FB-84 | After a "changed elsewhere" refusal, Cancel rebuilds the card from the stale preset, so the advised "Cancel and Edit again" is refused every time. | `frontend/js/pages/presets.js`, `save`, `replaceCard` | presets-and-perf.md |
| FB-85 | The list reload that follows an import discards editors open on other cards. | `presets.js`, `runApply` | presets-and-perf.md |
| FB-86 | An import that overwrites a preset does not re-check that the stored row is unchanged, and does not arm when the file removes a stored prompt. | `presets.js`, `applyImportPlan` | presets-and-perf.md |
| FB-87 | The preset editor's prompt box is missing from the `field-sizing` block, so its height cap is not applied where that property is supported. | `frontend/css/app.css`, `.preset-edit__prompt` | presets-and-perf.md |
| FB-88 | "No profiling data yet" appears only when the request fails, because the server never sends an empty breakdown. | `frontend/js/pages/perf.js`, `renderProfile` | presets-and-perf.md |
| FB-89 | Both reads behind the observability panel swallow their failures, so a refused config read shows "off". | `perf.js`, `loadObservability` | presets-and-perf.md |
| FB-90 | Refresh never reloads the profile. | `perf.js`, `setup` | presets-and-perf.md |

Embeddings:

| ID | Claim | Where | Page |
|---|---|---|---|
| FB-91 | Stop's failure path does not check that the run it stopped is still the current one, so a "no such request" answer can replace a newer run's status. Between Lab batches it always gets that answer. | `frontend/js/pages/embeddings.js`, the Stop handler | embeddings.md |
| FB-92 | Compare's Stop aborts nothing on the client, so a busy retry still goes out after Stop. | `embeddings.js`, `execute` | embeddings.md |
| FB-93 | A file that is too large or of an unknown type is reported as "not in the chosen folder". | `embeddings.js`, the folder loader | embeddings.md |
| FB-94 | Adding an item before the model list has loaded throws, uncaught, with nothing shown. | `embeddings.js`, `add` | embeddings.md |
| FB-95 | Neither mode checks the model's text-length limit before sending. | `embeddings.js`; `frontend/js/embed-kit.js`, `planBatches` | embeddings.md |
| FB-96 | The sample loader's count includes one file twice, and the number of sample cases is written out by hand in two places. | `embeddings.js`, the sample loader | embeddings.md |

Comments and docs that say something false:

| ID | What it says | Where |
|---|---|---|
| FB-97 | Says `samplerParams` drops a zero; it keeps it. | `frontend/js/settings.js`, `sync` in `buildSettingsPanel` |
| FB-98 | Mention "Display prefs", removed in v2.0.38. | `frontend/js/pages/models.js` and `frontend/js/pages/perf.js`, `setup` |
| FB-99 | Names `_gc_media`; the function is `_gc_orphans`. | `frontend/js/pages/chat.js`, the comment in `buildEditEl` |
| FB-100 | Cites a sample-set file that is not in the tree. | `frontend/js/embed-kit.js`, the comment on `SAMPLE_SET` |
| FB-101 | Says video windows overlap; the server makes consecutive ones. | `docs/frontend_v3_user_guide.md`, the Embeddings section |
| FB-102 | Three false comments: the reload route's description of a default, the `saved` baseline comment, and the load panel's header. | `src/heylook_llm/admin_api.py`, `reload_model`; `frontend/js/model-config.js`, `createModelConfigEditor`; `frontend/js/load-panel.js` |

## Reported by a reviewer, not re-checked

These came from the review and were not confirmed against the current tree,
either because the file has changed since or because they depend on timing.

| ID | Claim | Where to look |
|---|---|---|
| FB-34 | On the Presets page, the Escape hotkey clicks the first Cancel on the page, which discards an edited prompt with no confirmation. Every other destructive action there arms first. | `frontend/js/pages/presets.js`, the `onEscape` passed to `registerPageHotkeys` |
| FB-35 | A hide-time save is dispatched ahead of the save chain, so an older save already in flight can land after it and win. | `frontend/js/document-writer.js`, `putSystemPrompt` |
| FB-36 | Stopping the e2e server with `SIGKILL` signals `uv`, which may leave its Python child running. | `tests/e2e/lib/server.mjs`, `stop` |
