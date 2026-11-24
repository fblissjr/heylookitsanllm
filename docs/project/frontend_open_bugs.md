# Frontend: open bugs

last updated: 2026-11-24

Defects found by the 2026-11-24 review of `frontend/` that are still open. What
that review fixed is in `CHANGELOG.md` (2.0.279, 2.0.281, 2.0.282, 2.0.283, 2.0.284) and is not
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
| FB-04 | While editing a long message with the keyboard up, Save and Cancel sit below the keyboard. The editor's height cap follows the layout viewport, which the keyboard does not shrink. | `app.css`, `.message-edit textarea` max-height; `chat.js`, `growCap` in `buildEditEl` | needs a device | Cap from the visual viewport. The two caps must move together (`frontend/js/utils.js`, `autoGrow`). |
| FB-12 | Nothing limits overscroll on the page or its scrollers, so a pull down on the phone can reload the page in the middle of a generation. | `app.css`: `overscroll-behavior` appears on three inner panes only | needs a device | `overscroll-behavior-y: none` on `html`, `contain` on the message list, the drawer and the list panes. |

## Phone layout and styles

| ID | What happens | Where | Status | Smallest fix |
|---|---|---|---|---|
| FB-17 | The home-screen manifest hard-codes the dark theme colours, while `index.html` carries a light and a dark pair that a unit test checks. | `frontend/manifest.json` | confirmed; what the launch screen shows in light mode needs a device | Derive it, or accept one colour and say so in the manifest's neighbour comment. |
| FB-19 | A phone held sideways is wider than the phone breakpoint and gets the desktop layout: sidebar, list pane and a thread narrower than in portrait. | `app.css`, the `(max-width: 767px)` breakpoint | needs a device | Add a height condition to the phone breakpoint. The value is repeated across the file, so this is several edits. |

## Other pages and shared modules

| ID | What happens | Where | Status | Smallest fix |
|---|---|---|---|---|
| FB-21 | Any action on the Models page rebuilds every row and the open config editor. With a panel open that re-posts the fit check, forces a fresh system-metrics read, and drops focus from the field being typed in, which closes the phone keyboard. | `frontend/js/pages/models.js`, the callers of `renderModelList`; `frontend/js/model-config.js`, `fitMeter.refresh` at the end of `createModelConfigEditor` | confirmed | Update the busy row in place, as `onFitGate` already does, and rebuild a panel only when its own model's row changed. |
| FB-22 | A notebook system prompt typed before any notebook exists is held only in page state and is lost on reload. Chat parks its equivalent in browser storage. | `frontend/js/pages/notebook.js`, the prompt section's `set` | confirmed; it needs zero notebooks to bite | Park it the way chat does, through `frontend/js/storage.js`. |

## Docs and comments that state something false

| ID | What it says | Where |
|---|---|---|
| FB-30 | Describes a fixed `#bottom-nav`, removed in v2.0.212. | `tests/e2e/ios-sim.mjs`, header comment |
| FB-31 | Says the e2e harness shells out to it. The multi-engine arms that did were removed in v2.0.281. | `tests/helpers/engines.py`, docstring |
| FB-32 | Lists the pages as chat, notebook, models and perf. Presets and embeddings exist. | `.claude/rules/frontend.md`, Structure |

## Found while mapping the flows

Writing the flow maps in `docs/wiki/frontend/` turned up these. Each was read
from the code by the page's writer and none was run; treat them as leads. The
page named in the last column maps the flow each one sits in.

Chat:

| ID | Claim | Where | Page |
|---|---|---|---|
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
| FB-67 | Sign out revokes the session before the page's hide-time saves run, so a write still pending at that moment is refused. Inferred from the order of the code. | `frontend/js/api.js`, `logoutSession` | shell-and-lifecycle.md |
| FB-68 | The "stale cached module" hint is shown even when the page's module failed to load because of a sign-in refusal or a server that is down. | `frontend/js/app.js`, `staleModuleHint` | shell-and-lifecycle.md |
| FB-70 | The sign-in page's "misconfigured key" branches cannot be reached, because the gate answers first. | `src/heylook_llm/auth_api.py`, `login_page`, `login_submit` | shell-and-lifecycle.md |

Notebook:

| ID | Claim | Where | Page |
|---|---|---|---|
| FB-71 | While a notebook loads, the editor still shows the previous one and accepts typing. A save in that window writes the old title and body to the new notebook. | `frontend/js/pages/notebook.js`, `selectNotebook` | notebook.md |
| FB-76 | A notebook deleted on another device stays open, and every save then fails. | `notebook.js`, `refreshAfterResume` | notebook.md |

Models page:

| ID | Claim | Where | Page |
|---|---|---|---|
| FB-79 | Saving a discovered model's config answers with `source: "config"`, so its row loses "no entry" until the list is fetched again. | `src/heylook_llm/admin_api.py`, `update_model_config` | models-and-config.md |
| FB-81 | The fit gate is judged on unsaved drafts and stays after the panel closes. | `models.js`, `onFitGate` | models-and-config.md |
| FB-83 | Embedding metadata in the engine view would be labelled "live" and its arrays dropped. | `frontend/js/engine.js`, `slotRows` | models-and-config.md |

Perf:

| ID | Claim | Where | Page |
|---|---|---|---|
| FB-88 | "No profiling data yet" appears only when the request fails, because the server never sends an empty breakdown. | `frontend/js/pages/perf.js`, `renderProfile` | presets-and-perf.md |

Embeddings:

| ID | Claim | Where | Page |
|---|---|---|---|
| FB-93 | A file that is too large or of an unknown type is reported as "not in the chosen folder". | `embeddings.js`, the folder loader | embeddings.md |
| FB-95 | Neither mode checks the model's text-length limit before sending. | `embeddings.js`; `frontend/js/embed-kit.js`, `planBatches` | embeddings.md |

Comments and docs that say something false:

| ID | What it says | Where |
|---|---|---|
| FB-97 | Says `samplerParams` drops a zero; it keeps it. | `frontend/js/settings.js`, `sync` in `buildSettingsPanel` |
| FB-101 | Says video windows overlap; the server makes consecutive ones. | `docs/frontend_v3_user_guide.md`, the Embeddings section |
| FB-102 | Three false comments: the reload route's description of a default, the `saved` baseline comment, and the load panel's header. | `src/heylook_llm/admin_api.py`, `reload_model`; `frontend/js/model-config.js`, `createModelConfigEditor`; `frontend/js/load-panel.js` |

## Reported by a reviewer, not re-checked

These came from the review and were not confirmed against the current tree,
either because the file has changed since or because they depend on timing.

| ID | Claim | Where to look |
|---|---|---|
| FB-35 | A hide-time save is dispatched ahead of the save chain, so an older save already in flight can land after it and win. | `frontend/js/document-writer.js`, `putSystemPrompt` |
| FB-36 | Stopping the e2e server with `SIGKILL` signals `uv`, which may leave its Python child running. | `tests/e2e/lib/server.mjs`, `stop` |
