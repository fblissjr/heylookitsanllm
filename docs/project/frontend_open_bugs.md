# Frontend: open bugs

last updated: 2026-11-24

Defects found by the 2026-11-24 review of `frontend/` that are still open. What
that review fixed is in `CHANGELOG.md` (2.0.279, 2.0.281, 2.0.282, 2.0.283, 2.0.284, 2.0.285) and is not
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
| FB-04 | While editing a long message with the keyboard up, Save and Cancel sit below the keyboard. The editor's height cap follows the layout viewport, which the keyboard does not shrink. | `app.css`, `.message-edit textarea` max-height; `chat.js`, `growCap` in `buildEditEl` | needs a device | Cap from the visual viewport. The two caps must move together (`frontend/js/utils.js`, `autoGrow`). |
| FB-12 | Nothing limits overscroll on the page or its scrollers, so a pull down on the phone can reload the page in the middle of a generation. | `app.css`: `overscroll-behavior` appears on three inner panes only | needs a device | `overscroll-behavior-y: none` on `html`, `contain` on the message list, the drawer and the list panes. |

## Phone layout and styles

| ID | What happens | Where | Status | Smallest fix |
|---|---|---|---|---|
| FB-17 | The home-screen manifest hard-codes the dark theme colours, while `index.html` carries a light and a dark pair that a unit test checks. | `frontend/manifest.json` | confirmed; what the launch screen shows in light mode needs a device | Derive it, or accept one colour and say so in the manifest's neighbour comment. |
| FB-19 | A phone held sideways is wider than the phone breakpoint and gets the desktop layout: sidebar, list pane and a thread narrower than in portrait. | `app.css`, the `(max-width: 767px)` breakpoint | needs a device | Add a height condition to the phone breakpoint. The value is repeated across the file, so this is several edits. |

## Found while mapping the flows

Writing the flow maps in `docs/wiki/frontend/` turned up these. Each was read
from the code by the page's writer and none was run; treat them as leads. The
page named in the last column maps the flow each one sits in.

Chat:

| ID | Claim | Where | Page |
|---|---|---|---|
| FB-39 | After the first send creates a conversation, the page takes it as the open one without checking that nothing else was opened during the create. A conversation opened in that moment is overridden, and its prompt can be written onto the new one. | `chat.js`, `send`, after `createConversation` | chat-generation.md |
| FB-44 | New from the no-conversation state carries the typed text and drops staged attachments without saying so. | `chat.js`, `newConversation` | chat-documents.md |
| FB-50 | Stop's local abort on a "no active run" answer assumes the generate request never reached the server. If it did, the run continues detached. | `chat.js`, `stopStream` | chat-generation.md |
| FB-57 | The composer's prompt preview leaves staged attachments out without saying so, and does not flush pending settings or prompt writes first. | `chat.js`, `previewNextPrompt` | chat-model-and-settings.md |
| FB-59 | With thinking off and a depth level that means off, the first press of the thinking button leaves thinking off. Latent. | `chat.js`, the thinking button's click handler | chat-model-and-settings.md |

Shell and shared modules:

| ID | Claim | Where | Page |
|---|---|---|---|
| FB-67 | Sign out revokes the session before the page's hide-time saves run, so a write still pending at that moment is refused. Inferred from the order of the code. | `frontend/js/api.js`, `logoutSession` | shell-and-lifecycle.md |
| FB-68 | The "stale cached module" hint is shown even when the page's module failed to load because of a sign-in refusal or a server that is down. | `frontend/js/app.js`, `staleModuleHint` | shell-and-lifecycle.md |
| FB-70 | The sign-in page's "misconfigured key" branches cannot be reached, because the gate answers first. | `src/heylook_llm/auth_api.py`, `login_page`, `login_submit` | shell-and-lifecycle.md |

Embeddings:

| ID | Claim | Where | Page |
|---|---|---|---|
| FB-95 | Neither mode checks the model's text-length limit before sending. | `embeddings.js`; `frontend/js/embed-kit.js`, `planBatches` | embeddings.md |

## Reported by a reviewer, not re-checked

These came from the review and were not confirmed against the current tree,
either because the file has changed since or because they depend on timing.

| ID | Claim | Where to look |
|---|---|---|
| FB-35 | A hide-time save is dispatched ahead of the save chain, so an older save already in flight can land after it and win. | `frontend/js/document-writer.js`, `putSystemPrompt` |
| FB-36 | Stopping the e2e server with `SIGKILL` signals `uv`, which may leave its Python child running. | `tests/e2e/lib/server.mjs`, `stop` |
