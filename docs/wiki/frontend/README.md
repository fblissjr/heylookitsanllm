# Frontend flow maps

last updated: 2026-11-24

These pages map what the frontend does, flow by flow: for each thing a person
or the browser can do, which functions run in which order, what each reads and
writes, which requests go out, what the user sees, every guard that can refuse,
every `await` and what is re-checked after it, and how each failure ends.

They sit one level below [frontend_core.md](../frontend_core.md), which says
how the frontend is built and why. Come here when you are about to change a
flow, or when something misbehaves and you need to know what ran.

## The pages

| Page | What it maps |
|---|---|
| [shell-and-lifecycle.md](shell-and-lifecycle.md) | Cold boot, routing, the `createPage` contract, one API call end to end, sign in and out, browser storage, the settings drawer shell, the hotkeys, and the shared primitives in `utils.js`. Every other page stands on this one. |
| [chat-documents.md](chat-documents.md) | The chat page from boot to an open conversation: the list, select, new, clone, rename, delete, the resume sync, composer drafts, the remembered conversation, and the no-conversation state. |
| [chat-generation.md](chat-generation.md) | One message from Send to a saved reply: `send`, `startStream`, the event stream, the painter, every branch of `finishGenerate`, Stop in both forms, abandoning a run, a run the page is not subscribed to, and a table of every way a run ends. |
| [chat-messages.md](chat-messages.md) | The message list: rendering and node reuse, the anatomy of a row, the scroll functions, Copy, the editor and its three saves, Regenerate and Delete. |
| [chat-model-and-settings.md](chat-model-and-settings.md) | Which model a conversation uses and how it loads, the engine chip and readout, the sampler settings from the panel to the wire, the system prompt, presets as chat uses them, and the prompt preview. |
| [chat-attachments.md](chat-attachments.md) | Images and audio: the three ways in, every gate, image preparation, cost and Fit, how a staged file becomes a stored block, media inside the editor, and what a model switch does to each. |
| [notebook.md](notebook.md) | The notebook page: the list, typing and the debounced save with all its flushes, Generate, Stop, and the resume sync. |
| [models-and-config.md](models-and-config.md) | The Models page and the per-model config editor: load, unload and reload, the schema-driven fields, drafts and Save, the fit meter and gate, the chat-template panel, and the engine, load-panel and context-select components. |
| [presets-and-perf.md](presets-and-perf.md) | The Presets page with import and export, and the Perf page with its observability settings and log viewer. |
| [embeddings.md](embeddings.md) | The Embeddings page: Compare, the Embedding Lab, and every function in `embed-kit.js`. |

## Where to look for a given action

| You are looking at | Start in |
|---|---|
| The page does not come up, or a page fails to mount | shell-and-lifecycle.md, "Cold boot" and "Route change" |
| A request fails or shows the wrong message | shell-and-lifecycle.md, the API call flow, then the "Requests" table of the page that made it |
| Opening, switching, creating or deleting a conversation | chat-documents.md |
| Something changed while the phone was locked or the tab was away | chat-documents.md, "Resume" |
| Pressing Send, a reply that stops, stalls or vanishes, Stop | chat-generation.md, and its table of endings |
| "Still generating on the server" | chat-generation.md, "A run the page is not subscribed to" |
| The list scrolls wrongly, a row is rebuilt, an editor loses its draft | chat-messages.md |
| Editing a message, Save & Regenerate, Save & Continue | chat-messages.md |
| Switching model, Load, the readout line | chat-model-and-settings.md |
| A setting that did not reach the model, a preset that did not apply | chat-model-and-settings.md |
| An image or audio clip that was refused, lost or not sent | chat-attachments.md |
| A model's config, its chat template, the fit meter | models-and-config.md |

## How to read a page

Every page has the same sections in the same order: the files and what each
owns, a state table, an entry-point table, the flows, the requests the area
makes, the exact strings the user sees, the invariants and where each is
enforced, the calls into other areas, the open bugs that touch the area, and
what was not mapped or not certain.

Code is named by file and symbol, never by line, so a page survives edits that
only move lines. Strings the user sees are quoted exactly. A page says why
something is the way it is only by linking to
[sharp_edges.md](../../architecture/sharp_edges.md).

## How far to trust them

The pages were written on 2026-11-24 against the tree at commit `7181110f`, each
by a reader that read its files in full. Two checks followed. A script looked
up every file, function, CSS class, route and quoted string each page names in
the source, and found none that does not exist. A sample of flows was then read
against the code step by step: selecting a conversation, and the watcher for a
run the page is not subscribed to. Both matched.

That is a check that the names are real and that a sample of sequences is
right. It is not a proof that every step of every flow is in the right order.
When a page and the code disagree, the code is right and the page needs the
fix. The last section of each page lists what its writer could not determine,
which is mostly what only a browser or a phone can answer.

## Keeping them true

A change to a flow moves its page in the same commit, the same rule
`AGENTS.md` sets for the rest of the wiki. A step that is no longer true is
worse than a missing one, so delete what you cannot keep current rather than
leaving it.

Defects found while mapping are not recorded here. They are in
[frontend_open_bugs.md](../../project/frontend_open_bugs.md).
