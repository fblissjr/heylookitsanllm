// Shared per-document system-prompt editor -- the drawer section for any
// page whose document carries a system prompt (chat conversations,
// notebooks). Extracted from chat.js's original per-conversation prompt
// section, which fixed a real data-loss bug (see the commit-per-keystroke
// comment below); this factory is now the only place that logic lives, so a
// page can't drift from it by copy-paste.
//
// adapter = {
//   owner():                string|null  -- the document id this edit belongs to
//   get():                  string|null  -- the document's current prompt (may be '' or null)
//   set(value, ownerId):    void         -- commit to page state (per keystroke)
//   persist(value, ownerId):void         -- write to the server (debounced)
//   onEdit?():              void         -- optional, fires after every keystroke
//                                           (preset-bar drift tracking)
// }
//
// Returns { element, setValue, flush }:
//   element   the <details> to mount as a drawer section
//   setValue  external sync (document switch, preset apply): repaints the
//             textarea AND re-anchors which document new edits target
//   flush     force any pending debounced persist to fire now -- call before
//             switching the active document out from under a long-lived
//             instance (see notebook.js's selectNotebook)

import { createEl, autoGrow, debounce } from './utils.js';

const normalize = (raw) => raw.trim() || null;

export function createPromptSection(ctx, adapter) {
  const input = createEl('textarea', {
    class: 'sysprompt-input', rows: 6,
    placeholder: 'Optional system prompt for this document…',
    value: adapter.get() ?? '',
  });

  // The document this widget is currently anchored to. A page that builds a
  // fresh instance per document (chat: one per drawer build) captures this
  // once and never calls setValue; a page that reuses one instance across
  // documents (notebook) calls setValue on every switch, which re-anchors
  // this to the live owner -- so the anchor tracks "live" there in effect.
  let builtFor = adapter.owner();

  // Commit state per keystroke, debounce only the persist. The old
  // save-on-blur commit lost the text whenever the field was removed under
  // focus (drawer close via Escape/hashchange): a removed textarea never
  // fires `change`, and a preset saved in that window captured
  // system_prompt=null, so applying it later erased the prompt outright.
  const schedulePersist = debounce((value) => {
    // null: a pre-create draft, or -- for a reused instance -- no document
    // loaded yet. State rides along until something gives it a home.
    if (builtFor != null) adapter.persist(value, builtFor);
  }, 400);

  const currentValue = () => normalize(input.value);

  input.addEventListener('input', () => {
    autoGrow(input, 480);
    const value = currentValue();
    // A stale editor (document changed under an unrebuilt/unresynced widget)
    // must not write onto the NEW document -- the edit still belongs to the
    // one it was typed for.
    if (adapter.owner() === builtFor) adapter.set(value, builtFor);
    schedulePersist(value);
    adapter.onEdit?.();
  });
  // blur flushes immediately so a follow-up action (preset save/apply, a
  // document switch) can never race the debounce timer.
  input.addEventListener('change', () => schedulePersist.flush(currentValue()));

  // grow to fit existing content once the drawer has appended it
  queueMicrotask(() => autoGrow(input, 480));

  // Always expanded: a collapsed-when-empty details hid the field behind an
  // extra click and read as "my prompt disappeared".
  const element = createEl('details', { class: 'sysprompt', open: true }, [
    createEl('summary', {}, ['System prompt']),
    input,
  ]);

  function setValue(value) {
    input.value = value ?? '';
    autoGrow(input, 480);
    // this sync IS the document-switch point for a reused instance
    builtFor = adapter.owner();
    // an external sync must not be followed by a stale write racing behind it
    schedulePersist.cancel();
  }

  function flush() {
    schedulePersist.flush(currentValue());
  }

  // On teardown FLUSH, never cancel: navigating away within the debounce
  // window is the same data-loss shape as the drawer closing under focus --
  // the text is typed, the user believes it is saved, and no `change` fires
  // for a removed field. The write targets the captured document id, so it
  // stays correct after the mount is gone; flush() is a no-op when nothing
  // is pending.
  ctx.onTeardown(() => flush());

  return { element, setValue, flush };
}
