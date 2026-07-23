// Shared preset bar -- the drawer section for any page whose document carries
// a system prompt + sampler params (chat conversations, notebooks). One
// grammar everywhere:
//
//   - the <select> is INERT: it records the selection, prefills the save-as
//     name, and drives the drift line -- it never touches the document
//   - Apply is an explicit button that COPIES the preset onto the document
//     (LM Studio semantics -- no live binding; later edits don't touch the
//     preset until Save), armed-confirmed ("Replace prompt?") only when it
//     would replace a differing non-empty prompt -- sampler knobs are
//     trivially recoverable, the prompt is typed work
//   - Save snapshots the current prompt + the whole sampler panel under the
//     typed name; upsert-by-name is decided against a FRESH list (the local
//     cache can hide a name the server has -> 409, or list one it no longer
//     has -> 404)
//   - the drift line says what Apply/Save would DO to the selected preset,
//     updated in place -- the drawer's focus guard means a rebuild can't be
//     relied on while the user is typing in a field
//
// Presets are global (one /v1/presets store); the prompt side is the page's
// document, adapted via getPrompt/setPrompt. The bar subscribes to sampler
// changes itself (onSettingsChange, torn down with the mount); the page owns
// what the bar can't see: calling updateDrift() from its prompt-input
// handler, wiring onDrawerOpen into its drawer contribution, and -- when it
// renders the applied-preset chip -- supplying docId/onIndicator, calling
// refresh() eagerly at mount (the chip needs preset names before the
// drawer's first lazy fetch), and calling syncIndicator() at EVERY point the
// active document changes (select/create/delete, including failure paths).

import { createEl, armedConfirm } from './utils.js';
import { api } from './api.js';
import { applySettings, snapshotSettings, PARAM_META, onSettingsChange } from './settings.js';
import * as drawer from './settings-drawer.js';

// adapter = {
//   getPrompt():        string|null  -- the document's current system prompt
//   setPrompt(v|null):  void         -- apply copies the preset's prompt here
//   onStatus(text, isError?): void   -- the page's status line
//   docId?():           string|null  -- the active document (conversation/
//                                       notebook) id; enables the indicator
//   onIndicator?(info): void         -- applied-preset chip feed: null, or
//                                       { name, edited } for the active doc
//   getStamp?():        string|null  -- the active document's stored
//                                       applied_preset_id
//   setStamp?(id|null): void         -- persist it (the page owns the write,
//                                       same division as setPrompt)
// }
export function createPresetBar(ctx, { getPrompt, setPrompt, onStatus, docId, onIndicator, getStamp, setStamp }) {
  let presets = [];
  let presetId = null;   // select-box state only -- applying copies, not binds
  let driftEl = null;    // latest built section's line; detached writes are harmless
  // The stamp -- which preset a document EXPLICITLY had applied/saved onto
  // it -- lives on the DOCUMENT (getStamp/setStamp -> applied_preset_id), so
  // provenance survives a reload and is the same on every device, like every
  // other piece of per-document state in v3. What is stored stays strictly
  // explicit: apply/save write it, delete clears it, and NOTHING else. A
  // document whose state merely equals a preset is reported live by
  // indicatorInfo() and never written -- a stored inference could bind the
  // WRONG doc's state to a doc id (mid-switch, failed load) and then persist
  // as a false "(edited)" claim. A stamp naming a preset that no longer
  // exists is self-healing: it simply falls through to inference below.

  const fingerprint = () => JSON.stringify(presets.map((p) => [p.id, p.name, p.updated_at]));
  const selected = () => presets.find((p) => p.id === presetId);

  // Resolves true when the list (or the selection's validity) actually
  // changed, so cosmetic repaints can be skipped.
  async function refresh() {
    const before = fingerprint();
    try {
      const res = await api.listPresets({ signal: ctx.signal });
      presets = res.presets ?? [];
    } catch (err) {
      if (ctx.alive) onStatus(`Could not load presets: ${err.message}`, true);
    }
    let changed = fingerprint() !== before;
    if (!presets.some((p) => p.id === presetId)) {
      changed ||= presetId !== null;
      presetId = null;
    }
    return changed;
  }

  async function refreshAndRepaint() {
    await refresh();
    if (ctx.alive) drawer.requestRebuild({ force: true });
  }

  // Drawer onOpen hook: lazily refresh the list, repaint only if it changed.
  function onDrawerOpen() {
    refresh().then((changed) => {
      if (ctx.alive && changed) drawer.requestRebuild();
    });
  }

  // Does the selected preset match the live state (document prompt + the
  // whole sampler panel)? Field-by-field over PARAM_META, not JSON compare --
  // key order round-trips through the server and can't be trusted.
  function matchesState(preset) {
    if ((preset.system_prompt ?? null) !== (getPrompt() ?? null)) return false;
    const now = snapshotSettings();
    const saved = preset.params ?? {};
    return Object.keys(PARAM_META).every((k) => (now[k] ?? null) === (saved[k] ?? null));
  }

  // Would applying overwrite a non-empty document prompt with something
  // different? (The one destructive thing Apply can do.)
  function wouldReplacePrompt() {
    const preset = selected();
    const prompt = getPrompt();
    return Boolean(preset && prompt && (preset.system_prompt ?? null) !== prompt);
  }

  // The applied-preset info for the active document. An explicit stamp
  // (apply/save) tracks drift ("edited"); without one, an exact state match
  // is reported live -- true in effect under copy semantics -- but NOT
  // stored, so a coincidental or stale-state match can never turn into a
  // persistent false claim (it disappears the moment state diverges).
  function indicatorInfo() {
    const doc = docId?.();
    if (!doc) return null;
    const stamped = presets.find((p) => p.id === getStamp?.());
    if (stamped) return { name: stamped.name, edited: !matchesState(stamped) };
    const match = presets.find((p) => matchesState(p));
    return match ? { name: match.name, edited: false } : null;
  }

  // Feed the page's chip. Public: pages call it on document switch/create --
  // the drawer may be closed then, so the drift-line path can't be relied on.
  function syncIndicator() {
    onIndicator?.(indicatorInfo());
  }

  function updateDrift() {
    syncIndicator(); // chip tracks the same edits the drift line does
    if (!driftEl) return;
    const preset = selected();
    const next = !preset ? ''
      : matchesState(preset)
        ? 'Matches current settings.'
        : 'Differs from current settings -- Apply copies it here, Save overwrites it.';
    // write-on-change: this runs per keystroke in the prompt editors
    if (driftEl.textContent !== next) driftEl.textContent = next;
    if (driftEl.hidden !== !preset) driftEl.hidden = !preset;
  }

  // The bar owns the sampler half of drift-tracking (settings.js is global,
  // no page mediation needed); a consumer can't forget it and go stale.
  // After a drawer close the last section is detached -- drop the reference
  // so the dead subtree can be collected and later changes cost one check.
  ctx.onTeardown(onSettingsChange(ctx.guard(() => {
    // No early return: with the drawer closed the drift line is gone, but
    // the applied-preset chip still needs the sampler-edit sync.
    if (driftEl && !driftEl.isConnected) driftEl = null;
    updateDrift();
  })));

  function apply() {
    const preset = selected();
    if (preset) {
      applySettings(preset.params ?? {});
      setPrompt(preset.system_prompt ?? null);
      if (docId?.()) setStamp?.(preset.id);
      onStatus(`Preset "${preset.name}" applied.`);
    }
    // Force: the Apply button lives in the drawer, so the focus guard would
    // otherwise skip the repaint that shows the applied values.
    drawer.requestRebuild({ force: true });
  }

  async function save(name) {
    name = name.trim();
    if (!name) return;
    const body = { name, system_prompt: getPrompt(), params: snapshotSettings() };
    try {
      await refresh();
      if (!ctx.alive) return;
      const existing = presets.find((p) => p.name === name);
      const saved = existing
        ? await api.updatePreset(existing.id, body)
        : await api.createPreset(body);
      if (!ctx.alive) return;
      // the server just returned the row -- patch the (just-refreshed) local
      // list instead of fetching it a second time
      const idx = presets.findIndex((p) => p.id === saved.id);
      if (idx >= 0) presets[idx] = saved;
      else presets.unshift(saved);
      presetId = saved.id;
      // saving snapshots the current doc state -- the doc IS this preset now
      if (docId?.()) setStamp?.(saved.id);
      drawer.requestRebuild({ force: true });
      onStatus(`Preset "${name}" ${existing ? 'updated' : 'saved'}.`);
    } catch (err) {
      if (ctx.alive) onStatus(`Preset save failed: ${err.message}`, true);
    }
  }

  async function remove() {
    if (!presetId) return;
    const removedId = presetId;
    try {
      await api.deletePreset(removedId);
    } catch (err) {
      if (ctx.alive) onStatus(`Preset delete failed: ${err.message}`, true);
      return;
    }
    if (!ctx.alive) return;
    presetId = null;
    // Only the ACTIVE document is cleared. Other documents may still name the
    // deleted preset, which is harmless -- indicatorInfo resolves stamps
    // against the live preset list, so a dangling id reads as "no stamp".
    if (getStamp?.() === removedId) setStamp?.(null);
    await refreshAndRepaint();
    syncIndicator(); // rebuild no-ops while the drawer is closed -- sync anyway
  }

  function buildSection() {
    const current = selected();

    // aria-label, not just title: `title` is a tooltip whose exposure as an
    // accessible name is inconsistent, and neither control has a visible
    // label to associate (the bar is deliberately one compact row).
    const select = createEl('select', {
      title: 'Select a saved preset', 'aria-label': 'Select a saved preset',
    }, [
      createEl('option', { value: '' }, ['Presets…']),
      ...presets.map((p) => createEl('option', { value: p.id }, [p.name])),
    ]);
    select.value = presetId ?? '';

    const applyBtn = armedConfirm(
      createEl('button', {
        class: 'btn btn--sm', disabled: !current,
        title: 'Copy this preset here (prompt + sampler settings)',
      }, ['Apply']),
      apply,
      'Replace prompt?',
      wouldReplacePrompt,
    );
    const delBtn = armedConfirm(
      createEl('button', { class: 'btn btn--sm btn--ghost', disabled: !current }, ['Del']),
      remove,
    );

    // Save under the typed name: matches an existing preset -> overwrite it,
    // new name -> create. Picking a preset pre-fills its name for overwrite.
    const nameInput = createEl('input', {
      class: 'input', placeholder: 'Save as…', value: current?.name ?? '',
      'aria-label': 'Preset name to save as',
    });
    const saveBtn = createEl('button', { class: 'btn btn--sm' }, ['Save']);
    saveBtn.addEventListener('click', () => save(nameInput.value));
    nameInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') save(nameInput.value);
    });

    select.addEventListener('change', () => {
      presetId = select.value || null;
      const p = selected();
      nameInput.value = p?.name ?? '';
      applyBtn.disabled = delBtn.disabled = !p;
      updateDrift();
    });

    // .preset-drift is the E2E hook; styling rides the shared settings-note.
    // role=status: the line flips live (Matches/Differs) -- announced, not
    // just shown (DESIGN.md §7).
    driftEl = createEl('div', {
      class: 'preset-drift settings-note muted small', hidden: true, role: 'status',
      title: 'Presets are copies: Apply stamps the preset onto this document; '
        + 'later edits here never change the preset until you Save it again.',
    });
    updateDrift();

    return createEl('div', { class: 'preset-section' }, [
      createEl('h3', {}, ['Preset']),
      createEl('div', { class: 'preset-row' }, [select, applyBtn, delBtn]),
      driftEl,
      createEl('div', { class: 'preset-row' }, [nameInput, saveBtn]),
    ]);
  }

  return { buildSection, onDrawerOpen, updateDrift, refresh, syncIndicator };
}
