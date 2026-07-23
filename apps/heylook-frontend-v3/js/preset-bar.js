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
// document, adapted via getPrompt/setPrompt. The page owns calling
// updateDrift() from its prompt-input handler and its onSettingsChange
// listener, and refresh() from its drawer onOpen.

import { createEl, armedConfirm } from './utils.js';
import { api } from './api.js';
import { applySettings, snapshotSettings, PARAM_META } from './settings.js';
import * as drawer from './settings-drawer.js';

// adapter = {
//   getPrompt():        string|null  -- the document's current system prompt
//   setPrompt(v|null):  void         -- apply copies the preset's prompt here
//   onStatus(text, isError?): void   -- the page's status line
// }
export function createPresetBar(ctx, { getPrompt, setPrompt, onStatus }) {
  let presets = [];
  let presetId = null;   // select-box state only -- applying copies, not binds
  let driftEl = null;    // latest built section's line; detached writes are harmless

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

  function updateDrift() {
    if (!driftEl) return;
    const preset = selected();
    if (!preset) {
      driftEl.hidden = true;
      driftEl.textContent = '';
      return;
    }
    driftEl.hidden = false;
    driftEl.textContent = matchesState(preset)
      ? 'Matches current settings.'
      : 'Differs from current settings -- Apply copies it here, Save overwrites it.';
  }

  function apply() {
    const preset = selected();
    if (preset) {
      applySettings(preset.params ?? {});
      setPrompt(preset.system_prompt ?? null);
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
      presetId = saved.id;
      await refreshAndRepaint();
      if (ctx.alive) onStatus(`Preset "${name}" ${existing ? 'updated' : 'saved'}.`);
    } catch (err) {
      if (ctx.alive) onStatus(`Preset save failed: ${err.message}`, true);
    }
  }

  async function remove() {
    if (!presetId) return;
    try {
      await api.deletePreset(presetId);
    } catch (err) {
      if (ctx.alive) onStatus(`Preset delete failed: ${err.message}`, true);
      return;
    }
    if (!ctx.alive) return;
    presetId = null;
    await refreshAndRepaint();
  }

  function buildSection() {
    const current = selected();

    const select = createEl('select', { title: 'Select a saved preset' }, [
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

    driftEl = createEl('div', {
      class: 'preset-drift muted small', hidden: true,
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

  return { buildSection, refresh, updateDrift };
}
