// Shared preset section -- the drawer section for any page whose document
// carries a system prompt + sampler params (chat conversations, notebooks).
//
// The drawer holds what the DOCUMENT runs on, and this section says where
// that came from and offers the two moves between a document and the preset
// store. Managing presets (edit, rename, duplicate, delete) is the Presets
// page's job (pages/presets.js), and that split is the point (owner,
// 2026-10-05): this section used to carry a preset <select>, a read-only
// preview of the selected preset, Save (overwrite), Save as new, Delete and a
// drift line, beside the document's own prompt box, and it was hard to say
// which prompt was in force or what each button would write where.
//
//   - TWO VERBS on the drawer's face, "Apply preset…" and "Save…", each
//     opening a sheet in place. No button here appears or disappears with the
//     document's state (owner, 2026-10-05: a better design, never at the cost
//     of added complexity; a state-dependent control is complexity)
//   - ONE PATH HERE OVERWRITES A STORED PRESET, and it cannot be aimed. Apply
//     copies preset -> document. The Save sheet always offers "Save as a new
//     preset" (document -> a NEW preset; a name in use is refused), and,
//     only while the document is stamped with X and differs from it, "Update
//     X": the document written back to the preset it came from. That is the
//     iterate loop (apply, tune the prompt in a conversation, write it back),
//     and its target is a property of the DOCUMENT, never of a control: no
//     select and no name box feeds it. The old Save took its target from a
//     select that browsing moved and a name box the select pre-filled, and
//     that cost a 35k-character prompt on 2026-08-28. Update names X on its
//     face and refuses when X's stored row moved since this drawer read it.
//     It arms in one case only: when the document has no prompt and X has
//     one, because that write leaves X in the list but inert ("my preset
//     disappeared"), the rule every version of this section has kept. Do not
//     give Update a way to choose a target
//   - the PROVENANCE LINE answers "which preset is this document running, and
//     is it still that preset?" without opening anything: "No preset", "From
//     preset X", or "X, modified: prompt and two knobs (...)". It reads the
//     document's stamp (applied_preset_id) and compares the document's bag to
//     that preset -- it is the old drift line's comparison, asked about the
//     one preset that matters instead of whichever one a select was parked on
//   - Apply is a PICKER: a list of every preset showing its own prompt, with
//     one Apply button per entry. You read what a preset holds before it is
//     copied anywhere, which is what the old preview existed for. It copies
//     (LM Studio semantics -- no live binding), and arms ("Replace prompt?")
//     only when it would replace a differing non-empty prompt: sampler knobs
//     are trivially recoverable, the prompt is typed work
//   - the prompt is an OVERRIDE BOX: a preset OWNS a system prompt and
//     carries it onto whatever it is applied to, but a preset with NO prompt
//     changes nothing -- the document keeps its own. Empty means "does not
//     speak for the prompt", never "set it to empty" (owner rule 2026-08-11)
//     -- see presetPrompt() below. Such a preset reads "settings only"
//     wherever it is named
//   - a preset is a COPY, never a link: editing the document never changes
//     the preset, and editing the preset (on its page) never changes the
//     document. After either, the provenance line says "modified"
//   - a NEW document made from an open one starts as that document's stamped
//     preset -- prompt + params + stamp -- via presetForNewDoc(), or blank.
//     With no document open the drawer is the draft and the next document is
//     created from exactly what it shows; Apply is the only way a preset gets
//     in, and it stamps the draft.
//
// Presets are global (one /v1/presets store); the prompt side is the page's
// document, adapted via getPrompt/setPrompt. The section subscribes to
// sampler changes itself (onSettingsChange, torn down with the mount); the
// page owns what it can't see: calling updateProvenance() from its
// prompt-input handler, wiring onDrawerOpen into its drawer contribution, and
// -- when it renders the applied-preset chip -- supplying docId/onIndicator,
// calling refresh() eagerly at mount (the chip needs preset names before the
// drawer's first lazy fetch), and calling syncIndicator() at EVERY point the
// active document changes (select/create/delete, including failure paths).

import { createEl, armedConfirm } from './utils.js';
import { api } from './api.js';
import { applySettings, snapshotSettings, samplerParams, withoutInapplicable, reconcileSettings, droppedNote, PARAM_META, onSettingsChange } from './settings.js';
import * as drawer from './settings-drawer.js';

const COUNT_WORDS = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
  'eleven', 'twelve'];
const countWord = (n) => COUNT_WORDS[n] ?? String(n);

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
//   model?():           {caps, thinking}|null -- the selected model
//                                       (settings.rowModel): a save keeps what
//                                       it can use, an apply removes the rest
//                                       and says so, provenance ignores it
//   noun?:              string       -- what the page calls its document
//                                       ('conversation', 'notebook'), for the
//                                       Save sheet's own words
// }
// A preset holds what the panel showed when it was saved (owner, 2026-09-27):
// saving on one model and applying on another removes what the second cannot
// use, named on the status line, rather than keeping it silently unused.
export function createPresetBar(ctx, { getPrompt, setPrompt, onStatus, docId, onIndicator, getStamp, setStamp,
                                       model = () => null, noun = 'document' }) {
  let presets = [];
  let provEl = null;  // latest built section's provenance line; detached writes are harmless
  let syncUpdate = null; // latest built section's "Update X" option: offered or not, with the line
  // The stamp -- which preset a document EXPLICITLY had applied/saved onto
  // it -- lives on the DOCUMENT (getStamp/setStamp -> applied_preset_id), so
  // provenance survives a reload and is the same on every device, like every
  // other piece of per-document state. What is stored stays strictly
  // explicit: Apply and Save as new write it and NOTHING else does. A
  // document whose state merely equals some preset is not claimed for it
  // (that inference fed the chip until Phase 3b; a second source of
  // "which preset is this" is how the chip and the drawer came to disagree).
  // A stamp naming a preset that no longer exists is self-healing: it
  // resolves to nothing here and the document reads "No preset".

  const fingerprint = () => JSON.stringify(presets.map((p) => [p.id, p.name, p.updated_at]));

  // The preset the document is stamped with, from the live list.
  const stamped = () => presets.find((p) => p.id === (getStamp?.() ?? null));

  // A preset's system_prompt is an OVERRIDE, not a value (owner rule
  // 2026-08-11): the preset OWNS a prompt and carries it onto whatever it is
  // applied to, but an EMPTY one means "this preset does not speak for the
  // prompt" -- applying it leaves the document's prompt exactly as it was,
  // rather than blanking it. Consequences, all deliberate: only a preset that
  // actually carries a prompt can replace one (so the armed "Replace prompt?"
  // is the true test of loss), a promptless preset can never eat typed work,
  // and provenance ignores the prompt for such a preset because it makes no
  // claim about it.
  const presetPrompt = (p) => p?.system_prompt || null;

  // The preset a NEW document should start from when one is created FROM an
  // open document (owner decision 2026-08-11: the preset is the unit of
  // continuity across documents): the open document's stamped preset, as the
  // PRESET stores it, not as the document has since drifted. Returns null
  // without one -- the page then starts the document blank, never from the
  // open document's own values (owner, 2026-09-26: they leaked into every new
  // chat). Starting-as counts as an apply, so the page passes the id as
  // applied_preset_id at create.
  //
  // With NO document open the page does not ask: the drawer then IS the next
  // document, and it is created from exactly what the drawer shows -- a
  // preset reaches it through Apply, like anywhere else.
  const presetForNewDoc = () => (docId?.() ? (stamped() ?? null) : null);

  // Resolves true when the list actually changed, so cosmetic repaints can be
  // skipped.
  async function refresh() {
    const before = fingerprint();
    try {
      const res = await api.listPresets({ signal: ctx.signal });
      presets = res.presets ?? [];
    } catch (err) {
      if (ctx.alive) onStatus(`Could not load presets: ${err.message}`, true);
    }
    return fingerprint() !== before;
  }

  // Drawer onOpen hook: lazily refresh the list, repaint only if it changed.
  // The provenance line is repainted DIRECTLY as well, because the rebuild is
  // skipped whenever focus is in the drawer body.
  function onDrawerOpen() {
    refresh().then((changed) => {
      if (!ctx.alive || !changed) return;
      drawer.requestRebuild();
      updateProvenance();
    });
  }

  // Which knobs differ between a preset and the live panel. Field-by-field
  // over PARAM_META, not JSON compare -- key order round-trips through the
  // server and can't be trusted.
  function samplerDrift(preset) {
    // Only what THIS model can use: a preset saved on another model is still
    // the one running when its other keys were removed on apply.
    const now = snapshotSettings();
    const saved = withoutInapplicable(preset.params, model());
    return Object.keys(PARAM_META).filter((k) => (now[k] ?? null) !== (saved[k] ?? null));
  }

  // Has anything this preset SPEAKS FOR changed? In its two halves, because
  // the provenance line names which -- one word covering both "you nudged
  // temperature" and "you rewrote the whole prompt" read as alarming after a
  // trivial edit and identical after a total rewrite.
  //
  // A promptless preset makes no claim about the prompt, so `prompt` is false
  // for it by construction -- the override-box rule falls out of presetPrompt()
  // rather than being restated here.
  function driftParts(preset) {
    const incoming = presetPrompt(preset);
    const fields = samplerDrift(preset);
    return {
      prompt: Boolean(incoming && incoming !== (getPrompt() ?? null)),
      // Which knobs, by their panel labels.
      fields: fields.map((k) => PARAM_META[k].label.toLowerCase()),
    };
  }

  // THE answer to "which preset is this document running, and is it still
  // that preset?". One function: the drawer's line, the bar chip and the
  // picker's "applied here" mark all read it, so they cannot disagree.
  // Null = no preset. The draft (no document open) has a stamp of its own,
  // written by Apply; the pages clear it whenever they enter that state.
  function provenance() {
    const preset = stamped();
    if (!preset) return null;
    const { prompt, fields } = driftParts(preset);
    return {
      id: preset.id,
      name: preset.name,
      carriesPrompt: Boolean(presetPrompt(preset)),
      prompt,
      fields,
      modified: prompt || fields.length > 0,
    };
  }

  // The same answer in words: a bold lead that stands alone, and the detail.
  function provenanceText() {
    const p = provenance();
    if (!p) return { lead: 'No preset', detail: '' };
    if (!p.modified) {
      return {
        lead: `From preset ${p.name}`,
        detail: p.carriesPrompt ? '' : ' (settings only: the prompt is not the preset\'s)',
      };
    }
    const what = [];
    if (p.prompt) what.push('prompt');
    if (p.fields.length) {
      what.push(`${countWord(p.fields.length)} ${p.fields.length === 1 ? 'knob' : 'knobs'} (${p.fields.join(', ')})`);
    }
    return { lead: `${p.name}, modified`, detail: `: ${what.join(' and ')}` };
  }

  // Would applying `preset` overwrite a non-empty document prompt with
  // something different? (The one destructive thing Apply can do.) A preset
  // carrying no prompt overrides nothing, so it is never destructive and
  // never arms.
  function wouldReplacePrompt(preset) {
    const incoming = presetPrompt(preset);
    const prompt = getPrompt();
    return Boolean(incoming && prompt && incoming !== prompt);
  }

  // The applied-preset chip's feed, for the active document.
  function indicatorInfo() {
    if (!docId?.()) return null;
    const p = provenance();
    return p ? { name: p.name, edited: p.modified } : null;
  }

  // What is in force for the SYSTEM PROMPT specifically, for a page that
  // surfaces it outside the drawer. Separate from indicatorInfo(), which
  // answers the whole-document question (prompt + samplers): a user asking
  // "what prompt am I running, and is it still the preset's?" is not served
  // by a chip that also flips on a temperature nudge. Only a preset that
  // CARRIES a prompt counts as its source -- a promptless one overrides
  // nothing, so it can neither claim the prompt nor be "modified" from.
  function promptState() {
    const prompt = getPrompt() ?? null;
    const preset = stamped();
    const source = presetPrompt(preset);
    return {
      prompt,
      presetName: source ? preset.name : null,
      modified: Boolean(source && prompt !== source),
    };
  }

  // Feed the page's chip. Public: pages call it on document switch/create --
  // the drawer may be closed then, so the provenance path can't be relied on.
  function syncIndicator() {
    onIndicator?.(indicatorInfo());
  }

  function paintProvenance(el) {
    const { lead, detail } = provenanceText();
    const [leadEl, detailEl] = el.children;
    // write-on-change: this runs per keystroke in the prompt editors
    if (leadEl.textContent !== lead) leadEl.textContent = lead;
    if (detailEl.textContent !== detail) detailEl.textContent = detail;
  }

  function updateProvenance() {
    syncIndicator(); // the chip tracks the same edits the line does
    if (provEl) paintProvenance(provEl);
    syncUpdate?.(); // the Save sheet offers "Update X" only while the line reads "X, modified"
  }

  // The section owns the sampler half of provenance-tracking (settings.js is
  // global, no page mediation needed); a consumer can't forget it and go
  // stale. After a drawer close the last section is detached -- drop the
  // reference so the dead subtree can be collected.
  ctx.onTeardown(onSettingsChange(ctx.guard(() => {
    // No early return: with the drawer closed the line is gone, but the
    // applied-preset chip still needs the sampler-edit sync.
    if (provEl && !provEl.isConnected) { provEl = null; syncUpdate = null; }
    updateProvenance();
  })));

  function apply(preset) {
    applySettings(preset.params ?? {});
    const removed = droppedNote(reconcileSettings(model()));
    // Override box: carry the prompt when the preset has one, otherwise
    // leave whatever the document (or the model's own default) uses.
    const incoming = presetPrompt(preset);
    if (incoming) setPrompt(incoming);
    // A draft (no document yet) is stamped too: the page carries it into
    // the document the first send creates.
    setStamp?.(preset.id);
    onStatus((incoming
      ? `Preset "${preset.name}" applied.`
      : `Preset "${preset.name}" applied. It carries no system prompt, so this one is unchanged.`)
      + (removed ? ` ${removed}` : ''));
    // Force: the Apply button lives in the drawer, so the focus guard would
    // otherwise skip the repaint that shows the applied values (and closes
    // the picker).
    drawer.requestRebuild({ force: true });
    // Explicitly, not via the settings-change listener: that fires from
    // applySettings BEFORE the prompt and stamp are written above, so relying
    // on it would paint the chips from pre-apply state (and not fire at all
    // for a preset carrying no sampler params).
    syncIndicator();
  }

  // One sentence, one place: the local check and the server's 409 say it.
  const nameTakenNote = (name) =>
    `A preset named "${name}" already exists. Pick another name, or edit that one on the Presets page.`;

  // Create under the typed name. Never overwrites: a taken name is a REFUSAL,
  // not a silent clobber. Decided against a FRESH list, because the local
  // cache can miss a name another device just added (and the server's own 409
  // is the real backstop). Resolves true when a preset was created.
  async function saveAsNew(name) {
    name = name.trim();
    if (!name) return false;
    try {
      await refresh();
      if (!ctx.alive) return false;
      if (presets.some((p) => p.name === name)) {
        // NO rebuild: force:true bypasses the focus guard, replaces the
        // section, and the name you just typed is gone -- with the keyboard
        // closed, on a phone -- at the exact moment you are told to pick
        // another.
        onStatus(nameTakenNote(name), true);
        return false;
      }
      const saved = await api.createPreset({
        name, system_prompt: getPrompt(), params: samplerParams(model()),
      });
      if (!ctx.alive) return false;
      presets.unshift(saved);
      // saving snapshots the current doc state -- the doc IS this preset now
      // (a draft included, as in apply)
      setStamp?.(saved.id);
      drawer.requestRebuild({ force: true });
      syncIndicator();
      onStatus(`Preset "${saved.name}" saved.`);
      return true;
    } catch (err) {
      if (!ctx.alive) return false;
      onStatus(err.status === 409 ? nameTakenNote(name) : `Preset save failed: ${err.message}`, true);
      return false;
    }
  }

  // Write the document back to the preset it is stamped with: the iterate
  // loop's second half. `built` is the stamped preset as the section that
  // owns the option read it; everything is re-asked here, because the click
  // is the only moment that matters and a hidden option can still be clicked
  // by a script.
  async function updateStamped(built) {
    const now = provenance();
    // Stamped with THIS preset, and differing from it: the only state in
    // which there is anything to write, and the only one the option shows in.
    if (!now || now.id !== built.id || !now.modified) return;
    try {
      // Refetch FIRST. The write keeps no history, so it must not land on a
      // row that changed after the line above was computed from it.
      await refresh();
      if (!ctx.alive) return;
      const target = presets.find((p) => p.id === built.id);
      if (!target) {
        onStatus(`Preset "${built.name}" no longer exists: it was deleted elsewhere.`, true);
        drawer.requestRebuild({ force: true });
        syncIndicator();
        return;
      }
      if (target.updated_at !== built.updated_at) {
        onStatus(`"${target.name}" changed elsewhere since this drawer read it. Nothing was written: `
          + 'look at it under Apply preset, then save again if you still mean to.', true);
        drawer.requestRebuild({ force: true });
        syncIndicator();
        return;
      }
      // No `name`: PUT patches only the fields it is given, and re-sending a
      // cached name reverts a rename made on another device.
      const prompt = getPrompt() ?? null;
      const saved = await api.updatePreset(target.id, {
        system_prompt: prompt, params: samplerParams(model()),
      });
      if (!ctx.alive) return;
      presets[presets.indexOf(target)] = saved;
      drawer.requestRebuild({ force: true });
      syncIndicator();
      // An empty prompt is written as none (the override-box rule): the
      // preset stays in the list and carries settings only. Say so, because
      // that is the write that reads as "my preset disappeared" later.
      onStatus(prompt
        ? `Preset "${saved.name}" updated from this one.`
        : `Preset "${saved.name}" updated from this one. It now carries no system prompt (settings only).`);
    } catch (err) {
      if (!ctx.alive) return;
      onStatus(`Preset update failed: ${err.message}`, true);
    }
  }

  // One entry of the Apply picker: the preset's name, what it carries, its
  // OWN prompt, and the button that copies it here. The text shown is the
  // object applied, so what was read is what lands.
  function buildOption(preset, current) {
    const carries = presetPrompt(preset);
    const applyBtn = armedConfirm(
      createEl('button', {
        type: 'button', class: 'btn btn--sm',
        title: carries ? 'Copy this preset here (prompt + settings)' : 'Copy this preset\'s settings here',
        'aria-label': `Apply preset ${preset.name}`,
      }, ['Apply']),
      () => apply(preset),
      'Replace prompt?',
      () => wouldReplacePrompt(preset),
      // What this Apply would do: copy THIS preset over THIS prompt. Either
      // half moving voids the arm (the prompt box is another drawer section
      // this one gets no events from).
      () => JSON.stringify([preset.id, getPrompt() ?? null]),
    );
    const marks = [];
    if (!carries) marks.push('settings only');
    if (current?.id === preset.id) marks.push(current.modified ? 'applied here, modified since' : 'applied here');
    return createEl('div', { class: 'preset-option', dataset: { name: preset.name } }, [
      createEl('div', { class: 'preset-option__head' }, [
        createEl('span', { class: 'preset-option__name' }, [preset.name]),
        ...marks.map((m) => createEl('span', { class: 'preset-option__mark' }, [m])),
        applyBtn,
      ]),
      carries
        ? createEl('div', { class: 'preset-option__prompt' }, [carries])
        : createEl('div', { class: 'preset-option__prompt preset-option__prompt--none' }, [
          'Carries no system prompt: applying it changes the settings and leaves the prompt as it is.',
        ]),
    ]);
  }

  function buildSection() {
    // role=status: the line changes live as the prompt or a knob is edited
    // -- announced, not just shown (DESIGN.md §7). .preset-provenance is the
    // E2E hook.
    provEl = createEl('div', {
      class: 'preset-provenance', role: 'status',
      title: 'A preset is a copy. Applying one stamps it here; later edits on either side '
        + 'never reach the other.',
    }, [
      createEl('strong', { class: 'preset-provenance__lead' }),
      createEl('span', { class: 'preset-provenance__detail muted' }),
    ]);
    paintProvenance(provEl);

    // ---- the Apply sheet: every preset, with its own prompt ---------------
    const pickerEl = createEl('div', {
      class: 'preset-sheet preset-picker', id: 'preset-picker', hidden: true,
      role: 'group', 'aria-label': 'Presets to apply',
    });
    const fillPicker = () => {
      const current = provenance();
      pickerEl.replaceChildren(...(presets.length
        ? presets.map((p) => buildOption(p, current))
        : [createEl('div', { class: 'settings-note muted small' }, [
          'No presets yet. Save makes one from what is shown here.',
        ])]));
    };

    // ---- the Save sheet: where this document's bag can be written ---------
    // "Update X" is built only for a document that HAS a stamped preset,
    // bound to that preset, and offered only while the document differs from
    // it. It sits first because it is the answer when it applies at all.
    const built = stamped();
    const blanks = () => Boolean(presetPrompt(presets.find((p) => p.id === built?.id)) && !getPrompt());
    const updateBtn = built ? armedConfirm(
      createEl('button', {
        type: 'button', class: 'btn btn--sm preset-save__update', hidden: true,
        title: `Overwrites the stored preset "${built.name}"`,
      }, [`Update ${built.name} with this ${noun}'s prompt and settings`]),
      () => updateStamped(built),
      `Remove ${built.name}'s prompt?`,
      // The one write here that costs something nothing on screen shows: an
      // empty prompt box written over a prompt the preset stores.
      blanks,
      // Destination, payload and the stored row being replaced: the prompt is
      // edited in another drawer section this one gets no events from, and a
      // list refresh can move the row. Any of them moving voids the arm.
      () => JSON.stringify([getStamp?.() ?? null, getPrompt() ?? null,
        presets.find((p) => p.id === built.id)?.updated_at ?? null]),
    ) : null;
    syncUpdate = updateBtn ? () => {
      const now = provenance();
      const offer = Boolean(now && now.id === built.id && now.modified);
      if (!offer || !blanks()) updateBtn.disarm();
      if (updateBtn.hidden === offer) updateBtn.hidden = !offer;
    } : null;
    syncUpdate?.();

    // Creates only. Enter in the name box goes straight to the create, which
    // is safe because there is no arm to get past: this cannot overwrite.
    const nameInput = createEl('input', {
      class: 'input', placeholder: 'Name for a new preset',
      'aria-label': 'Name for a new preset',
    });
    const createBtn = createEl('button', { type: 'button', class: 'btn btn--sm' }, ['Save as a new preset']);
    createBtn.addEventListener('click', () => saveAsNew(nameInput.value));
    nameInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') saveAsNew(nameInput.value);
    });
    const saveEl = createEl('div', {
      class: 'preset-sheet preset-save', id: 'preset-save', hidden: true,
      role: 'group', 'aria-label': 'Save this prompt and these settings',
    }, [
      updateBtn,
      createEl('div', { class: 'preset-row' }, [nameInput, createBtn]),
    ]);

    // ---- the two verbs ----------------------------------------------------
    const pickBtn = createEl('button', {
      type: 'button', class: 'btn btn--sm', 'aria-expanded': 'false', 'aria-controls': 'preset-picker',
      title: 'Choose a preset to copy here. The list shows each preset\'s own prompt first.',
    }, ['Apply preset…']);
    const saveBtn = createEl('button', {
      type: 'button', class: 'btn btn--sm', 'aria-expanded': 'false', 'aria-controls': 'preset-save',
      title: 'Store this prompt and these settings as a preset',
    }, ['Save…']);
    // One sheet at a time, under the verb that opened it.
    const show = (which) => {
      for (const [btn, sheet] of [[pickBtn, pickerEl], [saveBtn, saveEl]]) {
        const open = sheet === which;
        sheet.hidden = !open;
        btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      }
    };
    pickBtn.addEventListener('click', () => {
      if (!pickerEl.hidden) { show(null); return; }
      // Shown from the cache at once, then again if the store has moved: a
      // preset edited on its page a moment ago must read as it is now.
      fillPicker();
      show(pickerEl);
      refresh().then((changed) => { if (ctx.alive && changed && !pickerEl.hidden) fillPicker(); });
    });
    saveBtn.addEventListener('click', () => {
      if (!saveEl.hidden) { show(null); return; }
      show(saveEl);
    });

    return createEl('div', { class: 'preset-section' }, [
      provEl,
      createEl('div', { class: 'preset-row' }, [
        pickBtn, saveBtn,
        // A real link: the hash change closes the drawer and the router
        // mounts the page, the same way a nav item does.
        createEl('a', { class: 'preset-manage', href: '#/presets' }, ['Manage presets']),
      ]),
      saveEl,
      pickerEl,
    ]);
  }

  return { buildSection, onDrawerOpen, updateProvenance, refresh, syncIndicator, presetForNewDoc, promptState };
}

// The bar chip's one renderer -- fed by onIndicator above, so it lives here
// rather than in each page.
export function paintPresetChip(chip, info) {
  chip.hidden = !info;
  chip.textContent = info ? (info.edited ? `${info.name} (modified)` : info.name) : '';
}
