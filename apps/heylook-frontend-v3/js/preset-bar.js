// Shared preset bar -- the drawer section for any page whose document carries
// a system prompt + sampler params (chat conversations, notebooks). One
// grammar everywhere:
//
//   - the <select> is INERT toward the document: it records the selection,
//     prefills the save-as name, and drives the drift line -- it never
//     writes. It does READ, though: with no explicit pick it shows the
//     document's applied_preset_id, so the control that asks "which preset
//     is this?" can answer it (it used to say "Presets…" on a document that
//     was running one, which is what sent people clicking through it)
//   - Apply is an explicit button that COPIES the preset onto the document
//     (LM Studio semantics -- no live binding; later edits don't touch the
//     preset until Save), armed-confirmed ("Replace prompt?") only when it
//     would replace a differing non-empty prompt -- sampler knobs are
//     trivially recoverable, the prompt is typed work
//   - BOTH directions are armed, and Save is the one that matters: Apply
//     overwrites the DOCUMENT (re-apply to get it back), Save overwrites the
//     STORED PRESET with an UPDATE that keeps no history. See
//     wouldOverwritePresetPrompt() for the three questions it asks, and why
//     the iterate loop is deliberately exempt
//   - re-aiming DISARMS: an arm is a promise about one target, so changing
//     the select or the name box cancels a pending confirm
//   - the prompt is an OVERRIDE BOX: a preset OWNS a system prompt and
//     carries it onto whatever it is applied to, but a preset with NO prompt
//     changes nothing -- the conversation keeps its own prompt (or the
//     model's default). Empty means "does not speak for the prompt", never
//     "set it to empty" (owner rule 2026-08-11) -- see presetPrompt() below
//   - Save snapshots the current prompt + the whole sampler panel under the
//     typed name; upsert-by-name is decided against a FRESH list (the local
//     cache can hide a name the server has -> 409, or list one it no longer
//     has -> 404). The ARM, unlike the save, is decided against the local
//     list -- armedConfirm needs a synchronous answer. See
//     wouldOverwritePresetPrompt() for what that costs and where
//   - the section carries a read-only preview of the SELECTED preset's own
//     prompt. The per-document prompt box below is a DIFFERENT thing and
//     says so in its own label; without the preview there was no way to see
//     what a preset held short of applying it
//   - the drift line says what Apply/Save would DO to the selected preset,
//     updated in place -- the drawer's focus guard means a rebuild can't be
//     relied on while the user is typing in a field
//   - a NEW document is the one exception to "the select never touches the
//     document": it starts as the selected (or, after a reload, the stamped)
//     preset -- prompt + params + stamp -- via presetForNewDoc(). Existing
//     documents still change only on an explicit Apply.
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
  // Select-box state only -- applying copies, it never binds. THREE states,
  // not two: `undefined` = no explicit pick yet, so the select FOLLOWS the
  // active document's stamp; `null` = explicitly deselected; an id = an
  // explicit pick. See effectiveId().
  let presetId;
  let selectionDoc = null; // the document the explicit pick was made on
  let driftEl = null;    // latest built section's line; detached writes are harmless
  let previewEl = null;  // read-only view of the SELECTED preset's own prompt
  let previewSummaryEl = null;
  let previewBodyEl = null;
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

  // Which preset the select shows. An explicit pick wins, but only for the
  // document it was made ON -- switching documents falls back to the new
  // document's stamp, so the dropdown reports what the document is actually
  // running rather than sitting on "Presets…" (or, worse, on the previous
  // document's pick while the prompt box below shows this document's text).
  // A stamp naming a deleted preset resolves to nothing -- the same
  // self-healing described in the stamp note above.
  const effectiveId = () => {
    const doc = docId?.() ?? null;
    if (presetId !== undefined && selectionDoc === doc) return presetId;
    return doc ? (getStamp?.() ?? null) : (presetId ?? null);
  };
  const selected = () => presets.find((p) => p.id === effectiveId());

  // Record an explicit pick. Always paired with the document it was made on
  // -- an id without one would leak the pick onto the next document.
  function pick(id) {
    presetId = id;
    selectionDoc = docId?.() ?? null;
  }

  // A preset's system_prompt is an OVERRIDE, not a value (owner rule
  // 2026-08-11): the preset OWNS a prompt and carries it onto whatever it is
  // applied to, but an EMPTY one means "this preset does not speak for the
  // prompt" -- applying it leaves the document's prompt exactly as it was,
  // rather than blanking it. Consequences, all deliberate: only a preset that
  // actually carries a prompt can replace one (so the armed "Replace prompt?"
  // is the true test of loss), a promptless preset can never eat typed work,
  // and drift/match ignore the prompt for such a preset because it makes no
  // claim about it. This is also what defuses the accidental blank Save: a
  // preset stored with no prompt is inert, not destructive.
  const presetPrompt = (p) => p?.system_prompt || null;

  // The preset a NEW document should start from (owner decision 2026-08-11:
  // the selected preset is the unit of continuity across documents). The
  // explicit bar selection wins; else the active document's stamp, so the
  // behavior survives a reload (selection is session state, the stamp is
  // durable). Returns null when neither exists -- the page then creates the
  // document exactly as before. Starting-as counts as an apply, so the page
  // passes the id as applied_preset_id at create.
  // (effectiveId already folds the stamp in behind an explicit pick, so
  // selected() alone expresses both halves of that rule.)
  const presetForNewDoc = () => selected() ?? null;

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
    // Only an EXPLICIT pick is cleared when it goes missing; a stamp that no
    // longer resolves is left alone and self-heals through effectiveId.
    if (presetId != null && !presets.some((p) => p.id === presetId)) {
      presetId = null;
      changed = true;
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
  function samplersMatch(preset) {
    const now = snapshotSettings();
    const saved = preset.params ?? {};
    return Object.keys(PARAM_META).every((k) => (now[k] ?? null) === (saved[k] ?? null));
  }

  // DRIFT sense, for a preset the document explicitly carries: has anything
  // this preset SPEAKS FOR changed? A promptless preset makes no claim about
  // the prompt, so editing the prompt cannot make it "(edited)".
  function matchesState(preset) {
    const incoming = presetPrompt(preset);
    if (incoming && incoming !== (getPrompt() ?? null)) return false;
    return samplersMatch(preset);
  }

  // IDENTITY sense, for the unstamped fallback in indicatorInfo(): does this
  // preset account for the document's WHOLE state, prompt included? Strictly
  // stricter than matchesState on purpose -- the drift sense would let a
  // promptless preset at default samplers "match" any default conversation,
  // so the chip would claim provenance for a hand-typed prompt the preset
  // never carried. Presets that lost their prompt to the v1.62.3 bug are
  // exactly the ones that would have made that false claim.
  function equalsState(preset) {
    return (preset.system_prompt ?? null) === (getPrompt() ?? null) && samplersMatch(preset);
  }

  // Would applying overwrite a non-empty document prompt with something
  // different? (The one destructive thing Apply can do.) A preset carrying no
  // prompt overrides nothing, so it is never destructive and never arms.
  function wouldReplacePrompt() {
    const incoming = presetPrompt(selected());
    const prompt = getPrompt();
    return Boolean(incoming && prompt && incoming !== prompt);
  }

  // The mirror of wouldReplacePrompt, for the direction that ACTUALLY loses
  // work. Apply overwrites the DOCUMENT, which is recoverable -- the preset
  // is still there to re-apply. Save overwrites the STORED PRESET with an
  // UPDATE that keeps no history, so a preset's prompt is gone the moment it
  // is written over. The guard was on the recoverable direction only; that
  // inverted the owner's only-loss rule and cost a 35k-character prompt on
  // 2026-08-28, when picking a preset from the select (which pre-fills this
  // very name box) and pressing Save wrote the DOCUMENT's prompt over it.
  //
  // Arms when Save would not leave a stored prompt as it is -- but NOT for
  // the loop this bar exists to serve. Three questions, in order:
  //
  //  1. Is there anything to lose? A new name, or a preset that carries no
  //     prompt, has nothing at stake -- a confirm there only trains
  //     click-through, which is what makes the real one worthless.
  //  2. Would this BLANK it? A Save from a promptless document writes NULL,
  //     and an override-box preset with no prompt is present but inert --
  //     "my preset disappeared". That always arms, including on the preset
  //     the document is running: clearing the box is not editing it.
  //  3. Otherwise, is the document already RUNNING this preset (and is that
  //     the preset the select is showing)? Then this is apply -> edit -> save
  //     it back, the iterate loop, and it stays one click. Saving onto a
  //     preset the document is NOT running is the shape that lost a 35k-char
  //     prompt on 2026-08-28 -- that arms.
  //
  // The KNOWN BOUNDARY of (3), accepted rather than patched: the stamp is
  // written by apply/save and cleared only by delete, so a document whose
  // prompt is later replaced WHOLESALE -- retyped, or adopted from another
  // device on resume -- still counts as running that preset, and saving
  // writes the new prompt over the old with no confirm. That is the trade (3)
  // exists to make; "the document is running P and you saved it" is the
  // loop's own definition, and the alternative is a drift-magnitude
  // threshold, which is a number nobody can defend. If this ever needs
  // closing, clear the STAMP when the prompt stops resembling the preset --
  // do not add a percentage here.
  //
  // The stamp is only meaningful with an active document (a stamp read with
  // none is the PREVIOUS document's -- same trap promptState() guards), and
  // a page that supplies no getStamp simply never earns the exemption.
  //
  // Deliberately resolved against the LOCAL list: save() re-fetches for
  // 409/404 correctness, not for this, and armedConfirm needs a synchronous
  // answer. That is free for questions (1) and (2) -- a stale list can only
  // mis-ARM there. Question (3) is the exception and the only branch that
  // trusts IDENTITY: it resolves a name to an id, so a list stale about which
  // row owns a name (deleted and recreated, or renamed, on another device) can
  // grant the exemption against the wrong row and let one overwrite through
  // unconfirmed. Narrow -- it needs a concurrent rename/recreate of the exact
  // name the document is stamped with -- and there is no synchronous fetch to
  // fix it with, so the mitigation is to keep (3) the only identity branch.
  function wouldOverwritePresetPrompt(name) {
    const existing = presets.find((p) => p.name === name.trim());
    const stored = existing?.system_prompt || null;
    if (!stored) return false;                       // (1) nothing to lose
    const incoming = getPrompt() ?? null;
    if (incoming === stored) return false;           // (1) no change either
    if (!incoming) return true;                      // (2) blanking always arms
    // (3) the iterate loop -- and it has to be the loop in FULL: the document
    // is running this preset AND the select is showing it. Requiring the
    // selection closes the divergence where a hand-typed name aims Save at a
    // preset the preview and drift line are not describing -- the exemption
    // would fire against the stamp while every visible readout named something
    // else, which is the same "what am I actually overwriting?" confusion the
    // preview was added to end.
    const stamp = docId?.() ? (getStamp?.() ?? null) : null;
    return !(stamp && existing.id === stamp && existing.id === effectiveId());
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
    // equalsState, not matchesState: an inferred (unstamped) claim must
    // account for the prompt too -- see the two comments above.
    const match = presets.find((p) => equalsState(p));
    return match ? { name: match.name, edited: false } : null;
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
    // Same docId guard as indicatorInfo(): a stamp read with no active
    // document is the PREVIOUS document's (deleting the last conversation
    // leaves the page's appliedPresetId set), which would label a freshly
    // typed draft "<old preset> (modified)".
    const stamped = docId?.() ? presets.find((p) => p.id === getStamp?.()) : null;
    const source = presetPrompt(stamped);
    return {
      prompt,
      presetName: source ? stamped.name : null,
      modified: Boolean(source && prompt !== source),
    };
  }

  // Feed the page's chip. Public: pages call it on document switch/create --
  // the drawer may be closed then, so the drift-line path can't be relied on.
  function syncIndicator() {
    onIndicator?.(indicatorInfo());
  }

  // The selected preset's OWN prompt, read-only. Without it there was no way
  // to see what a preset holds short of applying it, so browsing meant
  // clicking through the select -- whose section sits directly above the
  // DOCUMENT's prompt box, which never changes with the selection. Every
  // preset therefore appeared to carry whatever the open document carried
  // ("it copied it over to all of them"), and that reading is what turned
  // "let me see what that preset holds" into a Save that overwrote it.
  // No isConnected guard here (unlike the settings-change path's driftEl
  // check): both callers -- buildSection, and the select handler bound to
  // that same build -- run against the current build's element, and at build
  // time it is not appended yet, so an isConnected test would discard the
  // very element it is about to paint.
  function paintPreview() {
    if (!previewEl) return;
    const preset = selected();
    previewEl.hidden = !preset;
    if (!preset) return;
    const stored = preset.system_prompt || null;
    previewSummaryEl.textContent = stored
      ? `"${preset.name}" system prompt — ${stored.length.toLocaleString()} characters`
      : `"${preset.name}" carries no system prompt`;
    previewBodyEl.textContent = stored
      ?? 'Applying it leaves this document\'s prompt exactly as it is.';
    previewBodyEl.classList.toggle('preset-preview__body--none', !stored);
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
      // Override box: carry the prompt when the preset has one, otherwise
      // leave whatever the conversation (or the model's own default) uses.
      const incoming = presetPrompt(preset);
      if (incoming) setPrompt(incoming);
      if (docId?.()) setStamp?.(preset.id);
      onStatus(incoming
        ? `Preset "${preset.name}" applied.`
        : `Preset "${preset.name}" applied — it carries no system prompt, so this one is unchanged.`);
    }
    // Force: the Apply button lives in the drawer, so the focus guard would
    // otherwise skip the repaint that shows the applied values.
    drawer.requestRebuild({ force: true });
    // Explicitly, not via the settings-change listener: that fires from
    // applySettings BEFORE the prompt and stamp are written above, so relying
    // on it would paint the chips from pre-apply state (and not fire at all
    // for a preset carrying no sampler params).
    syncIndicator();
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
      pick(saved.id);
      // saving snapshots the current doc state -- the doc IS this preset now
      if (docId?.()) setStamp?.(saved.id);
      drawer.requestRebuild({ force: true });
      // Same reason apply() calls it: the document now names this preset, and
      // the bar chips are outside the drawer the rebuild above repaints.
      syncIndicator();
      onStatus(`Preset "${name}" ${existing ? 'updated' : 'saved'}.`);
    } catch (err) {
      if (ctx.alive) onStatus(`Preset save failed: ${err.message}`, true);
    }
  }

  async function remove() {
    const removedId = effectiveId();
    if (!removedId) return;
    try {
      await api.deletePreset(removedId);
    } catch (err) {
      if (ctx.alive) onStatus(`Preset delete failed: ${err.message}`, true);
      return;
    }
    if (!ctx.alive) return;
    pick(null); // explicit deselect, so the select does not fall back to the stamp
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
    select.value = effectiveId() ?? '';

    const applyBtn = armedConfirm(
      createEl('button', {
        class: 'btn btn--sm', disabled: !current,
        title: 'Copy this preset here (prompt + sampler settings)',
      }, ['Apply']),
      apply,
      'Replace prompt?',
      wouldReplacePrompt,
      // What Apply would do: copy THIS preset over THIS prompt. Either half
      // moving voids the arm.
      () => JSON.stringify([effectiveId(), getPrompt() ?? null]),
    );
    const delBtn = armedConfirm(
      createEl('button', { class: 'btn btn--sm btn--ghost', disabled: !current }, ['Del']),
      remove,
      'Confirm?',
      null,
      () => effectiveId(),
    );

    // Save under the typed name: matches an existing preset -> overwrite it,
    // new name -> create. Picking a preset pre-fills its name for overwrite.
    const nameInput = createEl('input', {
      class: 'input', placeholder: 'Save as…', value: current?.name ?? '',
      'aria-label': 'Preset name to save as',
    });
    const saveBtn = armedConfirm(
      createEl('button', { class: 'btn btn--sm' }, ['Save']),
      () => save(nameInput.value),
      'Overwrite prompt?',
      () => wouldOverwritePresetPrompt(nameInput.value),
      // Save's destination is the TYPED NAME and its payload is the document
      // prompt -- and that prompt is edited in another drawer section this bar
      // gets no events from, which is exactly why the check has to be re-read
      // at confirm time rather than wired to a control here.
      () => JSON.stringify([nameInput.value.trim(), getPrompt() ?? null]),
    );
    // Enter goes through the BUTTON, not straight to save() -- a second
    // entry point past the arm is the same hole with a keyboard on it.
    nameInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') saveBtn.click();
    });

    // Re-aiming disarms -- for HONESTY, not safety (the `target` callbacks
    // above already make a stale arm refuse to fire). A button still reading
    // "Overwrite prompt?" while aimed at a preset you just switched away from
    // is a lie. Each control disarms only the buttons IT re-aims: the select
    // moves all three, the name box moves Save alone -- disarming Apply from
    // the name box silently cancelled an apply the user was part-way through
    // confirming, just because they started typing a save-as name.
    select.addEventListener('change', () => {
      applyBtn.disarm(); saveBtn.disarm(); delBtn.disarm();
      pick(select.value || null);
      const p = selected();
      nameInput.value = p?.name ?? '';
      applyBtn.disabled = delBtn.disabled = !p;
      paintPreview();
      updateDrift();
    });
    nameInput.addEventListener('input', () => saveBtn.disarm());

    // .preset-drift is the E2E hook; styling rides the shared settings-note.
    // role=status: the line flips live (Matches/Differs) -- announced, not
    // just shown (DESIGN.md §7).
    driftEl = createEl('div', {
      class: 'preset-drift settings-note muted small', hidden: true, role: 'status',
      title: 'Presets are copies: Apply stamps the preset onto this document; '
        + 'later edits here never change the preset until you Save it again.',
    });
    // Collapsed by default: it answers "what does this preset hold?" on
    // demand without pushing the rest of the drawer off-screen on a phone.
    previewSummaryEl = createEl('summary', {});
    previewBodyEl = createEl('div', { class: 'preset-preview__body' });
    previewEl = createEl('details', { class: 'preset-preview' }, [
      previewSummaryEl, previewBodyEl,
    ]);
    paintPreview();
    updateDrift();

    return createEl('div', { class: 'preset-section' }, [
      createEl('h3', {}, ['Preset']),
      createEl('div', { class: 'preset-row' }, [select, applyBtn, delBtn]),
      driftEl,
      previewEl,
      createEl('div', { class: 'preset-row' }, [nameInput, saveBtn]),
    ]);
  }

  return { buildSection, onDrawerOpen, updateDrift, refresh, syncIndicator, presetForNewDoc, promptState };
}

// The bar chip's one renderer -- fed by onIndicator above, so it lives here
// rather than in each page. Chat and notebook carried byte-identical copies,
// both commented "the bar chip's ONE renderer", which was true of neither.
export function paintPresetChip(chip, info) {
  chip.hidden = !info;
  chip.textContent = info ? (info.edited ? `${info.name} (edited)` : info.name) : '';
}
