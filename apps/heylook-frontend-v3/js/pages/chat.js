// Chat: conversations sidebar + streaming thread. Markdown render path only
// (marked + DOMPurify via renderMarkdown -- no other text->HTML path).
//
// Invariants (Phase 2, plan_chat_orchestration.md -- the server-side saga):
// - GENERATION goes through POST /v1/conversations/{id}/generate: the server
//   builds the request from the store, anchors truncation by message id,
//   persists everything (completion, Stop, disconnect), and ends the stream
//   with the authoritative rows. The client's post-stream state is
//   ASSIGNMENT from heylook_saved's rows (adoptSavedRows -- synchronous, no
//   network), with resyncMessages (one GET) as the fallback for endings that
//   carry no rows; never position arithmetic.
// - Only renderMessages mutates the message LIST's DOM structure. The stream
//   painter writes text INSIDE the live node; startStream/finishGenerate/
//   handleStreamError never add or remove list nodes by hand -- a hand-removed
//   node is a frame with the response missing and a collapsed scrollHeight.
// - Regenerate / edit-regenerate / continue truncate NOTHING client-side:
//   the mirror drops the tail visually; the server commits its truncation
//   only together with the row it produced, so a failed or empty generation
//   gets the rows back at the end-of-stream reconcile.
// - Message delete is a SINGLE-row DELETE (v1.73.0): exactly the clicked
//   row goes; later rows keep their positions (gaps are fine by design).
// - Stop button = DELETE /{id}/generate (server aborts + persists partial;
//   the fetch stays open to receive the saved rows). Teardown/switch =
//   controller.abort() -- the server's disconnect path persists instead.

import { createPage } from '../page.js';
import { createEl, autoGrow, armedConfirm, beforeUnloadGuard, formatBytes, setStatus, dismissPaneOnOutsideClick } from '../utils.js';
import { api } from '../api.js';
import { streamGenerate, stopGenerate } from '../streaming.js';
import { renderMarkdown } from '../markdown.js';
import { samplerParams, snapshotSettings, bindDocumentParams, hydrateDocParams, getSetting, setSetting, onSettingsChange, PARAM_META } from '../settings.js';
import * as drawer from '../settings-drawer.js';
import { createPresetBar } from '../preset-bar.js';
import { createPromptSection } from '../prompt-section.js';

// A system prompt typed before any conversation exists has no owner: the
// server has nothing to attach it to, so it lived in page state alone and a
// reload -- or a trip to another page -- silently ate it. Sampler params in
// the same window survive, because settings.js parks them in localStorage,
// which is exactly why "everything else loads fine, just not the prompt".
// Park the draft the same way until a conversation adopts it (both create
// paths clear it). The follow-on this prevents is the worse half: with the
// box silently blank, a Save onto an existing preset name stored null over a
// good prompt.
const DRAFT_PROMPT_KEY = 'heylook.v3.chat.draft-prompt';

function readDraftPrompt() {
  try { return localStorage.getItem(DRAFT_PROMPT_KEY) || null; } catch { return null; }
}

function writeDraftPrompt(value) {
  try {
    if (value) localStorage.setItem(DRAFT_PROMPT_KEY, value);
    else localStorage.removeItem(DRAFT_PROMPT_KEY);
  } catch { /* storage full/unavailable -- the draft stays in memory only */ }
}

export default createPage({
  async setup(ctx) {
    const s = ctx.state;
    s.conversations = [];
    s.models = [];
    s.activeId = null;
    s.messages = [];
    s.systemPrompt = null;
    s.appliedPresetId = null;  // which preset this conversation is running
    s.stream = null;      // { controller, targetConvId, content, thinking, els, retries }
    s.editingId = null;
    s.msgNodes = new Map(); // renderMessages: message key -> { node, sig }

    buildSkeleton(ctx);
    // Shared preset bar (preset-bar.js), adapted to the active conversation.
    // The indicator feed drives the chat-bar chip (built in buildSkeleton).
    s.presetBar = createPresetBar(ctx, {
      getPrompt: () => s.systemPrompt,
      setPrompt: (v) => setSystemPrompt(ctx, v),
      onStatus: (text, isError) => showStatus(ctx, text, isError),
      docId: () => s.activeId,
      // One funnel for both chips: the bar calls this from syncIndicator
      // (document switch/create/delete) AND updateDrift (every prompt
      // keystroke + every sampler change), so neither chip can go stale
      // without the other noticing.
      onIndicator: (info) => { paintPresetChip(ctx, info); paintSysPromptChip(ctx); },
      getStamp: () => s.appliedPresetId,
      setStamp: (id) => setAppliedPreset(ctx, id),
    });
    // The chip needs preset names before the drawer's first lazy fetch.
    s.presetBar.refresh().then(() => { if (ctx.alive) s.presetBar.syncIndicator(); });
    // One throttle for the whole mount (it reads s.stream), not one per
    // stream -- per-stream throttles would pin each stream's closure in the
    // page's cleanup list for the mount lifetime.
    s.paint = ctx.throttle(() => paintStream(ctx));
    ctx.onTeardown(() => {
      if (s.stream) beforeUnloadGuard.disable();
    });

    // Chat's shared-drawer contribution: the preset bar + per-conversation
    // system-prompt editor lead the panel; full sampler controls; caps track
    // the selected model. onOpen lazily refreshes presets (fingerprint-diffed,
    // focus-guarded) so an open costs nothing until the list actually changed.
    const unregisterSettings = drawer.registerSettings({
      caps: () => currentCaps(ctx),
      samplers: 'enabled',
      sections: () => [s.presetBar.buildSection(), buildPromptSection(ctx).element],
      onOpen: s.presetBar.onDrawerOpen,
    });
    ctx.onTeardown(unregisterSettings);

    // Sampler knobs are per-conversation (like the system prompt) via the shared
    // per-document binding: a panel change persists to the active conversation's
    // `params`; hydrate on select is silent so this only fires on real edits +
    // preset applies.
    s.paramsBinder = bindDocumentParams({
      activeId: () => ctx.state.activeId,
      updateDoc: (id, body) => api.updateConversation(id, body),
      onError: (err) => showStatus(ctx, `Settings save failed: ${err.message}`, true),
    });
    ctx.onTeardown(s.paramsBinder);

    const [models, convList] = await Promise.all([
      api.listModels({ signal: ctx.signal }).catch(() => ({ data: [] })),
      api.listConversations({ signal: ctx.signal }).catch((err) => {
        if (ctx.alive) showStatus(ctx, `Could not load conversations: ${err.message}`, true);
        // `failed` distinguishes "no conversations" from "we don't know":
        // the draft-restore below must not fire on a transient network error,
        // which would drop a stale prompt on top of the error banner.
        return { conversations: [], failed: true };
      }),
    ]);
    if (!ctx.alive) return;

    s.models = models.data ?? [];
    s.conversations = convList.conversations ?? [];
    fillModelSelect(ctx);
    s.committedModelId = s.modelSelect.value || null;
    refreshThinkBtn(ctx);
    refreshAttachBtn(ctx);
    renderConvList(ctx);
    // Residency is a load-cost signal, not a gate -- fetched non-fatally and
    // AFTER first paint. Until it arrives the select shows plain ids and the
    // not-loaded warning stays silent (an honest "don't know", never a guess).
    refreshLoadedIds(ctx);

    if (s.conversations.length) {
      await selectConversation(ctx, s.conversations[0].id);
    } else if (!convList.failed) {
      // Genuinely no conversation owns a prompt -- restore the parked draft
      // so it survives reloads and page trips (the drawer reads
      // s.systemPrompt when it first renders, so setting it here is enough).
      s.systemPrompt = readDraftPrompt();
      renderMessages(ctx);
    }
    // First paint of the sysprompt chip: the preset-fed funnel above only
    // runs once the preset list lands (or never, if that fetch fails), and
    // the chip must state the prompt either way.
    paintSysPromptChip(ctx);
  },
});

// ---------------------------------------------------------------------------
// skeleton
// ---------------------------------------------------------------------------

function buildSkeleton(ctx) {
  const s = ctx.state;

  s.convListEl = createEl('div', { class: 'chat__convs-list' });
  const newBtn = createEl('button', { class: 'btn btn--sm' }, ['New']);
  newBtn.addEventListener('click', () => newConversation(ctx));

  const convPane = createEl('aside', { class: 'chat__convs' }, [
    createEl('div', { class: 'chat__convs-head' }, [
      createEl('h2', {}, ['Conversations']),
      newBtn,
    ]),
    s.convListEl,
  ]);

  s.modelSelect = createEl('select', { title: 'Model' });
  s.loadedIds = new Set();   // model ids with a resident process (admin list)
  s.loadedKnown = false;     // false until the first successful residency fetch
  s.committedModelId = null; // last COMMITTED selection (the select may show an unconfirmed target)
  s.modelSelect.addEventListener('change', () => {
    const from = s.committedModelId;
    const to = s.modelSelect.value;
    if (!to || to === from) return;
    // Pre-switch check, not post-switch discovery: cost and compatibility
    // are only actionable BEFORE the switch commits. A clean switch (target
    // resident, nothing in the conversation it can't take) commits silently.
    const warnings = switchWarnings(ctx, from, to);
    if (!warnings.length) {
      commitModelSwitch(ctx, to);
      return;
    }
    showSwitchWarning(ctx, to, warnings,
      () => commitModelSwitch(ctx, to),
      () => { s.modelSelect.value = from; showStatus(ctx, ''); });
  });

  // Pays the load deliberately, while reading, instead of discovering it
  // after hitting Send. Same server-owned load?warm=true the models page
  // uses; the status line speaks (buttons never spin).
  s.loadNowBtn = createEl('button', {
    class: 'btn btn--sm chat__load-btn', hidden: true,
    title: 'Load this model now so the first message does not pay for it',
  }, ['Load']);
  s.loadNowBtn.addEventListener('click', () => loadModelNow(ctx));

  const convsToggle = createEl('button', { class: 'btn btn--sm chat__convs-toggle' }, ['Chats']);
  convsToggle.addEventListener('click', () => s.rootEl.classList.toggle('chat--convs-open'));

  s.messagesInner = createEl('div', { class: 'chat__messages-inner' });
  s.messagesEl = createEl('div', { class: 'chat__messages' }, [s.messagesInner]);

  s.statusEl = createEl('div', { class: 'chat__status', role: 'status' });

  // Touch devices send via the button and the long placeholder wraps + clips
  // in the single-row textarea at phone widths -- keep the Enter hint desktop-only.
  s.textarea = createEl('textarea', {
    rows: 1,
    placeholder: matchMedia('(hover: none)').matches ? 'Message…' : 'Message… (Enter to send)',
  });
  s.textarea.addEventListener('input', () => autoGrow(s.textarea));
  s.textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send(ctx);
    }
  });
  // Attachments: images and (gguf models) audio, via picker, paste or drop.
  // The button is capability-gated like the thinking toggle -- hidden until
  // refreshAttachBtn sees a model with the vision and/or audio cap, and the
  // picker's accept list mirrors exactly what the model can take. All three
  // inputs funnel through addFiles -> addPendingFiles (one routine, so a cap
  // or limit fix lands once and can't be half-applied to one of them).
  s.pendingImages = [];
  s.pendingAudio = [];
  s.fileInput = createEl('input', { type: 'file', accept: 'image/*', multiple: true, hidden: true });
  s.fileInput.addEventListener('change', () => {
    addFiles(ctx, s.fileInput.files);
    s.fileInput.value = '';
  });
  s.attachBtn = createEl('button', {
    class: 'btn btn--icon', title: 'Attach images', 'aria-label': 'Attach images',
    hidden: true,
  });
  s.attachBtn.innerHTML = ICON_IMAGE;
  s.attachBtn.addEventListener('click', () => s.fileInput.click());

  // Composer thinking toggle: same cap gate + true/null semantics as the
  // drawer checkbox (unset follows the backend's per-model default). The
  // drawer is modal, so the two controls can't be edited concurrently --
  // onSettingsChange keeps this button honest after drawer edits.
  s.thinkBtn = createEl('button', {
    class: 'btn btn--icon', title: 'Thinking', 'aria-label': 'Toggle thinking',
    'aria-pressed': 'false', hidden: true,
  });
  s.thinkBtn.innerHTML = ICON_THINK;
  s.thinkBtn.addEventListener('click', () => {
    setSetting('enable_thinking', getSetting('enable_thinking') === true ? null : true);
  });
  ctx.onTeardown(onSettingsChange(ctx.guard(() => refreshThinkBtn(ctx))));
  s.attachStrip = createEl('div', { class: 'chat__attach', hidden: true });
  s.sendBtn = createEl('button', { class: 'btn btn--primary' }, ['Send']);
  s.sendBtn.addEventListener('click', () => (s.stream ? stopStream(ctx) : send(ctx)));

  // In-context settings entry: the sidebar-foot / bottom-nav gears are far
  // from the conversation, and the drawer holds chat's most-edited controls
  // (system prompt, presets). Same singleton drawer; focus returns here.
  const settingsBtn = createEl('button', {
    class: 'btn btn--icon chat__settings-btn', title: 'Settings', 'aria-label': 'Open settings',
  });
  settingsBtn.innerHTML = ICON_GEAR;
  settingsBtn.addEventListener('click', () => drawer.openSettings(settingsBtn));

  // Applied-preset chip: which preset this conversation is running (with an
  // "(edited)" suffix once it drifts). Fed by the preset bar's onIndicator;
  // clicking it opens the drawer at the preset controls.
  s.presetChip = createEl('button', {
    class: 'btn preset-chip chat__preset-chip', hidden: true,
    title: 'Preset applied to this conversation -- open settings',
  });
  s.presetChip.addEventListener('click', () => drawer.openSettings(s.presetChip));

  // System-prompt chip. NOT hidden when there is no prompt: "what am I
  // running?" has to be answerable at a glance, and an absent chip answers
  // nothing -- it reads the same whether there is no prompt or the UI simply
  // failed to show one (the exact ambiguity behind the disappearing-prompt
  // report). So the empty case says so out loud, and every state opens the
  // editor with the textarea focused.
  s.sysPromptChip = createEl('button', {
    class: 'btn preset-chip chat__sysprompt-chip',
  });
  s.sysPromptChip.addEventListener('click', () => {
    drawer.openSettings(s.sysPromptChip);
    // open() focuses the close button; move to the field the user asked for.
    // Scroll too -- the prompt section can sit below the fold on short panels.
    queueMicrotask(() => {
      const input = document.querySelector('.sysprompt-input');
      input?.focus();
      input?.scrollIntoView({ block: 'nearest' });
    });
  });

  const thread = createEl('section', { class: 'chat__thread' }, [
    createEl('header', { class: 'chat__bar' }, [
      convsToggle,
      s.modelSelect,
      s.loadNowBtn,
      s.presetChip,
      s.sysPromptChip,
      createEl('div', { class: 'chat__bar-spacer' }),
      settingsBtn,
    ]),
    s.messagesEl,
    s.statusEl,
    s.attachStrip,
    createEl('div', { class: 'chat__composer' }, [s.attachBtn, s.thinkBtn, s.fileInput, s.textarea, s.sendBtn]),
  ]);

  s.threadEl = thread;
  s.rootEl = createEl('div', { class: 'chat' }, [convPane, thread]);
  // Mobile: tapping the visible thread (outside the conversations pane + toggle)
  // dismisses the slide-in pane.
  dismissPaneOnOutsideClick(s.rootEl, 'chat--convs-open', '.chat__convs', '.chat__convs-toggle');
  wirePasteAttach(ctx);
  wireDropAttach(ctx, thread);
  ctx.el.append(s.rootEl);
}

// Static, trusted SVG markup (icons inherit currentColor from the button).
const ICON_IMAGE =
  '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" '
  + 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
  + '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/>'
  + '<path d="m21 15-5-5L5 21"/></svg>';
const ICON_GEAR =
  '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" '
  + 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
  + '<circle cx="12" cy="12" r="3"/>'
  + '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06'
  + 'a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09'
  + 'a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06'
  + 'a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09'
  + 'a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06'
  + 'a1.65 1.65 0 0 0 1.82.33h.01a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09'
  + 'a1.65 1.65 0 0 0 1 1.51h.01a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06'
  + 'a1.65 1.65 0 0 0-.33 1.82v.01a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09'
  + 'a1.65 1.65 0 0 0-1.51 1z"/></svg>';
const ICON_THINK =
  '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" '
  + 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
  + '<path d="M9 18h6"/><path d="M10 22h4"/>'
  + '<path d="M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.4 1 2.3h6c0-.9.4-1.8 1-2.3A7 7 0 0 0 12 2z"/></svg>';

// Residency rides the option label (● resident / ○ idle) so the load cost is
// visible in the act of choosing -- the only moment it can influence the
// choice. Values stay bare ids; titles spell the state out for AT/hover.
// Until the first residency fetch lands, labels are plain ids (no dot is an
// honest "don't know", a hollow dot would be a claim).
function fillModelSelect(ctx) {
  const s = ctx.state;
  const prev = s.modelSelect.value;
  s.modelSelect.replaceChildren(...s.models.map((m) => {
    const label = s.loadedKnown
      ? `${s.loadedIds.has(m.id) ? '●' : '○'} ${m.id}`
      : m.id;
    return createEl('option', {
      value: m.id,
      title: s.loadedKnown ? (s.loadedIds.has(m.id) ? `${m.id} — loaded` : `${m.id} — not loaded`) : m.id,
    }, [label]);
  }));
  if (prev && s.models.some((m) => m.id === prev)) s.modelSelect.value = prev;
}

// Non-fatal residency read (admin list). Refreshed at mount, after Load now,
// and after each completed generation -- a send can load the target AND evict
// the previous resident (max_loaded_models=1), so completion is exactly when
// the dots go stale. No polling: it fights the metrics cache and burns phone
// battery; these three moments are when the answer actually changes.
async function refreshLoadedIds(ctx) {
  const s = ctx.state;
  try {
    const data = await api.adminListModels({ signal: ctx.signal });
    if (!ctx.alive) return;
    s.loadedIds = new Set((data.models ?? []).filter((m) => m.loaded).map((m) => m.id));
    // Provider per model, for the one continuation asymmetry: user-role
    // continuation is MLX-only (llama-server prefills assistant turns only).
    s.providerById = new Map((data.models ?? []).map((m) => [m.id, m.provider]));
    s.loadedKnown = true;
  } catch {
    return; // keep the last known state; the UI never guesses residency
  }
  fillModelSelect(ctx);
  refreshLoadBtn(ctx);
  // Provider just became known, and an editor opened before it landed is
  // missing its Save & Continue button. Re-render so the row catches up here
  // rather than at whatever unrelated render happens next. Only rows whose
  // signature moved rebuild -- for a residency fetch that is the open editor
  // and nothing else -- and the rebuild carries the typed text across, so the
  // catch-up cannot cost anyone a draft.
  if (s.activeId) renderMessages(ctx);
}

function refreshLoadBtn(ctx) {
  const s = ctx.state;
  const id = s.modelSelect.value;
  s.loadNowBtn.hidden = !(s.loadedKnown && id && !s.loadedIds.has(id) && !s.loadNowBtn.dataset.busy);
}

async function loadModelNow(ctx) {
  const s = ctx.state;
  const id = s.modelSelect.value;
  if (!id || s.loadNowBtn.dataset.busy) return;
  s.loadNowBtn.dataset.busy = '1';
  s.loadNowBtn.disabled = true;
  showStatus(ctx, `Loading ${id}…`);
  try {
    const result = await api.adminLoadModel(id, true);
    if (!ctx.alive) return;
    if (result?.warm_error) {
      showStatus(ctx, `Loaded, but the warm-up generation failed: ${result.warm_error}`, true);
    } else {
      showStatus(ctx, result?.warm_ms != null
        ? `${id} loaded and warmed in ${(result.warm_ms / 1000).toFixed(1)}s.`
        : `${id} loaded.`);
    }
  } catch (err) {
    if (!ctx.alive) return;
    showStatus(ctx, `Load failed: ${err.message}`, true);
  }
  delete s.loadNowBtn.dataset.busy;
  s.loadNowBtn.disabled = false;
  if (ctx.alive) await refreshLoadedIds(ctx);
}

// What committing this switch would actually do to THIS conversation.
// Returned lines gate the confirm flow: empty = clean switch, commit
// silently. The thinking note is informational and deliberately does NOT
// trigger the flow on its own -- the toggle disappearing is its own visible
// signal, and a confirm dialog for a reversible toggle hide trains
// click-through (it rides along only when a real warning is already showing).
//
// Only LOSS gates: content this model cannot read. A cold target is NOT a
// warning -- changing model means paying for that model, so there is no
// decision to put behind a button, and the confirm was pure friction on the
// commonest path (nothing resident yet, where it also claimed an eviction
// that could not happen). Load cost is DISCLOSED instead, in three
// non-blocking places: the ○ in the option label, the Load button, and --
// the one that was actually missing -- a live status while the send waits
// on the load (sendMessage). Owner call 2026-08-11: state what is
// happening, do not ask permission for the inevitable.
function switchWarnings(ctx, from, to) {
  const s = ctx.state;
  const target = s.models.find((m) => m.id === to);
  const caps = target?.capabilities ?? [];
  const lines = [];

  const count = (type) => s.messages.reduce(
    (n, m) => n + (m.content_blocks?.filter((b) => b.type === type).length ?? 0), 0);
  const images = count('image');
  const audio = count('audio');
  if (images && !caps.includes('vision')) {
    lines.push(`This conversation has ${images} image${images === 1 ? '' : 's'}. `
      + 'They will be dropped from every request to this model — it cannot read them.');
  }
  if (audio && !caps.includes('audio')) {
    lines.push(`This conversation has ${audio} audio clip${audio === 1 ? '' : 's'}. `
      + 'They will be dropped from every request to this model — it cannot hear them.');
  }
  if (lines.length && getSetting('enable_thinking') === true) {
    const fromCaps = s.models.find((m) => m.id === from)?.capabilities ?? [];
    if (fromCaps.includes('thinking') && !caps.includes('thinking')) {
      lines.push('Thinking is unavailable on this model; the toggle will hide.');
    }
  }
  return lines;
}

// Inline in the chat status area, per the switching design -- NOT a modal (a
// modal would block reading the very conversation being decided about). The
// select keeps showing the unconfirmed target; Cancel reverts it.
function showSwitchWarning(ctx, to, lines, onConfirm, onCancel) {
  const s = ctx.state;
  const confirmBtn = createEl('button', { class: 'btn btn--sm' }, ['Switch anyway']);
  const cancelBtn = createEl('button', { class: 'btn btn--sm' }, ['Cancel']);
  confirmBtn.addEventListener('click', onConfirm);
  cancelBtn.addEventListener('click', onCancel);
  s.statusEl.replaceChildren(createEl('div', { class: 'chat__switch-warning' }, [
    createEl('div', {}, [`Switching to ${to}:`]),
    ...lines.map((l) => createEl('div', { class: 'chat__switch-line' }, [l])),
    createEl('div', { class: 'chat__switch-actions' }, [cancelBtn, confirmBtn]),
  ]));
}

function commitModelSwitch(ctx, to) {
  const s = ctx.state;
  // Stop an in-flight stream BEFORE model_id changes hands, so the old
  // model stops generating the moment the user switches away. NB this is
  // an abort, not a settled handoff: the partial still persists (server
  // disconnect path) into this same conversation after model_id is
  // rewritten, and messages carry no model column -- so a reader of
  // conversation.model_id still misattributes it. True per-message
  // attribution is G5 in the switching design (deferred until a
  // _SCHEMA_VERSION bump adds messages.model_id).
  if (s.stream) abortStream(ctx);
  s.committedModelId = to;
  // Say what this costs instead of asking whether to pay it. Silent when the
  // target is resident or residency is still unknown -- a guess would be
  // worse than nothing (same rule as the dots).
  showStatus(ctx, s.loadedKnown && !s.loadedIds.has(to)
    ? `${to} is not loaded — your first message loads it, or press Load to do it now.`
    : '');
  if (s.activeId) {
    api.updateConversation(s.activeId, { model_id: to })
      .catch((err) => console.warn('model_id save failed', err));
    const conv = s.conversations.find((c) => c.id === s.activeId);
    if (conv) conv.model_id = to;
  }
  // Capability-gated controls (enable_thinking) must track the model: force
  // an open drawer to rebuild here, not only on its next open.
  drawer.requestRebuild({ force: true });
  refreshThinkBtn(ctx);
  refreshAttachBtn(ctx);
  refreshLoadBtn(ctx);
  // Per-message drop disclosures depend on the current model's caps.
  renderMessages(ctx);
}

// Visible only for thinking-capable models (mirrors the drawer's requiresCap
// gate); pressed = explicit true. Re-run on model switch, conversation
// hydrate, and any settings change.
function refreshThinkBtn(ctx) {
  const s = ctx.state;
  // same gate as the drawer checkbox -- read the cap from PARAM_META so
  // the two can never disagree on the capability name
  s.thinkBtn.hidden = !currentCaps(ctx).includes(PARAM_META.enable_thinking.requiresCap);
  const on = getSetting('enable_thinking') === true;
  s.thinkBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
}

// Attach button follows the model's input modalities (capabilities, not the
// descriptive `modalities` field -- caps are what the server will actually
// serve). Hidden for text-only models; the picker accepts exactly what the
// model takes. Runs everywhere refreshThinkBtn runs.
function refreshAttachBtn(ctx) {
  const s = ctx.state;
  const caps = currentCaps(ctx);
  const vision = caps.includes('vision');
  const audio = caps.includes('audio');
  s.attachBtn.hidden = !(vision || audio);
  s.fileInput.accept = [vision && 'image/*', audio && 'audio/*'].filter(Boolean).join(',');
  const label = vision && audio ? 'Attach images or audio'
    : audio ? 'Attach audio' : 'Attach images';
  s.attachBtn.title = label;
  s.attachBtn.setAttribute('aria-label', label);
  // The drop overlay's text (CSS reads it via content: attr()). It names what
  // THIS model takes, including the text-only case -- an overlay that says
  // "drop images" over a model that can't read them would be a lie the user
  // only discovers after dropping.
  if (s.threadEl) {
    s.threadEl.dataset.dropLabel = vision && audio ? 'Drop images or audio to attach'
      : audio ? 'Drop audio to attach'
        : vision ? 'Drop images to attach'
          : 'This model takes text only';
  }
}

function currentCaps(ctx) {
  const model = ctx.state.models.find((m) => m.id === ctx.state.modelSelect.value);
  return model?.capabilities ?? [];
}

// ---------------------------------------------------------------------------
// system prompt (per-conversation) + presets (saved prompt/sampler bundles)
// ---------------------------------------------------------------------------

// The preset-apply write path (the bar's setPrompt adapter): state now, PUT
// if a conversation exists (explicit null clears server-side). With no active
// conversation the value rides along until a create gives it a home. The
// textarea has its own path -- per-keystroke state + debounced PUT via the
// shared prompt-section factory (buildPromptSection below); both converge on
// putSystemPrompt.
function setSystemPrompt(ctx, value) {
  const s = ctx.state;
  const changed = value !== s.systemPrompt;
  s.systemPrompt = value;
  if (!s.activeId) {
    writeDraftPrompt(value); // no conversation owns it yet -- park it
    return;
  }
  if (!changed) return; // no-op PUTs skipped (preset re-apply)
  putSystemPrompt(ctx, s.activeId, value);
}

// All system-prompt PUTs serialize through one per-mount chain: the
// textarea's blur-flush and a preset-apply's write are separate fetches
// issued milliseconds apart, and without ordering the stale one could land
// last server-side (pre-existing class, /code-review 2026-07-23; localhost
// keep-alive makes it near-unobservable, but the chain closes it outright).
function putSystemPrompt(ctx, convId, value) {
  const s = ctx.state;
  s.promptPutChain = (s.promptPutChain ?? Promise.resolve())
    .then(() => api.updateConversation(convId, { system_prompt: value }))
    .catch((err) => showStatus(ctx, `System prompt save failed: ${err.message}`, true));
}

// Chat builds a NEW section per drawer build (the drawer forces a rebuild on
// every conversation switch), so the shared factory's construction-time
// owner capture already gives per-conversation isolation for free -- no
// setValue call needed here.
// Persist the applied-preset stamp onto the active conversation. Explicit
// stamps only (the preset bar calls this from apply/save/delete) -- see the
// provenance note in preset-bar.js.
function setAppliedPreset(ctx, presetId) {
  const s = ctx.state;
  s.appliedPresetId = presetId;
  if (!s.activeId) return; // pre-create draft: the stamp rides in state only
  api.updateConversation(s.activeId, { applied_preset_id: presetId })
    .catch((err) => showStatus(ctx, `Preset stamp save failed: ${err.message}`, true));
}

function buildPromptSection(ctx) {
  const s = ctx.state;
  return createPromptSection(ctx, {
    owner: () => s.activeId,
    get: () => s.systemPrompt,
    set: (v) => {
      s.systemPrompt = v;
      // With no conversation the factory's persist is a deliberate no-op
      // (builtFor is null) -- park the draft here instead, per keystroke.
      if (!s.activeId) writeDraftPrompt(v);
    },
    persist: (v, id) => putSystemPrompt(ctx, id, v),
    onEdit: () => s.presetBar.updateDrift(), // prompt edits drift the selected preset live
  });
}


function showStatus(ctx, text, isError = false) {
  setStatus(ctx.state.statusEl, text, isError);
}

// Phase 0 loud guards (plan_chat_orchestration.md). A user action refused
// because of a live stream must SAY so: the silent returns these replace were
// indistinguishable from a broken button, worst during the pre-first-token
// window of a cold load when nothing visibly streams. role="status" on the
// line makes this audible to AT as well as visible.
function refuseWhileStreaming(ctx) {
  if (!ctx.state.stream) return false;
  // (The v1.64 pendingSave latch is gone with Phase 2: the server persists
  // BEFORE its stream ends, so there is no client-save window to guard.)
  showStatus(ctx, 'A response is still streaming — wait for it to finish, or press Stop.');
  return true;
}

// While an unsaved (id-less) row exists, the client's positions are
// known-divergent from the store: a send would mint a colliding server
// position, and any position-anchored mutation can truncate the wrong tail
// SERVER-side. Refuse loudly until Retry or Discard resolves the row.
function refuseWhileUnsaved(ctx) {
  if (!ctx.state.messages.some((m) => m.id == null)) return false;
  showStatus(ctx, 'A message is not saved yet — use its Retry save or Discard button first.', true);
  return true;
}

// The bar chip's one renderer (fed by the preset bar's onIndicator).
function paintPresetChip(ctx, info) {
  const chip = ctx.state.presetChip;
  chip.hidden = !info;
  chip.textContent = info ? (info.edited ? `${info.name} (edited)` : info.name) : '';
}

// The system-prompt chip: what prompt is in force, where it came from, and
// whether it still matches that source. Four states, each one a claim the
// user can check by clicking through to the text itself.
function paintSysPromptChip(ctx) {
  const s = ctx.state;
  const chip = s.sysPromptChip;
  if (!chip) return;
  const { prompt, presetName, modified } = s.presetBar.promptState();

  let label;
  if (!prompt) label = 'No system prompt';
  else if (!presetName) label = 'System prompt: custom';
  else label = `System prompt: ${presetName}${modified ? ' (modified)' : ''}`;

  chip.textContent = label;
  chip.classList.toggle('chat__sysprompt-chip--empty', !prompt);
  chip.classList.toggle('chat__sysprompt-chip--modified', Boolean(modified));
  // The full text belongs in the tooltip/accessible name, not the bar -- but
  // an unbounded prompt would make an unreadable tooltip, so cap the preview
  // and say it is one.
  const preview = prompt && prompt.length > 300 ? `${prompt.slice(0, 300)}…` : prompt;
  chip.title = prompt
    ? `${label} — click to edit\n\n${preview}`
    : 'No system prompt is in force — click to write one';
  chip.setAttribute('aria-label', prompt
    ? `${label}. Click to edit the system prompt.`
    : 'No system prompt. Click to write one.');
}

// ---------------------------------------------------------------------------
// conversations sidebar
// ---------------------------------------------------------------------------

function renderConvList(ctx) {
  const s = ctx.state;
  if (!s.conversations.length) {
    s.convListEl.replaceChildren(
      createEl('div', { class: 'empty-state' }, ['No conversations yet.']),
    );
    return;
  }
  s.convListEl.replaceChildren(...s.conversations.map((conv) => {
    const title = createEl('span', { class: 'conv-item__title', title: conv.title }, [conv.title]);
    title.addEventListener('dblclick', (e) => {
      e.stopPropagation();
      startRename(ctx, conv, title);
    });
    // dblclick is desktop-only (double-tap zooms on iOS); a reveal-on-hover/touch
    // button gives touch a rename path. Same visibility grammar as Del.
    const ren = createEl('button', { class: 'btn btn--sm btn--ghost conv-item__edit', title: 'Rename' }, ['Ren']);
    ren.addEventListener('click', (e) => {
      e.stopPropagation();
      startRename(ctx, conv, title);
    });
    const del = armedConfirm(
      createEl('button', { class: 'btn btn--sm btn--ghost conv-item__delete' }, ['Del']),
      () => deleteConversation(ctx, conv.id),
    );
    const item = createEl('div', {
      class: `conv-item${conv.id === s.activeId ? ' conv-item--active' : ''}`,
    }, [title, ren, del]);
    item.addEventListener('click', () => {
      selectConversation(ctx, conv.id);
      s.rootEl.classList.remove('chat--convs-open');
    });
    return item;
  }));
}

function startRename(ctx, conv, titleEl) {
  const input = createEl('input', { class: 'input conv-item__rename', value: conv.title });
  const commit = ctx.guard(async () => {
    const next = input.value.trim();
    input.replaceWith(titleEl);
    if (!next || next === conv.title) return;
    try {
      await api.updateConversation(conv.id, { title: next });
      if (!ctx.alive) return;
      conv.title = next;
      renderConvList(ctx);
    } catch (err) {
      showStatus(ctx, `Rename failed: ${err.message}`, true);
    }
  });
  input.addEventListener('blur', commit);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') input.blur();
    if (e.key === 'Escape') { input.value = conv.title; input.blur(); }
  });
  input.addEventListener('click', (e) => e.stopPropagation());
  titleEl.replaceWith(input);
  input.focus();
  input.select();
}

async function newConversation(ctx) {
  const s = ctx.state;
  clearPendingAttachments(ctx); // staged attachments belong to the conv they were picked in
  try {
    // The selected (or stamped) preset is the unit of continuity: a new
    // conversation STARTS as it -- prompt + params + stamp (starting-as is an
    // apply, so stamping at create is explicit, not inferred). Without one,
    // the old rules hold: an active conversation's prompt does NOT leak, and
    // sampler knobs carry forward from the current panel.
    const preset = s.presetBar.presetForNewDoc();
    const conv = await api.createConversation({
      title: 'New conversation',
      model_id: s.modelSelect.value || undefined,
      // a prompt drafted before ANY conversation exists still wins over the
      // preset -- it is the more explicit act
      system_prompt: (!s.activeId && s.systemPrompt) || preset?.system_prompt || undefined,
      params: preset ? { ...(preset.params ?? {}) } : snapshotSettings(),
      applied_preset_id: preset?.id,
    });
    if (!ctx.alive) return;
    // The draft (if any) just became this conversation's prompt -- it has a
    // real owner now, so stop parking it.
    writeDraftPrompt(null);
    s.conversations.unshift(conv);
    await selectConversation(ctx, conv.id);
    s.rootEl.classList.remove('chat--convs-open');
    s.textarea.focus();
  } catch (err) {
    if (ctx.alive) showStatus(ctx, `Create failed: ${err.message}`, true);
  }
}

async function deleteConversation(ctx, convId) {
  const s = ctx.state;
  // Deleting the conversation that is actively STREAMING: the CRUD gate
  // 409s while the claim is held, so stop it first -- server-side abort
  // (persists the partial, releases the claim) plus local fetch abort --
  // and give the unwind a moment before retrying once.
  if (s.stream?.targetConvId === convId) {
    stopGenerate(convId);
    abortStream(ctx);
  }
  try {
    await api.deleteConversation(convId);
  } catch (err) {
    if (err.status === 409) {
      await new Promise((r) => setTimeout(r, 600));
      try {
        await api.deleteConversation(convId);
      } catch (err2) {
        if (ctx.alive) showStatus(ctx, `Delete failed: ${err2.message}`, true);
        return;
      }
    } else {
      if (ctx.alive) showStatus(ctx, `Delete failed: ${err.message}`, true);
      return;
    }
  }
  if (!ctx.alive) return;
  s.conversations = s.conversations.filter((c) => c.id !== convId);
  if (s.activeId === convId) {
    abortStream(ctx);
    s.activeId = null;
    s.messages = [];
    s.systemPrompt = null;
    if (s.conversations.length) await selectConversation(ctx, s.conversations[0].id);
    else {
      drawer.requestRebuild({ force: true });
      s.presetBar.syncIndicator(); // no active doc -> chip clears
      renderMessages(ctx);
    }
  }
  if (ctx.alive) renderConvList(ctx);
}

async function selectConversation(ctx, convId) {
  const s = ctx.state;
  if (s.activeId === convId && s.messages.length) return;
  if (s.stream) abortStream(ctx); // partial still persists to its own conv (server disconnect path)
  s.activeId = convId;
  s.editingId = null;
  s.msgNodes = new Map(); // node reuse is per-document; never carry rows across
  clearPendingAttachments(ctx); // staged attachments belong to the conv they were picked in
  showStatus(ctx, '');
  renderConvList(ctx);
  try {
    const conv = await api.getConversation(convId, { signal: ctx.signal });
    if (!ctx.alive || s.activeId !== convId) return;
    // A conversation owns the prompt from here on, so the parked draft is
    // unreachable (newConversation with an active doc takes the preset, not
    // the draft). Drop it rather than let it outlive its moment: a forgotten
    // draft that survives until the last conversation is deleted would
    // silently become some future conversation's prompt.
    writeDraftPrompt(null);
    s.messages = conv.messages ?? [];
    s.systemPrompt = conv.system_prompt ?? null;
    s.appliedPresetId = conv.applied_preset_id ?? null;
    hydrateDocParams(conv);  // sampler panel <- this conversation (silent, no re-PUT)
    if (conv.model_id && s.models.some((m) => m.id === conv.model_id)) {
      s.modelSelect.value = conv.model_id;
    }
    // AFTER the select moves, never before: both read currentCaps(), which
    // reads modelSelect.value, so running them first describes the model we
    // just navigated AWAY from. That staleness was invisible while it only
    // mis-set a hidden attribute; the drop overlay makes it a visible promise
    // ("Drop images to attach" over a model that refuses the drop).
    refreshThinkBtn(ctx);    // silent hydrate skips onSettingsChange -- sync directly
    refreshAttachBtn(ctx);
    // Programmatic selection IS the committed model -- a restore never runs
    // the pre-switch warning flow (the conversation already lives there).
    s.committedModelId = s.modelSelect.value || null;
    refreshLoadBtn(ctx);
    // an open drawer shows the previous conversation's system prompt otherwise
    drawer.requestRebuild({ force: true });
    s.presetBar.syncIndicator(); // rebuild no-ops while the drawer is closed
    renderMessages(ctx);
    scrollMessages(ctx, true);
  } catch (err) {
    if (ctx.alive && s.activeId === convId) {
      showStatus(ctx, `Could not load conversation: ${err.message}`, true);
      // The doc never loaded: clear the PREVIOUS conversation's leftovers so
      // (a) nothing renders/writes under the wrong id, and (b) re-clicking
      // this conversation retries -- the select guard's messages.length
      // condition would otherwise no-op on the stale array forever.
      s.messages = [];
      s.systemPrompt = null;
      s.appliedPresetId = null;
      renderMessages(ctx);
      drawer.requestRebuild({ force: true });
      // resync so the chip doesn't keep claiming the previous conversation's
      // preset for the one that failed to load
      s.presetBar.syncIndicator();
    }
  }
}

// ---------------------------------------------------------------------------
// message rendering
// ---------------------------------------------------------------------------

// Render key + signature. `.message` carries `content-visibility: auto`, so a
// message that is off-screen only knows its `contain-intrinsic-size` estimate
// (3rem) until it has been laid out once -- and that measurement lives on the
// NODE. Rebuilding the list threw every measurement away, collapsing
// scrollHeight to a fraction of the real one mid-render; every pixel-based
// scroll after that (restore OR scrollTop = scrollHeight) then aimed at a
// thread that was about to grow underneath it, which is what dumped a long
// conversation near the top on send/edit/delete. So nodes are REUSED unless
// their content actually changed, and the list is reconciled in place.
function msgKey(msg) {
  return msg.id ?? `pos:${msg.position}`;
}

function blockFingerprint(b) {
  const src = b.source;
  if (!src) return b.type; // text block: no source to identify
  return src.type === 'url'
    ? `${b.type}:${src.url}`
    : `${b.type}:${src.media_type}:${src.data?.length ?? 0}`;
}

// Model-dependent chrome is scoped to the rows that actually carry it: caps
// reach a row only if it HAS media (they decide the drop disclosure), provider
// only an open editor (Save & Continue is MLX-only). Putting either in every
// row's signature invalidates the whole list the moment residency lands or the
// model changes -- rebuilding every node, which is the full detach this design
// exists to avoid.
function msgSignature(msg, { editing, capsKey, provider }) {
  return [
    msg.role,
    msg.position,
    editing ? `edit:${provider}` : 'view',
    hasMediaBlocks(msg) ? capsKey : '',
    msg.thinking ?? '',
    msg.content ?? '',
    // Media identity only: a text block carries no `source`, and its text is
    // already covered by msg.content. Fingerprint the source rather than
    // spelling it out -- a base64 image would put megabytes in this string.
    (msg.content_blocks ?? []).map(blockFingerprint).join(','),
    // NUL-joined, not space-joined: content and thinking both contain spaces,
    // so a space separator lets one field's tail read as the next field's head
    // -- two different messages, one signature, a stale row reused.
  ].join('\u0000');
}

// Move an in-progress edit from a row about to be discarded onto its
// replacement. Silent no-op when the old row was not an editor. Pairwise:
// an editor can hold two textareas now (thinking + response), and both are
// built from the same message, so the counts match by construction.
function carryEditorDraft(fromEl, toEl) {
  const from = [...(fromEl?.querySelectorAll?.('textarea') ?? [])];
  const to = [...(toEl?.querySelectorAll?.('textarea') ?? [])];
  if (!from.length || from.length !== to.length) return;
  from.forEach((f, i) => {
    to[i].value = f.value;
    to[i].selectionStart = f.selectionStart;
    to[i].selectionEnd = f.selectionEnd;
  });
}

// Place `nodes` as the parent's children, moving/keeping existing elements
// rather than replacing them (see msgSignature: a detach is what loses the
// laid-out height).
function reconcileChildren(parent, nodes) {
  // Drop departing children FIRST. Placing before removing walks a stale node
  // down the list, moving (= detaching) every node after it -- which loses the
  // very measurements this is here to keep: one edit-cancel re-laid-out the
  // whole tail and the list slammed to the bottom.
  const wanted = new Set(nodes);
  for (const child of [...parent.childNodes]) {
    if (!wanted.has(child)) child.remove();
  }
  nodes.forEach((node, i) => {
    const cur = parent.childNodes[i];
    if (cur !== node) parent.insertBefore(node, cur ?? null);
  });
}

function renderMessages(ctx) {
  const s = ctx.state;
  if (!s.activeId) {
    s.msgNodes = new Map();
    s.messagesInner.replaceChildren(
      createEl('div', { class: 'empty-state' },
        ['Send a message below to start a new conversation.']),
    );
    return;
  }
  const prev = s.msgNodes ?? new Map();
  const next = new Map();
  const capsKey = currentCaps(ctx).join('|');
  const provider = s.providerById?.get(s.modelSelect.value) ?? '?';
  // A live stream owns its own row (startStream appends the placeholder, and
  // for a continuation it removed the message's rendered row in favour of it).
  // Carry that node through the reconcile instead of dropping it: a render
  // mid-stream (model switch) used to detach the element the stream was still
  // painting into.
  const stream = s.stream?.targetConvId === s.activeId ? s.stream : null;
  const continued = stream?.continueMsg?.id ?? null;
  const nodes = s.messages
    .filter((msg) => continued == null || msg.id !== continued)
    .map((msg) => {
      const key = msgKey(msg);
      // `id != null` first: an unsaved row carries id null, and a bare
      // `editingId === msg.id` matches null-to-null -- which rendered the
      // save-failure fallback row as an open editor whose Save would PUT to
      // /messages/null.
      const editing = msg.id != null && msg.id === s.editingId;
      const sig = msgSignature(msg, { editing, capsKey, provider });
      const cached = prev.get(key);
      let node = cached?.sig === sig ? cached.node : null;
      if (!node) {
        node = editing ? buildEditEl(ctx, msg) : buildMessageEl(ctx, msg);
        // Rebuilding an OPEN editor (its model chrome changed) must not eat
        // what is in the box -- buildEditEl seeds from msg.content, which is
        // the SAVED text. Carry the live value + caret over instead.
        if (editing && cached) carryEditorDraft(cached.node, node);
      }
      next.set(key, { node, sig });
      return node;
    });
  if (stream?.els?.msgEl) nodes.push(stream.els.msgEl);
  s.msgNodes = next;
  reconcileChildren(s.messagesInner, nodes);
}

function buildThinkingEl(thinking, open = false) {
  const details = createEl('details', { class: 'thinking', open }, [
    createEl('summary', {}, ['Thinking']),
    createEl('div', { class: 'thinking__body' }, [thinking ?? '']),
  ]);
  return details;
}

function buildMessageEl(ctx, msg) {
  const content = createEl('div', { class: 'message-content' });
  if (msg.role === 'assistant') content.innerHTML = renderMarkdown(msg.content);
  else content.textContent = msg.content;

  const bubbleChildren = [];
  if (hasBlocks(msg, 'image')) {
    bubbleChildren.push(createEl('div', { class: 'message-images' },
      msg.content_blocks
        .filter((b) => b.type === 'image')
        .map((b) => createEl('img', {
          class: 'message-image',
          src: blockSourceUrl(b),
          alt: 'attached image',
        }))));
  }
  if (hasBlocks(msg, 'audio')) {
    bubbleChildren.push(createEl('div', { class: 'message-audio-list' },
      msg.content_blocks
        .filter((b) => b.type === 'audio')
        .map((b) => createEl('audio', {
          class: 'message-audio',
          controls: '',
          src: blockSourceUrl(b),
          'aria-label': 'attached audio',
        }))));
  }
  bubbleChildren.push(content);
  const bubble = createEl('div', { class: 'message-bubble' }, bubbleChildren);
  const children = [];
  if (msg.role === 'assistant' && msg.thinking) children.push(buildThinkingEl(msg.thinking));
  children.push(bubble);
  // Drop disclosure: media still renders (it's in the store) but is not sent
  // to the CURRENT model (toWireContent drops what the caps exclude). The
  // marker is what keeps the drop honest -- without it the model answers as
  // if the image were invisible and the user has no idea why.
  const caps = currentCaps(ctx);
  const dropped = [];
  const imgCount = msg.content_blocks?.filter((b) => b.type === 'image').length ?? 0;
  const audCount = msg.content_blocks?.filter((b) => b.type === 'audio').length ?? 0;
  if (imgCount && !caps.includes('vision')) {
    dropped.push(`${imgCount} image${imgCount === 1 ? '' : 's'}`);
  }
  if (audCount && !caps.includes('audio')) {
    dropped.push(`${audCount} audio clip${audCount === 1 ? '' : 's'}`);
  }
  if (dropped.length) {
    children.push(createEl('div', { class: 'message-drop-note muted small' },
      [`${dropped.join(' and ')} not sent to this model`]));
  }
  // Unsaved fallback row: say so ON the row, always visible (not
  // hover-gated -- the state must be legible on touch). While it exists,
  // send and every position-anchored op refuse loudly (refuseWhileUnsaved).
  if (msg.id == null) {
    children.push(createEl('div', { class: 'message-unsaved-note small', role: 'status' },
      ['Not saved — this text exists only on this screen. Retry save or Discard it.']));
  }
  children.push(buildActions(ctx, msg));

  return createEl('div', { class: `message message--${msg.role}` }, children);
}

function buildActions(ctx, msg) {
  const btn = (label, fn) => {
    const b = createEl('button', { class: 'btn btn--sm btn--ghost' }, [label]);
    b.addEventListener('click', fn);
    return b;
  };
  const actions = [];
  // Copy copies the flattened text; an image-only message has none to copy.
  if (msg.content) {
    actions.push(btn('Copy', () => navigator.clipboard?.writeText(msg.content).catch(() => {})));
  }
  // Unsaved fallback row (its save failed; the store never had it). The only
  // safe exits are client-side: re-POST it or discard it locally. The normal
  // actions are position-anchored server ops and stay off it -- Edit's save
  // would PUT to /messages/null, Delete/Regenerate would truncate against a
  // position the server never assigned.
  if (msg.id == null) {
    actions.push(btn('Retry save', () => retrySave(ctx, msg)));
    actions.push(armedConfirm(
      createEl('button', { class: 'btn btn--sm btn--ghost' }, ['Discard']),
      () => discardUnsaved(ctx, msg),
    ));
    return createEl('div', { class: 'message__actions' }, actions);
  }
  // Editing is text-only: the editor would replace content and silently drop
  // the media blocks. Image/audio messages get delete/regenerate, not edit.
  if (!hasMediaBlocks(msg)) {
    actions.push(btn('Edit', () => {
      // loud, not silent: renderMessages would orphan the stream placeholder
      if (refuseWhileStreaming(ctx)) return;
      ctx.state.editingId = msg.id;
      renderMessages(ctx);
    }));
  }
  if (msg.role === 'assistant') {
    actions.push(btn('Regenerate', () => regenerate(ctx, msg)));
  }
  actions.push(armedConfirm(
    createEl('button', { class: 'btn btn--sm btn--ghost' }, ['Delete']),
    () => deleteMessage(ctx, msg),
  ));
  return createEl('div', { class: 'message__actions' }, actions);
}

// Re-POST an unsaved row. On success the server row replaces it in place --
// its key changes from pos:N to the real id, so the node rebuilds saved.
async function retrySave(ctx, msg) {
  const s = ctx.state;
  if (refuseWhileStreaming(ctx)) return;
  const convId = s.activeId;
  try {
    const saved = await api.addMessage(convId, {
      role: msg.role, content: msg.content, thinking: msg.thinking || undefined,
    });
    if (!ctx.alive || s.activeId !== convId) return;
    const i = s.messages.indexOf(msg);
    if (i !== -1) s.messages[i] = saved; else s.messages.push(saved);
    renderMessages(ctx);
    showStatus(ctx, 'Message saved.');
  } catch (err) {
    if (ctx.alive) showStatus(ctx, `Save failed again: ${err.message}`, true);
  }
}

function discardUnsaved(ctx, msg) {
  const s = ctx.state;
  const i = s.messages.indexOf(msg);
  if (i !== -1) s.messages.splice(i, 1);
  renderMessages(ctx);
  showStatus(ctx, 'Unsaved message discarded.');
}

function buildEditEl(ctx, msg) {
  const s = ctx.state;
  const textarea = createEl('textarea', { value: msg.content, 'aria-label': 'Edit message' });
  // Size to the message, not a fixed pixel cap: long messages were getting a
  // barely-readable slit. Cap at ~60% of the viewport so the buttons stay on
  // screen; past that the textarea scrolls internally.
  const growCap = () => Math.max(400, Math.round(window.innerHeight * 0.6));
  textarea.addEventListener('input', () => autoGrow(textarea, growCap()));
  // Thinking is editable alongside content (owner ask 2026-08-13; the
  // backend always accepted MessageUpdate.thinking -- only the editor was
  // missing). Offered only when the message HAS thinking; capped shorter
  // than content so the response stays reachable under a long trace.
  const thinkCap = () => Math.max(160, Math.round(window.innerHeight * 0.3));
  let thinkArea = null;
  if (msg.role === 'assistant' && msg.thinking) {
    thinkArea = createEl('textarea', {
      class: 'message-edit__thinking', value: msg.thinking,
      'aria-label': 'Edit thinking',
    });
    thinkArea.addEventListener('input', () => autoGrow(thinkArea, thinkCap()));
  }
  const cancel = () => { s.editingId = null; renderMessages(ctx); };

  const save = async (regenerateAfter, continueAfter = false) => {
    // The truncate-then-stream branches are destructive; with a stream
    // already running they mangle the thread (the truncation commits, the
    // new stream silently no-ops, the old one paints into a detached node).
    // Same guards regenerate() and deleteMessage() carry -- loud, not silent.
    if ((regenerateAfter || continueAfter)
        && (refuseWhileStreaming(ctx) || refuseWhileUnsaved(ctx))) return;
    // Everything below is anchored to the conversation the edit belongs to,
    // captured NOW: s.activeId read after an await can be a different
    // conversation (switch mid-PUT), and a position-anchored mutation
    // against it would irreversibly rewrite the wrong thread.
    const convId = s.activeId;
    const next = textarea.value;
    const changes = {};
    if (next !== msg.content) changes.content = next;
    // empty = clear: the PUT sends null so the row's thinking column clears
    // rather than storing an empty string that still renders a block
    if (thinkArea && thinkArea.value !== msg.thinking) changes.thinking = thinkArea.value || null;
    try {
      if (Object.keys(changes).length) {
        const updated = await api.updateMessage(convId, msg.id, changes);
        if (!ctx.alive) return;
        // Keep content, blocks AND thinking in sync with the server's view --
        // on the row the LIST currently holds, not just this closure's
        // capture: a resync between opening the editor and Save replaces
        // s.messages with fresh objects, and mutating only the stale capture
        // repainted the pre-edit text (the "my edit vanished" report). The
        // capture is still updated too -- the regenerate/continue branches
        // below read msg.position/content.
        const live = s.messages.find((m) => m.id === msg.id);
        for (const target of live && live !== msg ? [msg, live] : [msg]) {
          target.content = updated.content;
          target.content_blocks = updated.content_blocks;
          target.thinking = updated.thinking;
        }
      }
      if (s.activeId !== convId) return; // switched away mid-save: edit saved, nothing destructive
      s.editingId = null;
      if (regenerateAfter) {
        // Everything after the edited message goes (visually -- the server
        // owns the real truncation). Anchor on the row after it when one
        // exists; with the edited row last, a plain append generates fresh.
        const nextMsg = s.messages.find((m) => m.position > msg.position);
        s.messages = s.messages.filter((m) => m.position <= msg.position);
        renderMessages(ctx);
        startStream(ctx, nextMsg
          ? { mode: 'regenerate', messageId: nextMsg.id }
          : { mode: 'append' });
      } else if (continueAfter) {
        // Continue FROM the edited text: prefill semantics on the server,
        // which merges the continuation back onto this very row.
        s.messages = s.messages.filter((m) => m.position <= msg.position);
        renderMessages(ctx);
        startStream(ctx, { mode: 'continue', messageId: msg.id, continueMsg: msg });
      } else {
        renderMessages(ctx);
      }
    } catch (err) {
      if (ctx.alive) showStatus(ctx, `Save failed: ${err.message}`, true);
      // the PUT/truncate saga may have half-applied -- adopt server truth
      if (ctx.alive) resyncMessages(ctx, convId);
    }
  };

  const buttons = [
    createEl('button', { class: 'btn btn--sm' }, ['Cancel']),
    createEl('button', { class: 'btn btn--sm btn--primary' }, ['Save']),
  ];
  buttons[0].addEventListener('click', cancel);
  buttons[1].addEventListener('click', () => save(false));
  if (msg.role === 'user') {
    const saveRegen = createEl('button', { class: 'btn btn--sm btn--primary' }, ['Save & Regenerate']);
    saveRegen.addEventListener('click', () => save(true));
    buttons.push(saveRegen);
  }
  // Continuation works for BOTH roles (an assistant message is finished; a
  // user message is co-written) -- but user-role continuation is MLX-only:
  // llama-server prefills assistant turns and has no user-turn spelling, so
  // the button hides rather than offering a guaranteed 400. FAIL CLOSED
  // while the provider is unknown (fetch pending or failed): showing the
  // button on a guess costs a destructive truncate before the 400 lands.
  const provider = s.providerById?.get(s.modelSelect.value);
  if (msg.id && (msg.role === 'assistant' || (provider != null && provider !== 'gguf'))) {
    const saveContinue = createEl('button', {
      class: 'btn btn--sm btn--primary',
      title: 'Save, drop everything after, and let the model finish this message',
    }, ['Save & Continue']);
    saveContinue.addEventListener('click', () => save(false, true));
    buttons.push(saveContinue);
  }

  const editChildren = [];
  if (thinkArea) {
    // visible captions, not placeholder text: with two boxes open the user
    // must be able to tell which is which at a glance
    editChildren.push(createEl('div', { class: 'message-edit__label muted small' }, ['Thinking']));
    editChildren.push(thinkArea);
    editChildren.push(createEl('div', { class: 'message-edit__label muted small' }, ['Response']));
  }
  editChildren.push(textarea, createEl('div', { class: 'message-edit__buttons' }, buttons));
  const el = createEl('div', { class: `message message--${msg.role}` }, [
    createEl('div', { class: 'message-edit' }, editChildren),
  ]);
  // rAF, not microtask: the initial grow needs layout to have happened, or
  // scrollHeight reads short and the editor opens as a slit.
  requestAnimationFrame(() => {
    if (thinkArea) autoGrow(thinkArea, thinkCap());
    autoGrow(textarea, growCap());
    textarea.focus();
  });
  return el;
}

function scrollMessages(ctx, force = false) {
  const el = ctx.state.messagesEl;
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
  if (!force && !nearBottom) return;
  el.scrollTop = el.scrollHeight;
  // A row added this tick is still at its `contain-intrinsic-size` estimate,
  // so scrollHeight reads short and this lands above the real bottom. Re-aim
  // once the browser has laid the new row out. The non-forced path (stream
  // paints, stream-end adoption) needs it too -- but re-checks nearBottom in
  // the frame, so a user who scrolled up between the two writes is left alone.
  requestAnimationFrame(() => {
    if (!ctx.alive) return;
    if (force || el.scrollHeight - el.scrollTop - el.clientHeight < 100) {
      el.scrollTop = el.scrollHeight;
    }
  });
}

// ---------------------------------------------------------------------------
// message mutations (position-based truncation)
// ---------------------------------------------------------------------------

// Phase 0 reconcile (plan_chat_orchestration.md): adopt the server's rows
// wholesale after a saga settles (stream persistence done, or a mutation
// failed part-way). The mirror has no other reconciliation point -- the
// select guard deliberately skips re-fetching the active conversation -- so
// without this, small divergences accumulate until a position-anchored op
// truncates the wrong tail. Unchanged rows keep their nodes (signature
// reuse), so a clean reconcile repaints nothing. Unsaved (id-less) rows
// survive adoption: the server doesn't have them, and vanishing text is
// worse than divergence (they also lock destructive ops via
// refuseWhileUnsaved, so surviving is safe).
async function resyncMessages(ctx, convId) {
  const s = ctx.state;
  let conv;
  try {
    conv = await api.getConversation(convId, { signal: ctx.signal });
  } catch {
    return; // best-effort: the next saga end (or a reselect) retries
  }
  if (!ctx.alive || s.activeId !== convId || s.stream) return;
  const serverRows = conv.messages ?? [];
  const unsaved = s.messages.filter((m) => m.id == null);
  let nextPos = (serverRows[serverRows.length - 1]?.position ?? -1) + 1;
  for (const m of unsaved) m.position = nextPos++;
  s.messages = [...serverRows, ...unsaved];
  renderMessages(ctx);
}

// Merge the heylook_saved rows into the mirror -- pure assignment, no
// network. What the mirror holds below the first saved position IS what the
// server kept (append: the prefix ending at the user turn; regenerate/
// continue: the tail the mirror hid is exactly what the commit deleted), so
// local rows at or past that position give way to the saved rows. A continue
// anchor comes back as its own updated row (same id, merged content). Id-less
// rows cannot exist here: refuseWhileUnsaved blocks every generation start
// while one is on screen.
function adoptSavedRows(ctx, rows) {
  const s = ctx.state;
  const savedIds = new Set(rows.map((r) => r.id));
  const minPos = Math.min(...rows.map((r) => r.position));
  s.messages = [
    ...s.messages.filter((m) => m.position < minPos && !savedIds.has(m.id)),
    ...rows,
  ];
}

function regenerate(ctx, msg) {
  const s = ctx.state;
  if (refuseWhileStreaming(ctx) || refuseWhileUnsaved(ctx)) return;
  // No client-side truncation: the server anchors on the message ID and
  // commits its truncation only together with the replacement row. The
  // mirror drops the tail VISUALLY; if the generation fails or produces
  // nothing, the end-of-stream reconcile brings the rows back.
  s.messages = s.messages.filter((m) => m.position < msg.position);
  renderMessages(ctx);
  scrollMessages(ctx, true);
  startStream(ctx, { mode: 'regenerate', messageId: msg.id });
}

// Delete means delete: exactly the clicked row goes, the rest of the thread
// stays (v1.73.0 -- the old spelling was a ?after truncation that silently
// destroyed everything below, which no button saying "Delete" should do).
// Later rows keep their positions; gaps are fine, nothing assumes density.
async function deleteMessage(ctx, msg) {
  const s = ctx.state;
  if (refuseWhileStreaming(ctx) || refuseWhileUnsaved(ctx)) return;
  const convId = s.activeId; // anchor before any await (switch mid-flight)
  try {
    await api.deleteMessage(convId, msg.id);
    if (!ctx.alive || s.activeId !== convId) return;
    s.messages = s.messages.filter((m) => m.id !== msg.id);
    renderMessages(ctx);
  } catch (err) {
    if (ctx.alive) showStatus(ctx, `Delete failed: ${err.message}`, true);
    if (ctx.alive) resyncMessages(ctx, convId);
  }
}

// ---------------------------------------------------------------------------
// image attachments
// ---------------------------------------------------------------------------

const MAX_ATTACH_IMAGES = 8;
const MAX_ATTACH_AUDIO = 2; // server-side gemma cap is 30s/clip; keep the strip sane

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

// Picker dispatch: the input's accept list is capability-driven, but a
// picker can't be trusted (drag/drop, "All files") -- split by MIME here.
// One staging routine parameterized per kind (the repo's shared-factory
// discipline: a cap/guard fix lands once, never in a per-kind twin).
const ATTACH_KINDS = {
  image: {
    mime: 'image/', cap: 'vision', stateKey: 'pendingImages', max: MAX_ATTACH_IMAGES,
    label: 'image', toEntry: (f, dataUrl) => ({ dataUrl, mediaType: f.type }),
  },
  audio: {
    mime: 'audio/', cap: 'audio', stateKey: 'pendingAudio', max: MAX_ATTACH_AUDIO,
    label: 'audio clip', toEntry: (f, dataUrl) => ({ dataUrl, mediaType: f.type, name: f.name }),
  },
};

// Anything file-shaped a paste or a drop could carry that we would stage.
function attachableFiles(files) {
  const kinds = Object.values(ATTACH_KINDS);
  return [...files].filter((f) => kinds.some((k) => f.type.startsWith(k.mime)));
}

// Safari has historically populated `items` but not `files` on a paste, so
// read both rather than picking one and being wrong on one browser.
function clipboardFiles(e) {
  const dt = e.clipboardData;
  if (!dt) return [];
  if (dt.files?.length) return [...dt.files];
  return [...(dt.items ?? [])]
    .filter((it) => it.kind === 'file')
    .map((it) => it.getAsFile())
    .filter(Boolean);
}

// Would any of these files actually be staged, or will every kind be refused
// for want of a capability? Paste needs to know BEFORE cancelling the default.
function willStageAny(ctx, files) {
  const caps = currentCaps(ctx);
  return Object.values(ATTACH_KINDS).some((k) => caps.includes(k.cap)
    && files.some((f) => f.type.startsWith(k.mime)));
}

// Paste-to-attach, on DOCUMENT rather than the chat root. Clicking a message
// leaves focus on document.body -- verified, not assumed -- and body is an
// ANCESTOR of the chat root, so a paste targeted there never reaches a
// listener mounted on the page. A root-scoped listener therefore only worked
// when a form field or a selection inside the thread held focus, which is the
// case that already worked via the composer. Document scope is what makes
// "click in the thread, then paste" actually reach here.
//
// Two guards earn their place at that scope:
//  - a paste aimed at any OTHER editable is left alone (the conversation
//    rename input, and the drawer's system-prompt box, which is a body child
//    outside #app and would otherwise be intercepted).
//  - preventDefault only when something will really be staged. A clipboard
//    payload can carry text AND an image (Excel, Word, most web pages);
//    cancelling on a model that refuses the image would eat the text too and
//    leave the user an error and an empty composer.
function wirePasteAttach(ctx) {
  document.addEventListener('paste', (e) => {
    const t = e.target;
    const editable = t?.isContentEditable
      || ['INPUT', 'TEXTAREA', 'SELECT'].includes(t?.tagName ?? '');
    if (editable && t !== ctx.state.textarea) return;
    const files = attachableFiles(clipboardFiles(e));
    if (!files.length) return;
    if (willStageAny(ctx, files)) e.preventDefault();
    addFiles(ctx, files);   // refuses loudly when it cannot stage
  }, { signal: ctx.signal });
}

// Drag-and-drop staging. Desktop-only by nature; the phone paths (attach
// button -> photo library, paste) are unaffected, so this adds an affordance
// without becoming a hover-only one. Three things break a naive version:
//   - dragenter AND dragover must preventDefault, or `drop` never fires and
//     the browser navigates away to the file, taking the page with it.
//   - dragleave fires when the pointer crosses into a CHILD element, so a
//     boolean "is dragging" flag makes the overlay flicker. Count depth.
//   - a drag of selected text within the page is not a file drag; without the
//     types check the overlay lights up on ordinary text selection drags.
function wireDropAttach(ctx, el) {
  const isFileDrag = (e) => [...(e.dataTransfer?.types ?? [])].includes('Files');
  const setOver = (on) => el.classList.toggle('chat__thread--dragover', on);
  let depth = 0;

  el.addEventListener('dragenter', (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    depth += 1;
    setOver(true);
  });
  el.addEventListener('dragover', (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
  });
  el.addEventListener('dragleave', (e) => {
    if (!isFileDrag(e)) return;
    depth = Math.max(depth - 1, 0);
    if (!depth) setOver(false);
  });
  el.addEventListener('drop', (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    depth = 0;
    setOver(false);
    addFiles(ctx, e.dataTransfer?.files ?? []);
  });

  // A file dropped just OUTSIDE the target still navigates the browser to it,
  // discarding an unsent message and every staged attachment. Swallow those at
  // the window. Registered with ctx.signal so they die with the page -- these
  // are the only listeners here that outlive the chat DOM.
  const swallow = (e) => { if (isFileDrag(e)) e.preventDefault(); };
  window.addEventListener('dragover', swallow, { signal: ctx.signal });
  window.addEventListener('drop', swallow, { signal: ctx.signal });
}

function addFiles(ctx, files) {
  const all = [...files];
  // A drop of a PDF, a .txt or a folder (File.type is '') matches no kind and
  // would otherwise vanish -- indistinguishable from a broken drop target.
  // Same principle as the capability refusal below: silence is the worst
  // answer available.
  if (all.length && !attachableFiles(all).length) {
    const caps = currentCaps(ctx);
    const takes = Object.values(ATTACH_KINDS)
      .filter((k) => caps.includes(k.cap)).map((k) => `${k.label}s`);
    showStatus(ctx, takes.length
      ? `Nothing attachable there -- ${takes.join(' or ')} only.`
      : 'This model takes text only.', true);
    return;
  }
  for (const kind of Object.values(ATTACH_KINDS)) {
    addPendingFiles(ctx, all.filter((f) => f.type.startsWith(kind.mime)), kind);
  }
}

async function addPendingFiles(ctx, files, kind) {
  if (!files.length) return;
  // Refuse at STAGING time. The send-side guard is for a DIFFERENT case --
  // media staged on a capable model, then a switch that loses the cap, where
  // "remove them or switch models" is the right words. Here nothing is staged
  // yet, so accepting a blob the user must then hunt down and clear is
  // strictly worse than saying no now. This is also the backstop for a picker
  // bypassed via "All files", and for paste/drop, which have no accept list
  // to bypass in the first place.
  if (!currentCaps(ctx).includes(kind.cap)) {
    showStatus(ctx, `This model does not take ${kind.label}s -- pick a model that does.`, true);
    return;
  }
  const pendingAtStart = ctx.state[kind.stateKey];
  const reads = await Promise.all([...files]
    .map((f) => fileToDataUrl(f)
      .then((dataUrl) => kind.toEntry(f, dataUrl))
      .catch(() => null)));  /* unreadable file -- skip */
  if (!ctx.alive) return;
  // send() and clearPendingAttachments REPLACE these arrays rather than
  // emptying them, so a send (or a conversation switch) during the read leaves
  // us holding an orphan: pushing into it would render nothing and lose the
  // file without a word. Drag-and-drop makes multi-megabyte reads routine, so
  // this window is no longer theoretical.
  const pending = ctx.state[kind.stateKey];
  if (pending !== pendingAtStart) {
    showStatus(ctx, `Attachment discarded -- the composer was cleared while it was still loading.`, true);
    return;
  }
  const usable = reads.filter(Boolean);
  if (!usable.length) return;
  const room = Math.max(kind.max - pending.length, 0);
  pending.push(...usable.slice(0, room));
  renderAttachStrip(ctx);
  // aria-live: chat__status is role="status" -- this announces the cap to
  // screen readers, not just the sighted strip.
  if (usable.length > room) {
    showStatus(ctx, `${kind.max} ${kind.label} max -- ${usable.length - room} not attached.`);
  }
}

function clearPendingAttachments(ctx) {
  ctx.state.pendingImages = [];
  ctx.state.pendingAudio = [];
  renderAttachStrip(ctx);
}

function renderAttachStrip(ctx) {
  const s = ctx.state;
  s.attachStrip.hidden = !s.pendingImages.length && !s.pendingAudio.length;
  const imageThumbs = s.pendingImages.map((img, i) => {
    // Real <button> + aria-label naming the target image: keyboard-reachable
    // and announced correctly even though the strip has no per-thumb text.
    const label = `Remove image ${i + 1}`;
    const remove = createEl('button', { class: 'attach-thumb__remove', title: label, 'aria-label': label }, ['×']);
    remove.addEventListener('click', () => {
      s.pendingImages.splice(i, 1);
      renderAttachStrip(ctx);
    });
    return createEl('div', { class: 'attach-thumb' }, [
      createEl('img', { src: img.dataUrl, alt: '' }),
      remove,
    ]);
  });
  const audioChips = s.pendingAudio.map((clip, i) => {
    const label = `Remove audio ${clip.name || i + 1}`;
    const remove = createEl('button', { class: 'attach-thumb__remove', title: label, 'aria-label': label }, ['×']);
    remove.addEventListener('click', () => {
      s.pendingAudio.splice(i, 1);
      renderAttachStrip(ctx);
    });
    return createEl('div', { class: 'attach-thumb attach-thumb--audio' }, [
      createEl('span', { class: 'attach-thumb__audio-name', title: clip.name || 'audio' },
        [clip.name || 'audio']),
      remove,
    ]);
  });
  s.attachStrip.replaceChildren(...imageThumbs, ...audioChips);
}

// Stored shape is Messages-style content blocks (what the server persists);
// hasMediaBlocks gates the flows that only make sense for text (edit).
function hasBlocks(msg, type) {
  return Boolean(msg.content_blocks?.some((b) => b.type === type));
}

function hasMediaBlocks(msg) {
  return hasBlocks(msg, 'image') || hasBlocks(msg, 'audio');
}

// One place that knows how a media block becomes a data/remote URL.
// IMAGES use it for BOTH render and wire (they must never disagree).
// Audio uses it for RENDER only -- the audio WIRE shape is deliberately
// different (input_audio takes RAW base64, a data: URI is rejected
// server-side), built inline in toWireContent.
function blockSourceUrl(b) {
  return b.source.type === 'url'
    ? b.source.url
    : `data:${b.source.media_type};base64,${b.source.data}`;
}

function buildContentBlocks(text, images, audio) {
  const blocks = images.map((img) => ({
    type: 'image',
    source: {
      type: 'base64',
      media_type: img.mediaType,
      data: img.dataUrl.slice(img.dataUrl.indexOf(',') + 1),
    },
  }));
  blocks.push(...audio.map((clip) => ({
    type: 'audio',
    source: {
      type: 'base64',
      media_type: clip.mediaType,
      data: clip.dataUrl.slice(clip.dataUrl.indexOf(',') + 1),
    },
  })));
  if (text) blocks.push({ type: 'text', text });
  return blocks;
}

// ---------------------------------------------------------------------------
// send + stream
// ---------------------------------------------------------------------------

async function send(ctx) {
  const s = ctx.state;
  const text = s.textarea.value.trim();
  const images = s.pendingImages;
  const audio = s.pendingAudio;
  if (!text && !images.length && !audio.length) return;
  // Loud, not silent: Enter during a stream (the button already reads Stop)
  // or while a row is unsaved must say why nothing was sent. The unsaved
  // case is load-bearing -- a send would mint a server position colliding
  // with the unsaved row's guessed one, and the request body would carry
  // text the store doesn't have.
  if (refuseWhileStreaming(ctx) || refuseWhileUnsaved(ctx)) return;
  if (!s.modelSelect.value) {
    showStatus(ctx, 'No models available.', true);
    return;
  }
  // A pending switch warning leaves the select on an unconfirmed target;
  // sending WITH that target selected is the strongest confirmation there
  // is -- commit it (which also clears the warning) rather than sending to
  // a model the conversation isn't labelled with.
  if (s.committedModelId && s.modelSelect.value !== s.committedModelId) {
    commitModelSwitch(ctx, s.modelSelect.value);
  }
  // Attachments staged on a capable model must not ride to a model that
  // lost the cap on switch -- fail loudly, keep them staged for a re-pick.
  // DELIBERATE ASYMMETRY with history media (dropped server-side by the
  // generate endpoint, disclosed per-message by buildMessageEl): staged
  // media is something the user just chose and can trivially un-choose, so
  // BLOCK; history is work already done, and refusing to talk until it's
  // deleted punishes the wrong thing, so history DROPS with disclosure.
  // Do not "fix" one site to match the other.
  const caps = currentCaps(ctx);
  if (images.length && !caps.includes('vision')) {
    showStatus(ctx, 'This model does not take images -- remove them or switch models.', true);
    return;
  }
  if (audio.length && !caps.includes('audio')) {
    showStatus(ctx, 'This model does not take audio -- remove it or switch models.', true);
    return;
  }

  const content = (images.length || audio.length)
    ? buildContentBlocks(text, images, audio) : text;
  const title = (text || (audio.length ? 'Audio message' : 'Image message')).slice(0, 50);
  s.textarea.value = '';
  autoGrow(s.textarea);
  s.pendingImages = [];
  s.pendingAudio = [];
  renderAttachStrip(ctx);
  showStatus(ctx, '');

  try {
    if (!s.activeId) {
      // a prompt typed (or preset applied) before the first send
      const sentPrompt = s.systemPrompt;
      const conv = await api.createConversation({
        title,
        model_id: s.modelSelect.value,
        system_prompt: sentPrompt || undefined,
        params: snapshotSettings(),  // the panel state this first message was sent with
      });
      if (!ctx.alive) return;
      writeDraftPrompt(null); // adopted by the conversation this send created
      s.conversations.unshift(conv);
      s.activeId = conv.id;
      s.messages = [];
      if (s.systemPrompt !== sentPrompt) {
        // prompt changed while the create was in flight -- it has a home now
        putSystemPrompt(ctx, conv.id, s.systemPrompt);
      } else {
        s.systemPrompt = conv.system_prompt ?? null;
      }
      // an open drawer's sysprompt textarea was built for activeId=null --
      // rebind it to the conversation that now owns the prompt
      drawer.requestRebuild({ force: true });
      s.presetBar.syncIndicator(); // the new conversation may match a preset
      renderConvList(ctx);
    } else {
      const conv = s.conversations.find((c) => c.id === s.activeId);
      if (conv && conv.title === 'New conversation' && !s.messages.length) {
        conv.title = title;
        api.updateConversation(s.activeId, { title: conv.title }).catch(() => {});
        renderConvList(ctx);
      }
    }

    // The user turn posts as plain CRUD (immediate render, loud failure);
    // the generate call then builds its prompt from the store, which
    // includes this row. (The endpoint could persist it too -- deliberately
    // unused so the typed text is on screen and saved before any
    // generation concern can interfere.)
    const convId = s.activeId; // anchor: a switch mid-POST must not leak into another conv
    const msg = await api.addMessage(convId, { role: 'user', content });
    if (!ctx.alive || s.activeId !== convId) return; // saved to its conv; reselect re-fetches it
    s.messages.push(msg);
    renderMessages(ctx);
    scrollMessages(ctx, true);
    startStream(ctx, { mode: 'append' });
  } catch (err) {
    if (!ctx.alive) return;
    // The message did NOT reach the store -- put the composer back exactly
    // as it was. Without this, the new mid-generation 409 (another device
    // streaming into this conversation) destroyed the typed text and
    // staged attachments client-side (review finding 2026-08-13).
    s.textarea.value = text;
    autoGrow(s.textarea);
    s.pendingImages = images;
    s.pendingAudio = audio;
    renderAttachStrip(ctx);
    showStatus(ctx, `Send failed: ${err.message}`, true);
  }
}

// The request body no longer exists client-side (Phase 2): the server
// builds it from the store -- system prompt, sampler bag (cap-gated), rows
// with history media dropped-and-counted for the target model. The
// per-message drop DISCLOSURE stays client-rendered (buildMessageEl), off
// the same capabilities, so the transcript states what the server omits.

function startStream(ctx, opts = {}) {
  const s = ctx.state;
  if (s.stream || !s.activeId) return;
  const { mode = 'append', messageId = null, continueMsg = null } = opts;

  const controller = ctx.linkedController();

  // Streaming placeholder message. A CONTINUATION streams into the tail of
  // an EXISTING message (continueMsg, the anchor): the placeholder is
  // seeded with its content and replaces its rendered element; the SERVER
  // merges the combined text back onto that same row.
  const baseContent = continueMsg?.content ?? '';
  const role = continueMsg?.role ?? 'assistant';
  const contentEl = createEl('div', { class: 'message-content' });
  if (baseContent) contentEl.innerHTML = renderMarkdown(baseContent);
  const thinkingEl = buildThinkingEl(continueMsg?.thinking ?? '', true);
  thinkingEl.hidden = !continueMsg?.thinking;
  const msgEl = createEl('div', { class: `message message--${role} message--streaming` }, [
    thinkingEl,
    createEl('div', { class: 'message-bubble' }, [contentEl]),
  ]);

  const stream = {
    controller,
    targetConvId: s.activeId,
    continueMsg,
    baseContent,
    baseThinking: continueMsg?.thinking ?? '',
    content: '',
    thinking: '',
    contentDirty: false,
    thinkingDirty: false,
    waiting: true, // no delta yet -- the wait status below is still on screen
    els: { msgEl, contentEl, thinkingBody: thinkingEl.querySelector('.thinking__body'), thinkingEl },
  };
  s.stream = stream;
  // List STRUCTURE goes through renderMessages only (it appends the stream
  // node and, for a continuation, excludes the anchor's own row). The painter
  // updates text INSIDE the node; nothing here touches the list by hand --
  // hand-placed nodes are what let stream end leave a frame with the
  // response missing.
  renderMessages(ctx);
  scrollMessages(ctx, true);
  s.sendBtn.textContent = 'Stop';
  beforeUnloadGuard.enable();

  // The dead air before the first token is the one moment the user cannot
  // tell a slow model from a hung one -- and on a cold target it is a
  // multi-GB load, the single longest wait in the app, previously shown as
  // an empty bubble. Name it. Cleared by the first delta below (thinking
  // counts: a reasoning model emits thinking first, and leaving "Loading…"
  // up while thinking streams would be a lie).
  const modelId = s.modelSelect.value;
  const cold = s.loadedKnown && modelId && !s.loadedIds.has(modelId);
  // Remembered so teardown can clear THIS line and nothing else: an abort
  // resolves asynchronously, so a status written after the abort (e.g. the
  // model-switch disclosure, when the user switches away mid-wait) would
  // otherwise be wiped by the dying stream's cleanup.
  stream.waitStatus = cold
    ? `Loading ${modelId}… (the first token follows the load — this can take a while)`
    : 'Waiting for the first token…';
  showStatus(ctx, stream.waitStatus);

  const isCurrent = () => s.stream === stream && s.activeId === stream.targetConvId;
  // Idempotent: fires on every delta, does work on the first.
  const firstDelta = () => {
    if (!stream.waiting) return;
    stream.waiting = false;
    if (ctx.alive && isCurrent()) showStatus(ctx, '');
    // The load just landed, so the dots and the Load button are stale NOW,
    // not only at completion -- a long generation would otherwise show a
    // resident model as idle for its whole duration.
    if (cold && ctx.alive) refreshLoadedIds(ctx);
  };

  // The stored conversation is the request's BASE, but the panel is the
  // user's live intent and its writes to the store are asynchronous
  // (debounced params PUT, fire-and-forget model PUT) -- a Send inside
  // that window would generate with stale store values. Two halves close
  // the race: overrides carry the SET panel values + current model, and
  // the debounced params PUT is FLUSHED first, because a CLEARED value is
  // expressed by absence and only the PUT can spell that (overrides
  // cannot un-set a stored key).
  const overrides = { model: s.modelSelect.value, ...samplerParams(currentCaps(ctx)) };
  const launch = () => streamGenerate(stream.targetConvId, { mode, message_id: messageId, overrides }, {
    signal: controller.signal,
    onToken: (_, full) => { firstDelta(); stream.sawEvent = true; stream.content = full; stream.contentDirty = true; if (ctx.alive) s.paint(); },
    onThinking: (_, full) => { firstDelta(); stream.sawEvent = true; stream.thinking = full; stream.thinkingDirty = true; if (ctx.alive) s.paint(); },
    onSaved: () => { stream.sawEvent = true; },
    onRetryWait: (wait) => {
      if (ctx.alive && isCurrent()) showStatus(ctx, `Server busy -- retrying in ${wait}s…`);
    },
    onComplete: (result) => {
      finishGenerate(ctx, stream, result);
      // A send can load the target AND evict the previous resident
      // (max_loaded_models=1) -- completion is when the residency dots and
      // the Load button go stale, so refresh them here rather than polling.
      if (ctx.alive) refreshLoadedIds(ctx);
    },
    onError: (err) => {
      // In-band typed error: the stream is still alive and heylook_saved
      // may still follow (a partial persists) -- surface it, let
      // onComplete do the cleanup. Transport/HTTP errors end the stream
      // with no onComplete, so they clean up here.
      if (err.inBand) {
        stream.inBandError = err;
        if (ctx.alive && isCurrent()) showStatus(ctx, `Generation failed: ${err.message}`, true);
        return;
      }
      handleStreamError(ctx, stream, err);
    },
  });
  // Flush the pending params PUT, then launch (usually an instant resolve).
  // Guarded: a teardown/switch during the flush must not launch a stream
  // for a page state that no longer exists.
  Promise.resolve(s.paramsBinder?.flush?.()).then(
    () => { if (ctx.alive && s.stream === stream) launch(); },
    () => { if (ctx.alive && s.stream === stream) launch(); },
  );
}

// Throttled painter (one per mount, created in setup): renders only the
// halves that changed since the last frame.
function paintStream(ctx) {
  const s = ctx.state;
  const stream = s.stream;
  if (!stream || s.activeId !== stream.targetConvId) return;
  if (stream.contentDirty) {
    stream.contentDirty = false;
    // A continuation paints prefix + new text as ONE markdown document --
    // rendering them separately would break constructs spanning the seam.
    stream.els.contentEl.innerHTML = renderMarkdown(stream.baseContent + stream.content);
  }
  if (stream.thinkingDirty) {
    stream.thinkingDirty = false;
    stream.els.thinkingEl.hidden = false;
    // Same seam rule as content: a continuation's seeded prior thinking must
    // survive the first new thinking delta, not vanish until finishStream.
    stream.els.thinkingBody.textContent = stream.baseThinking + stream.thinking;
  }
  scrollMessages(ctx);
}

// The Stop button: SERVER-side abort. The partial persists on the server
// and the open stream still delivers the saved rows, so the fetch is left
// alone -- aborting it would throw that final state away. EXCEPT when the
// server answers 404 (nothing active): the stream is in a client-side-only
// phase -- the 503-busy retry sleep, or the dispatch window before the
// claim -- and only a local abort actually stops it.
function stopStream(ctx) {
  const stream = ctx.state.stream;
  if (!stream) return;
  stopGenerate(stream.targetConvId).then((status) => {
    // 404 = nothing active server-side. Abort locally ONLY if this stream
    // has produced no events yet (the 503-retry sleep / dispatch window).
    // Once events flowed, a 404 means the server already FINISHED and
    // released its claim while the tail -- including heylook_saved -- is
    // still in flight to us; aborting then would report a completed
    // generation as "Stopped" and drop the saved rows (review 2026-08-13).
    if (status === 404 && ctx.state.stream === stream && !stream.sawEvent) {
      stream.controller.abort();
    }
  });
}

// Teardown / conversation switch / model switch: kill the fetch. The
// server's disconnect path persists the partial (detached task), and the
// next select of that conversation re-fetches whatever it saved.
function abortStream(ctx) {
  ctx.state.stream?.controller.abort();
}

function releaseStream(ctx, stream) {
  const s = ctx.state;
  // No-op after normal completion; drops the linkedController chain listener.
  stream.controller.abort();
  if (s.stream !== stream) return;
  s.stream = null;
  beforeUnloadGuard.disable();
  if (!ctx.alive) return;
  s.sendBtn.textContent = 'Send';
  // A wait line must not outlive its stream: a zero-token completion, or a
  // Stop pressed during the load, never reaches firstDelta and would strand
  // "Loading…" on screen for good. Clear it only while it is still the line
  // ON SCREEN -- anything written since (the model-switch disclosure, a retry
  // notice) belongs to whatever wrote it. Callers set their own message AFTER
  // this (handleStreamError, finishGenerate's status line), so this can't eat
  // an error either.
  if (stream.waiting && s.activeId === stream.targetConvId
      && s.statusEl.textContent === stream.waitStatus) {
    showStatus(ctx, '');
  }
}

// The server persisted everything before the stream ended (or its
// disconnect path is doing it) -- the client's job is display + adoption.
// No client persistence, no pendingSave window, no unsaved fallback row:
// the whole class of "the mirror wrote something the store didn't get"
// cannot occur on this path anymore.
//
// Adoption is ASSIGNMENT FROM THE SAVED EVENT (spec §4), synchronously: the
// streamed node is never removed by hand -- the same renderMessages pass
// that inserts the saved rows reconciles it away, so there is no frame with
// the response absent and no scrollHeight collapse (the old order removed
// the node, THEN awaited a wholesale GET: on an image conversation that
// round-trip re-downloaded every base64 block, and for its whole duration
// the response was gone and the viewport clamped upward). The GET survives
// only as the fallback for endings that carry no usable rows: transport
// death (no heylook_saved at all -- recovery loop below) and the
// failed/empty generation, where resync is what brings back the tail the
// regenerate/continue mirror hid.
async function finishGenerate(ctx, stream, { content, thinking, usage, aborted, saved }) {
  const s = ctx.state;
  releaseStream(ctx, stream);

  if (!ctx.alive || s.activeId !== stream.targetConvId) return;

  if (saved?.messages?.length) {
    adoptSavedRows(ctx, saved.messages);
    renderMessages(ctx);
  } else {
    await resyncMessages(ctx, stream.targetConvId);
    if (!ctx.alive || s.activeId !== stream.targetConvId) return;
  }
  scrollMessages(ctx);

  const endReason = saved?.end_reason;
  if (aborted || endReason === 'aborted') {
    showStatus(ctx, content ? 'Stopped -- partial response saved.' : 'Stopped.');
  } else if (endReason === 'error' || stream.inBandError) {
    // the in-band error status line stands; a partial (if any) is saved
  } else if (usage || saved) {
    const timing = saved?.timing;
    const parts = [`${usage?.output_tokens ?? '?'} tokens`];
    if (timing?.peak_memory_gb != null) parts.push(`${timing.peak_memory_gb.toFixed(2)} GB peak`);
    if (timing?.kv_cache_bytes != null) parts.push(`${formatBytes(timing.kv_cache_bytes)} KV`);
    if (timing?.draft_acceptance != null) parts.push(`draft ${(timing.draft_acceptance * 100).toFixed(0)}%`);
    showStatus(ctx, parts.join(' · '));
  }

  // A stream that ended WITHOUT its heylook_saved is not success (spec §4:
  // that event is always last) -- the transport died and the server is
  // persisting via its detached disconnect task, which the resync above may
  // have raced. Adopt again with backoff (the task can lose a DB-writer
  // race well past any single fixed delay), then close the loop on the
  // status line. A new stream supersedes the remaining retries -- its own
  // saga end re-adopts everything anyway.
  if (!saved && !aborted && (content || thinking) && !stream.inBandError) {
    showStatus(ctx, 'The stream ended without a save confirmation — recovering…', true);
    const retry = (delay, attemptsLeft) => setTimeout(async () => {
      if (!ctx.alive || s.activeId !== stream.targetConvId || s.stream) return;
      await resyncMessages(ctx, stream.targetConvId);
      if (!ctx.alive || s.activeId !== stream.targetConvId) return;
      if (attemptsLeft > 0) {
        retry(delay * 2.5, attemptsLeft - 1);
      } else if (s.statusEl.textContent.includes('recovering')) {
        showStatus(ctx, 'Recovered what the server saved.');
      }
    }, delay);
    retry(1000, 2);
  }
}

function handleStreamError(ctx, stream, err) {
  const s = ctx.state;
  releaseStream(ctx, stream);
  if (!ctx.alive || s.activeId !== stream.targetConvId) return;
  // s.stream is null now, so this reconcile drops the streaming node -- the
  // renderer owns list structure, the same rule as finishGenerate.
  renderMessages(ctx);
  showStatus(ctx, `Generation failed: ${err.message}`, true);
  // A transport death mid-generation: the server's disconnect path may
  // still have persisted a partial -- adopt whatever it saved (also
  // restores a tail hidden by a regenerate/continue mirror).
  resyncMessages(ctx, stream.targetConvId);
}
