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
import { MarkdownStream, appendPlainText } from '../markdown-stream.js';
import { prepareImage, blobToBase64, MAX_EDGE_PX } from '../image-prep.js';
import { samplerParams, displayWireFields, snapshotSettings, bindDocumentParams, hydrateDocParams, getSetting, setSetting, onSettingsChange, documentScopeNote, PARAM_META } from '../settings.js';
import * as drawer from '../settings-drawer.js';
import { createPresetBar, paintPresetChip } from '../preset-bar.js';
import { createPromptSection } from '../prompt-section.js';
import { createDocumentWriter } from '../document-writer.js';

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

// How often a streaming message repaints. One paint per animation frame is up
// to 120/s on a ProMotion phone; nobody reads at that rate, and every paint
// costs a markdown render plus a scroll write. ~15/s still reads as live text.
const PAINT_INTERVAL_MS = 66;

// How close to the tail still counts as "following along".
const STICK_SLACK_PX = 100;

// The display prefs this page HONORS -- one array, two uses: it declares what the
// drawer may offer (registerSettings `displayPrefs`) and it selects what goes on
// the wire (displayWireFields). Same list for both, so offering a control and
// sending it cannot come apart.
const DISPLAY_PREFS = ['show_special_tokens'];


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
      onIndicator: (info) => { paintPresetChip(s.presetChip, info); paintSysPromptChip(ctx); },
      getStamp: () => s.appliedPresetId,
      setStamp: (id) => setAppliedPreset(ctx, id),
    });
    // The chip needs preset names before the drawer's first lazy fetch.
    s.presetBar.refresh().then(() => { if (ctx.alive) s.presetBar.syncIndicator(); });
    // One throttle for the whole mount (it reads s.stream), not one per
    // stream -- per-stream throttles would pin each stream's closure in the
    // page's cleanup list for the mount lifetime.
    s.docWriter = createDocumentWriter({
      update: api.updateConversation,
      onError: (msg) => showStatus(ctx, msg, true),
    });
    s.paint = ctx.throttleTime(() => paintStream(ctx), PAINT_INTERVAL_MS);
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
      // The panel IS this conversation's stored params once one is open, and
      // the seed for the next new one when none is -- say which.
      scope: () => documentScopeNote('conversation', Boolean(s.activeId)),
      sections: () => [s.presetBar.buildSection(), buildPromptSection(ctx).element],
      onOpen: s.presetBar.onDrawerOpen,
      displayPrefs: DISPLAY_PREFS,
      // What the server resolves an UNSET thinking to for this model (the
      // admin row's thinking_default) -- labels the tri-state's "Model
      // default (on|off)". Null until the admin rows land.
      modelDefaults: () => ({ enable_thinking: currentThinkingDefault(ctx) }),
      // A ticked "Show special tokens" on a gguf model changes nothing:
      // llama-server never emits them (the reasoning split and the stop
      // token happen inside it). Say so under the box rather than let the
      // tick look broken -- the pref is global and stays editable.
      displayNotes: () => (currentProvider(ctx) === 'gguf'
        ? { show_special_tokens: `No effect on ${s.modelSelect.value}: llama-server never `
            + 'emits special tokens, so there is nothing to show or strip. Use Preview '
            + 'prompt to see the markers the template puts around each turn.' }
        : {}),
    });
    ctx.onTeardown(unregisterSettings);

    // Sampler knobs are per-conversation (like the system prompt) via the shared
    // per-document binding: a panel change persists to the active conversation's
    // `params`; hydrate on select is silent so this only fires on real edits +
    // preset applies.
    s.paramsBinder = bindDocumentParams({
      activeId: () => ctx.state.activeId,
      updateDoc: (id, body, opts) => api.updateConversation(id, body, opts),
      onError: (err) => showStatus(ctx, `Settings save failed: ${err.message}`, true),
      onHide: ctx.onHide, // the binder owns its hide flush, like its teardown flush
    });
    ctx.onTeardown(s.paramsBinder);
    ctx.onTeardown(() => clearPendingAttachments(ctx));

    // The page is a mirror of the store: re-adopt it when the tab comes back
    // (why: refreshAfterResume). Each debounced writer owns its own hide
    // flush (prompt-section.js, bindDocumentParams).
    ctx.onResume(ctx.guard(() => refreshAfterResume(ctx)));

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
  s.adminRows = new Map();   // id -> /v1/admin/models row (provider, config, context_*)
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

  // Context size for the NEXT load of a gguf model. A native <select> of
  // power-of-two steps, not a slider: the range is logarithmic (4k to 1M) and
  // a linear thumb cannot land on 32k; a select is also the one control that
  // is already right on a phone with no widgetry. "Auto" is llama-server's
  // own answer (sized from the model, fitted to memory) and is the default
  // -- the stored `ctx_size` preselects when one is set. Hidden for MLX,
  // which has no fixed context allocation. Choosing a different value shows
  // Load/Reload; the value is sent WITH the load and persisted server-side,
  // so the models page shows the same number afterwards.
  s.ctxSelect = createEl('select', {
    class: 'chat__ctx-select', hidden: true,
    title: 'Context size for the next load',
    'aria-label': 'Context size',
  });
  s.ctxSelect.addEventListener('change', () => refreshLoadBtn(ctx));

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
  // Toggles the EFFECTIVE state (explicit value, else the model's default):
  // pressed means the next reply thinks. Writing an explicit true/false
  // rather than cycling through "unset" keeps one tap = one visible change;
  // the drawer's tri-state is where "back to the model default" lives.
  s.thinkBtn.addEventListener('click', () => {
    setSetting('enable_thinking', !effectiveThinking(ctx));
  });
  ctx.onTeardown(onSettingsChange(ctx.guard(() => refreshThinkBtn(ctx))));
  // "What will the model see?" -- the exact prompt the next Send would
  // build, rendered by the model's own engine (v1.79.62). The typed draft is
  // rendered as the next user turn without being sent or saved.
  s.promptBtn = createEl('button', {
    class: 'btn btn--icon', title: 'Preview prompt — the exact text the model will be sent',
    'aria-label': 'Preview prompt',
  });
  s.promptBtn.innerHTML = ICON_PROMPT;
  s.promptBtn.addEventListener('click', () => previewNextPrompt(ctx));
  s.promptPreviewEl = createEl('div', { class: 'prompt-preview chat__prompt-preview', hidden: true });
  s.attachStrip = createEl('div', { class: 'chat__attach', hidden: true });
  s.sendBtn = createEl('button', { class: 'btn btn--primary' }, ['Send']);
  // Three states, not two. A generation now OUTLIVES the response that
  // started it, so this conversation can be generating with no local stream
  // object -- you left mid-generation and came back, or another device
  // started it. Stop still has to reach it, or a runaway 27B has no off
  // switch from the surface that shows it running.
  s.sendBtn.addEventListener('click', () => {
    if (s.stream) return stopStream(ctx);
    if (s.remoteGenerating) return stopRemote(ctx);
    return send(ctx);
  });

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
      s.ctxSelect,
      s.loadNowBtn,
      s.presetChip,
      s.sysPromptChip,
      createEl('div', { class: 'chat__bar-spacer' }),
      settingsBtn,
    ]),
    s.messagesEl,
    s.statusEl,
    s.attachStrip,
    s.promptPreviewEl,
    createEl('div', { class: 'chat__composer' }, [s.attachBtn, s.thinkBtn, s.promptBtn, s.fileInput, s.textarea, s.sendBtn]),
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
// An eye: "show me what the model sees".
const ICON_PROMPT =
  '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" '
  + 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
  + '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>';
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
    // The whole row, for the context control: provider gates it, config
    // carries the stored ctx_size, context_length is the ceiling,
    // context_running is what the resident process actually got.
    s.adminRows = new Map((data.models ?? []).map((m) => [m.id, m]));
    s.loadedKnown = true;
  } catch {
    return; // keep the last known state; the UI never guesses residency
  }
  fillModelSelect(ctx);
  refreshLoadBtn(ctx);
  refreshThinkBtn(ctx);
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
  refreshCtxSelect(ctx);
  // Cold: Load pays the load now. Resident with a different context chosen:
  // Reload restarts the process at the new size -- the one case a loaded
  // model has a reason to show the button at all.
  const cold = isCold(ctx, id);
  const changed = ctxChoiceChanged(ctx);
  s.loadNowBtn.textContent = cold ? 'Load' : 'Reload';
  s.loadNowBtn.title = cold
    ? 'Load this model now so the first message does not pay for it'
    : 'Restart this model with the chosen context size';
  s.loadNowBtn.hidden = !((cold || changed) && !s.loadNowBtn.dataset.busy);
}

// --- context size control -------------------------------------------------

const CTX_MIN = 4096;
const CTX_FALLBACK_MAX = 262144; // ceiling when the header did not say

function fmtCtx(n) {
  return n >= 1048576 ? `${(n / 1048576).toFixed(n % 1048576 ? 1 : 0)}M` : `${Math.round(n / 1024)}k`;
}

// Power-of-two steps from 4k up to the model's training context (or the
// fallback), plus the ceiling itself when it is not a power of two (Qwen3's
// 40960) and the stored value when it is off-grid -- the select must be able
// to SHOW what is stored, or Auto would be preselected over a real value.
function ctxStepsFor(row) {
  const max = row?.context_length || CTX_FALLBACK_MAX;
  const steps = [];
  for (let n = CTX_MIN; n <= max; n *= 2) steps.push(n);
  if (!steps.includes(max)) steps.push(max);
  const stored = row?.config?.ctx_size;
  if (stored && !steps.includes(stored)) steps.push(stored);
  return steps.sort((a, b) => a - b);
}

function refreshCtxSelect(ctx) {
  const s = ctx.state;
  const id = s.modelSelect.value;
  const row = s.adminRows.get(id);
  const gguf = row?.provider === 'gguf';
  s.ctxSelect.hidden = !gguf;
  if (!gguf) {
    // Forget the last gguf model's facts: a hidden control must not come
    // back describing a different model's running context.
    delete s.ctxSelect.dataset.sig;
    s.ctxSelect.title = 'Context size for the next load';
    return;
  }
  const stored = row.config?.ctx_size ?? '';
  const running = row.loaded ? row.context_running : null;
  // Rebuild only when the model (or its facts) moved; an untouched rebuild
  // would throw away a choice the user just made.
  const sig = `${id}|${stored}|${row.context_length ?? ''}|${running ?? ''}`;
  if (s.ctxSelect.dataset.sig === sig) return;
  s.ctxSelect.dataset.sig = sig;
  const autoLabel = running && !stored ? `Auto (${fmtCtx(running)})` : 'Auto';
  const options = [createEl('option', { value: '' }, [autoLabel])];
  for (const n of ctxStepsFor(row)) {
    const tag = n === row.context_length ? ' (max)' : '';
    options.push(createEl('option', { value: String(n) }, [`${fmtCtx(n)}${tag}`]));
  }
  s.ctxSelect.replaceChildren(...options);
  s.ctxSelect.value = stored ? String(stored) : '';
  s.ctxSelect.title = running
    ? `Context size for the next load — running with ${fmtCtx(running)} now`
    : 'Context size for the next load';
}

// The chosen value differs from what is STORED for the model -- the only
// change that means anything, since the stored value is what a load uses.
function ctxChoiceChanged(ctx) {
  const s = ctx.state;
  const row = s.adminRows.get(s.modelSelect.value);
  if (row?.provider !== 'gguf' || s.ctxSelect.hidden) return false;
  const stored = row.config?.ctx_size ?? '';
  return String(stored) !== s.ctxSelect.value;
}

// What to send with the load: the chosen size, 0 for Auto. Null for a model
// the control does not apply to, so the plain load route is used.
function ctxChoiceToSend(ctx) {
  const s = ctx.state;
  const row = s.adminRows.get(s.modelSelect.value);
  if (row?.provider !== 'gguf') return null;
  return s.ctxSelect.value ? Number(s.ctxSelect.value) : 0;
}

async function loadModelNow(ctx) {
  const s = ctx.state;
  const id = s.modelSelect.value;
  if (!id || s.loadNowBtn.dataset.busy) return;
  s.loadNowBtn.dataset.busy = '1';
  s.loadNowBtn.disabled = true;
  s.modelSelect.disabled = true;
  s.ctxSelect.disabled = true;
  // gguf goes through the server-owned reload WITH the context choice (the
  // server persists it and skips the restart when nothing changed); every
  // other provider keeps the plain load.
  const ctxSize = ctxChoiceToSend(ctx);
  const restarting = !isCold(ctx, id) && ctxSize != null;
  showStatus(ctx, restarting
    ? `Restarting ${id} with ${ctxSize ? fmtCtx(ctxSize) : 'auto'} context…`
    : `Loading ${id}…`);
  let summary = null;
  try {
    const result = ctxSize != null
      ? await api.adminReloadModel(id, true, ctxSize)
      : await api.adminLoadModel(id, true);
    if (!ctx.alive) return;
    if (result?.warm_error) {
      showStatus(ctx, `Loaded, but the warm-up generation failed: ${result.warm_error}`, true);
    } else {
      summary = result?.warm_ms != null
        ? `${id} loaded and warmed in ${(result.warm_ms / 1000).toFixed(1)}s.`
        : `${id} loaded.`;
      showStatus(ctx, summary);
    }
  } catch (err) {
    if (!ctx.alive) return;
    showStatus(ctx, `Load failed: ${err.message}`, true);
  } finally {
    delete s.loadNowBtn.dataset.busy;
    s.loadNowBtn.disabled = false;
    s.modelSelect.disabled = false;
    s.ctxSelect.disabled = false;
  }
  if (!ctx.alive) return;
  await refreshLoadedIds(ctx);
  // Say what the process actually got, once the row can tell us: for Auto
  // that is the only place the number exists.
  const running = s.adminRows.get(id)?.context_running;
  if (summary && running && ctx.alive) showStatus(ctx, `${summary} Context ${fmtCtx(running)}.`);
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
  // Thinking loss RIDES an existing warning but never raises one on its own
  // (see thinkingLossNote): losing a capability destroys nothing, and
  // only-loss-gates says disclose that rather than interrupt a routine switch.
  if (lines.length && thinkingLossNote(ctx, from, to)) {
    lines.push('Thinking is unavailable on this model; the toggle will hide.');
  }
  return lines;
}

// The same fact, for the case that has no warning to ride: a text-only
// conversation switching from a thinking model to a plain one used to say
// NOTHING, because the line above was gated on `lines.length` -- so a real
// capability loss was announced only when it happened to coincide with media
// being dropped. Returns the note, or null when nothing is lost.
function thinkingLossNote(ctx, from, to) {
  const s = ctx.state;
  if (getSetting('enable_thinking') !== true) return null;
  const fromCaps = s.models.find((m) => m.id === from)?.capabilities ?? [];
  const toCaps = s.models.find((m) => m.id === to)?.capabilities ?? [];
  if (!fromCaps.includes('thinking') || toCaps.includes('thinking')) return null;
  return `Thinking is unavailable on ${to} — the toggle will hide.`;
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
  // A local, not page state: thinkingLossNote needs where we came FROM, and
  // committedModelId is about to become the destination. It was briefly an
  // `s.lastCommittedModelId`, which read as meaningful elsewhere while only
  // ever being used ten lines below -- and switchWarnings(ctx, from, to)
  // right above already takes `from` as a parameter. Computed BEFORE the
  // abort, because the abort resolves later and `from` is gone by then.
  const from = s.committedModelId;
  const thinking = thinkingLossNote(ctx, from, to);

  // Abandoning, not stopping: the run detaches and commits into THIS
  // conversation under the old model. finishGenerate says so (abandonNote),
  // which also carries the load-cost clause -- so this path writes no line of
  // its own that the async abort would overwrite a beat later. But the
  // THINKING note is not in abandonNote and is not recoverable later, so it
  // rides along on the stream. Suppressing both was how a mid-stream switch
  // silently reintroduced the capability loss v1.79.27 exists to close.
  const abandoning = Boolean(s.stream);
  if (s.stream) {
    s.stream.pendingNote = thinking;
    abortStream(ctx, ABANDON.MODEL);
  }
  s.committedModelId = to;
  // Say what this costs instead of asking whether to pay it. Silent when the
  // target is resident or residency is still unknown -- a guess would be
  // worse than nothing (same rule as the dots).
  if (!abandoning) {
    // Both facts, whichever apply: what this costs, and what it takes away.
    // The thinking note is the ONLY announcement when the conversation has no
    // media for switchWarnings to have warned about.
    const notes = [];
    if (isCold(ctx, to)) {
      notes.push(`${to} is not loaded — your first message loads it, or press Load to do it now.`);
    }
    if (thinking) notes.push(thinking);
    showStatus(ctx, notes.join(' '));
  }
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
  // same gate as the drawer control -- read the cap from PARAM_META so
  // the two can never disagree on the capability name
  s.thinkBtn.hidden = !currentCaps(ctx).includes(PARAM_META.enable_thinking.requiresCap);
  const explicit = getSetting('enable_thinking');
  const on = effectiveThinking(ctx);
  s.thinkBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
  s.thinkBtn.title = explicit == null
    ? `Thinking: ${on ? 'on' : 'off'} (model default)`
    : `Thinking: ${on ? 'on' : 'off'}`;
}

// The model's own thinking default: on the /v1/models row since v1.79.63
// (so it is known from the first paint), the admin row as a fallback.
function currentThinkingDefault(ctx) {
  const s = ctx.state;
  const id = s.modelSelect.value;
  const listed = s.models.find((m) => m.id === id)?.thinking_default;
  if (listed === true || listed === false) return listed;
  return s.adminRows?.get(id)?.thinking_default ?? null;
}

function currentProvider(ctx) {
  const s = ctx.state;
  return s.providerById?.get(s.modelSelect.value) ?? null;
}

// Whether the next reply thinks: the panel's explicit value, else the
// model's default as the server reports it. The same resolution the server
// runs (request > models.toml > capability), read off its answer rather
// than re-derived.
function effectiveThinking(ctx) {
  const explicit = getSetting('enable_thinking');
  if (explicit === true || explicit === false) return explicit;
  return currentThinkingDefault(ctx) === true;
}

// ---------------------------------------------------------------------------
// prompt preview: the exact string the model will be fed (v1.79.62)
// ---------------------------------------------------------------------------

// Special-token spellings across the families this server serves, for
// HIGHLIGHTING only -- the text is the engine's own render, shown verbatim;
// the marks just make <|im_start|> stand out from prose. Unmatched markers
// still show as text, so a family this misses loses nothing but colour.
const SPECIAL_TOKEN_RE = new RegExp(
  '(<\\|[^<>\\n]{1,48}\\|>|<\\|[^<>\\n]{1,48}>|<[^<>\\n]{1,48}\\|>|</?think>'
  + '|<start_of_turn>|<end_of_turn>|<bos>|<eos>|<s>|</s>|\\[INST\\]|\\[/INST\\]|\\[/?THINK\\])', 'g');

function highlightSpecials(text) {
  const frag = document.createDocumentFragment();
  let last = 0;
  for (const m of text.matchAll(SPECIAL_TOKEN_RE)) {
    if (m.index > last) frag.append(document.createTextNode(text.slice(last, m.index)));
    frag.append(createEl('mark', { class: 'prompt-preview__tok' }, [m[0]]));
    last = m.index + m[0].length;
  }
  if (last < text.length) frag.append(document.createTextNode(text.slice(last)));
  return frag;
}

// Render a preview response into `host` (a .prompt-preview container).
function paintPromptPreview(host, body, onClose) {
  const closeBtn = createEl('button', { class: 'btn btn--sm btn--ghost' }, ['Close']);
  closeBtn.addEventListener('click', onClose);
  const what = body.continuation === 'thinking'
    ? 'Resumes inside the open thinking block'
    : body.continuation === 'content'
      ? 'Continues the response after the closed thinking block'
      : 'The next reply generates after this';
  const pre = createEl('pre', { class: 'prompt-preview__text' });
  pre.append(highlightSpecials(body.prompt));
  host.replaceChildren(
    createEl('div', { class: 'prompt-preview__head' }, [
      createEl('span', { class: 'prompt-preview__title' }, ['What the model will see']),
      createEl('span', { class: 'muted small' }, [
        `${body.model_id} · ${body.provider} · ${body.char_count.toLocaleString()} chars · ${what}`,
      ]),
      closeBtn,
    ]),
    pre,
  );
  host.hidden = false;
}

function paintPromptPreviewError(host, message, onClose) {
  const closeBtn = createEl('button', { class: 'btn btn--sm btn--ghost' }, ['Close']);
  closeBtn.addEventListener('click', onClose);
  host.replaceChildren(
    createEl('div', { class: 'prompt-preview__head' }, [
      createEl('span', { class: 'prompt-preview__title' }, ['Prompt preview']),
      createEl('span', { class: 'error-note' }, [message]),
      closeBtn,
    ]),
  );
  host.hidden = false;
}

// Composer entry: the draft as the next user turn (append mode).
async function previewNextPrompt(ctx) {
  const s = ctx.state;
  if (!s.activeId) { showStatus(ctx, 'Start a conversation to preview its prompt.', true); return; }
  const host = s.promptPreviewEl;
  const close = () => { host.hidden = true; host.replaceChildren(); };
  const draft = s.textarea.value.trim();
  try {
    const body = await api.previewPrompt(s.activeId, {
      mode: 'append',
      user_content: draft || undefined,
      overrides: { model: s.modelSelect.value, ...samplerParams(currentCaps(ctx)) },
    });
    if (!ctx.alive) return;
    paintPromptPreview(host, body, close);
  } catch (err) {
    if (ctx.alive) paintPromptPreviewError(host, err.message, close);
  }
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

// Thin delegates to the shared writer (document-writer.js). What stays here is
// only what is genuinely this page's: which api function to call, and the
// pre-create guard below.
function putSystemPrompt(ctx, docId, value, opts) {
  ctx.state.docWriter.putSystemPrompt(docId, value, opts);
}

function setAppliedPreset(ctx, presetId) {
  const s = ctx.state;
  s.appliedPresetId = presetId;
  if (!s.activeId) return; // pre-create draft: the stamp rides in state only
  s.docWriter.setAppliedPreset(s.activeId, presetId);
}

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

// Chat builds a NEW section per drawer build (the drawer forces a rebuild on
// every conversation switch), so the shared factory's construction-time
// owner capture already gives per-conversation isolation for free -- no
// setValue call needed here.
// Persist the applied-preset stamp onto the active conversation. Explicit
// stamps only (the preset bar calls this from apply/save/delete) -- see the
// provenance note in preset-bar.js.

function buildPromptSection(ctx) {
  const s = ctx.state;
  // One live section at a time: releasing the one this render replaces
  // flushes its pending write and drops its hide hook, so hooks don't pile
  // up across renders and two sections never both answer one hide.
  s.promptSection?.release();
  s.promptSection = createPromptSection(ctx, {
    owner: () => s.activeId,
    get: () => s.systemPrompt,
    set: (v) => {
      s.systemPrompt = v;
      // With no conversation the factory's persist is a deliberate no-op
      // (builtFor is null) -- park the draft here instead, per keystroke.
      if (!s.activeId) writeDraftPrompt(v);
    },
    persist: (v, id, opts) => putSystemPrompt(ctx, id, v, opts),
    onEdit: () => s.presetBar.updateDrift(), // prompt edits drift the selected preset live
    label: 'System prompt for this conversation',
  });
  return s.promptSection;
}


// True while a switch warning owns the status line. showStatus assigns
// textContent, which deletes the warning AND its Cancel / Switch-anyway
// buttons -- leaving the select on an unconfirmed model with no way to
// confirm or cancel, and the next send going to that model. An async abandon
// note landing a beat later did exactly that.
function switchWarningOnScreen(ctx) {
  return Boolean(ctx.state.statusEl?.querySelector('.chat__switch-warning'));
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
  if (!ctx.state.stream && ctx.state.remoteGenerating) {
    showStatus(ctx, 'This conversation is still generating on the server — press Stop, or wait for it to finish.');
    return true;
  }
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
    // NOT an armed confirm (owner call, v1.79.37). Only LOSS gates: a clone
    // destroys nothing and one Del undoes it, so the second tap was buying
    // inconvenience, not safety -- and it sat immediately beside Del, whose arm
    // does guard a loss, teaching that these buttons ask twice as a matter of
    // course. The outcome is disclosed instead ("Conversation cloned.").
    const copy = createEl('button',
      { class: 'btn btn--sm btn--ghost conv-item__clone', title: 'Clone conversation' }, ['Copy']);
    copy.addEventListener('click', (e) => {
      e.stopPropagation();   // armedConfirm used to do this; the row is clickable
      cloneConversation(ctx, conv.id);
    });
    const del = armedConfirm(
      createEl('button', { class: 'btn btn--sm btn--ghost conv-item__delete' }, ['Del']),
      () => deleteConversation(ctx, conv.id),
    );
    const item = createEl('div', {
      class: `conv-item${conv.id === s.activeId ? ' conv-item--active' : ''}`,
    }, [title, ren, copy, del]);
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
    // AWAITED, and the reason follows the answer. `stopGenerate` resolves to
    // the STATUS: 200 = stopped, 404 = nothing was active because it already
    // finished -- both mean the run genuinely ended server-side, which is what
    // ABANDON.DELETE asserts and, since the strongest reason now wins, the
    // strongest thing this set can assert. `null` is the DELETE never
    // arriving: the run is still going as far as anyone here knows, so claim
    // nothing. `stopRemote` has modelled this for the same endpoint since
    // v1.79.29 and the two must not diverge.
    //
    // Awaiting also fixes the 409 retry it sits above: the conversation DELETE
    // used to race the stop it depends on, so the 600ms retry was covering for
    // an ordering this function could simply have had.
    const status = await stopGenerate(convId);
    abortStream(ctx, status === null ? ABANDON.TEARDOWN : ABANDON.DELETE);
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
    abortStream(ctx, ABANDON.DELETE);
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

async function cloneConversation(ctx, convId) {
  const s = ctx.state;
  if (s.stream?.targetConvId === convId) {
    showStatus(ctx, 'Cannot clone while generation is in progress', true);
    return;
  }
  // Re-entry guard, NOT a resurrected confirm. armedConfirm used to coalesce a
  // double-tap for free (the first tap only armed), so dropping it in v1.79.37
  // silently made two fast taps two clones -- plus two racing selectConversation
  // awaits whose interleaving decided which one ended up active. The button is
  // small and sits on a phone, where double-tap is the norm, not the accident.
  //
  // PAGE-LEVEL, not per-conversation id. A per-id Set closes the double-tap and
  // leaves the race its own comment names: Clone on row A then row B inside the
  // request window is still two clones and two racing selectConversation awaits
  // deciding the active document -- the identical symptom, one row over. Only
  // one clone may be in flight at a time, which is also what the user means by
  // the single "Cloning conversation…" status this shows.
  //
  // Guarding HERE rather than by disabling the button: the row is re-rendered by
  // the list refresh below, so a disabled flag set on the old node would be
  // dropped mid-flight, and any future caller gets the guard for free.
  if (s.cloneInFlight) return;
  s.cloneInFlight = true;
  try {
    showStatus(ctx, 'Cloning conversation…');
    const cloned = await api.cloneConversation(convId);
    if (!ctx.alive) return;
    // RE-LIST rather than unshift the response. `POST /clone` returns the FULL
    // conversation -- every message, and the content blocks with it -- while
    // `s.conversations` holds LIST-shaped rows: the server deliberately drops
    // system_prompt and params there and adds `generating`. Unshifting the
    // clone put one row of a different shape among the others -- it pinned the
    // whole cloned message list in memory for the session (the cost the list
    // trim exists to avoid), and it carried no `generating` key, so
    // `refreshAfterResume`'s `'generating' in activeRow` test answered
    // differently for it than for every other row.
    // Re-listing rather than hand-copying the list fields here: WHICH fields
    // are list-shaped is the server's decision (`_CONV_LIST_NAMES`, derived by
    // subtraction so a new column joins only deliberately), and a second copy
    // of that list in the client is the drift this repo keeps paying for.
    const listed = await api.listConversations({ signal: ctx.signal }).catch(() => null);
    if (!ctx.alive) return;
    if (listed?.conversations) s.conversations = listed.conversations;
    else s.conversations.unshift(cloned);  // list fetch failed: a stale shape beats no row
    renderConvList(ctx);
    await selectConversation(ctx, cloned.id);
    s.rootEl.classList.remove('chat--convs-open');
    showStatus(ctx, 'Conversation cloned.');
  } catch (err) {
    if (ctx.alive) showStatus(ctx, `Clone failed: ${err.message}`, true);
  } finally {
    s.cloneInFlight = false;
  }
}

async function selectConversation(ctx, convId) {
  const s = ctx.state;
  if (s.activeId === convId && s.messages.length) return;
  // Named BEFORE the abort, and disclosed after the load below: finishGenerate
  // returns early once activeId has moved, so it is not going to say this for
  // us. The run itself is unharmed -- it detaches and commits the whole reply
  // into the conversation being left.
  // `|| 'That conversation'`, not `?? null`: an untitled conversation (empty
  // string, or one missing from the list) is falsy, and the disclosure the
  // user guide promises would silently not appear at all.
  const leaving = s.stream
    ? (s.conversations.find((c) => c.id === s.stream.targetConvId)?.title || 'That conversation')
    : null;
  if (s.stream) abortStream(ctx, ABANDON.CONVERSATION);
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
    adoptConversationMeta(ctx, conv);
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
    // Last, so the empty-status reset above cannot swallow it.
    if (leaving) {
      showStatus(ctx, `"${leaving}" keeps generating — it will finish on the server `
        + 'and be there when you come back.');
    }
  } catch (err) {
    if (ctx.alive && s.activeId === convId) {
      showStatus(ctx, `Could not load conversation: ${err.message}`, true);
      // The doc never loaded: clear the PREVIOUS conversation's leftovers so
      // (a) nothing renders/writes under the wrong id, and (b) re-clicking
      // this conversation retries -- the select guard's messages.length
      // condition would otherwise no-op on the stale array forever.
      s.messages = [];
      // Through the one adopter, with an empty row: the sampler panel must
      // not keep the PREVIOUS conversation's params under this id either --
      // the next knob edit would PUT them onto it.
      adoptConversationMeta(ctx, {});
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

// Render key + signature. Reconcile, never rebuild. The original reason was
// `content-visibility: auto` -- an off-screen row knew only its 3rem estimate
// until laid out once, that measurement lived on the NODE, and rebuilding
// threw it away, collapsing scrollHeight mid-render. That mechanism went with
// the feature, but the rule stands on its own: a rebuild drops open editors
// and their unsaved drafts. Historically it also dumped a long
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
function msgSignature(msg, { editing, capsKey, provider, modelNote }) {
  return [
    msg.role,
    msg.position,
    editing ? `edit:${provider}` : 'view',
    hasMediaBlocks(msg) ? capsKey : '',
    // Which model produced the row -- rendered only while the thread MIXES
    // models, so the note string (not the raw stamp) is what the signature
    // carries: rows rebuild exactly when the label appears/disappears.
    modelNote,
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
  // Per-message model attribution (schema v7 model_id), shown only when the
  // thread actually MIXES models -- a single-model thread labelling every
  // row would be noise stating the header's select.
  const mixedModels = new Set(s.messages.map((m) => m.model_id).filter(Boolean)).size > 1;
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
      const modelNote = mixedModels && msg.model_id ? msg.model_id : '';
      const sig = msgSignature(msg, { editing, capsKey, provider, modelNote });
      const cached = prev.get(key);
      let node = cached?.sig === sig ? cached.node : null;
      if (!node) {
        node = editing ? buildEditEl(ctx, msg) : buildMessageEl(ctx, msg, modelNote);
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

function buildMessageEl(ctx, msg, modelNote = '') {
  const content = createEl('div', { class: 'message-content' });
  if (msg.role === 'assistant') content.innerHTML = renderMarkdown(msg.content);
  else content.textContent = msg.content;

  const bubbleChildren = [];
  if (hasBlocks(msg, 'image')) {
    bubbleChildren.push(createEl('div', { class: 'message-images' },
      msg.content_blocks
        .filter((b) => b.type === 'image')
        // decoding="async" keeps the decode off the main thread. NO
        // loading="lazy" here, deliberately: a lazily-loaded image has no
        // height until it arrives, and WebKit has no scroll anchoring to
        // absorb the shift -- the same reason `content-visibility` was
        // removed from `.message` altogether (css/app.css). A height that
        // arrives late is the whole bug class. Revisit only alongside stored
        // image dimensions, which would let the box be reserved up front.
        .map((b) => createEl('img', {
          class: 'message-image',
          src: blockSourceUrl(b),
          alt: 'attached image',
          decoding: 'async',
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
  // to the CURRENT model (the server's _wire_content, conversation_generate_api.py,
  // drops what the caps exclude). The
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
  // Attribution note (renderMessages passes it only while the thread mixes
  // models): which model produced this row.
  if (modelNote) {
    children.push(createEl('div', { class: 'message-model-note muted small' }, [modelNote]));
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
      role: msg.role,
      content: msg.content_blocks?.length ? msg.content_blocks : (msg.content ?? ''),
      thinking: msg.thinking || undefined,
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
  const cancel = () => { s.editingId = null; renderMessages(ctx); keepRowInView(ctx, msg); };

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
        keepRowInView(ctx, msg);
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
  // "Preview prompt": what the editor's primary action WOULD send, rendered
  // from the boxes as they are now (nothing saved). An assistant row
  // previews Save & Continue (continue mode from this row); a user row
  // previews Save & Regenerate (regenerate from the row after it, or a
  // plain append when it is last) with the edited text overlaid. The marker
  // question ("what wraps the thinking?") is answered by the render itself.
  let previewHost = null;
  if (msg.id) {
    previewHost = createEl('div', { class: 'prompt-preview message-edit__preview', hidden: true });
    const previewBtn = createEl('button', {
      class: 'btn btn--sm btn--ghost',
      title: msg.role === 'assistant'
        ? 'Show the exact text the model would be sent by Save & Continue, from these boxes as they are now'
        : 'Show the exact text the model would be sent by Save & Regenerate, with this box as it is now',
    }, ['Preview prompt']);
    previewBtn.addEventListener('click', async () => {
      const close = () => { previewHost.hidden = true; previewHost.replaceChildren(); };
      const edits = { message_id: msg.id, content: textarea.value };
      if (thinkArea) edits.thinking = thinkArea.value || null;
      const nextMsg = s.messages.find((m) => m.position > msg.position);
      const shape = msg.role === 'assistant'
        ? { mode: 'continue', message_id: msg.id }
        : nextMsg ? { mode: 'regenerate', message_id: nextMsg.id } : { mode: 'append' };
      try {
        const body = await api.previewPrompt(s.activeId, {
          ...shape, edits,
          overrides: { model: s.modelSelect.value, ...samplerParams(currentCaps(ctx)) },
        });
        if (!ctx.alive) return;
        paintPromptPreview(previewHost, body, close);
      } catch (err) {
        if (ctx.alive) paintPromptPreviewError(previewHost, err.message, close);
      }
    });
    buttons.unshift(previewBtn);
  }
  if (msg.role === 'user') {
    const saveRegen = createEl('button', { class: 'btn btn--sm btn--primary' }, ['Save & Regenerate']);
    saveRegen.addEventListener('click', () => save(true));
    buttons.push(saveRegen);
  }
  // Continuation works for BOTH roles (an assistant message is finished; a
  // user message is co-written) -- but user-role continuation is MLX-only:
  // llama-server prefills assistant turns and has no user-turn spelling. Still
  // FAIL CLOSED (never fire it on a guess -- the truncate is destructive and
  // lands before the 400), but DISABLED WITH A REASON rather than absent: a
  // button that silently comes and goes with the model reads as arbitrary, and
  // the reason is invisible at exactly the moment you look for it.
  if (msg.id) {
    const provider = s.providerById?.get(s.modelSelect.value);
    const blocked = msg.role === 'user'
      ? (provider == null
          ? 'Checking what this model supports…'
          : provider === 'gguf'
            ? `Continuing your own message is not supported on ${s.modelSelect.value} — that needs an MLX model.`
            : null)
      : null;
    const saveContinue = createEl('button', {
      class: 'btn btn--sm btn--primary',
      disabled: Boolean(blocked),
      title: blocked ?? 'Save, drop everything after, and let the model finish this message',
    }, ['Save & Continue']);
    if (!blocked) saveContinue.addEventListener('click', () => save(false, true));
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
  if (previewHost) editChildren.push(previewHost);
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

// The STRUCTURAL scroll: a row was added, removed or rebuilt.
function scrollMessages(ctx, force = false) {
  const el = ctx.state.messagesEl;
  if (!force && !followingTail(ctx)) return;
  el.scrollTop = el.scrollHeight;
  ctx.state.pinnedTop = el.scrollTop;
  // One write, no re-aim. A row added this tick lays out at its REAL height
  // immediately, so scrollHeight is already right when we read it. A second
  // frame used to re-aim after the browser got round to laying the row out --
  // necessary only while `content-visibility: auto` let a fresh row report a
  // 3rem estimate. That is gone (css/app.css) and this went with it.
}

// Is the reader at the tail, i.e. should the view follow the generation?
//
// The streaming painter calls this BEFORE it mutates the message and pins
// AFTER, and that order is load-bearing twice over. It is CHEAP before the
// write, because layout is still clean from the previous paint, so the reads
// are cache hits rather than a forced re-layout -- and a forced layout walks
// every row in the conversation, since nothing skips off-screen rows any more
// (css/app.css). And it is only HONEST
// before the write: measured after, a single paint that appends more than the
// slack -- a code block, a table -- reads as "the reader has scrolled away"
// and would strand the view above the tail for the rest of the generation.
//
// A cached flag was tried instead and is wrong: nothing writes it reliably
// while a stream runs. Pinning writes `scrollTop`, but the resulting scroll
// events are coalesced to a handful across a whole generation (measured), so
// the flag goes stale exactly when the viewport changes underneath it -- which
// on a phone is every time the keyboard opens.
function followingTail(ctx) {
  const s = ctx.state;
  const el = s.messagesEl;
  // The gap alone cannot tell "the reader scrolled away" from "the gap opened
  // underneath them" -- and the second happens for two reasons here: one paint
  // appending more than the slack (a code block, a table), and a VIEWPORT
  // RESIZE shrinking clientHeight, which on a phone is every keyboard. Both
  // stranded the view mid-generation, measured.
  //
  // scrollTop is the honest discriminator, the same one scrollMessages uses
  // for a row growing past its intrinsic-size estimate: growth and resize move
  // scrollHeight and clientHeight, never scrollTop. So if scrollTop is still
  // where we last pinned it, the reader has not moved and we keep following,
  // whatever the gap says.
  // A VIEWPORT RESIZE is the case the gap gets wrong and this gets right: on
  // a phone the keyboard shrinks clientHeight by a few hundred px in one step,
  // opening a gap far past the slack while the reader has not moved at all.
  // Growth and resize move scrollHeight and clientHeight; only the reader
  // moves scrollTop. So if scrollTop is still where we last pinned it, keep
  // following whatever the gap says.
  //
  // Briefly removed along with content-visibility on an A/B that came back
  // clean -- against a reproducer whose stub truncated the thread and never
  // exercised a resize. At a realistic tail position it stranded 771px on
  // every run. Both changes were needed; only one of them was the root cause.
  if (ctx.state.pinnedTop != null
      && Math.abs(el.scrollTop - ctx.state.pinnedTop) < 2) return true;
  return el.scrollHeight - el.scrollTop - el.clientHeight < STICK_SLACK_PX;
}

// The STREAMING pin: one write. Rows lay out at their real height, so there is
// nothing to read back and nothing to re-aim at.
function pinToTail(ctx) {
  const el = ctx.state.messagesEl;
  el.scrollTop = el.scrollHeight;
  // Post-clamp: what the write actually landed on, which is what
  // followingTail compares a later read against.
  ctx.state.pinnedTop = el.scrollTop;
}

// After an editor closes (Save or Cancel) the row it stood in is rebuilt as a
// view row, and on a phone the keyboard that was open for the textarea goes
// away with it. WebKit restores the scroll offsets it moved for the keyboard
// asynchronously, and the restore lands the row partly under the composer
// (measured 174px low on an iOS 26.5 simulator with content-visibility off,
// which is now the only mode) -- so it reads as "my message vanished until I
// tapped around". Re-aim
// at the row with the least movement that brings it fully into view: once
// now, once after the viewport has had time to settle. `nearest` never moves
// a row that is already visible, so the desktop path is a no-op.
function keepRowInView(ctx, msg) {
  const aim = () => {
    if (!ctx.alive) return;
    ctx.state.msgNodes?.get(msgKey(msg))?.node?.scrollIntoView({ block: 'nearest' });
  };
  requestAnimationFrame(aim);
  setTimeout(aim, 250);
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
// Returns 'generating' | 'idle' | 'unknown' -- THREE answers, because falsy
// used to mean three different things and callers read all of them as "the
// server is done". A failed GET returned undefined; the stale-context guard
// returned false; only the third was an actual answer. The recovery loop then
// announced "Recovered what the server saved." having never reached the
// server. Anything that decides on this MUST distinguish 'unknown' -- it is
// the absence of an answer, not a negative one.
async function resyncMessages(ctx, convId) {
  const s = ctx.state;
  let conv;
  try {
    conv = await api.getConversation(convId, { signal: ctx.signal });
  } catch {
    return 'unknown'; // best-effort: the next saga end (or a reselect) retries
  }
  if (!ctx.alive || s.activeId !== convId || s.stream) return 'unknown';
  mergeServerRows(ctx, conv.messages ?? []);
  // Since v1.79.12 a run outlives the response that started it, so "our
  // stream ended" no longer implies "the server is done". Adopt what the
  // server says and hand it back, because the difference decides whether
  // there is anything to recover.
  setRemoteGenerating(ctx, Boolean(conv.generating));
  return conv.generating ? 'generating' : 'idle';
}

// The one way server rows replace the mirror outside a stream end: saved
// rows wholesale, id-less rows re-seated after them (see resyncMessages).
function mergeServerRows(ctx, serverRows) {
  const s = ctx.state;
  const unsaved = s.messages.filter((m) => m.id == null);
  // An open editor on a row the server no longer has (deleted elsewhere):
  // keep the local row, and the draft in it, rather than vanish typed work
  // without a word. Save will 404 loudly and resync; Cancel drops it.
  const orphanEditor = s.editingId != null && !serverRows.some((r) => r.id === s.editingId)
    ? s.messages.find((m) => m.id === s.editingId) ?? null
    : null;
  if (orphanEditor) {
    showStatus(ctx, 'The message you are editing was deleted elsewhere — your draft is kept until you cancel or save.', true);
  }
  const rows = orphanEditor
    ? [...serverRows, orphanEditor].sort((a, b) => a.position - b.position)
    : [...serverRows];
  let nextPos = (rows[rows.length - 1]?.position ?? -1) + 1;
  for (const m of unsaved) m.position = nextPos++;
  s.messages = [...rows, ...unsaved];
  renderMessages(ctx);
}

// The per-conversation state a fetched row carries into the page, in ONE
// place for both adopters (select and resume) -- the next stored field is
// added here or it is silently missing on the path nobody retests. Rows,
// scroll and the model select stay with the caller: select moves the select
// to the conversation's model, resume deliberately does not (a model the
// user picked since is newer than the store's stamp).
// `keepPrompt`: leave s.systemPrompt alone (a resume while the prompt is being
// typed -- the box holds the only state newer than the store).
function adoptConversationMeta(ctx, conv, { keepPrompt = false } = {}) {
  const s = ctx.state;
  if (!keepPrompt) s.systemPrompt = conv.system_prompt ?? null;
  s.appliedPresetId = conv.applied_preset_id ?? null;
  hydrateDocParams(conv);  // sampler panel <- this conversation (silent, no re-PUT)
  setRemoteGenerating(ctx, Boolean(conv.generating));
}

// A run this page is not subscribed to. Server-owned generation means the
// answer keeps coming while you are elsewhere; this is the surface that says
// so and offers the way out. Cleared the moment a LOCAL stream takes over --
// the local one owns the button from then on.
// Every line that claims a run is still going, and the one predicate that
// recognises them. The clear-branch used to test startsWith('Still
// generating'), which the switch-model wording does not match -- so that note
// could never be cleared and the page went on claiming a reply was coming for
// an idle conversation. A shared prefix plus one predicate means a new
// wording cannot come apart from its own eraser.
const GENERATING_PREFIX = 'Still generating on the server';
const MODEL_SWITCH_PREFIX = 'The reply in flight keeps generating';
const isGeneratingNote = (text) =>
  text.startsWith(GENERATING_PREFIX) || text.startsWith(MODEL_SWITCH_PREFIX);

// The composer button has exactly two REST states and they are written from
// two places (a remote flag changing, and a local stream releasing). One
// speller, because a title that drifts from its text is how "Stop" came to
// mean two different things with nothing on screen distinguishing them.
// The LOCAL stream's Stop is deliberately not here: it says something else
// ("keep what has been generated so far") and is written where it is set.
function setSendButton(s, remoteGenerating) {
  s.sendBtn.textContent = remoteGenerating ? 'Stop' : 'Send';
  // The button says the same word for two different situations. Only the
  // tooltip can separate "stop what you are watching" from "stop a run you
  // are not subscribed to" without a second control.
  s.sendBtn.title = remoteGenerating
    ? 'Stop the run finishing on the server for this conversation'
    : '';
}

function setRemoteGenerating(ctx, on) {
  const s = ctx.state;
  // A LOCAL stream owns the button and the status line, so a remote flag is
  // not merely ignored here -- it must not be RECORDED either. The server
  // correctly reports our own in-flight run as generating, so a resume during
  // it used to latch this true beneath the stream; releaseStream resets the
  // button but not this flag, and afterwards Send silently performed a Stop
  // (and the `=== on` guard above made the latch self-perpetuating, swallowing
  // the later honest value as a no-op). The local stream's own end re-derives
  // this through resyncMessages.
  //
  // OWNERSHIP, not existence. A stream aimed at another conversation owns THAT
  // conversation's button, not this one's -- so a switch INTO a generating
  // conversation, made while the abandoned stream is still unwinding, must
  // still be allowed to say so. In practice the unwind wins that race (it is
  // microtasks; the switch awaits a GET), which is why no symptom was ever
  // seen -- but "no symptom" is a fact about scheduling, and the sentence
  // above is a claim about ownership. They should not have been the same line.
  if (s.stream && s.stream.targetConvId === s.activeId) return;
  if (s.remoteGenerating === on) return;
  s.remoteGenerating = on;
  setSendButton(s, on);
  if (on) {
    // NOT "started elsewhere": since abandoning a run is routine and now
    // disclosed, the commonest way to see this line is a run THIS tab started
    // and walked away from. Claiming another origin was a guess, and a wrong
    // one most of the time.
    showStatus(ctx, `${GENERATING_PREFIX} — Stop ends it and keeps the partial.`);
  } else if (isGeneratingNote(s.statusEl.textContent)) {
    showStatus(ctx, '');
  }
}

// Poll the store a few times with backoff until the run has actually
// finished. Not a general polling loop: it is bounded, it only runs after an
// explicit Stop, and it stops the moment the server says the run is done --
// the alternative is a partial that stays invisible until the user happens to
// reselect the conversation.
function resyncUntilSettled(ctx, convId, delay = 250, attemptsLeft = 4) {
  setTimeout(async () => {
    if (!ctx.alive || ctx.state.activeId !== convId || ctx.state.stream) return;
    const state = await resyncMessages(ctx, convId);
    if (!ctx.alive || ctx.state.activeId !== convId) return;
    // Only a real 'idle' ends the poll. 'unknown' means we never got an
    // answer, so keep trying rather than announcing a stop we cannot see.
    if (state === 'idle') {
      if (ctx.state.statusEl.textContent === 'Stopping…') showStatus(ctx, 'Stopped.');
      return;
    }
    if (attemptsLeft > 0) resyncUntilSettled(ctx, convId, delay * 2, attemptsLeft - 1);
  }, delay);
}

// Stop a run this page never subscribed to. The server aborts and commits the
// partial exactly as it does for the local Stop; the row lands on the next
// refresh rather than through a stream we are not reading.
function stopRemote(ctx) {
  const s = ctx.state;
  const convId = s.activeId;
  if (!convId) return;
  showStatus(ctx, 'Stopping…');
  // stopGenerate resolves to the STATUS (null on network failure, 404 when
  // the server has nothing active). Reporting "stopped" for all three told
  // the user a run had been stopped that was still going -- the one thing
  // this button exists to promise.
  stopGenerate(convId).then((status) => {
    if (!ctx.alive || s.activeId !== convId) return;
    if (status === null) {
      showStatus(ctx, 'Could not reach the server to stop it — still generating.', true);
      return;  // leave the flag ON: as far as we know it is still running
    }
    // 404 = the server has nothing active: it already finished. Not an error,
    // but not a stop either -- adopt the truth from the store either way.
    setRemoteGenerating(ctx, false);
    // The abort is asynchronous server-side: the provider has to observe the
    // flag, unwind, and win the DB writer queue. One immediate resync almost
    // always precedes the commit, so the partial would stay invisible until a
    // reselect. Re-check until the run actually clears.
    resyncUntilSettled(ctx, convId);
  });
}

// What the sidebar shows, in order (order already encodes recency); not
// updated_at itself, which every local write bumps without the page noticing.
const convListFingerprint = (convs) => JSON.stringify(convs.map((c) => [c.id, c.title]));

// Re-adopt the store after the tab comes back (ctx.onResume). The page is a
// mirror with no other invalidation: nothing polls, and the select guard
// deliberately skips re-fetching the active conversation. That is fine while
// the tab runs continuously; it is not fine on a phone, where Safari freezes
// a backgrounded tab and later resumes it with the heap it had -- hours old,
// and every whole-value write from it (prompt keystroke, sampler PUT, preset
// Save snapshot) would re-play that stale state over whatever happened since.
//
// Cost-shaped: the list and the presets are cheap; the conversation body is
// not (every row, base64 media inline), so it is fetched only when the
// list's `updated_at` for the active conversation differs from the one the
// page holds -- every message write touches it server-side, so an equal
// stamp means nothing to adopt. A local PUT bumps the server's stamp without
// updating the page's copy, which costs one extra fetch on the next resume
// and nothing else. Two things are never touched: a live stream's rows (the
// stream owns them; heylook_saved reconciles), and the prompt while it is
// being TYPED -- the box holds the one local state newer than the store, and
// its own debounce will write it. Every other drawer field commits on change
// and re-reads the panel, so the stamp and params adopt wherever focus is.
//
// The page commits the new `updated_at` only once everything it covers has
// been adopted. Committing it first would turn one failed body fetch (phone
// radio half up) or one deferred adoption into a permanent "unchanged": the
// select guard never refetches the active conversation, so nothing else
// would ever retry.
async function refreshAfterResume(ctx) {
  const s = ctx.state;
  if (s.resumeSync) return; // one in flight; a burst of events is one resume
  s.resumeSync = true;
  try {
    const convId = s.activeId;
    const [presetsChanged, list] = await Promise.all([
      s.presetBar.refresh(),
      api.listConversations({ signal: ctx.signal }).catch(() => null),
    ]);
    if (!ctx.alive) return;
    let docChanged = false;
    if (list?.conversations) {
      const held = s.conversations.find((c) => c.id === convId)?.updated_at;
      const fresh = list.conversations.find((c) => c.id === convId)?.updated_at;
      const unchanged = Boolean(held && fresh && held === fresh);
      let adopted = unchanged;
      if (convId && !unchanged) {
        const conv = await api.getConversation(convId, { signal: ctx.signal }).catch(() => null);
        if (!ctx.alive) return;
        if (conv && s.activeId === convId) {
          docChanged = true;
          const typingPrompt = document.activeElement?.classList.contains('sysprompt-input') ?? false;
          adoptConversationMeta(ctx, conv, { keepPrompt: typingPrompt });
          refreshThinkBtn(ctx);
          const rowsMerged = !s.stream;
          if (rowsMerged) mergeServerRows(ctx, conv.messages ?? []);
          adopted = !typingPrompt && rowsMerged;
        }
      }
      // A sidebar rename commits on blur against the conv object it was
      // started on; swapping the list under it orphans that commit, and
      // WebKit removes the input without a blur at all. Leave the list alone
      // until the rename settles -- the next resume catches up.
      // The list carries `generating` for every row; adopt the ACTIVE one
      // even when the body was not refetched. Reaching it only through
      // adoptConversationMeta (which runs only when updated_at moved) left a
      // real stick: a run that generated nothing never moves updated_at, so
      // the composer read "Stop" and refused every send until the user
      // navigated away and back.
      const activeRow = list.conversations.find((c) => c.id === convId);
      if (activeRow && 'generating' in activeRow) {
        setRemoteGenerating(ctx, Boolean(activeRow.generating));
      }
      const renaming = Boolean(s.convListEl.querySelector('.conv-item__rename'));
      if (!renaming) {
        const before = convListFingerprint(s.conversations);
        s.conversations = list.conversations.map((c) =>
          (c.id === convId && !adopted && held) ? { ...c, updated_at: held } : c);
        if (convListFingerprint(s.conversations) !== before) renderConvList(ctx);
      }
    }
    if (presetsChanged || docChanged) {
      s.presetBar.syncIndicator(); // repaints both chips via onIndicator
      drawer.requestRebuild();     // focus-guarded: never under a field being edited
    }
  } finally {
    s.resumeSync = false;
  }
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


// Picker dispatch: the input's accept list is capability-driven, but a
// picker can't be trusted (drag/drop, "All files") -- split by MIME here.
// One staging routine parameterized per kind (the repo's shared-factory
// discipline: a cap/guard fix lands once, never in a per-kind twin).
// A staged entry holds a BLOB and an object URL for the thumbnail -- never
// base64. Base64 is 1.33x the bytes and it used to be produced at staging time
// and retained for as long as the attachment sat in the composer; it is now
// produced at send (buildContentBlocks) and lives only as long as the request
// body. `prepare` is per-kind because only images can be downscaled.
const ATTACH_KINDS = {
  image: {
    mime: 'image/', cap: 'vision', stateKey: 'pendingImages', max: MAX_ATTACH_IMAGES,
    label: 'image',
    prepare: async (f) => {
      const { blob, mediaType, resized } = await prepareImage(f);
      return { blob, previewUrl: URL.createObjectURL(blob), mediaType, resized, sourceBytes: f.size };
    },
  },
  audio: {
    mime: 'audio/', cap: 'audio', stateKey: 'pendingAudio', max: MAX_ATTACH_AUDIO,
    label: 'audio clip',
    // Audio is passed through: there is no cheap, lossless equivalent of a
    // resolution cap, and the 30s/clip server-side limit already bounds it.
    prepare: async (f) => ({ blob: f, previewUrl: URL.createObjectURL(f), mediaType: f.type, name: f.name }),
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
    .map((f) => kind.prepare(f).catch(() => null)));  /* unreadable file -- skip */
  if (!ctx.alive) {
    for (const r of reads) {
      if (r?.previewUrl) URL.revokeObjectURL(r.previewUrl);
    }
    return;
  }
  // send() and clearPendingAttachments REPLACE these arrays rather than
  // emptying them, so a send (or a conversation switch) during the read leaves
  // us holding an orphan: pushing into it would render nothing and lose the
  // file without a word. Drag-and-drop makes multi-megabyte reads routine, so
  // this window is no longer theoretical.
  const pending = ctx.state[kind.stateKey];
  if (pending !== pendingAtStart) {
    for (const r of reads) {
      if (r?.previewUrl) URL.revokeObjectURL(r.previewUrl);
    }
    showStatus(ctx, `Attachment discarded -- the composer was cleared while it was still loading.`, true);
    return;
  }
  const usable = reads.filter(Boolean);
  if (!usable.length) return;
  const room = Math.max(kind.max - pending.length, 0);
  const excess = usable.slice(room);
  for (const ex of excess) {
    if (ex?.previewUrl) URL.revokeObjectURL(ex.previewUrl);
  }
  const staged = usable.slice(0, room);
  pending.push(...staged);
  renderAttachStrip(ctx);
  // Disclosure, not a prompt: the resolution cap is a cost the user pays
  // silently otherwise, and a confirm on every photo would only train
  // click-through. Says it once per staging batch, and only when it happened.
  const shrunk = staged.filter((e) => e.resized).length;
  if (shrunk) {
    showStatus(ctx, `${shrunk === 1 ? 'Image' : `${shrunk} images`} scaled down to ${MAX_EDGE_PX}px for upload.`);
  }
  // aria-live: chat__status is role="status" -- this announces the cap to
  // screen readers, not just the sighted strip.
  if (usable.length > room) {
    showStatus(ctx, `${kind.max} ${kind.label} max -- ${usable.length - room} not attached.`);
  }
}

function clearPendingAttachments(ctx) {
  for (const img of ctx.state.pendingImages ?? []) {
    if (img.previewUrl) URL.revokeObjectURL(img.previewUrl);
  }
  for (const clip of ctx.state.pendingAudio ?? []) {
    if (clip.previewUrl) URL.revokeObjectURL(clip.previewUrl);
  }
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
      const [removed] = s.pendingImages.splice(i, 1);
      if (removed?.previewUrl) URL.revokeObjectURL(removed.previewUrl);
      renderAttachStrip(ctx);
    });
    return createEl('div', { class: 'attach-thumb' }, [
      createEl('img', { src: img.previewUrl, alt: '', decoding: 'async' }),
      remove,
    ]);
  });
  const audioChips = s.pendingAudio.map((clip, i) => {
    const label = `Remove audio ${clip.name || i + 1}`;
    const remove = createEl('button', { class: 'attach-thumb__remove', title: label, 'aria-label': label }, ['×']);
    remove.addEventListener('click', () => {
      const [removed] = s.pendingAudio.splice(i, 1);
      if (removed?.previewUrl) URL.revokeObjectURL(removed.previewUrl);
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

// One place that knows how a media block becomes a renderable URL. RENDER
// only since Phase 2: the wire is built server-side (_wire_content in
// conversation_generate_api.py -- data URLs for images, RAW base64 for
// audio). Since schema v7 stored blocks usually carry url sources
// (/v1/conversations/{id}/media/{id}); the base64 branch survives for
// just-staged previews and any pre-v7-shaped payload.
function blockSourceUrl(b) {
  return b.source.type === 'url'
    ? b.source.url
    : `data:${b.source.media_type};base64,${b.source.data}`;
}

// Base64 is minted HERE and nowhere else. It used to exist three times over at
// send -- the data URL retained since staging, the slice taken to build these
// blocks, and the JSON string of the whole body -- for every attachment, at
// full camera-roll resolution. Now the staged bytes are already capped and the
// base64 lives only as long as the request.
async function buildContentBlocks(text, images, audio) {
  // In PARALLEL: eight staged photos read one after another put eight full
  // FileReader round trips in series at exactly the moment the user is waiting
  // on the send. Promise.all costs the slowest read instead of their sum;
  // block order within each kind is all that has to hold, and it does.
  const encode = (entry, type) => blobToBase64(entry.blob).then((data) => ({
    type, source: { type: 'base64', media_type: entry.mediaType, data },
  }));
  const blocks = await Promise.all([
    ...images.map((img) => encode(img, 'image')),
    ...audio.map((clip) => encode(clip, 'audio')),
  ]);
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

  const title = (text || (audio.length ? 'Audio message' : 'Image message')).slice(0, 50);
  s.textarea.value = '';
  autoGrow(s.textarea);
  s.pendingImages = [];
  s.pendingAudio = [];
  renderAttachStrip(ctx);
  showStatus(ctx, '');

  try {
    // Inside the try: encoding is real work that can fail, and the restore
    // path below is exactly what a failure here needs.
    const content = (images.length || audio.length)
      ? await buildContentBlocks(text, images, audio) : text;
    if (!ctx.alive) {
      for (const img of images) { if (img.previewUrl) URL.revokeObjectURL(img.previewUrl); }
      for (const clip of audio) { if (clip.previewUrl) URL.revokeObjectURL(clip.previewUrl); }
      return;
    }
    if (!s.activeId) {
      // a prompt typed (or preset applied) before the first send
      const preset = s.presetBar.presetForNewDoc();
      const sentPrompt = s.systemPrompt || preset?.system_prompt;
      const appliedPresetId = s.appliedPresetId || preset?.id;
      const promptBeforeCreate = s.systemPrompt;
      const conv = await api.createConversation({
        title,
        model_id: s.modelSelect.value || undefined,
        system_prompt: sentPrompt || undefined,
        params: (!s.appliedPresetId && preset?.params) ? { ...(preset.params ?? {}) } : snapshotSettings(),
        applied_preset_id: appliedPresetId || undefined,
      });
      if (!ctx.alive) return;
      writeDraftPrompt(null); // adopted by the conversation this send created
      s.appliedPresetId = conv.applied_preset_id ?? null;
      s.conversations.unshift(conv);
      s.activeId = conv.id;
      s.messages = [];
      if (s.systemPrompt !== promptBeforeCreate) {
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
    for (const img of images) { if (img.previewUrl) URL.revokeObjectURL(img.previewUrl); }
    for (const clip of audio) { if (clip.previewUrl) URL.revokeObjectURL(clip.previewUrl); }
    if (!ctx.alive || s.activeId !== convId) return; // saved to its conv; reselect re-fetches it
    // Idempotent by id: a resume merge that landed while the POST was in
    // flight may already hold this row (the server's list saw the bump).
    if (!s.messages.some((m) => m.id === msg.id)) s.messages.push(msg);
    renderMessages(ctx);
    scrollMessages(ctx, true);
    startStream(ctx, { mode: 'append' });
  } catch (err) {
    if (!ctx.alive) {
      for (const img of images) { if (img.previewUrl) URL.revokeObjectURL(img.previewUrl); }
      for (const clip of audio) { if (clip.previewUrl) URL.revokeObjectURL(clip.previewUrl); }
      return;
    }
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
  const md = new MarkdownStream(contentEl);
  // A continuation's prior text is seeded through the SAME renderer, so its
  // blocks are already classified and committed before the first delta -- the
  // anchor's content is never re-parsed again for the life of the stream.
  if (baseContent) md.render(baseContent);
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
    md,
    // What the thinking box already holds, so a paint appends the delta
    // instead of rewriting the whole string (buildThinkingEl seeded it).
    thinkingWritten: (continueMsg?.thinking ?? '').length,
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
  s.remoteGenerating = false;  // the local stream owns the button now
  s.sendBtn.textContent = 'Stop';
  s.sendBtn.title = 'Stop this run and keep what has been generated so far';
  beforeUnloadGuard.enable();

  // The dead air before the first token is the one moment the user cannot
  // tell a slow model from a hung one -- and on a cold target it is a
  // multi-GB load, the single longest wait in the app, previously shown as
  // an empty bubble. Name it. Cleared by the first delta below (thinking
  // counts: a reasoning model emits thinking first, and leaving "Loading…"
  // up while thinking streams would be a lie).
  const modelId = s.modelSelect.value;
  const cold = isCold(ctx, modelId);
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
  // displayWireFields() is spread in BESIDE overrides, never into it: overrides
  // is the sampler bag the server layers over the document's stored params, and
  // a display pref landing in there would be persisted as generation state.
  const launch = () => streamGenerate(stream.targetConvId,
    { mode, message_id: messageId, overrides, ...displayWireFields(DISPLAY_PREFS) }, {
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
  // Measured before anything below mutates the DOM -- see followingTail.
  const following = followingTail(ctx);
  if (stream.contentDirty) {
    stream.contentDirty = false;
    // Prefix + new text are still ONE markdown document -- constructs spanning
    // the seam were the reason the old painter re-rendered the whole thing
    // every frame. MarkdownStream keeps that guarantee by choosing a cut no
    // construct can span, instead of by never cutting.
    stream.md.render(stream.baseContent + stream.content);
  }
  if (stream.thinkingDirty) {
    stream.thinkingDirty = false;
    stream.els.thinkingEl.hidden = false;
    // Same seam rule as content: a continuation's seeded prior thinking must
    // survive the first new thinking delta, not vanish until finishStream.
    stream.thinkingWritten = appendPlainText(stream.els.thinkingBody,
      stream.baseThinking + stream.thinking, stream.thinkingWritten);
  }
  if (following) pinToTail(ctx);
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
      // ONLY here does userStopped belong. It marks a LOCAL abort that really
      // does mean the run is over, so the end may say "Stopped" -- and this is
      // the one outcome of the four where nothing else can say it (a 200 is
      // covered by end_reason 'aborted').
      //
      // It used to be set unconditionally at the top of this function, which
      // re-opened the exact regression the 2026-08-13 review fixed from the
      // other side: on a 404 WITH events the server had already finished, the
      // stream completes with the whole answer, and the flag made the UI
      // report "Stopped -- partial response saved." over a complete response.
      // A null status (the DELETE never reached the server) lied the same way
      // about a run that was still going.
      stream.userStopped = true;
      stream.controller.abort();
      return;
    }
    // The DELETE never arrived (offline, 5xx). The run is still going, the
    // stream keeps painting, and the button still reads Stop -- so saying
    // nothing invites a second press that does exactly as little. stopRemote
    // already models this; the two make the same promise about the same
    // endpoint and must not diverge.
    if (status === null && ctx.state.stream === stream) {
      showStatus(ctx, 'Could not reach the server to stop it — still generating.', true);
    }
  });
}

// Conversation switch, model switch, delete: kill the fetch. NOT teardown --
// page.js's unmount() aborts ctx.signal and linkedController chains the
// stream's controller off it, so leaving the page never comes through here.
// `teardown` is the WEAKEST claim rather than a dead one: it is the default
// that makes a reason-less call safe, and deleteConversation now passes it
// deliberately for the case where the stop never reached the server and there
// is nothing to assert. This ends our SUBSCRIPTION, not the run -- the server detaches the generation and it
// finishes and commits the WHOLE answer (conversation_generate_api._Run), so
// the next select of that conversation gets the complete reply, not a
// truncated one. Only an explicit Stop (DELETE .../generate -> abort_event)
// keeps just the partial. Do not re-describe this as "persists the partial";
// it did work that way once, and the stale comment made walking away look
// lossy in the user guide until the server was read.
// The reasons, as a frozen set rather than four bare literals across five
// call sites and two consumers -- a typo'd reason silently reads as "not that
// one" at every consumer, which is the quietest possible failure.
const ABANDON = Object.freeze({
  TEARDOWN: 'teardown',              // the weakest claim: 'we do not know' (see below)
  CONVERSATION: 'switch-conversation',
  MODEL: 'switch-model',
  DELETE: 'delete',                  // genuinely ended server-side
});

// How STRONG a claim each reason makes about the run's fate. Only DELETE says
// the run really ended server-side; the two switches say the opposite (it
// detaches and finishes); TEARDOWN says nothing, being the default nothing
// sets. Ranked because abortStream can be called twice for one stream and was
// last-writer-wins, so the second call SILENTLY WEAKENED the first: delete
// aborts the stream and then selects the next conversation, which aborts it
// again as `switch-conversation`. Nothing reads DELETE today, which is exactly
// why the downgrade was invisible -- a reason is a claim, and a claim that can
// be quietly overwritten by a later, weaker one is not one the next consumer
// can trust.
const ABANDON_RANK = Object.freeze({
  [ABANDON.TEARDOWN]: 0,
  [ABANDON.CONVERSATION]: 1,
  [ABANDON.MODEL]: 1,
  [ABANDON.DELETE]: 2,
});

// Is `id` a model the first message would have to load? Silent when residency
// is unknown -- a guess is worse than nothing (same rule as the dots). One
// derivation: it had grown copies in startStream, commitModelSwitch and
// abandonNote.
function isCold(ctx, id) {
  const s = ctx.state;
  return Boolean(s.loadedKnown && id && !s.loadedIds.has(id));
}

function abortStream(ctx, reason = ABANDON.TEARDOWN) {
  const stream = ctx.state.stream;
  if (!stream) return;
  // WHY we let go decides what the end is allowed to claim. "Stopped" is a
  // statement about the SERVER, and only two of these reasons make it true.
  // The STRONGEST reason wins, not the last one -- see ABANDON_RANK. Equal
  // ranks still take the later value: two switches are both "abandoned", and
  // the newer one describes where the user actually is.
  const held = ABANDON_RANK[stream.abandonReason] ?? -1;
  if ((ABANDON_RANK[reason] ?? 0) >= held) stream.abandonReason = reason;
  stream.controller.abort();
}

// What to say when we dropped the subscription and the run did NOT stop. This
// is the app's one disclosure of its best behaviour: the generation detaches,
// finishes, and commits the whole answer (conversation_generate_api._Run), so
// walking away costs nothing -- and nobody could discover that, because the
// only place it was ever mentioned was a status line you saw if you happened
// to come back mid-run.
function abandonNote(ctx, stream) {
  const s = ctx.state;
  if (stream.abandonReason === ABANDON.MODEL) {
    // committedModelId, NOT modelSelect.value: the select can be showing an
    // UNCONFIRMED target (showSwitchWarning leaves it there pending Cancel /
    // Switch anyway), and this sentence promises what the next message uses.
    const next = s.committedModelId ?? s.modelSelect.value;
    const cold = isCold(ctx, next);
    return `${MODEL_SWITCH_PREFIX} on the previous model and saves to `
      + `this conversation. Your next message ${cold ? 'loads' : 'uses'} ${next}.`
      // Carried from commitModelSwitch: a capability the switch takes away is
      // not derivable here (the `from` model is long gone) and nothing else
      // will ever say it.
      + (stream.pendingNote ? ` ${stream.pendingNote}` : '');
  }
  return `${GENERATING_PREFIX} — the reply will be saved to this conversation.`;
}

function releaseStream(ctx, stream) {
  const s = ctx.state;
  // No-op after normal completion; drops the linkedController chain listener.
  stream.controller.abort();
  if (s.stream !== stream) return;
  s.stream = null;
  beforeUnloadGuard.disable();
  if (!ctx.alive) return;
  // Restore from the CONVERSATION's state, not from a literal. This stream is
  // over, but the document on screen may be one the server is still generating
  // for -- switch away from a live run into another live run and the abandoned
  // stream's unwind lands last, so a hardcoded 'Send' would erase the only
  // control for the run that is still going. It is the same fact
  // setRemoteGenerating already holds, so read it rather than re-derive it.
  setSendButton(s, s.remoteGenerating);
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

  // `serverState` is the AUTHORITATIVE answer to "is the run still going",
  // and it is the whole reason the branch below needs no proxies. It was
  // already being fetched here and thrown away.
  let serverState = null;
  if (saved?.messages?.length) {
    adoptSavedRows(ctx, saved.messages);
    renderMessages(ctx);
  } else {
    serverState = await resyncMessages(ctx, stream.targetConvId);
    if (!ctx.alive || s.activeId !== stream.targetConvId) return;
  }
  scrollMessages(ctx);

  const endReason = saved?.end_reason;
  // WHAT HAPPENED IS THE SERVER'S ANSWER, NOT OURS.
  //
  // `heylook_saved` is always the LAST event (spec §4), so `saved` present
  // means the run ENDED and `end_reason` says how -- whatever our fetch did.
  // `saved` absent means the transport died, and then `serverState` (the GET
  // above) is the only thing that knows whether a run is still going.
  //
  // This used to be a three-input boolean derived from client-side proxies
  // (`aborted`, `stream.userStopped`, `stream.abandonReason`), and every
  // lifecycle bug in v1.79.26-.29 was a symptom of that: each new abort path
  // had to be hand-classified, and the classification was silently wrong the
  // moment it drifted. Two independent reviews found bugs in it inside a
  // week. The proxies survive only where the server cannot answer.
  //
  // Order is failure-first: a run that failed and was then stopped is still a
  // failure, and "Stopped." used to paint over the red line in normal colour.
  const ended = Boolean(saved) || serverState === 'idle';
  if (endReason === 'error' || stream.inBandError) {
    // the in-band error status line stands; a partial (if any) is saved
  } else if (endReason === 'aborted' || (ended && stream.userStopped)) {
    // userStopped is only reachable via stopStream's 404-with-no-events
    // branch, where there is no server answer to have -- see stopStream.
    showStatus(ctx, content ? 'Stopped -- partial response saved.' : 'Stopped.');
  } else if (serverState === 'generating') {
    // The server says a run is still going and it is not ours any more.
    if (!switchWarningOnScreen(ctx)) showStatus(ctx, abandonNote(ctx, stream));
  } else if (aborted && !saved && serverState !== 'idle') {
    // No answer either way (`unknown`): we let go of a run whose end we never
    // saw. Say the reply may still be coming rather than claim it stopped --
    // the run detaches and commits, so the optimistic reading is the true one
    // far more often than not.
    if (!switchWarningOnScreen(ctx)) showStatus(ctx, abandonNote(ctx, stream));
  } else if (usage || saved) {
    const timing = saved?.timing;
    const parts = [`${usage?.output_tokens ?? '?'} tokens`];
    if (timing?.peak_memory_gb != null) parts.push(`${timing.peak_memory_gb.toFixed(2)} GB peak`);
    if (timing?.kv_cache_bytes != null) parts.push(`${formatBytes(timing.kv_cache_bytes)} KV`);
    if (timing?.draft_acceptance != null) parts.push(`draft ${(timing.draft_acceptance * 100).toFixed(0)}%`);
    showStatus(ctx, parts.join(' · '));
  }

  // A stream that ended WITHOUT its heylook_saved is not success (spec §4:
  // that event is always last) -- the transport died. Two different things
  // that used to look identical: the server FINISHED and its commit raced
  // our resync, or the server is STILL GENERATING, because a run now
  // outlives the response that started it. Only the first has anything to
  // recover; announcing "recovered" during the second states that a partial
  // answer is the whole one. resyncMessages reports which it is.
  // A new stream supersedes the remaining retries -- its own saga end
  // re-adopts everything anyway.
  if (!saved && !aborted && (content || thinking) && !stream.inBandError) {
    showStatus(ctx, 'The stream ended without a save confirmation — recovering…', true);
    const retry = (delay, attemptsLeft) => setTimeout(async () => {
      if (!ctx.alive || s.activeId !== stream.targetConvId || s.stream) return;
      const state = await resyncMessages(ctx, stream.targetConvId);
      if (!ctx.alive || s.activeId !== stream.targetConvId) return;
      // Still working: nothing to recover, and no backoff could outlast it.
      // setRemoteGenerating has already put the honest line on screen and
      // turned the button into Stop.
      if (state === 'generating') return;
      // 'unknown' is NOT a recovery: the fetch never reached the server, so
      // keep retrying and never fall through to the success line below.
      if (state === 'unknown') {
        if (attemptsLeft > 0) retry(delay * 2.5, attemptsLeft - 1);
        return;
      }
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
