// Context size for the NEXT load of a gguf model (v1.79.61), extracted from
// chat.js. A native <select> of power-of-two steps, not a slider: the range
// is logarithmic (4k to 1M) and a linear thumb cannot land on 32k; a select
// is also the one control that is already right on a phone with no
// widgetry. "Auto" is llama-server's own answer (sized from the model,
// fitted to memory) and is the default -- the stored `ctx_size` preselects
// when one is set. Hidden for MLX, which has no fixed context allocation.
// Choosing a different value shows Load/Reload on the page; the value is
// sent WITH the load and persisted server-side, so the models page shows
// the same number afterwards.
//
// The steps are a convenience, not the range: "Custom…" reveals a number
// input for any value the server will accept, because the useful size is a
// property of the machine and the moment (what else is resident, how long the
// prompts are) and no fixed ladder can name it. A committed custom value
// becomes a real option, so it survives a rebuild and preselects afterwards --
// ctxStepsFor already carried the off-grid stored value for exactly this.
//
// createContextSelect({ currentModelId, adminRow, onChange }) -> {
//   element        the wrapper to mount (class chat__ctx), holding the
//                  <select class="chat__ctx-select"> and the custom input
//   refresh()      re-read the current model's admin row and rebuild the
//                  options if its facts moved (see the signature rule below)
//   changed()      the chosen value differs from the model's STORED ctx_size
//   choiceToSend() the size to send with a load: N, 0 for Auto, null when
//                  the control does not apply to the current model
//   setEnabled(on) enable/disable both inputs. The wrapper is a span, so the
//                  page cannot reach for `.disabled` on the element itself.
// }
//   currentModelId()  the id the page's model select shows
//   adminRow(id)      that model's /v1/admin/models row (provider gates the
//                     control, config carries the stored ctx_size,
//                     context_length is the ceiling, context_running is what
//                     the resident process actually got), or undefined
//   onChange()        fired on a user pick, so the page can re-decide whether
//                     its Load/Reload button shows

import { createEl, formatTokens } from './utils.js';

const CTX_MIN = 4096;
const CTX_FALLBACK_MAX = 262144; // ceiling when the header did not say

// The floor a CUSTOM value may reach -- GGUFModelConfig.ctx_size is `ge=512`,
// so anything smaller is a 422 from the server rather than a small context.
// Deliberately below CTX_MIN: the ladder starts at 4k because that is where
// the useful sizes begin, not because 2k is illegal.
const CTX_ABS_MIN = 512;

// Sentinel option value. Not a number, so it can never be mistaken for a size.
const CUSTOM = 'custom';

const IDLE_TITLE = 'Context size for the next load';

// Power-of-two steps from 4k up to the model's training context (or the
// fallback), plus the ceiling itself when it is not a power of two (Qwen3's
// 40960) and the stored value when it is off-grid -- the select must be able
// to SHOW what is stored, or Auto would be preselected over a real value.
export function ctxStepsFor(row) {
  const max = row?.context_length || CTX_FALLBACK_MAX;
  const steps = [];
  for (let n = CTX_MIN; n <= max; n *= 2) steps.push(n);
  if (!steps.includes(max)) steps.push(max);
  const stored = row?.config?.ctx_size;
  if (stored && !steps.includes(stored)) steps.push(stored);
  return steps.sort((a, b) => a - b);
}

export function createContextSelect({ currentModelId, adminRow, onChange }) {
  const select = createEl('select', {
    class: 'chat__ctx-select',
    title: IDLE_TITLE,
    'aria-label': 'Context size',
  });
  // type=number so a phone gets the numeric keypad. Kept OUT of the tab order
  // and out of the accessibility tree while hidden by `hidden` itself, which
  // is why it is toggled with .hidden and never with style.display.
  const custom = createEl('input', {
    class: 'chat__ctx-custom', hidden: true,
    type: 'number', step: '1', inputMode: 'numeric',
    title: 'Custom context size, in tokens',
    'aria-label': 'Custom context size in tokens',
  });
  const element = createEl('span', { class: 'chat__ctx', hidden: true }, [select, custom]);

  // The last value committed through the custom box but not yet stored
  // server-side. Re-offered on rebuild: ctxStepsFor can only know the STORED
  // value, and silently dropping a number the user typed is worse than
  // dropping a dropdown pick they can make again in one click.
  let pending = null;
  // What the select showed before "Custom…" was chosen, so Escape and an
  // unparseable entry both have somewhere to go back to.
  let beforeCustom = '';

  function ceilingFor(row) {
    return row?.context_length || CTX_FALLBACK_MAX;
  }

  // Put `n` in the list as a real option (sorted, before "Custom…") and
  // select it. Idempotent: an existing option is reused rather than doubled.
  function selectValue(n, row) {
    const value = String(n);
    if (![...select.options].some((o) => o.value === value)) {
      const tag = n === row?.context_length ? ' (max)' : '';
      const option = createEl('option', { value }, [`${formatTokens(n)}${tag}`]);
      const after = [...select.options].find(
        (o) => o.value !== CUSTOM && o.value !== '' && Number(o.value) > n);
      select.insertBefore(option, after ?? select.querySelector(`option[value="${CUSTOM}"]`));
    }
    select.value = value;
  }

  function closeCustom() {
    custom.hidden = true;
    custom.value = '';
  }

  function commitCustom() {
    const row = adminRow(currentModelId());
    const raw = Number.parseInt(custom.value, 10);
    if (!Number.isFinite(raw)) {
      // Nothing usable typed: put the previous choice back rather than
      // leaving the select parked on the sentinel, which would read as a
      // selection and make `changed()` answer about a value that is not one.
      select.value = beforeCustom;
      closeCustom();
      return;
    }
    // CLAMP rather than refuse. The number that ends up in the select is the
    // number that will be sent, so the correction is visible at the moment it
    // happens -- a silent refusal would leave the reader believing they had
    // asked for something they had not.
    const n = Math.min(Math.max(raw, CTX_ABS_MIN), ceilingFor(row));
    pending = n;
    selectValue(n, row);
    closeCustom();
    onChange?.();
  }

  select.addEventListener('change', () => {
    if (select.value !== CUSTOM) {
      closeCustom();
      onChange?.();
      return;
    }
    // Opening the box decides nothing yet, so no onChange here: firing it
    // would show Load/Reload for a value that does not exist.
    const row = adminRow(currentModelId());
    custom.min = String(CTX_ABS_MIN);
    custom.max = String(ceilingFor(row));
    custom.value = String(
      row?.config?.ctx_size || (row?.loaded ? row.context_running : null) || CTX_MIN);
    custom.hidden = false;
    custom.focus();
    custom.select();
  });

  // `change` covers blur and the phone keyboard's Done; Enter is bound too
  // because a bare Enter in a lone number input submits nothing here and
  // would otherwise feel dead. Escape abandons.
  custom.addEventListener('change', commitCustom);
  custom.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); commitCustom(); }
    else if (e.key === 'Escape') { select.value = beforeCustom; closeCustom(); }
  });

  function refresh() {
    const id = currentModelId();
    const row = adminRow(id);
    const gguf = row?.provider === 'gguf';
    element.hidden = !gguf;
    if (!gguf) {
      // Forget the last gguf model's facts: a hidden control must not come
      // back describing a different model's running context.
      delete element.dataset.sig;
      select.title = IDLE_TITLE;
      pending = null;
      closeCustom();
      return;
    }
    const stored = row.config?.ctx_size ?? '';
    const running = row.loaded ? row.context_running : null;
    // Rebuild only when the model (or its facts) moved; an untouched rebuild
    // would throw away a choice the user just made.
    const sig = `${id}|${stored}|${row.context_length ?? ''}|${running ?? ''}`;
    if (element.dataset.sig === sig) return;
    const modelChanged = (element.dataset.sig ?? '').split('|')[0] !== id;
    if (modelChanged) pending = null;  // another model's number means nothing here
    element.dataset.sig = sig;
    const autoLabel = running && !stored ? `Auto (${formatTokens(running)})` : 'Auto';
    const options = [createEl('option', { value: '' }, [autoLabel])];
    const steps = ctxStepsFor(row);
    if (pending && !steps.includes(pending)) steps.push(pending);
    for (const n of steps.sort((a, b) => a - b)) {
      const tag = n === row.context_length ? ' (max)' : '';
      options.push(createEl('option', { value: String(n) }, [`${formatTokens(n)}${tag}`]));
    }
    options.push(createEl('option', { value: CUSTOM }, ['Custom…']));
    select.replaceChildren(...options);
    select.value = stored ? String(stored) : (pending ? String(pending) : '');
    closeCustom();
    select.title = running
      ? `${IDLE_TITLE} — running with ${formatTokens(running)} now`
      : IDLE_TITLE;
  }

  // The chosen value differs from what is STORED for the model -- the only
  // change that means anything, since the stored value is what a load uses.
  function changed() {
    const row = adminRow(currentModelId());
    if (row?.provider !== 'gguf' || element.hidden) return false;
    // The sentinel is an open editor, not a choice.
    if (select.value === CUSTOM) return false;
    const stored = row.config?.ctx_size ?? '';
    return String(stored) !== select.value;
  }

  // What to send with the load: the chosen size, 0 for Auto. Null for a model
  // the control does not apply to, so the plain load route is used.
  function choiceToSend() {
    const row = adminRow(currentModelId());
    if (row?.provider !== 'gguf') return null;
    // Same rule as changed(): mid-edit is not a choice, so send what is
    // stored and leave the model where it is.
    if (select.value === CUSTOM) return row.config?.ctx_size ?? 0;
    return select.value ? Number(select.value) : 0;
  }

  function setEnabled(on) {
    select.disabled = !on;
    custom.disabled = !on;
  }

  // Remembered on every user interaction with the select, so the sentinel is
  // never what Escape restores.
  select.addEventListener('mousedown', () => { beforeCustom = select.value; });
  select.addEventListener('keydown', () => { beforeCustom = select.value; });
  select.addEventListener('focus', () => { beforeCustom = select.value; });

  return { element, refresh, changed, choiceToSend, setEnabled };
}
