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
// createContextSelect({ currentModelId, adminRow, onChange }) -> {
//   element        the <select> to mount (class chat__ctx-select)
//   refresh()      re-read the current model's admin row and rebuild the
//                  options if its facts moved (see the signature rule below)
//   changed()      the chosen value differs from the model's STORED ctx_size
//   choiceToSend() the size to send with a load: N, 0 for Auto, null when
//                  the control does not apply to the current model
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
  const element = createEl('select', {
    class: 'chat__ctx-select', hidden: true,
    title: IDLE_TITLE,
    'aria-label': 'Context size',
  });
  element.addEventListener('change', () => onChange?.());

  function refresh() {
    const id = currentModelId();
    const row = adminRow(id);
    const gguf = row?.provider === 'gguf';
    element.hidden = !gguf;
    if (!gguf) {
      // Forget the last gguf model's facts: a hidden control must not come
      // back describing a different model's running context.
      delete element.dataset.sig;
      element.title = IDLE_TITLE;
      return;
    }
    const stored = row.config?.ctx_size ?? '';
    const running = row.loaded ? row.context_running : null;
    // Rebuild only when the model (or its facts) moved; an untouched rebuild
    // would throw away a choice the user just made.
    const sig = `${id}|${stored}|${row.context_length ?? ''}|${running ?? ''}`;
    if (element.dataset.sig === sig) return;
    element.dataset.sig = sig;
    const autoLabel = running && !stored ? `Auto (${formatTokens(running)})` : 'Auto';
    const options = [createEl('option', { value: '' }, [autoLabel])];
    for (const n of ctxStepsFor(row)) {
      const tag = n === row.context_length ? ' (max)' : '';
      options.push(createEl('option', { value: String(n) }, [`${formatTokens(n)}${tag}`]));
    }
    element.replaceChildren(...options);
    element.value = stored ? String(stored) : '';
    element.title = running
      ? `${IDLE_TITLE} — running with ${formatTokens(running)} now`
      : IDLE_TITLE;
  }

  // The chosen value differs from what is STORED for the model -- the only
  // change that means anything, since the stored value is what a load uses.
  function changed() {
    const row = adminRow(currentModelId());
    if (row?.provider !== 'gguf' || element.hidden) return false;
    const stored = row.config?.ctx_size ?? '';
    return String(stored) !== element.value;
  }

  // What to send with the load: the chosen size, 0 for Auto. Null for a model
  // the control does not apply to, so the plain load route is used.
  function choiceToSend() {
    const row = adminRow(currentModelId());
    if (row?.provider !== 'gguf') return null;
    return element.value ? Number(element.value) : 0;
  }

  return { element, refresh, changed, choiceToSend };
}
