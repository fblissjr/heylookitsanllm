// Load settings for the NEXT load of a model (plan W1): one control per field
// the server tags `load_setting` in /v1/admin/model-options, generated from
// that list and never listed here, so a field added server-side appears with
// no frontend change. Shown beside the model select, and only for a provider
// that has load settings (gguf today; MLX has none).
//
// `ctx_size` keeps its own control (context-select.js: a power-of-two ladder
// plus Custom…). Every other field gets a native <select> whose first option
// is Auto, labelled with what auto resolved to when the running process said
// ("flash attn: auto (on)"), then the field's own enum values. Choosing Auto
// is the reset: it sends null and the server drops the stored key, so a
// default is never written (plan principle 1). The title carries the
// setting's reason and provenance from the engine contract, which is where
// "where did this value come from" is answered.
//
// Inline in the bar, not behind a toggle: the context select was reported
// missing on a phone once (2026-09-04, pinned by the render suite), and a
// panel that has to be opened first would bring that back.
//
// createLoadPanel({ currentModelId, adminRow, loadFields, onChange }) -> {
//   element          the wrapper to mount (class chat__load)
//   refresh()        re-read the current model's row and rebuild what moved
//   changed()        any choice differs from what the model has STORED
//   choicesToSend()  {field: value|null} to send with a reload, or null when
//                    the current model has no load settings (plain load)
//   describe()       a short phrase for the status line ("64K context, flash
//                    attn off"), or '' when everything is Auto
//   setEnabled(on)
// }
//   currentModelId()   the id the page's model select shows
//   adminRow(id)       that model's /v1/admin/models row, or undefined
//   loadFields(prov)   that provider's model-options fields with
//                      load_setting true ([] while the schema is unknown)
//   onChange()         fired on a user pick

import { createEl, formatTokens } from './utils.js';
import { createContextSelect } from './context-select.js';

const CTX = 'ctx_size';

const labelFor = (name) => name.replace(/_/g, ' ');

export function createLoadPanel({ currentModelId, adminRow, loadFields, onChange }) {
  const ctxSelect = createContextSelect({ currentModelId, adminRow, onChange });
  const extras = createEl('span', { class: 'chat__load-extras' });
  const element = createEl('span', { class: 'chat__load' }, [ctxSelect.element, extras]);
  // name -> <select>, for the fields other than ctx_size on the current model
  let selects = new Map();

  const fieldsNow = () => {
    const row = adminRow(currentModelId());
    return row ? loadFields(row.provider) : [];
  };

  function refresh() {
    ctxSelect.refresh();
    const row = adminRow(currentModelId());
    const fields = row ? loadFields(row.provider).filter((f) => f.name !== CTX && f.enum) : [];
    const settings = row?.engine?.settings ?? {};
    // Rebuild only when the model or its facts moved: an untouched rebuild
    // would throw away a pick the user just made.
    const sig = JSON.stringify([currentModelId(), fields.map((f) => [
      f.name, row?.config?.[f.name] ?? null, settings[f.name]?.value ?? null])]);
    if (extras.dataset.sig === sig) return;
    extras.dataset.sig = sig;
    selects = new Map();
    extras.replaceChildren(...fields.map((f) => {
      const setting = settings[f.name];
      const label = labelFor(f.name);
      const resolved = setting?.auto;
      const auto = resolved && resolved !== 'auto' ? `${label}: auto (${resolved})` : `${label}: auto`;
      const select = createEl('select', {
        class: 'chat__load-select', 'data-field': f.name,
        'aria-label': label,
        title: [f.description, setting && `Now: ${setting.value ?? 'auto'} (${setting.reason})`]
          .filter(Boolean).join('\n\n'),
      }, [
        createEl('option', { value: '' }, [auto]),
        ...f.enum.map((v) => createEl('option', { value: String(v) }, [`${label}: ${v}`])),
      ]);
      select.value = row?.config?.[f.name] ?? '';
      select.addEventListener('change', () => onChange?.());
      selects.set(f.name, select);
      return select;
    }));
  }

  function changed() {
    if (ctxSelect.changed()) return true;
    const row = adminRow(currentModelId());
    return [...selects].some(([name, sel]) => sel.value !== String(row?.config?.[name] ?? ''));
  }

  function choicesToSend() {
    const out = {};
    // The context select gates itself on the provider, so its pick is sent
    // even while the option schema is unknown (not fetched yet, or the fetch
    // failed); otherwise a pick shown on screen would load as a plain load.
    const n = ctxSelect.choiceToSend();
    if (n !== null) out[CTX] = n || null;  // the select's 0 is Auto
    for (const f of fieldsNow()) {
      if (f.name !== CTX && selects.has(f.name)) out[f.name] = selects.get(f.name).value || null;
    }
    return Object.keys(out).length ? out : null;
  }

  function describe(choices) {
    if (!choices) return '';
    return Object.entries(choices).filter(([, v]) => v != null)
      .map(([k, v]) => (k === CTX ? `${formatTokens(v)} context` : `${labelFor(k)} ${v}`))
      .join(', ');
  }

  function setEnabled(on) {
    ctxSelect.setEnabled(on);
    for (const sel of selects.values()) sel.disabled = !on;
  }

  return { element, refresh, changed, choicesToSend, describe, setEnabled };
}
