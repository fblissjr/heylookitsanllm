// Prompt preview: the exact string the model will be fed (v1.79.62),
// extracted from chat.js. The body is a POST /v1/conversations/{id}/prompt
// response -- the engine's own render (llama-server /apply-template, or the
// MLX tokenizer's chat template through the same builder generation uses),
// shown VERBATIM; nothing here interprets the text. Chat paints it in two
// places, the composer's eye button (what the next Send would send) and the
// editor's Preview prompt (what Save & Continue / Save & Regenerate would
// send), through these same two functions so the two panels cannot drift.
//
//   paintPromptPreview(host, body, onClose)        render a response into host
//   paintPromptPreviewError(host, message, onClose) render a failure into host
//   host is a .prompt-preview container the page owns; both unhide it, and
//   onClose is the page's own hide-and-clear.

import { createEl } from './utils.js';

// Special-token spellings across the families this server serves, for
// HIGHLIGHTING only -- the text is the engine's own render, shown verbatim;
// the marks just make <|im_start|> stand out from prose. Unmatched markers
// still show as text, so a family this misses loses nothing but colour.
const SPECIAL_TOKEN_RE = new RegExp(
  '(<\\|[^<>\\n]{1,48}\\|>|<\\|[^<>\\n]{1,48}>|<[^<>\\n]{1,48}\\|>|</?think>'
  + '|<start_of_turn>|<end_of_turn>|<bos>|<eos>|<s>|</s>|\\[INST\\]|\\[/INST\\]|\\[/?THINK\\])', 'g');

export function highlightSpecials(text) {
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
export function paintPromptPreview(host, body, onClose) {
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

export function paintPromptPreviewError(host, message, onClose) {
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
