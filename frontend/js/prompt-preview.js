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

// The markers to highlight are the MODEL's own added tokens that occur in the
// prompt (`markers` on the preview response, read from its tokenizer files),
// longest first so `</think>` is not read as `<` + `/think>`. The text itself
// is the engine's render, shown verbatim; a marker list the server could not
// read leaves the text plain, never guessed at.
export function highlightSpecials(text, markers = []) {
  const frag = document.createDocumentFragment();
  if (!markers.length) {
    frag.append(document.createTextNode(text));
    return frag;
  }
  const escape = (m) => m.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(markers.map(escape).join('|'), 'g');
  let last = 0;
  for (const m of text.matchAll(re)) {
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
  pre.append(highlightSpecials(body.prompt, body.markers ?? []));
  const children = [
    createEl('div', { class: 'prompt-preview__head' }, [
      createEl('span', { class: 'prompt-preview__title' }, ['What the model will see']),
      createEl('span', { class: 'muted small' }, [
        `${body.model_id} · ${body.provider} · ${body.char_count.toLocaleString()} chars · ${what}`,
      ]),
      closeBtn,
    ]),
  ];
  // Media that IS being sent but cannot appear in this render (the MLX vision
  // path has no text-only render). Said out loud, and said as "not shown
  // here", never as a drop: the picture reaches the model. Without this line
  // the panel is a prompt with no image in it, under a heading that promises
  // it is what the model will see -- which reads as the image having been
  // lost, and is the opposite of the truth.
  const un = body.unrendered_media ?? {};
  const missing = [
    un.images ? `${un.images} image${un.images === 1 ? '' : 's'}` : null,
    un.audio ? `${un.audio} audio clip${un.audio === 1 ? '' : 's'}` : null,
  ].filter(Boolean);
  if (missing.length) {
    children.push(createEl('div', { class: 'prompt-preview__note muted small', role: 'note' }, [
      `${missing.join(' and ')} will be sent but cannot be shown here — `
      + `${body.provider === 'mlx' ? 'MLX renders the text template only' : 'this engine has no text render for media'}.`,
    ]));
  }
  // Media the model will NOT receive is a different statement, and stays one.
  const dropped = body.dropped_media ?? {};
  const lost = [
    dropped.images ? `${dropped.images} image${dropped.images === 1 ? '' : 's'}` : null,
    dropped.audio ? `${dropped.audio} audio clip${dropped.audio === 1 ? '' : 's'}` : null,
  ].filter(Boolean);
  if (lost.length) {
    children.push(createEl('div', { class: 'prompt-preview__note muted small', role: 'note' }, [
      `${lost.join(' and ')} not sent — this model cannot read them.`,
    ]));
  }
  children.push(pre);
  host.replaceChildren(...children);
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
