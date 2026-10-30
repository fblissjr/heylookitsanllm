// Shared keyboard shortcuts across the workbench:
// - `?` (outside editable fields): toggles the keyboard shortcuts dialog
// - `Cmd/Ctrl + Enter`: triggers page-level submit (Chat Send / Notebook Generate)
// - `Escape`: dismisses page-level overlays or stops an active generation when
//   no modal dialog or settings drawer is open.

import { createEl } from './utils.js';

let dialogEl = null;

function isEditableTarget(el) {
  if (!el || !(el instanceof HTMLElement)) return false;
  if (el.isContentEditable) return true;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
}

function isModalOpen() {
  if (document.querySelector('dialog[open]')) return true;
  if (document.querySelector('.drawer--open')) return true;
  return false;
}

const SHORTCUT_ROWS = [
  { keys: ['⌘ / Ctrl', 'Enter'], desc: 'Send message (Chat) or Generate continuation (Notebook)' },
  { keys: ['Enter'], desc: 'Send message in Chat composer (Shift + Enter for newline)' },
  { keys: ['Esc'], desc: 'Stop active generation or close open drawer / panel / dialog' },
  { keys: ['?'], desc: 'Toggle this keyboard shortcuts sheet' },
];

export function mountHotkeysModal() {
  if (dialogEl) return dialogEl;

  const list = createEl('dl', { class: 'hotkeys-dialog__list' },
    SHORTCUT_ROWS.flatMap((row) => [
      createEl('dt', { class: 'hotkeys-dialog__keys' },
        row.keys.map((k) => createEl('kbd', { class: 'hotkeys-dialog__kbd' }, [k]))),
      createEl('dd', { class: 'hotkeys-dialog__desc' }, [row.desc]),
    ]));

  const form = createEl('form', { method: 'dialog', class: 'hotkeys-dialog__box' }, [
    createEl('header', { class: 'hotkeys-dialog__head' }, [
      createEl('h2', { id: 'hotkeys-dialog-title' }, ['Keyboard shortcuts']),
      createEl('button', {
        type: 'submit',
        class: 'btn btn--ghost btn--sm hotkeys-dialog__close',
        'aria-label': 'Close keyboard shortcuts',
      }, ['Close']),
    ]),
    list,
  ]);

  dialogEl = createEl('dialog', {
    class: 'hotkeys-dialog',
    closedby: 'any',
    'aria-labelledby': 'hotkeys-dialog-title',
  }, [form]);

  // Fallback for browsers without native <dialog closedby="any"> support (e.g. Safari).
  if (!('closedBy' in HTMLDialogElement.prototype)) {
    dialogEl.addEventListener('click', (event) => {
      if (event.target !== dialogEl) return;
      const rect = dialogEl.getBoundingClientRect();
      const isDialogContent = (
        rect.top <= event.clientY
        && event.clientY <= rect.top + rect.height
        && rect.left <= event.clientX
        && event.clientX <= rect.left + rect.width
      );
      if (isDialogContent) return;
      dialogEl.close();
    });
  }

  document.body.append(dialogEl);

  document.addEventListener('keydown', (e) => {
    if (e.defaultPrevented) return;
    if (e.key === '?' && !e.metaKey && !e.ctrlKey && !e.altKey) {
      if (isEditableTarget(document.activeElement)) return;
      e.preventDefault();
      if (dialogEl.open) dialogEl.close();
      else if (typeof dialogEl.showModal === 'function') dialogEl.showModal();
    }
  });

  window.addEventListener('hashchange', () => {
    if (dialogEl?.open) dialogEl.close();
  });

  return dialogEl;
}

export function registerPageHotkeys(ctx, { onSubmit, onEscape } = {}) {
  const onKeyDown = (e) => {
    if (!ctx.alive || e.defaultPrevented) return;
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && typeof onSubmit === 'function') {
      if (isModalOpen()) return;
      e.preventDefault();
      onSubmit(e);
      return;
    }
    if (e.key === 'Escape' && typeof onEscape === 'function') {
      if (isModalOpen()) return;
      if (onEscape(e) === true) {
        e.preventDefault();
      }
    }
  };
  document.addEventListener('keydown', onKeyDown);
  ctx.onTeardown(() => document.removeEventListener('keydown', onKeyDown));
}
