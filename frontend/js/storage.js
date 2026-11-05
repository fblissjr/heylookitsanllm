// Unified browser-local storage module (localStorage + retired IndexedDB sweep).
// Nothing else in `frontend/js/` may reference `localStorage`, `sessionStorage`, or `indexedDB`
// (`tests/unit/test_frontend_storage_guard.py` enforces this).
//
// Why one module:
// - Safari throws in private mode, quota is finite, and iOS evicts script-writable
//   storage for origins not visited recently.
// - Keeping one registry (`STORAGE_REGISTRY`) ensures startup sweeps and status readouts
//   derive from a single source of truth instead of hand-copied key or database names.

export const LS_PREFIX = 'heylook.';

export const STORAGE_KEYS = Object.freeze({
  CHAT_DRAFT_PROMPT: 'chat.draft-prompt',
  CHAT_LAST_CONVERSATION: 'chat.last-conversation',
});

// Single registry of every browser-stored key.
export const STORAGE_REGISTRY = Object.freeze([
  Object.freeze({
    kind: 'localStorage',
    key: STORAGE_KEYS.CHAT_DRAFT_PROMPT,
    fullKey: `${LS_PREFIX}${STORAGE_KEYS.CHAT_DRAFT_PROMPT}`,
    feature: 'chat',
    holds: 'System-prompt draft typed before any conversation exists',
  }),
  Object.freeze({
    kind: 'localStorage',
    key: STORAGE_KEYS.CHAT_LAST_CONVERSATION,
    fullKey: `${LS_PREFIX}${STORAGE_KEYS.CHAT_LAST_CONVERSATION}`,
    feature: 'chat',
    holds: 'ID of the conversation last viewed in this browser',
  }),
]);

const REGISTERED_LS_KEYS = new Set(
  STORAGE_REGISTRY.filter((e) => e.kind === 'localStorage').map((e) => e.key),
);

function assertRegisteredKey(key) {
  if (!REGISTERED_LS_KEYS.has(key)) {
    throw new Error(`Unregistered browser storage key "${key}"; add it to STORAGE_REGISTRY in storage.js.`);
  }
}

export function lsRead(key) {
  assertRegisteredKey(key);
  try { return localStorage.getItem(LS_PREFIX + key); } catch { return null; }
}

export function lsWrite(key, value) {
  assertRegisteredKey(key);
  try {
    if (value === null || value === undefined || value === '') localStorage.removeItem(LS_PREFIX + key);
    else localStorage.setItem(LS_PREFIX + key, value);
  } catch { /* private mode / quota -- in-memory only */ }
}

// Keys this app used to write and no longer reads. Swept once at boot.
export const RETIRED_KEYS = Object.freeze([
  'heylook-v3-settings',    // sampler bag: de-persisted in v2.0.38
  'heylook-v3-display',     // display prefs: removed with show_special_tokens
  'heylook-v3-scan-paths',  // one-off scan paths: no longer remembered
]);

// Keys that were renamed, not retired: [old, new]. Moved once at boot without overwriting new data.
export const MOVED_KEYS = Object.freeze([
  Object.freeze(['heylook.v3.chat.draft-prompt', `${LS_PREFIX}${STORAGE_KEYS.CHAT_DRAFT_PROMPT}`]),
]);

// IndexedDB databases this app used to open before retiring browser Media Finder (`v2.0.243`).
// Deleted at boot.
export const RETIRED_DATABASES = Object.freeze([
  'embed-kit:media-finder',
  'heylook',
]);

export function sweepRetiredStorage() {
  for (const [from, to] of MOVED_KEYS) {
    try {
      const held = localStorage.getItem(from);
      if (held !== null && localStorage.getItem(to) === null) localStorage.setItem(to, held);
      if (held !== null) localStorage.removeItem(from);
    } catch { /* storage unavailable */ }
  }
  for (const key of RETIRED_KEYS) {
    try { localStorage.removeItem(key); } catch { /* nothing to do */ }
  }
  for (const dbName of RETIRED_DATABASES) {
    try {
      if (typeof indexedDB !== 'undefined') indexedDB.deleteDatabase(dbName);
    } catch { /* indexedDB unavailable */ }
  }
}
