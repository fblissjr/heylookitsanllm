// Unified browser-local storage module (localStorage + single `heylook` IndexedDB).
// Nothing else in `frontend/js/` may reference `localStorage`, `sessionStorage`, or `indexedDB`
// (`tests/unit/test_frontend_storage_guard.py` enforces this).
//
// Why one module:
// - Safari throws in private mode, quota is finite, and iOS evicts script-writable
//   storage for origins not visited recently.
// - Keeping one registry (`STORAGE_REGISTRY`) ensures startup sweeps, status readouts,
//   and per-feature clear actions derive from a single source of truth instead of
//   hand-copied key or database names, and clearing one feature never wipes another's keys.

export const LS_PREFIX = 'heylook.';
export const IDB_NAME = 'heylook';
export const IDB_VERSION = 1;

export const STORAGE_KEYS = Object.freeze({
  CHAT_DRAFT_PROMPT: 'chat.draft-prompt',
  CHAT_LAST_CONVERSATION: 'chat.last-conversation',
});

export const IDB_STORES = Object.freeze({
  MEDIA_FINDER: 'media-finder',
});

// Single registry of every browser-stored key and IndexedDB object store.
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
  Object.freeze({
    kind: 'indexedDB',
    db: IDB_NAME,
    store: IDB_STORES.MEDIA_FINDER,
    keyPath: 'id',
    indexes: Object.freeze([Object.freeze({ name: 'space', keyPath: 'space' })]),
    feature: 'media-finder',
    holds: 'Media Finder vectors, thumbnails, and file metadata keyed by configuration space',
  }),
]);

const REGISTERED_LS_KEYS = new Set(
  STORAGE_REGISTRY.filter((e) => e.kind === 'localStorage').map((e) => e.key),
);
const REGISTERED_IDB_STORES = STORAGE_REGISTRY.filter((e) => e.kind === 'indexedDB');

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

// IndexedDB databases this app used to open before unifying under `heylook`.
// Deleted at boot (no migration; users re-index).
export const RETIRED_DATABASES = Object.freeze([
  'embed-kit:media-finder',
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

/**
 * Open a registered object store inside the unified `heylook` IndexedDB database.
 * Each record carries `{ id, fileId?, space, vec: Float32Array, ...meta }`.
 */
export async function openVectorStore(storeName = IDB_STORES.MEDIA_FINDER) {
  const entry = REGISTERED_IDB_STORES.find((e) => e.store === storeName);
  if (!entry) {
    throw new Error(`Unregistered IndexedDB store "${storeName}"; add it to STORAGE_REGISTRY in storage.js.`);
  }
  const db = await new Promise((resolve, reject) => {
    const open = indexedDB.open(IDB_NAME, IDB_VERSION);
    open.onupgradeneeded = () => {
      const database = open.result;
      for (const spec of REGISTERED_IDB_STORES) {
        if (!database.objectStoreNames.contains(spec.store)) {
          const store = database.createObjectStore(spec.store, { keyPath: spec.keyPath });
          for (const idx of spec.indexes || []) {
            store.createIndex(idx.name, idx.keyPath, { unique: false });
          }
        }
      }
    };
    open.onsuccess = () => resolve(open.result);
    open.onerror = () => reject(open.error);
  });
  const tx = (mode) => db.transaction(storeName, mode).objectStore(storeName);
  const wrap = (req) => new Promise((res, rej) => {
    req.onsuccess = () => res(req.result);
    req.onerror = () => rej(req.error);
  });
  return {
    ids: async (space) => new Set(await wrap(tx('readonly').index('space').getAllKeys(space))),
    all: (space) => wrap(tx('readonly').index('space').getAll(space)),
    put: (records) => new Promise((res, rej) => {
      const t = db.transaction(storeName, 'readwrite');
      const s = t.objectStore(storeName);
      for (const r of records) s.put(r);
      t.oncomplete = () => res();
      t.onerror = () => rej(t.error);
      t.onabort = () => rej(t.error);
    }),
    clear: (space) => new Promise((res, rej) => {
      const t = db.transaction(storeName, 'readwrite');
      const req = t.objectStore(storeName).index('space').openKeyCursor(IDBKeyRange.only(space));
      req.onsuccess = () => {
        const c = req.result;
        if (c) { t.objectStore(storeName).delete(c.primaryKey); c.continue(); }
      };
      t.oncomplete = () => res();
      t.onerror = () => rej(t.error);
      t.onabort = () => rej(t.error);
    }),
    clearAll: () => new Promise((res, rej) => {
      const t = db.transaction(storeName, 'readwrite');
      t.objectStore(storeName).clear();
      t.oncomplete = () => { db.close(); res(); };
      t.onerror = () => rej(t.error);
      t.onabort = () => rej(t.error);
    }),
    spaces: async () => {
      const counts = new Map();
      const all = await wrap(tx('readonly').getAll());
      for (const r of all) counts.set(r.space, (counts.get(r.space) ?? 0) + 1);
      return counts;
    },
    spaceFileCounts: async () => {
      const filesBySpace = new Map();
      const all = await wrap(tx('readonly').getAll());
      for (const r of all) {
        let set = filesBySpace.get(r.space);
        if (!set) { set = new Set(); filesBySpace.set(r.space, set); }
        set.add(r.fileId || String(r.id).split('#t=')[0]);
      }
      const counts = new Map();
      for (const [space, set] of filesBySpace.entries()) counts.set(space, set.size);
      return counts;
    },
    close: () => db.close(),
  };
}
