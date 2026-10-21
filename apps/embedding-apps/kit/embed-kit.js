// embed-kit: the client half of heylook's /v1/embeddings that every app here shares.
// A request queue with bounded batches, a persistent vector store that records its
// provenance, and a flat cosine search. The server has no index; this is it.

export const MAX_ITEMS = 16;              // server: items per request
export const MAX_BATCH_BYTES = 48 * 2 ** 20; // under the server's 64 MB request media cap
export const MAX_FILE_BYTES = 32 * 2 ** 20;  // server: bytes per media block
export const KINDS = ['image', 'audio', 'video'];

export const kindOf = (file) => {
  const k = (file.type || '').split('/')[0];
  return KINDS.includes(k) ? k : null;
};

export const newRequestId = () => `kit-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

async function fail(response) {
  let message = `${response.status} ${response.statusText}`;
  try {
    const body = await response.json();
    const detail = body.detail ?? body.error?.message ?? body.error ?? body.message;
    if (detail) message = typeof detail === 'string' ? detail : JSON.stringify(detail);
  } catch { /* keep the status line */ }
  const error = new Error(message);
  error.status = response.status;
  return error;
}

/** Models that serve embeddings, with the engine's own description of them. */
export async function listEmbeddingModels() {
  const response = await fetch('/v1/models');
  if (!response.ok) throw await fail(response);
  const body = await response.json();
  return (body.data ?? []).filter((m) => m.engine?.embedding);
}

export function cancel(requestId) {
  return fetch(`/v1/requests/${encodeURIComponent(requestId)}`, { method: 'DELETE' }).catch(() => {});
}

export function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(',')[1]);
    reader.onerror = () => reject(new Error(`Cannot read ${file.name}`));
    reader.readAsDataURL(file);
  });
}

export async function mediaBlock(file) {
  const type = kindOf(file);
  if (!type) throw new Error(`Unsupported file type: ${file.name}`);
  return { type, source: { type: 'base64', media_type: file.type, data: await fileToBase64(file) } };
}

export const textBlock = (text) => ({ type: 'text', text });

/** One request. items: [{content:[blocks], prompt_name?}]. Returns the server's result. */
export async function embed({ model, dimensions, items, requestId = newRequestId(), signal }) {
  const response = await fetch('/v1/embeddings', {
    method: 'POST', signal,
    headers: { 'Content-Type': 'application/json', 'X-Request-ID': requestId },
    body: JSON.stringify({ model, dimensions, input: items }),
  });
  if (!response.ok) throw await fail(response);
  return response.json();
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** embed() that waits out a busy server (429/503) a few times. */
export async function embedWithRetry(args, tries = 4) {
  for (let attempt = 0; ; attempt++) {
    try { return await embed(args); }
    catch (error) {
      if (![429, 503].includes(error.status) || attempt >= tries || args.signal?.aborted) throw error;
      await sleep(1500 * (attempt + 1));
    }
  }
}

/** Split work into requests that respect the item and byte limits, in order. */
export function planBatches(entries, maxItems = MAX_ITEMS, maxBytes = MAX_BATCH_BYTES) {
  const batches = [];
  let current = [], bytes = 0;
  for (const entry of entries) {
    if (current.length && (current.length >= maxItems || bytes + entry.bytes > maxBytes)) {
      batches.push(current); current = []; bytes = 0;
    }
    current.push(entry); bytes += entry.bytes;
  }
  if (current.length) batches.push(current);
  return batches;
}

// ---- persistence -----------------------------------------------------------

/** A space names one embedding configuration. Vectors from different spaces never mix. */
export const spaceId = ({ model, dimensions, prompt }) => `${model}|${dimensions}|${prompt || '-'}`;

function request(r) {
  return new Promise((resolve, reject) => { r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error); });
}

export async function openStore(name) {
  const db = await new Promise((resolve, reject) => {
    const open = indexedDB.open(`embed-kit:${name}`, 1);
    open.onupgradeneeded = () => {
      const store = open.result.createObjectStore('vectors', { keyPath: 'id' });
      store.createIndex('space', 'space');
    };
    open.onsuccess = () => resolve(open.result);
    open.onerror = () => reject(open.error);
  });
  const tx = (mode) => db.transaction('vectors', mode).objectStore('vectors');
  return {
    async put(records) {
      const t = db.transaction('vectors', 'readwrite');
      for (const r of records) t.objectStore('vectors').put(r);
      await new Promise((res, rej) => { t.oncomplete = res; t.onerror = () => rej(t.error); });
    },
    ids: async (space) => new Set(await request(tx('readonly').index('space').getAllKeys(space))),
    all: (space) => request(tx('readonly').index('space').getAll(space)),
    spaces: async () => [...new Set((await request(tx('readonly').getAll())).map((r) => r.space))],
    async clear(space) {
      const keys = await request(tx('readonly').index('space').getAllKeys(space));
      const t = db.transaction('vectors', 'readwrite');
      keys.forEach((k) => t.objectStore('vectors').delete(k));
      await new Promise((res, rej) => { t.oncomplete = res; t.onerror = () => rej(t.error); });
    },
  };
}

// ---- search ----------------------------------------------------------------

/** Server vectors are L2-normalised, so cosine is a dot product. Records hold Float32Array `vec`. */
export function topK(query, records, k = 24) {
  const q = Float32Array.from(query);
  const scored = records.map((record) => {
    let dot = 0;
    const v = record.vec;
    for (let i = 0; i < q.length; i++) dot += q[i] * v[i];
    return { record, score: dot };
  });
  return scored.sort((a, b) => b.score - a.score).slice(0, k);
}

// ---- thumbnails ------------------------------------------------------------

const THUMB = 192;

function toThumb(source, width, height) {
  const scale = Math.min(1, THUMB / Math.max(width, height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  canvas.getContext('2d').drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.7);
}

/** A small data URL for an image or video file, or null when the browser cannot decode it. */
export async function thumbnail(file, kind) {
  try {
    if (kind === 'image') {
      const bitmap = await createImageBitmap(file);
      const url = toThumb(bitmap, bitmap.width, bitmap.height);
      bitmap.close();
      return url;
    }
    if (kind === 'video') {
      const url = URL.createObjectURL(file);
      try {
        const video = document.createElement('video');
        video.muted = true; video.preload = 'auto'; video.src = url;
        await new Promise((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error('timeout')), 5000);
          video.onloadeddata = () => { video.currentTime = Math.min(0.1, (video.duration || 1) / 2); };
          video.onseeked = () => { clearTimeout(timer); resolve(); };
          video.onerror = () => { clearTimeout(timer); reject(new Error('decode')); };
        });
        return toThumb(video, video.videoWidth, video.videoHeight);
      } finally { URL.revokeObjectURL(url); }
    }
  } catch { /* fall through */ }
  return null;
}
