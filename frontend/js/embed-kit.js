// Shared client primitives for #/embeddings (Compare and Embedding Lab):
// MIME/kind resolution, base64 encoding, bounded batch planning, thumbnails,
// and retrieval benchmark metrics.

import { httpError, requestId } from './api.js';

export const KINDS = ['image', 'audio', 'video'];

export function maxBytesFor(info, kind) {
  const cap = info?.limits?.media_bytes?.[kind];
  if (!Number.isInteger(cap) || cap <= 0) throw new Error(`Missing media_bytes limit for ${kind}`);
  return cap;
}

const EXT_MIME = {
  png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp',
  gif: 'image/gif', bmp: 'image/bmp', avif: 'image/avif',
  mp3: 'audio/mpeg', wav: 'audio/wav', flac: 'audio/flac', ogg: 'audio/ogg',
  m4a: 'audio/mp4', aac: 'audio/aac', opus: 'audio/opus',
  mp4: 'video/mp4', webm: 'video/webm', mov: 'video/quicktime', mkv: 'video/x-matroska',
};

export const mimeOf = (file) => {
  if (file?.type) return file.type;
  const ext = (file?.name || '').split('.').pop()?.toLowerCase();
  return EXT_MIME[ext] || '';
};

export const kindOf = (file) => {
  const k = mimeOf(file).split('/')[0];
  return KINDS.includes(k) ? k : null;
};

export function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(',')[1]);
    reader.onerror = () => reject(new Error(`Cannot read ${file.name}`));
    reader.readAsDataURL(file);
  });
}

export async function mediaBlock(file, windows = null) {
  const type = kindOf(file);
  if (!type) throw new Error(`Unsupported file type: ${file.name}`);
  return {
    type,
    source: { type: 'base64', media_type: mimeOf(file), data: await fileToBase64(file) },
    ...(type === 'video' && windows ? { windows } : {}),
  };
}

export const textBlock = (text) => ({ type: 'text', text });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function embedBatch({ model, dimensions, items, reqId = requestId(), signal }) {
  const res = await fetch('/v1/embeddings', {
    method: 'POST',
    signal,
    headers: { 'Content-Type': 'application/json', 'X-Request-ID': reqId },
    body: JSON.stringify({ model, dimensions, input: items }),
  });
  if (!res.ok) throw await httpError(res);
  return res.json();
}

export async function embedWithRetry(args, tries = 4) {
  for (let attempt = 0; ; attempt++) {
    try {
      return await embedBatch(args);
    } catch (error) {
      if (![429, 503].includes(error.status) || attempt >= tries || args.signal?.aborted) throw error;
      await sleep(1500 * (attempt + 1));
    }
  }
}

export function planBatches(entries, limits) {
  const maxItems = limits?.items;
  const reqBytes = limits?.request_media_bytes;
  if (!Number.isInteger(maxItems) || maxItems <= 0 || !Number.isInteger(reqBytes) || reqBytes <= 0) {
    throw new Error('Missing embedding request limits');
  }
  // 0.75 leaves headroom for 4/3 base64 expansion plus JSON framing under request_media_bytes.
  const maxBytes = Math.floor(reqBytes * 0.75);
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

// ---- Thumbnails ------------------------------------------------------------

const THUMB = 192;

function toThumb(source, width, height) {
  const scale = Math.min(1, THUMB / Math.max(width, height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  canvas.getContext('2d').drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.7);
}

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

// ---- Retrieval evaluation metrics ------------------------------------------

export function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na && nb ? dot / Math.sqrt(na * nb) : 0;
}

export function scoreItem(query, entry) {
  if (Array.isArray(entry) || ArrayBuffer.isView(entry)) return cosine(query, entry);
  let best = entry?.embedding ? cosine(query, entry.embedding) : -Infinity;
  if (Array.isArray(entry?.windows)) {
    for (const w of entry.windows) {
      if (w?.embedding) best = Math.max(best, cosine(query, w.embedding));
    }
  }
  return best;
}

export function evaluateCase(query, pool, positives, excluded = new Set()) {
  const qVec = Array.isArray(query) || ArrayBuffer.isView(query) ? query : query?.embedding;
  let bestPos = -Infinity, bestNeg = -Infinity, bestPosIndex = -1, bestNegIndex = -1;
  const scores = pool.map((entry) => scoreItem(qVec, entry));
  scores.forEach((score, i) => {
    if (excluded.has(i)) return;
    if (positives.has(i)) { if (score > bestPos) { bestPos = score; bestPosIndex = i; } }
    else if (score > bestNeg) { bestNeg = score; bestNegIndex = i; }
  });
  if (bestPosIndex < 0) throw new Error('A case needs at least one positive.');
  const rank = 1 + scores.filter((score, i) => !excluded.has(i) && !positives.has(i) && score > bestPos).length;
  return {
    rank, bestPos, bestNeg: bestNegIndex < 0 ? null : bestNeg, bestNegIndex, bestPosIndex,
    margin: bestNegIndex < 0 ? null : bestPos - bestNeg,
  };
}

export function summarize(results) {
  const n = results.length;
  const margins = results.map((r) => r.margin).filter((m) => m != null);
  return {
    n,
    r1: results.filter((r) => r.rank <= 1).length / n,
    r3: results.filter((r) => r.rank <= 3).length / n,
    mrr: results.reduce((sum, r) => sum + 1 / r.rank, 0) / n,
    meanMargin: margins.length ? margins.reduce((a, b) => a + b, 0) / margins.length : null,
    minMargin: margins.length ? Math.min(...margins) : null,
  };
}

export function buildPlan(set, known) {
  if (!set || !Array.isArray(set.cases) || !set.cases.length) throw new Error('set.json needs a non-empty "cases" array.');
  const pool = [], index = new Map(), queries = [], cases = [], seenIds = new Set();
  const norm = (raw, where) => {
    const item = typeof raw === 'string' ? { text: raw } : raw;
    if (!item || typeof item !== 'object' || Array.isArray(item)) throw new Error(`${where}: expected a string or {text, file, windows}.`);
    const extra = Object.keys(item).filter((k) => k !== 'text' && k !== 'file' && k !== 'windows');
    if (extra.length) throw new Error(`${where}: unknown field "${extra[0]}".`);
    const text = item.text == null ? undefined : String(item.text);
    const file = item.file == null ? undefined : String(item.file);
    const windows = item.windows == null || item.windows === '' ? undefined : String(item.windows);
    if (windows && windows !== 'fps' && windows !== 'content') throw new Error(`${where}: windows must be "fps" or "content".`);
    if (windows && (!file || text?.trim())) throw new Error(`${where}: windows requires a single video file without text.`);
    if (!text?.trim() && !file) throw new Error(`${where}: needs text or a file.`);
    if (file && !known(file)) throw new Error(`${where}: file "${file}" is not in the chosen folder.`);
    return { ...(text?.trim() ? { text } : {}), ...(file ? { file } : {}), ...(windows ? { windows } : {}) };
  };
  const keyOf = (item) => JSON.stringify([item.text ?? null, item.file ?? null, item.windows ?? null]);
  const add = (raw, where) => {
    const item = norm(raw, where);
    const key = keyOf(item);
    if (!index.has(key)) { index.set(key, pool.length); pool.push(item); }
    return index.get(key);
  };
  set.cases.forEach((c, i) => {
    const id = String(c.id ?? `case ${i + 1}`);
    if (seenIds.has(id)) throw new Error(`Duplicate case id "${id}".`);
    seenIds.add(id);
    const where = (part) => `${id} ${part}`;
    const query = norm(c.query, where('query'));
    const positives = (Array.isArray(c.positives) ? c.positives : []).map((p, j) => add(p, where(`positives[${j}]`)));
    const negatives = (Array.isArray(c.negatives) ? c.negatives : []).map((p, j) => add(p, where(`negatives[${j}]`)));
    if (!positives.length) throw new Error(`${id}: needs at least one positive.`);
    if (positives.some((p) => negatives.includes(p))) throw new Error(`${id}: the same item is both a positive and a negative.`);
    queries.push(query);
    cases.push({ id, query, positives, negatives, excluded: [] });
  });
  cases.forEach((c) => {
    const own = index.get(keyOf(c.query));
    if (own == null) return;
    if (c.positives.includes(own)) throw new Error(`${c.id}: the query is identical to one of its own positives.`);
    c.excluded = [own];
  });
  return { queries, pool, cases };
}

// ---- Built-in sample dataset (matches embedding-lab/sample/set.json) -------

const SAMPLE_SET = {
  name: 'Sample: text retrieval with hard negatives, plus four generated shapes',
  cases: [
    {
      id: 'sql-join',
      query: 'how do I combine rows from two tables in SQL',
      positives: ['A JOIN clause merges rows from two tables based on a related column, for example INNER JOIN users ON users.id = orders.user_id.'],
      negatives: ['A carpenter joint such as a dovetail joins two pieces of wood without nails.', 'SQL indexes speed up lookups by keeping a sorted structure beside the table.'],
    },
    {
      id: 'sourdough',
      query: 'why is my sourdough starter not rising',
      positives: ['A sluggish starter is usually too cold or underfed; keep it near 24 C and feed it flour and water at regular intervals.'],
      negatives: ['Sourdough bread gets its tang from lactic and acetic acids produced during long fermentation.', 'To brown a loaf evenly, bake it in a preheated Dutch oven.'],
    },
    {
      id: 'python-copy',
      query: 'make an independent copy of a nested list in python',
      positives: ['copy.deepcopy(items) recursively duplicates nested containers so changes to the copy never touch the original.'],
      negatives: ['A shallow copy such as items[:] still shares the inner lists with the original.', 'Pythons are non-venomous constrictors found in Africa and Asia.'],
    },
    {
      id: 'river',
      query: 'a quiet walk beside a river at dawn',
      positives: ['Mist hung over the water as we strolled along the bank in the early morning light.'],
      negatives: ['The river basin supplies drinking water to four million people.', 'At noon the market square filled with shouting vendors.'],
    },
    {
      id: 'text-to-image-red',
      query: 'a solid red circle on a white background',
      positives: [{ file: 'red-circle.png' }],
      negatives: [{ file: 'blue-square.png' }, { file: 'green-triangle.png' }, { file: 'black-cross.png' }],
    },
    {
      id: 'text-to-image-blue',
      query: 'a blue square',
      positives: [{ file: 'blue-square.png' }],
      negatives: [{ file: 'red-circle.png' }, { file: 'green-triangle.png' }, { file: 'black-cross.png' }],
    },
    {
      id: 'text-to-image-triangle',
      query: 'a green triangle pointing up',
      positives: [{ file: 'green-triangle.png' }],
      negatives: [{ file: 'red-circle.png' }, { file: 'blue-square.png' }],
    },
    {
      id: 'image-to-text',
      query: { file: 'black-cross.png' },
      positives: ['a black plus sign on a white background'],
      negatives: ['a solid red circle on a white background', 'a blue square'],
    },
  ],
};

async function drawShapePng(name, drawFn) {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, 128, 128);
  drawFn(ctx);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
  return new File([blob], name, { type: 'image/png' });
}

export async function buildSampleDataset() {
  const files = new Map();
  files.set('red-circle.png', await drawShapePng('red-circle.png', (c) => {
    c.fillStyle = 'crimson';
    c.beginPath(); c.arc(64, 64, 44, 0, Math.PI * 2); c.fill();
  }));
  files.set('blue-square.png', await drawShapePng('blue-square.png', (c) => {
    c.fillStyle = 'royalblue';
    c.fillRect(24, 24, 80, 80);
  }));
  files.set('green-triangle.png', await drawShapePng('green-triangle.png', (c) => {
    c.fillStyle = 'forestgreen';
    c.beginPath(); c.moveTo(64, 20); c.lineTo(108, 104); c.lineTo(20, 104); c.closePath(); c.fill();
  }));
  files.set('black-cross.png', await drawShapePng('black-cross.png', (c) => {
    c.fillStyle = 'black';
    c.fillRect(52, 20, 24, 88);
    c.fillRect(20, 52, 88, 24);
  }));
  const plan = buildPlan(SAMPLE_SET, (p) => files.has(p));
  return { setName: SAMPLE_SET.name, files, plan };
}
