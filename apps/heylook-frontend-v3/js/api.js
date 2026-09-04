// JSON API layer. Every wrapper is generated from ROUTES below; streaming
// chat lives in streaming.js, not here.

export function requestId() {
  try { return crypto.randomUUID(); }
  catch { return `req-${Math.random().toString(36).slice(2)}-${Date.now()}`; }
}

// Normalized HTTP error with .status/.code/.retryAfter -- the one contract
// that retry and error-surfacing logic keys on. Shared with streaming.js.
export async function httpError(res) {
  let detail = res.statusText;
  let code = null;
  let structured = null;
  try {
    const data = await res.json();
    // FastAPI's `detail` is not always a string. The fit route answers
    // {detail:{field, error}}, and stringifying that gives "[object Object]",
    // so the structured reason has to survive on the error rather than be
    // flattened into the message. Kept as `err.detail` for callers that can
    // render a field-scoped reason; `message` stays a string for the rest.
    structured = (data.detail && typeof data.detail === 'object') ? data.detail : null;
    const detailText = structured ? (structured.error ?? null) : data.detail;
    detail = data.error?.message || detailText || data.error?.code || detail;
    code = data.error?.code ?? null;
  } catch { /* non-JSON error body */ }
  const err = new Error(detail);
  err.status = res.status;
  err.detail = structured;
  err.code = code;
  err.retryAfter = Number(res.headers.get('Retry-After')) || null;
  return err;
}

// `keepalive`: the request outlives the document -- for the last-moment
// flushes a page makes on hide/pagehide, where a plain fetch is cancelled by
// the unload. Bodies must stay under the browser's ~64KB keepalive cap.
export async function request(method, path, body, { signal, keepalive = false } = {}) {
  const headers = { 'X-Request-ID': requestId() };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const res = await fetch(path, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal,
    keepalive,
  });
  if (!res.ok) throw await httpError(res);
  return res.status === 204 ? null : res.json();
}

// name: [method, buildPath, hasBody] -- when hasBody, the call's last
// argument (after buildPath's params) is the JSON body.
const ROUTES = {
  // models + system
  listModels:        ['GET', () => '/v1/models'],
  systemMetrics:     ['GET', (force) => `/v1/system/metrics${force ? '?force_refresh=true' : ''}`],
  perfProfile:       ['GET', (range) => `/v1/performance/profile/${range}`],
  clearAllData:      ['POST', () => '/v1/data/clear'],

  // conversations
  listConversations: ['GET', () => '/v1/conversations'],
  createConversation:['POST', () => '/v1/conversations', true],
  getConversation:   ['GET', (id) => `/v1/conversations/${id}`],
  cloneConversation: ['POST', (id) => `/v1/conversations/${id}/clone`, true],
  updateConversation:['PUT', (id) => `/v1/conversations/${id}`, true],
  deleteConversation:['DELETE', (id) => `/v1/conversations/${id}`],
  addMessage:        ['POST', (id) => `/v1/conversations/${id}/messages`, true],
  updateMessage:     ['PUT', (id, msgId) => `/v1/conversations/${id}/messages/${msgId}`, true],
  // What generate WOULD send, rendered by the model's own engine (v1.79.62).
  // Same mode/message_id/overrides vocabulary as generate; persists nothing;
  // 409 when the model is not resident (a preview never loads a model).
  previewPrompt:     ['POST', (id) => `/v1/conversations/${id}/prompt`, true],
  deleteMessage:     ['DELETE', (id, msgId) => `/v1/conversations/${id}/messages/${msgId}`],

  // presets (saved system prompt + sampler bundles)
  listPresets:       ['GET', () => '/v1/presets'],
  createPreset:      ['POST', () => '/v1/presets', true],
  updatePreset:      ['PUT', (id) => `/v1/presets/${id}`, true],
  deletePreset:      ['DELETE', (id) => `/v1/presets/${id}`],

  // j-space (Jacobian lens interpretability)
  jspaceModels:      ['GET', () => '/v1/jspace/models'],
  jspaceAnalyze:     ['POST', () => '/v1/jspace/analyze', true],

  // notebooks
  listNotebooks:     ['GET', () => '/v1/notebooks'],
  createNotebook:    ['POST', () => '/v1/notebooks', true],
  getNotebook:       ['GET', (id) => `/v1/notebooks/${id}`],
  updateNotebook:    ['PUT', (id) => `/v1/notebooks/${id}`, true],
  deleteNotebook:    ['DELETE', (id) => `/v1/notebooks/${id}`],

  // admin models
  adminListModels:   ['GET', () => '/v1/admin/models'],
  // warm=true additionally runs a 1-token generation through the real
  // generation path -- the server-owned readiness call (v1.38.0), so the
  // first real message doesn't pay the Metal kernel JIT.
  // NOT an admin route since v1.79.48: loading is what a generate request
  // already does, so it is gated like inference. Name kept -- every caller
  // is the models/chat Load button, and renaming buys nothing.
  adminLoadModel:    ['POST', (id, warm) => `/v1/models/${encodeURIComponent(id)}/load${warm ? '?warm=true' : ''}`],
  adminUnloadModel:  ['POST', (id) => `/v1/admin/models/${encodeURIComponent(id)}/unload`],
  // ONE server-owned unload+load(+warm): a browser-driven pair could strand
  // the model unloaded if the tab died between the calls. Load's shape.
  // `ctxSize` (gguf only, v1.79.61): the context to load with, persisted as
  // the model's `ctx_size` config by the server -- ONE writer, the same
  // models.toml write the models page's editor makes. 0 = Auto (unset). The
  // server makes the unchanged-and-resident case a plain load, so sending
  // the same choice again does not restart a warm process.
  adminReloadModel:  ['POST', (id, warm, ctxSize) => {
    const q = [];
    if (warm) q.push('warm=true');
    if (ctxSize != null) q.push(`ctx_size=${encodeURIComponent(ctxSize)}`);
    return `/v1/admin/models/${encodeURIComponent(id)}/reload${q.length ? '?' + q.join('&') : ''}`;
  }],
  adminScan:         ['POST', () => '/v1/admin/models/scan', true],
  adminImport:       ['POST', () => '/v1/admin/models/import', true],
  // The [scan] watch folders -- what the server DISCOVERS models from. Since
  // v1.69.0 a model under one of these is served with no models.toml entry,
  // so this list is the primary way to add models; import is the fallback for
  // a model that lives somewhere else. PUT reloads the router and answers
  // with models_served, the observable consequence of the edit.
  adminScanConfig:   ['GET', () => '/v1/admin/models/scan-config'],
  adminSetScanConfig:['PUT', () => '/v1/admin/models/scan-config', true],
  // The per-provider option schema (field type/bounds/enum/default + the
  // `effect` class saying WHEN a change takes effect). NOT under
  // /v1/admin/models -- that router's {model_id:path} would eat the path.
  adminModelOptions: ['GET', () => '/v1/admin/model-options'],
  // PATCH body {config:{key:value|null}}; null means "unset -- back to the
  // default" (the key is removed from models.toml). Response carries
  // reload_required_fields (+ warning when the post-save reload failed).
  adminUpdateModel:  ['PATCH', (id) => `/v1/admin/models/${encodeURIComponent(id)}`, true],
  // Server-computed memory fit for a model + candidate (unsaved) config
  // edits. Read-only; the meter renders the response verbatim and NEVER
  // computes fit client-side (design doc §5).
  adminModelFit:     ['POST', (id) => `/v1/admin/models/${encodeURIComponent(id)}/fit`, true],
};

function makeCall(method, buildPath, hasBody) {
  return (...args) => {
    const pathArgs = args.slice(0, buildPath.length);
    const body = hasBody ? args[buildPath.length] : undefined;
    const opts = args[buildPath.length + (hasBody ? 1 : 0)] || {};
    return request(method, buildPath(...pathArgs), body, opts);
  };
}

export const api = Object.fromEntries(
  Object.entries(ROUTES).map(([name, [method, buildPath, hasBody]]) =>
    [name, makeCall(method, buildPath, hasBody)]),
);
