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
  try {
    const data = await res.json();
    detail = data.error?.message || data.detail || data.error?.code || detail;
    code = data.error?.code ?? null;
  } catch { /* non-JSON error body */ }
  const err = new Error(detail);
  err.status = res.status;
  err.code = code;
  err.retryAfter = Number(res.headers.get('Retry-After')) || null;
  return err;
}

export async function request(method, path, body, { signal } = {}) {
  const headers = { 'X-Request-ID': requestId() };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const res = await fetch(path, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal,
  });
  if (!res.ok) throw await httpError(res);
  return res.status === 204 ? null : res.json();
}

// name: [method, buildPath, hasBody] -- when hasBody, the call's last
// argument (after buildPath's params) is the JSON body.
const ROUTES = {
  // models + system
  listModels:        ['GET', () => '/v1/models'],
  capabilities:      ['GET', () => '/v1/capabilities'],
  systemMetrics:     ['GET', (force) => `/v1/system/metrics${force ? '?force_refresh=true' : ''}`],
  perfProfile:       ['GET', (range) => `/v1/performance/profile/${range}`],
  clearAllData:      ['POST', () => '/v1/data/clear'],

  // conversations
  listConversations: ['GET', () => '/v1/conversations'],
  createConversation:['POST', () => '/v1/conversations', true],
  getConversation:   ['GET', (id) => `/v1/conversations/${id}`],
  updateConversation:['PUT', (id) => `/v1/conversations/${id}`, true],
  deleteConversation:['DELETE', (id) => `/v1/conversations/${id}`],
  addMessage:        ['POST', (id) => `/v1/conversations/${id}/messages`, true],
  updateMessage:     ['PUT', (id, msgId) => `/v1/conversations/${id}/messages/${msgId}`, true],
  deleteMessagesAfter:['DELETE', (id, pos) => `/v1/conversations/${id}/messages?after=${pos}`],

  // presets (saved system prompt + sampler bundles)
  listPresets:       ['GET', () => '/v1/presets'],
  createPreset:      ['POST', () => '/v1/presets', true],
  updatePreset:      ['PUT', (id) => `/v1/presets/${id}`, true],
  deletePreset:      ['DELETE', (id) => `/v1/presets/${id}`],

  // notebooks
  listNotebooks:     ['GET', () => '/v1/notebooks'],
  createNotebook:    ['POST', () => '/v1/notebooks', true],
  getNotebook:       ['GET', (id) => `/v1/notebooks/${id}`],
  updateNotebook:    ['PUT', (id) => `/v1/notebooks/${id}`, true],
  deleteNotebook:    ['DELETE', (id) => `/v1/notebooks/${id}`],

  // admin models
  adminListModels:   ['GET', () => '/v1/admin/models'],
  adminLoadModel:    ['POST', (id) => `/v1/admin/models/${encodeURIComponent(id)}/load`],
  adminUnloadModel:  ['POST', (id) => `/v1/admin/models/${encodeURIComponent(id)}/unload`],
  adminScan:         ['POST', () => '/v1/admin/models/scan', true],
  adminImport:       ['POST', () => '/v1/admin/models/import', true],
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
