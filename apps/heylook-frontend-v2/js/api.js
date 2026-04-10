// API client -- fetch wrappers for backend endpoints

const BASE = ''  // Same origin; override for dev

export async function request(method, path, body) {
  const reqId = crypto.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json', 'X-Request-ID': reqId },
  }
  if (body !== undefined) opts.body = JSON.stringify(body)
  const res = await fetch(`${BASE}${path}`, opts)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `${res.status} ${res.statusText}`)
  }
  return res.json()
}

// Conversations
export const listConversations = () => request('GET', '/v1/conversations')
export const getConversation = (id) => request('GET', `/v1/conversations/${id}`)
export const createConversation = (data = {}) => request('POST', '/v1/conversations', data)
export const updateConversation = (id, data) => request('PUT', `/v1/conversations/${id}`, data)
export const deleteConversation = (id) => request('DELETE', `/v1/conversations/${id}`)

// Messages
export const appendMessage = (convId, data) => request('POST', `/v1/conversations/${convId}/messages`, data)
export const updateMessage = (convId, msgId, data) => request('PUT', `/v1/conversations/${convId}/messages/${msgId}`, data)
export const truncateMessages = (convId, afterPos) => request('DELETE', `/v1/conversations/${convId}/messages?after=${afterPos}`)

// Models
export const listModels = () => request('GET', '/v1/models')
export const listAdminModels = () => request('GET', '/v1/admin/models')
export const loadModel = (id) => request('POST', `/v1/admin/models/${id}/load`)
export const unloadModel = (id) => request('POST', `/v1/admin/models/${id}/unload`)
export const scanModels = (opts = {}) => request('POST', '/v1/admin/models/scan', opts)
export const importModels = (data) => request('POST', '/v1/admin/models/import', data)

// Notebooks
export const listNotebooks = () => request('GET', '/v1/notebooks')
export const getNotebook = (id) => request('GET', `/v1/notebooks/${id}`)
export const createNotebook = (data = {}) => request('POST', '/v1/notebooks', data)
export const updateNotebook = (id, data) => request('PUT', `/v1/notebooks/${id}`, data)
export const deleteNotebook = (id) => request('DELETE', `/v1/notebooks/${id}`)

// System
export const getCapabilities = () => request('GET', '/v1/capabilities')
export const getMetrics = () => request('GET', '/v1/system/metrics')
