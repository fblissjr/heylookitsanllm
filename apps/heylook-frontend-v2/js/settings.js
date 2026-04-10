// Shared sampler settings -- persisted to localStorage
// null values mean "use backend/model default" -- only explicitly set values are sent

const STORAGE_KEY = 'heylook-v2-settings'

const DEFAULTS = {
  temperature: null,
  top_p: null,
  top_k: null,
  max_tokens: null,
}

let _cache = null

function load() {
  if (_cache) return _cache
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    _cache = raw ? { ...DEFAULTS, ...JSON.parse(raw) } : { ...DEFAULTS }
  } catch {
    _cache = { ...DEFAULTS }
  }
  return _cache
}

function save() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(_cache))
  } catch { /* localStorage full or unavailable */ }
}

export function getSettings() {
  return { ...load() }
}

export function getSetting(key) {
  return load()[key]
}

export function updateSetting(key, value) {
  load()
  _cache[key] = value
  save()
}

export function updateSettings(obj) {
  load()
  Object.assign(_cache, obj)
  save()
}

export function resetSettings() {
  _cache = { ...DEFAULTS }
  save()
}

/** Build request params from settings, omitting null (backend-default) values. */
export function samplerParams() {
  const s = load()
  const params = {}
  if (s.temperature != null) params.temperature = s.temperature
  if (s.top_p != null) params.top_p = s.top_p
  if (s.top_k != null && s.top_k > 0) params.top_k = s.top_k
  if (s.max_tokens != null) params.max_tokens = s.max_tokens
  return params
}

export { DEFAULTS }
