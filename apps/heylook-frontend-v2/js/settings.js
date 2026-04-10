// Shared sampler settings -- persisted to localStorage

const STORAGE_KEY = 'heylook-v2-settings'

const DEFAULTS = {
  temperature: 0.7,
  top_p: 1.0,
  top_k: 0,
  max_tokens: 2048,
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

export { DEFAULTS }
