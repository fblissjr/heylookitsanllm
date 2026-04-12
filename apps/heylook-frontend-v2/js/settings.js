// Shared sampler settings -- persisted to localStorage
// null values mean "use backend/model default" -- only explicitly set values are sent
//
// Backend default cascade: global defaults -> thinking mode overrides ->
// models.toml per-model config -> request params. Sending null lets the
// backend apply its model-specific defaults.

const STORAGE_KEY = 'heylook-v2-settings'

const DEFAULTS = {
  temperature: null,
  top_p: null,
  top_k: null,
  min_p: null,
  max_tokens: null,
  repetition_penalty: null,
  repetition_context_size: null,
  presence_penalty: null,
  seed: null,
  enable_thinking: null,
}

// Metadata for the settings panel UI
export const PARAM_META = {
  temperature:             { label: 'Temperature',      min: 0, max: 2,   step: 0.1,  section: 'core' },
  max_tokens:              { label: 'Max Tokens',       min: 1, max: 8192, step: 64,  section: 'core', type: 'int' },
  top_p:                   { label: 'Top P',            min: 0, max: 1,   step: 0.05, section: 'core' },
  top_k:                   { label: 'Top K',            min: 0, max: 200, step: 1,    section: 'core', type: 'int' },
  min_p:                   { label: 'Min P',            min: 0, max: 1,   step: 0.01, section: 'advanced' },
  repetition_penalty:      { label: 'Repetition Penalty', min: 0.1, max: 2, step: 0.05, section: 'advanced' },
  repetition_context_size: { label: 'Rep. Context Size', min: 1, max: 100, step: 1,   section: 'advanced', type: 'int' },
  presence_penalty:        { label: 'Presence Penalty',  min: 0, max: 2,   step: 0.1,  section: 'advanced' },
  seed:                    { label: 'Seed',             min: 0, max: 999999, step: 1,  section: 'advanced', type: 'int' },
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
  if (s.min_p != null) params.min_p = s.min_p
  if (s.max_tokens != null) params.max_tokens = s.max_tokens
  if (s.repetition_penalty != null) params.repetition_penalty = s.repetition_penalty
  if (s.repetition_context_size != null) params.repetition_context_size = s.repetition_context_size
  if (s.presence_penalty != null && s.presence_penalty > 0) params.presence_penalty = s.presence_penalty
  if (s.seed != null) params.seed = s.seed
  if (s.enable_thinking != null) params.enable_thinking = s.enable_thinking
  return params
}

export { DEFAULTS }
