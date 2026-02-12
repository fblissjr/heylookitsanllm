import { useState } from 'react'
import type { AdminModelConfig } from '../types'

/** Fields that can be changed at runtime vs needing reload */
const RUNTIME_FIELDS = new Set([
  'temperature', 'top_p', 'top_k', 'min_p', 'max_tokens',
  'repetition_penalty', 'presence_penalty', 'enable_thinking',
  'repetition_context_size',
])

/** MLX-specific config fields in display order */
const MLX_FIELDS = [
  { key: 'model_path', label: 'Model Path', type: 'text', readOnly: true },
  { key: 'vision', label: 'Vision', type: 'bool' },
  { key: 'temperature', label: 'Temperature', type: 'number', step: 0.1, min: 0, max: 2 },
  { key: 'top_p', label: 'Top P', type: 'number', step: 0.05, min: 0, max: 1 },
  { key: 'top_k', label: 'Top K', type: 'number', min: 0 },
  { key: 'min_p', label: 'Min P', type: 'number', step: 0.01, min: 0, max: 1 },
  { key: 'max_tokens', label: 'Max Tokens', type: 'number', min: 1 },
  { key: 'repetition_penalty', label: 'Rep. Penalty', type: 'number', step: 0.05, min: 0.1, max: 2 },
  { key: 'presence_penalty', label: 'Presence Penalty', type: 'number', step: 0.1, min: 0, max: 2 },
  { key: 'cache_type', label: 'Cache Type', type: 'select', options: ['standard', 'rotating', 'quantized'] },
  { key: 'max_kv_size', label: 'Max KV Size', type: 'number', min: 0 },
  { key: 'kv_bits', label: 'KV Bits', type: 'number', min: 1, max: 8 },
  { key: 'kv_group_size', label: 'KV Group Size', type: 'number' },
  { key: 'quantized_kv_start', label: 'Quantized KV Start', type: 'number' },
  { key: 'enable_thinking', label: 'Enable Thinking', type: 'bool' },
  { key: 'draft_model_path', label: 'Draft Model Path', type: 'text' },
  { key: 'num_draft_tokens', label: 'Draft Tokens', type: 'number' },
] as const

/** GGUF-specific config fields */
const GGUF_FIELDS = [
  { key: 'model_path', label: 'Model Path', type: 'text', readOnly: true },
  { key: 'vision', label: 'Vision', type: 'bool' },
  { key: 'mmproj_path', label: 'MM Proj Path', type: 'text' },
  { key: 'chat_format', label: 'Chat Format', type: 'text' },
  { key: 'chat_format_template', label: 'Chat Template', type: 'text' },
  { key: 'n_gpu_layers', label: 'GPU Layers', type: 'number' },
  { key: 'n_ctx', label: 'Context Size', type: 'number' },
  { key: 'temperature', label: 'Temperature', type: 'number', step: 0.1, min: 0, max: 2 },
  { key: 'top_p', label: 'Top P', type: 'number', step: 0.05, min: 0, max: 1 },
  { key: 'top_k', label: 'Top K', type: 'number', min: 0 },
  { key: 'min_p', label: 'Min P', type: 'number', step: 0.01, min: 0, max: 1 },
  { key: 'max_tokens', label: 'Max Tokens', type: 'number', min: 1 },
  { key: 'repetition_penalty', label: 'Rep. Penalty', type: 'number', step: 0.05, min: 0.1, max: 2 },
] as const

/** STT-specific config fields */
const STT_FIELDS = [
  { key: 'model_path', label: 'Model Path', type: 'text', readOnly: true },
  { key: 'chunk_duration', label: 'Chunk Duration (s)', type: 'number' },
  { key: 'overlap_duration', label: 'Overlap Duration (s)', type: 'number' },
  { key: 'use_local_attention', label: 'Local Attention', type: 'bool' },
  { key: 'local_attention_context', label: 'Local Attn Context', type: 'number' },
  { key: 'fp32', label: 'FP32 Mode', type: 'bool' },
] as const

type FieldDef = { key: string; label: string; type: string; readOnly?: boolean; step?: number; min?: number; max?: number; options?: readonly string[] }

function getFieldsForProvider(provider: string): FieldDef[] {
  switch (provider) {
    case 'mlx': return MLX_FIELDS as unknown as FieldDef[]
    case 'gguf':
    case 'llama_cpp': return GGUF_FIELDS as unknown as FieldDef[]
    case 'mlx_stt': return STT_FIELDS as unknown as FieldDef[]
    default: return MLX_FIELDS as unknown as FieldDef[]
  }
}

interface Props {
  model: AdminModelConfig
  onUpdate: (updates: Record<string, unknown>) => Promise<void>
}

export function ProviderConfigForm({ model, onUpdate }: Props) {
  const [editState, setEditState] = useState<Record<string, unknown>>({})
  const [saving, setSaving] = useState(false)

  const fields = getFieldsForProvider(model.provider)
  const hasPending = Object.keys(editState).length > 0

  const getValue = (key: string) => {
    if (key in editState) return editState[key]
    return model.config[key]
  }

  const handleChange = (key: string, value: unknown) => {
    setEditState((prev) => ({ ...prev, [key]: value }))
  }

  const handleSave = async () => {
    if (!hasPending) return
    setSaving(true)
    try {
      await onUpdate(editState)
      setEditState({})
    } finally {
      setSaving(false)
    }
  }

  const handleReset = () => setEditState({})

  return (
    <div className="px-4 pb-3 space-y-2">
      {fields.map((field) => {
        const value = getValue(field.key)
        const isRuntime = RUNTIME_FIELDS.has(field.key)
        const isModified = field.key in editState

        return (
          <div key={field.key} className="flex items-center gap-2">
            <label className="text-xs text-gray-400 w-32 shrink-0 truncate" title={field.label}>
              {field.label}
              {!isRuntime && !field.readOnly && (
                <span className="ml-1 text-amber-400" title="Requires reload">*</span>
              )}
            </label>
            <div className="flex-1 min-w-0">
              {field.readOnly ? (
                <span className="text-xs text-gray-500 dark:text-gray-400 truncate block" title={String(value ?? '')}>
                  {String(value ?? '')}
                </span>
              ) : field.type === 'bool' ? (
                <button
                  onClick={() => handleChange(field.key, !value)}
                  className={`px-2 py-0.5 text-xs rounded ${
                    value
                      ? 'bg-green-500/20 text-green-400'
                      : 'bg-gray-500/20 text-gray-400'
                  } ${isModified ? 'ring-1 ring-primary' : ''}`}
                >
                  {value ? 'true' : 'false'}
                </button>
              ) : field.type === 'select' ? (
                <select
                  value={String(value ?? '')}
                  onChange={(e) => handleChange(field.key, e.target.value)}
                  className={`w-full px-2 py-0.5 text-xs bg-gray-100 dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100 ${isModified ? 'ring-1 ring-primary' : ''}`}
                >
                  {field.options?.map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              ) : field.type === 'number' ? (
                <input
                  type="number"
                  value={value != null ? String(value) : ''}
                  step={field.step}
                  min={field.min}
                  max={field.max}
                  onChange={(e) => {
                    const v = e.target.value === '' ? null : Number(e.target.value)
                    handleChange(field.key, v)
                  }}
                  className={`w-full px-2 py-0.5 text-xs bg-gray-100 dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100 ${isModified ? 'ring-1 ring-primary' : ''}`}
                />
              ) : (
                <input
                  type="text"
                  value={String(value ?? '')}
                  onChange={(e) => handleChange(field.key, e.target.value)}
                  className={`w-full px-2 py-0.5 text-xs bg-gray-100 dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100 ${isModified ? 'ring-1 ring-primary' : ''}`}
                />
              )}
            </div>
          </div>
        )
      })}

      {hasPending && (
        <div className="flex items-center gap-2 pt-2">
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-3 py-1 text-xs font-medium text-white bg-primary hover:bg-primary-hover rounded transition-colors disabled:opacity-50"
          >
            {saving ? 'Saving...' : 'Save Changes'}
          </button>
          <button
            onClick={handleReset}
            className="px-3 py-1 text-xs text-gray-400 hover:text-gray-200"
          >
            Reset
          </button>
        </div>
      )}

      <p className="text-[10px] text-gray-500 pt-1">
        Fields marked with * require model reload after change.
      </p>
    </div>
  )
}
