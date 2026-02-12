import { useState, useEffect } from 'react'
import { useModelsStore } from '../stores/modelsStore'
import type { AdminModelConfig } from '../types'
import { PresetSelector } from './PresetSelector'
import { ProviderConfigForm } from './ProviderConfigForm'
import { ModelStatusCard } from './ModelStatusCard'

export function ModelDetail() {
  const configs = useModelsStore((s) => s.configs)
  const selectedId = useModelsStore((s) => s.selectedId)
  const updateConfig = useModelsStore((s) => s.updateConfig)
  const removeConfig = useModelsStore((s) => s.removeConfig)
  const toggleEnabled = useModelsStore((s) => s.toggleEnabled)

  const model = configs.find((c) => c.id === selectedId)

  if (!model) {
    return (
      <div className="h-full flex items-center justify-center text-gray-400 text-sm">
        Select a model to view details
      </div>
    )
  }

  return <ModelDetailInner key={model.id} model={model} updateConfig={updateConfig} removeConfig={removeConfig} toggleEnabled={toggleEnabled} />
}

interface InnerProps {
  model: AdminModelConfig
  updateConfig: (id: string, updates: Record<string, unknown>) => Promise<string[]>
  removeConfig: (id: string) => Promise<void>
  toggleEnabled: (id: string) => Promise<void>
}

function ModelDetailInner({ model, updateConfig, removeConfig, toggleEnabled }: InnerProps) {
  const [description, setDescription] = useState(model.description ?? '')
  const [tags, setTags] = useState(model.tags.join(', '))
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [reloadFields, setReloadFields] = useState<string[]>([])
  const [confirmRemove, setConfirmRemove] = useState(false)
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    status: true,
    presets: true,
    sampling: false,
    provider: false,
    meta: false,
  })

  useEffect(() => {
    setDescription(model.description ?? '')
    setTags(model.tags.join(', '))
    setDirty(false)
    setReloadFields([])
    setConfirmRemove(false)
  }, [model.id, model.description, model.tags])

  const toggleSection = (key: string) => {
    setExpandedSections((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      const updates: Record<string, unknown> = {}
      if (description !== (model.description ?? '')) updates.description = description
      const newTags = tags.split(',').map((t) => t.trim()).filter(Boolean)
      if (JSON.stringify(newTags) !== JSON.stringify(model.tags)) updates.tags = newTags
      const fields = await updateConfig(model.id, updates)
      setReloadFields(fields)
      setDirty(false)
    } finally {
      setSaving(false)
    }
  }

  const handleConfigUpdate = async (configUpdates: Record<string, unknown>) => {
    setSaving(true)
    try {
      const fields = await updateConfig(model.id, { config: configUpdates })
      setReloadFields(fields)
    } finally {
      setSaving(false)
    }
  }

  const handleRemove = async () => {
    await removeConfig(model.id)
  }

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 shrink-0">
        <div className="flex items-center justify-between">
          <div className="min-w-0">
            <h2 className="text-base font-semibold text-gray-900 dark:text-gray-100 truncate">{model.id}</h2>
            <span className="text-xs text-gray-400 uppercase">{model.provider}</span>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={() => toggleEnabled(model.id)}
              className={`px-2 py-1 text-xs font-medium rounded ${
                model.enabled
                  ? 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                  : 'bg-gray-500/20 text-gray-400 hover:bg-gray-500/30'
              }`}
            >
              {model.enabled ? 'Enabled' : 'Disabled'}
            </button>
          </div>
        </div>
      </div>

      {/* Reload warning */}
      {reloadFields.length > 0 && (
        <div className="px-4 py-2 bg-amber-500/10 border-b border-amber-500/20 text-xs text-amber-400">
          Changes to {reloadFields.join(', ')} require model reload to take effect.
        </div>
      )}

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto">
        {/* Status section */}
        <Section title="Status" expanded={expandedSections.status} onToggle={() => toggleSection('status')}>
          <ModelStatusCard model={model} />
        </Section>

        {/* Presets section */}
        <Section title="Presets" expanded={expandedSections.presets} onToggle={() => toggleSection('presets')}>
          <PresetSelector modelId={model.id} />
        </Section>

        {/* Provider config */}
        <Section title="Configuration" expanded={expandedSections.provider} onToggle={() => toggleSection('provider')}>
          <ProviderConfigForm model={model} onUpdate={handleConfigUpdate} />
        </Section>

        {/* Metadata */}
        <Section title="Metadata" expanded={expandedSections.meta} onToggle={() => toggleSection('meta')}>
          <div className="space-y-3 px-4 pb-3">
            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1">Description</label>
              <input
                type="text"
                value={description}
                onChange={(e) => { setDescription(e.target.value); setDirty(true) }}
                className="w-full px-2 py-1.5 text-sm bg-gray-100 dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1">Tags (comma-separated)</label>
              <input
                type="text"
                value={tags}
                onChange={(e) => { setTags(e.target.value); setDirty(true) }}
                className="w-full px-2 py-1.5 text-sm bg-gray-100 dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100"
              />
            </div>
            {dirty && (
              <button
                onClick={handleSave}
                disabled={saving}
                className="px-3 py-1.5 text-sm font-medium text-white bg-primary hover:bg-primary-hover rounded transition-colors disabled:opacity-50"
              >
                {saving ? 'Saving...' : 'Save Metadata'}
              </button>
            )}
          </div>
        </Section>

        {/* Danger zone */}
        <div className="px-4 py-4 border-t border-gray-200 dark:border-gray-700">
          {confirmRemove ? (
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-400">Remove this model from config?</span>
              <button
                onClick={handleRemove}
                className="px-2 py-1 text-xs font-medium text-white bg-red-600 hover:bg-red-700 rounded"
              >
                Confirm
              </button>
              <button
                onClick={() => setConfirmRemove(false)}
                className="px-2 py-1 text-xs text-gray-400 hover:text-gray-200"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={() => setConfirmRemove(true)}
              className="text-xs text-red-400 hover:text-red-300"
            >
              Remove from config (files stay on disk)
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function Section({
  title,
  expanded,
  onToggle,
  children,
}: {
  title: string
  expanded: boolean
  onToggle: () => void
  children: React.ReactNode
}) {
  return (
    <div className="border-b border-gray-200 dark:border-gray-700">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-4 py-2.5 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800/50"
      >
        {title}
        <svg
          className={`w-4 h-4 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {expanded && <div>{children}</div>}
    </div>
  )
}
