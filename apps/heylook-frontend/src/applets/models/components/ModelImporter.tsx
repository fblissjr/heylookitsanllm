import { useState } from 'react'
import { useModelsStore } from '../stores/modelsStore'
import type { ScannedModel } from '../types'

export function ModelImporter() {
  const importOpen = useModelsStore((s) => s.importOpen)
  const setImportOpen = useModelsStore((s) => s.setImportOpen)
  const scanResults = useModelsStore((s) => s.scanResults)
  const scanning = useModelsStore((s) => s.scanning)
  const importing = useModelsStore((s) => s.importing)
  const scanForModels = useModelsStore((s) => s.scanForModels)
  const importModels = useModelsStore((s) => s.importModels)
  const profiles = useModelsStore((s) => s.profiles)

  const [customPath, setCustomPath] = useState('')
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [selectedProfile, setSelectedProfile] = useState('balanced')
  const [step, setStep] = useState<'scan' | 'select'>('scan')

  if (!importOpen) return null

  const handleScan = async () => {
    const paths = customPath.trim() ? [customPath.trim()] : []
    await scanForModels({ paths, scan_hf_cache: true })
    setStep('select')
    // Pre-select models that aren't already configured
    const unconfigured = scanResults.filter((m) => !m.already_configured)
    setSelectedIds(new Set(unconfigured.map((m) => m.id)))
  }

  const handleImport = async () => {
    const selected = scanResults.filter((m) => selectedIds.has(m.id))
    await importModels({
      models: selected.map((m) => ({
        id: m.id,
        path: m.path,
        provider: m.provider,
        vision: m.vision,
        size_gb: m.size_gb,
        tags: m.tags,
        description: m.description,
        config: { model_path: m.path, vision: m.vision },
      })),
      profile: selectedProfile,
    })
  }

  const toggleId = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const selectAll = () => {
    const ids = scanResults.filter((m) => !m.already_configured).map((m) => m.id)
    setSelectedIds(new Set(ids))
  }

  const selectNone = () => setSelectedIds(new Set())

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/60 z-50" onClick={() => setImportOpen(false)} />

      {/* Modal */}
      <div className="fixed inset-4 md:inset-x-auto md:inset-y-8 md:max-w-2xl md:mx-auto bg-white dark:bg-surface-dark rounded-xl shadow-2xl z-50 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="px-5 py-3 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between shrink-0">
          <h2 className="text-base font-semibold text-gray-900 dark:text-gray-100">Import Models</h2>
          <button
            onClick={() => setImportOpen(false)}
            className="p-1 rounded text-gray-400 hover:text-gray-200"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5">
          {step === 'scan' ? (
            <ScanStep
              customPath={customPath}
              setCustomPath={setCustomPath}
              scanning={scanning}
              onScan={handleScan}
            />
          ) : (
            <SelectStep
              results={scanResults}
              selectedIds={selectedIds}
              toggleId={toggleId}
              selectAll={selectAll}
              selectNone={selectNone}
              profiles={profiles}
              selectedProfile={selectedProfile}
              setSelectedProfile={setSelectedProfile}
            />
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-gray-200 dark:border-gray-700 flex items-center justify-between shrink-0">
          {step === 'select' && (
            <button
              onClick={() => setStep('scan')}
              className="text-sm text-gray-400 hover:text-gray-200"
            >
              Back
            </button>
          )}
          <div className="ml-auto flex gap-2">
            <button
              onClick={() => setImportOpen(false)}
              className="px-3 py-1.5 text-sm text-gray-400 hover:text-gray-200"
            >
              Cancel
            </button>
            {step === 'scan' ? (
              <button
                onClick={handleScan}
                disabled={scanning}
                className="px-4 py-1.5 text-sm font-medium text-white bg-primary hover:bg-primary-hover rounded-lg transition-colors disabled:opacity-50"
              >
                {scanning ? 'Scanning...' : 'Scan'}
              </button>
            ) : (
              <button
                onClick={handleImport}
                disabled={importing || selectedIds.size === 0}
                className="px-4 py-1.5 text-sm font-medium text-white bg-primary hover:bg-primary-hover rounded-lg transition-colors disabled:opacity-50"
              >
                {importing ? 'Importing...' : `Import ${selectedIds.size} Models`}
              </button>
            )}
          </div>
        </div>
      </div>
    </>
  )
}

function ScanStep({
  customPath,
  setCustomPath,
  scanning,
  onScan,
}: {
  customPath: string
  setCustomPath: (v: string) => void
  scanning: boolean
  onScan: () => void
}) {
  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600 dark:text-gray-300">
        Scan your filesystem for models that can be imported. The HuggingFace cache
        and the project modelzoo directory are always scanned.
      </p>
      <div>
        <label className="block text-xs font-medium text-gray-400 mb-1">
          Additional path to scan (optional)
        </label>
        <input
          type="text"
          value={customPath}
          onChange={(e) => setCustomPath(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') onScan() }}
          placeholder="/path/to/models"
          className="w-full px-3 py-2 text-sm bg-gray-100 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400"
        />
      </div>
      {scanning && (
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          Scanning for models...
        </div>
      )}
    </div>
  )
}

function SelectStep({
  results,
  selectedIds,
  toggleId,
  selectAll,
  selectNone,
  profiles,
  selectedProfile,
  setSelectedProfile,
}: {
  results: ScannedModel[]
  selectedIds: Set<string>
  toggleId: (id: string) => void
  selectAll: () => void
  selectNone: () => void
  profiles: { name: string; description: string }[]
  selectedProfile: string
  setSelectedProfile: (p: string) => void
}) {
  const newModels = results.filter((m) => !m.already_configured)
  const existingModels = results.filter((m) => m.already_configured)

  return (
    <div className="space-y-4">
      {/* Profile selector */}
      <div>
        <label className="block text-xs font-medium text-gray-400 mb-1">Profile to apply</label>
        <select
          value={selectedProfile}
          onChange={(e) => setSelectedProfile(e.target.value)}
          className="w-full px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100"
        >
          {profiles.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name} -- {p.description}
            </option>
          ))}
        </select>
      </div>

      {/* Selection controls */}
      <div className="flex items-center gap-3 text-xs">
        <span className="text-gray-400">{newModels.length} new models found</span>
        <button onClick={selectAll} className="text-primary hover:underline">Select all</button>
        <button onClick={selectNone} className="text-gray-400 hover:underline">Clear</button>
      </div>

      {/* Model list */}
      <div className="space-y-1 max-h-80 overflow-y-auto">
        {newModels.map((model) => (
          <ModelScanRow
            key={model.id}
            model={model}
            selected={selectedIds.has(model.id)}
            onToggle={() => toggleId(model.id)}
          />
        ))}
      </div>

      {existingModels.length > 0 && (
        <div className="text-xs text-gray-500">
          {existingModels.length} models already configured (skipped)
        </div>
      )}
    </div>
  )
}

function ModelScanRow({
  model,
  selected,
  onToggle,
}: {
  model: ScannedModel
  selected: boolean
  onToggle: () => void
}) {
  return (
    <button
      onClick={onToggle}
      className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors ${
        selected
          ? 'bg-primary/10 border border-primary/30'
          : 'bg-gray-50 dark:bg-gray-800/50 border border-transparent hover:border-gray-300 dark:hover:border-gray-600'
      }`}
    >
      <div className={`w-4 h-4 rounded border-2 flex items-center justify-center shrink-0 ${
        selected ? 'border-primary bg-primary' : 'border-gray-400'
      }`}>
        {selected && (
          <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
          </svg>
        )}
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">{model.id}</div>
        <div className="text-[11px] text-gray-400 flex gap-2">
          <span className="uppercase">{model.provider}</span>
          <span>{model.size_gb.toFixed(1)} GB</span>
          {model.vision && <span>vision</span>}
          {model.quantization && <span>{model.quantization}</span>}
        </div>
      </div>
    </button>
  )
}
