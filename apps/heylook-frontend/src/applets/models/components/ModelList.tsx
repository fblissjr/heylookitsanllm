import clsx from 'clsx'
import { useModelsStore } from '../stores/modelsStore'
import type { AdminModelConfig } from '../types'

function statusPill(model: AdminModelConfig) {
  if (!model.enabled) return { label: 'Disabled', color: 'bg-gray-500' }
  if (model.loaded) return { label: 'Loaded', color: 'bg-green-500' }
  return { label: 'Available', color: 'bg-blue-500' }
}

function matchesSearch(model: AdminModelConfig, query: string): boolean {
  if (!query) return true
  const q = query.toLowerCase()
  return (
    model.id.toLowerCase().includes(q) ||
    (model.description?.toLowerCase().includes(q) ?? false) ||
    model.tags.some((t) => t.toLowerCase().includes(q)) ||
    model.provider.toLowerCase().includes(q)
  )
}

function matchesFilters(
  model: AdminModelConfig,
  filters: { provider: string[]; status: string[]; capability: string[] }
): boolean {
  if (filters.provider.length > 0 && !filters.provider.includes(model.provider)) return false
  if (filters.status.length > 0) {
    const status = !model.enabled ? 'disabled' : model.loaded ? 'loaded' : 'available'
    if (!filters.status.includes(status)) return false
  }
  if (filters.capability.length > 0) {
    if (!filters.capability.some((c) => model.capabilities.includes(c))) return false
  }
  return true
}

export function ModelList() {
  const configs = useModelsStore((s) => s.configs)
  const selectedId = useModelsStore((s) => s.selectedId)
  const setSelectedId = useModelsStore((s) => s.setSelectedId)
  const searchQuery = useModelsStore((s) => s.searchQuery)
  const setSearchQuery = useModelsStore((s) => s.setSearchQuery)
  const filters = useModelsStore((s) => s.filters)
  const setImportOpen = useModelsStore((s) => s.setImportOpen)

  const filtered = configs
    .filter((m) => matchesSearch(m, searchQuery))
    .filter((m) => matchesFilters(m, filters))

  return (
    <div className="flex flex-col h-full">
      {/* Search + Import */}
      <div className="p-3 space-y-2 border-b border-gray-200 dark:border-gray-700">
        <input
          type="text"
          placeholder="Search models..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 focus:outline-none focus:ring-1 focus:ring-primary text-gray-900 dark:text-gray-100 placeholder-gray-400"
        />
        <button
          onClick={() => setImportOpen(true)}
          className="w-full px-3 py-1.5 text-sm font-medium text-white bg-primary hover:bg-primary-hover rounded-lg transition-colors"
        >
          Import Models
        </button>
      </div>

      {/* Filter chips */}
      <FilterChips />

      {/* Model list */}
      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="p-4 text-center text-sm text-gray-400">
            {configs.length === 0 ? 'No models configured' : 'No models match filters'}
          </div>
        ) : (
          filtered.map((model) => {
            const pill = statusPill(model)
            return (
              <button
                key={model.id}
                onClick={() => setSelectedId(model.id)}
                className={clsx(
                  'w-full px-3 py-2.5 text-left border-b border-gray-100 dark:border-gray-800 transition-colors',
                  selectedId === model.id
                    ? 'bg-primary/10 border-l-2 border-l-primary'
                    : 'hover:bg-gray-50 dark:hover:bg-gray-800/50'
                )}
              >
                <div className="flex items-center justify-between mb-0.5">
                  <span className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate pr-2">
                    {model.id}
                  </span>
                  <span className={clsx('px-1.5 py-0.5 text-[10px] font-medium text-white rounded-full shrink-0', pill.color)}>
                    {pill.label}
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-[11px] text-gray-400">
                  <span className="uppercase">{model.provider}</span>
                  {model.config.vision as boolean && <span>| vision</span>}
                  {model.tags.length > 0 && (
                    <span className="truncate">| {model.tags.slice(0, 3).join(', ')}</span>
                  )}
                </div>
              </button>
            )
          })
        )}
      </div>

      {/* Footer count */}
      <div className="px-3 py-1.5 text-[11px] text-gray-400 border-t border-gray-200 dark:border-gray-700 shrink-0">
        {filtered.length} of {configs.length} models
      </div>
    </div>
  )
}

function FilterChips() {
  const filters = useModelsStore((s) => s.filters)
  const setFilters = useModelsStore((s) => s.setFilters)

  const hasFilters = filters.provider.length > 0 || filters.status.length > 0 || filters.capability.length > 0

  if (!hasFilters) return null

  return (
    <div className="flex flex-wrap gap-1 px-3 py-1.5 border-b border-gray-200 dark:border-gray-700">
      {filters.provider.map((p) => (
        <Chip
          key={`p-${p}`}
          label={p}
          onRemove={() => setFilters({ provider: filters.provider.filter((x) => x !== p) })}
        />
      ))}
      {filters.status.map((s) => (
        <Chip
          key={`s-${s}`}
          label={s}
          onRemove={() => setFilters({ status: filters.status.filter((x) => x !== s) })}
        />
      ))}
      {filters.capability.map((c) => (
        <Chip
          key={`c-${c}`}
          label={c}
          onRemove={() => setFilters({ capability: filters.capability.filter((x) => x !== c) })}
        />
      ))}
    </div>
  )
}

function Chip({ label, onRemove }: { label: string; onRemove: () => void }) {
  return (
    <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 text-[10px] font-medium bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded">
      {label}
      <button onClick={onRemove} className="ml-0.5 hover:text-accent-red">x</button>
    </span>
  )
}
