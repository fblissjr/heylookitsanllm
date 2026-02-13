import { useMemo } from 'react'
import clsx from 'clsx'
import { useModelsStore } from '../stores/modelsStore'
import type { AdminModelConfig, ModelFilter, SortConfig } from '../types'

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

function matchesFilters(model: AdminModelConfig, filters: ModelFilter): boolean {
  if (filters.provider.length > 0 && !filters.provider.includes(model.provider)) return false
  if (filters.status.length > 0) {
    const status = !model.enabled ? 'disabled' : model.loaded ? 'loaded' : 'available'
    if (!filters.status.includes(status)) return false
  }
  if (filters.capability.length > 0) {
    if (!filters.capability.some((c) => model.capabilities.includes(c))) return false
  }
  if (filters.tag.length > 0) {
    if (!filters.tag.some((t) => model.tags.includes(t))) return false
  }
  return true
}

function statusRank(model: AdminModelConfig): number {
  if (model.loaded) return 0
  if (model.enabled) return 1
  return 2
}

function sortModels(models: AdminModelConfig[], sort: SortConfig): AdminModelConfig[] {
  const dir = sort.direction === 'asc' ? 1 : -1
  return [...models].sort((a, b) => {
    switch (sort.field) {
      case 'name':
        return dir * a.id.localeCompare(b.id)
      case 'provider': {
        const cmp = a.provider.localeCompare(b.provider)
        return cmp !== 0 ? dir * cmp : a.id.localeCompare(b.id)
      }
      case 'status': {
        const cmp = statusRank(a) - statusRank(b)
        return cmp !== 0 ? dir * cmp : a.id.localeCompare(b.id)
      }
      default:
        return 0
    }
  })
}

type SortOption = { label: string; field: SortConfig['field']; direction: SortConfig['direction'] }

const SORT_OPTIONS: SortOption[] = [
  { label: 'Name (A-Z)', field: 'name', direction: 'asc' },
  { label: 'Name (Z-A)', field: 'name', direction: 'desc' },
  { label: 'Provider', field: 'provider', direction: 'asc' },
  { label: 'Status (Loaded first)', field: 'status', direction: 'asc' },
]

function sortOptionKey(opt: SortOption): string {
  return `${opt.field}:${opt.direction}`
}

export function ModelList() {
  const configs = useModelsStore((s) => s.configs)
  const selectedId = useModelsStore((s) => s.selectedId)
  const setSelectedId = useModelsStore((s) => s.setSelectedId)
  const searchQuery = useModelsStore((s) => s.searchQuery)
  const setSearchQuery = useModelsStore((s) => s.setSearchQuery)
  const filters = useModelsStore((s) => s.filters)
  const sortConfig = useModelsStore((s) => s.sortConfig)
  const setSortConfig = useModelsStore((s) => s.setSortConfig)
  const setImportOpen = useModelsStore((s) => s.setImportOpen)

  const allTags = useMemo(() => {
    const tags = new Set<string>()
    for (const m of configs) {
      for (const t of m.tags) tags.add(t)
    }
    return Array.from(tags).sort()
  }, [configs])

  const filtered = useMemo(() => {
    const result = configs
      .filter((m) => matchesSearch(m, searchQuery))
      .filter((m) => matchesFilters(m, filters))
    return sortModels(result, sortConfig)
  }, [configs, searchQuery, filters, sortConfig])

  const currentSortKey = sortOptionKey({ label: '', field: sortConfig.field, direction: sortConfig.direction })

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
        <div className="flex gap-2">
          <select
            value={currentSortKey}
            onChange={(e) => {
              const opt = SORT_OPTIONS.find((o) => sortOptionKey(o) === e.target.value)
              if (opt) setSortConfig({ field: opt.field, direction: opt.direction })
            }}
            className="flex-1 px-2 py-1.5 text-xs bg-gray-100 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 focus:outline-none focus:ring-1 focus:ring-primary"
          >
            {SORT_OPTIONS.map((opt) => (
              <option key={sortOptionKey(opt)} value={sortOptionKey(opt)}>
                {opt.label}
              </option>
            ))}
          </select>
          <button
            onClick={() => setImportOpen(true)}
            className="px-3 py-1.5 text-xs font-medium text-white bg-primary hover:bg-primary-hover rounded-lg transition-colors shrink-0"
          >
            Import
          </button>
        </div>
      </div>

      {/* Tag filter chips */}
      {allTags.length > 0 && <TagChips allTags={allTags} />}

      {/* Active filter chips */}
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

function TagChips({ allTags }: { allTags: string[] }) {
  const filters = useModelsStore((s) => s.filters)
  const setFilters = useModelsStore((s) => s.setFilters)
  const activeTags = filters.tag

  function toggleTag(tag: string) {
    const next = activeTags.includes(tag)
      ? activeTags.filter((t) => t !== tag)
      : [...activeTags, tag]
    setFilters({ tag: next })
  }

  return (
    <div className="flex flex-wrap gap-1 px-3 py-1.5 border-b border-gray-200 dark:border-gray-700">
      {allTags.map((tag) => (
        <button
          key={tag}
          onClick={() => toggleTag(tag)}
          className={clsx(
            'px-1.5 py-0.5 text-[10px] font-medium rounded transition-colors',
            activeTags.includes(tag)
              ? 'bg-primary text-white'
              : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-300 dark:hover:bg-gray-600'
          )}
        >
          {tag}
        </button>
      ))}
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
