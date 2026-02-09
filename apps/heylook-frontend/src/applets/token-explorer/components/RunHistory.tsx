import { useCallback } from 'react'
import { useExplorerStore } from '../stores/explorerStore'
import { RunHistoryCard } from './RunHistoryCard'
import { RunHistoryList } from '../../../components/composed/RunHistoryList'
import type { ExplorerRun } from '../types'

export function RunHistory() {
  const runs = useExplorerStore((s) => s.runs)
  const activeRunId = useExplorerStore((s) => s.activeRunId)
  const selectRun = useExplorerStore((s) => s.selectRun)
  const removeRun = useExplorerStore((s) => s.removeRun)
  const clearRuns = useExplorerStore((s) => s.clearRuns)

  const handleSelect = useCallback((id: string) => selectRun(id), [selectRun])
  const handleRemove = useCallback((id: string) => removeRun(id), [removeRun])

  const renderCard = useCallback(
    (run: ExplorerRun, isActive: boolean) => (
      <RunHistoryCard
        run={run}
        isActive={isActive}
        onSelect={handleSelect}
        onRemove={handleRemove}
      />
    ),
    [handleSelect, handleRemove]
  )

  return (
    <RunHistoryList
      runs={runs}
      activeRunId={activeRunId}
      onSelect={handleSelect}
      onRemove={handleRemove}
      onClear={clearRuns}
      renderCard={renderCard}
    />
  )
}
