import { useEffect } from 'react'
import { useComparisonStore } from '../stores/comparisonStore'
import { LeftPanel } from './LeftPanel'
import { ComparisonGrid } from './ComparisonGrid'
import { EmptyState } from '../../../components/primitives/EmptyState'
import { ScaleIcon } from '../../../components/icons'
import { AppletLayout } from '../../../components/layout/AppletLayout'

export function ComparisonView() {
  const activeRunId = useComparisonStore((s) => s.activeRunId)
  const runs = useComparisonStore((s) => s.runs)
  const stopRun = useComparisonStore((s) => s.stopRun)

  const activeRun = activeRunId ? runs.find((r) => r.id === activeRunId) : null

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      const state = useComparisonStore.getState()
      if (state.activeRunId) {
        const run = state.runs.find((r) => r.id === state.activeRunId)
        if (run?.status === 'running') {
          state.stopRun(state.activeRunId)
        }
      }
    }
  }, [])

  // Keyboard: Escape stops active run
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return

      if (e.key === 'Escape' && activeRun?.status === 'running' && activeRunId) {
        e.preventDefault()
        stopRun(activeRunId)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [activeRun, activeRunId, stopRun])

  return (
    <AppletLayout leftPanel={<LeftPanel />} leftPanelWidth="w-80">
      {/* Main content */}
      <div className="flex-1 overflow-hidden flex items-center justify-center">
        {activeRun ? (
          <ComparisonGrid run={activeRun} />
        ) : (
          <EmptyState
            icon={<ScaleIcon className="w-8 h-8 text-gray-400" />}
            title="No comparison running"
            description="Select models, enter a prompt, and click Compare to see side-by-side results."
          />
        )}
      </div>
    </AppletLayout>
  )
}
