import { useCallback, useEffect } from 'react'
import { ExplorerForm } from './ExplorerForm'
import { TokenStream } from './TokenStream'
import { TokenDetail } from './TokenDetail'
import { RunHistory } from './RunHistory'
import { useExplorerStore } from '../stores/explorerStore'
import { AppletLayout } from '../../../components/layout/AppletLayout'

export function TokenExplorerView() {
  const activeRunId = useExplorerStore((s) => s.activeRunId)
  const runs = useExplorerStore((s) => s.runs)
  const selectedTokenIndex = useExplorerStore((s) => s.selectedTokenIndex)
  const selectToken = useExplorerStore((s) => s.selectToken)
  const stopRun = useExplorerStore((s) => s.stopRun)
  const activeRun = activeRunId ? runs.find((r) => r.id === activeRunId) : null

  const selectedToken =
    activeRun && selectedTokenIndex !== null
      ? activeRun.tokens[selectedTokenIndex] ?? null
      : null

  const handleTokenClick = useCallback(
    (index: number) => {
      selectToken(selectedTokenIndex === index ? null : index)
    },
    [selectToken, selectedTokenIndex],
  )

  // Keyboard navigation: arrows to move between tokens, Escape to deselect
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (!activeRun || activeRun.tokens.length === 0) return

      // Don't handle if focus is in an input/textarea
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return

      if (e.key === 'ArrowRight') {
        e.preventDefault()
        const next = selectedTokenIndex === null ? 0 : Math.min(selectedTokenIndex + 1, activeRun.tokens.length - 1)
        selectToken(next)
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        const prev = selectedTokenIndex === null ? 0 : Math.max(selectedTokenIndex - 1, 0)
        selectToken(prev)
      } else if (e.key === 'Escape') {
        e.preventDefault()
        selectToken(null)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [activeRun, selectedTokenIndex, selectToken])

  // Cleanup AbortController on unmount
  useEffect(() => {
    return () => {
      stopRun()
    }
  }, [stopRun])

  const leftPanel = (
    <>
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
          Token Explorer
        </h1>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
          Visualize token probabilities as they stream
        </p>
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        <ExplorerForm />
        <RunHistory />
      </div>
    </>
  )

  return (
    <AppletLayout leftPanel={leftPanel} leftPanelWidth="w-80">
      {/* Main panel: token stream + detail */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {activeRun ? (
          <>
            {/* Status bar */}
            <div className="px-6 py-2 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
              <div className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-2">
                {activeRun.status === 'streaming' && (
                  <>
                    <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                    Streaming...
                  </>
                )}
                {activeRun.status === 'completed' && (
                  <span>{activeRun.tokens.length} tokens</span>
                )}
                {activeRun.status === 'stopped' && (
                  <span>{activeRun.tokens.length} tokens (stopped)</span>
                )}
                {activeRun.status === 'error' && (
                  <span className="text-red-500">Error: {activeRun.error}</span>
                )}
              </div>
              <div className="flex items-center gap-3 text-xs text-gray-400">
                {activeRun.totalDuration && (
                  <span>{(activeRun.totalDuration / 1000).toFixed(1)}s</span>
                )}
                {selectedTokenIndex !== null && (
                  <span className="text-gray-500 dark:text-gray-400">
                    Use arrow keys to navigate, Esc to deselect
                  </span>
                )}
              </div>
            </div>

            {/* Token stream */}
            <div className="flex-1 overflow-y-auto p-6">
              <div className="max-w-4xl">
                <TokenStream
                  tokens={activeRun.tokens}
                  selectedIndex={selectedTokenIndex}
                  isStreaming={activeRun.status === 'streaming'}
                  thinkingTokenCount={activeRun.thinkingTokenCount}
                  onTokenClick={handleTokenClick}
                />
              </div>
            </div>

            {/* Token detail panel */}
            {selectedToken && <TokenDetail token={selectedToken} />}
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-400 dark:text-gray-500">
            <div className="text-center">
              <p className="text-sm">Enter a prompt and click Explore Tokens to begin</p>
              <p className="text-xs mt-1">Tokens will stream in with probability coloring</p>
            </div>
          </div>
        )}
      </div>
    </AppletLayout>
  )
}
