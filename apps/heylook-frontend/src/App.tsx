import { useEffect, useState, lazy, Suspense } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { AppShell } from './components/layout/AppShell'
import { Layout } from './components/layout/Layout'
import { ChatView } from './applets/chat/components/ChatView'
import { ConfirmDeleteModal } from './applets/chat/components/ConfirmDeleteModal'
import { useModelStore } from './stores/modelStore'

const BatchView = lazy(() =>
  import('./applets/batch').then((m) => ({ default: m.BatchView }))
)

const TokenExplorerView = lazy(() =>
  import('./applets/token-explorer').then((m) => ({ default: m.TokenExplorerView }))
)

const ComparisonView = lazy(() =>
  import('./applets/model-comparison').then((m) => ({ default: m.ComparisonView }))
)

const PerformanceView = lazy(() =>
  import('./applets/performance').then((m) => ({ default: m.PerformanceView }))
)

const NotebookView = lazy(() =>
  import('./applets/notebook').then((m) => ({ default: m.NotebookView }))
)

function LazyFallback() {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
    </div>
  )
}

function App() {
  const [isConnected, setIsConnected] = useState<boolean | null>(null)
  const fetchModels = useModelStore((state) => state.fetchModels)
  const fetchCapabilities = useModelStore((state) => state.fetchCapabilities)

  useEffect(() => {
    // Check connection and fetch models + capabilities on startup
    const init = async () => {
      try {
        await fetchModels()
        // Fetch capabilities in parallel (non-blocking)
        fetchCapabilities()
        setIsConnected(true)
      } catch {
        setIsConnected(false)
      }
    }
    init()
  }, [fetchModels, fetchCapabilities])

  if (isConnected === null) {
    return (
      <div className="h-screen flex items-center justify-center bg-background-dark">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          <p className="text-gray-400 text-sm">Connecting to server...</p>
        </div>
      </div>
    )
  }

  if (isConnected === false) {
    return (
      <div className="h-screen flex items-center justify-center bg-background-dark">
        <div className="flex flex-col items-center gap-4 text-center px-4">
          <div className="w-12 h-12 rounded-full bg-accent-red/20 flex items-center justify-center">
            <svg className="w-6 h-6 text-accent-red" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <div>
            <h2 className="text-xl font-semibold text-white mb-2">Connection Failed</h2>
            <p className="text-gray-400 text-sm max-w-md">
              Could not connect to the heylookitsanllm server at localhost:8080.
              Make sure the server is running with <code className="text-primary">heylookllm</code>
            </p>
          </div>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded-lg transition-colors"
          >
            Retry Connection
          </button>
        </div>
      </div>
    )
  }

  return (
    <>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/chat" element={
            <Layout>
              <ChatView />
            </Layout>
          } />
          <Route path="/batch" element={
            <Suspense fallback={<LazyFallback />}>
              <BatchView />
            </Suspense>
          } />
          <Route path="/explore" element={
            <Suspense fallback={<LazyFallback />}>
              <TokenExplorerView />
            </Suspense>
          } />
          <Route path="/compare" element={
            <Suspense fallback={<LazyFallback />}>
              <ComparisonView />
            </Suspense>
          } />
          <Route path="/perf" element={
            <Suspense fallback={<LazyFallback />}>
              <PerformanceView />
            </Suspense>
          } />
          <Route path="/notebook" element={
            <Suspense fallback={<LazyFallback />}>
              <NotebookView />
            </Suspense>
          } />
          <Route path="*" element={<Navigate to="/chat" replace />} />
        </Route>
      </Routes>
      <ConfirmDeleteModal />
    </>
  )
}

export default App
