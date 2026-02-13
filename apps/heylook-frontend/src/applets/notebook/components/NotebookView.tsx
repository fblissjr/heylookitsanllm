import { useEffect } from 'react'
import { LeftPanel } from './LeftPanel'
import { Editor } from './Editor'
import { useNotebookStore } from '../stores/notebookStore'
import { AppletLayout } from '../../../components/layout/AppletLayout'

export function NotebookView() {
  const loadFromDB = useNotebookStore((s) => s.loadFromDB)
  const createDocument = useNotebookStore((s) => s.createDocument)
  const documents = useNotebookStore((s) => s.documents)
  const loaded = useNotebookStore((s) => s.loaded)
  const stopGeneration = useNotebookStore((s) => s.stopGeneration)

  // Load persisted documents on mount
  useEffect(() => {
    loadFromDB()
  }, [loadFromDB])

  // Cmd+N: new document
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'n') {
        e.preventDefault()
        createDocument()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [createDocument])

  // Cmd+S: force save
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault()
        useNotebookStore.getState().saveToDB()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopGeneration()
    }
  }, [stopGeneration])

  // Auto-create a document if none exist after loading completes
  useEffect(() => {
    if (loaded && documents.length === 0) {
      createDocument()
    }
  }, [loaded, documents.length, createDocument])

  return (
    <AppletLayout leftPanel={<LeftPanel />} leftPanelWidth="w-72">
      <Editor />
    </AppletLayout>
  )
}
