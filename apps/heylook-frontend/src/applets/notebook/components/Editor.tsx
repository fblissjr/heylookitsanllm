import { useRef, useCallback, useEffect, useState } from 'react'
import { StopIcon } from '../../../components/icons'
import { useModelStore } from '../../../stores/modelStore'
import { useNotebookStore } from '../stores/notebookStore'

export function Editor() {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const [cursorPosition, setCursorPosition] = useState(0)

  const activeDocumentId = useNotebookStore((s) => s.activeDocumentId)
  const documents = useNotebookStore((s) => s.documents)
  const generation = useNotebookStore((s) => s.generation)
  const error = useNotebookStore((s) => s.error)
  const updateContent = useNotebookStore((s) => s.updateContent)
  const startGeneration = useNotebookStore((s) => s.startGeneration)
  const stopGeneration = useNotebookStore((s) => s.stopGeneration)

  const loadedModel = useModelStore((s) => s.loadedModel)

  const activeDoc = activeDocumentId
    ? documents.find((d) => d.id === activeDocumentId)
    : null

  const isGenerating = generation?.isGenerating ?? false

  const handleContentChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      if (!activeDocumentId) return
      updateContent(activeDocumentId, e.target.value)
    },
    [activeDocumentId, updateContent]
  )

  const handleCursorChange = useCallback(() => {
    if (textareaRef.current) {
      setCursorPosition(textareaRef.current.selectionStart)
    }
  }, [])

  const handleGenerate = useCallback(() => {
    const pos = textareaRef.current?.selectionStart ?? cursorPosition
    startGeneration(pos)
  }, [cursorPosition, startGeneration])

  // Move cursor to end of generated text when generation completes
  useEffect(() => {
    if (generation && !generation.isGenerating && textareaRef.current) {
      const endPos = generation.insertPosition + generation.generatedLength
      textareaRef.current.selectionStart = endPos
      textareaRef.current.selectionEnd = endPos
      textareaRef.current.focus()
    }
  }, [generation?.isGenerating])

  // Keyboard shortcuts
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // Cmd+Enter: generate
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault()
        if (isGenerating) {
          stopGeneration()
        } else {
          handleGenerate()
        }
      }
      // Escape: stop generation
      if (e.key === 'Escape' && isGenerating) {
        e.preventDefault()
        stopGeneration()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isGenerating, stopGeneration, handleGenerate])

  if (!activeDoc) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-400 dark:text-gray-500">
        <div className="text-center">
          <p className="text-sm">No document selected</p>
          <p className="text-xs mt-1">Create a new document or select one from the sidebar</p>
        </div>
      </div>
    )
  }

  const contextChars = textareaRef.current
    ? textareaRef.current.selectionStart
    : cursorPosition

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Editor area */}
      <div className="flex-1 overflow-hidden">
        <textarea
          ref={textareaRef}
          value={activeDoc.content}
          onChange={handleContentChange}
          onSelect={handleCursorChange}
          onClick={handleCursorChange}
          onKeyUp={handleCursorChange}
          placeholder="Start typing... The model will continue from your cursor position."
          disabled={isGenerating}
          className="w-full h-full resize-none p-6 bg-transparent text-gray-100 font-mono text-sm leading-relaxed placeholder:text-gray-600 focus:outline-none disabled:opacity-70"
          spellCheck={false}
        />
      </div>

      {/* Status bar + controls */}
      <div className="flex-shrink-0 border-t border-gray-700 px-4 py-2 flex items-center justify-between">
        {/* Status */}
        <div className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-3">
          {isGenerating ? (
            <>
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span>
                Generating from cursor ({contextChars.toLocaleString()} chars of context)
                {generation && ` -- ${generation.generatedLength} chars`}
              </span>
            </>
          ) : (
            <>
              <span>{activeDoc.content.length.toLocaleString()} chars</span>
              <span>cursor: {contextChars.toLocaleString()}</span>
              {loadedModel && <span>{loadedModel.id}</span>}
            </>
          )}
        </div>

        {/* Error */}
        {error && (
          <span className="text-xs text-red-400 mx-3 truncate max-w-xs">
            {error}
          </span>
        )}

        {/* Generate / Stop button */}
        <div className="flex items-center gap-2">
          {isGenerating ? (
            <button
              onClick={stopGeneration}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
            >
              <StopIcon className="w-3.5 h-3.5" />
              Stop
              <kbd className="ml-1 text-[10px] opacity-60">Esc</kbd>
            </button>
          ) : (
            <button
              onClick={handleGenerate}
              disabled={!loadedModel || !activeDoc.content.trim()}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-primary/20 text-primary hover:bg-primary/30 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Generate
              <kbd className="ml-1 text-[10px] opacity-60">Cmd+Enter</kbd>
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
