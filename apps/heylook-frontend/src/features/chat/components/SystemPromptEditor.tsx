import { useState, useCallback, useRef, useEffect } from 'react'
import clsx from 'clsx'

interface SystemPromptEditorProps {
  systemPrompt: string | undefined
  conversationId: string  // Kept for future use (e.g., tracking per-conversation edit state)
  onUpdate: (systemPrompt: string, shouldRegenerate: boolean) => Promise<void>
  disabled?: boolean
  hasMessages: boolean
}

export function SystemPromptEditor({
  systemPrompt,
  conversationId: _conversationId,
  onUpdate,
  disabled,
  hasMessages
}: SystemPromptEditorProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editValue, setEditValue] = useState(systemPrompt || '')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Sync edit value when systemPrompt changes externally
  useEffect(() => {
    if (!isEditing) {
      setEditValue(systemPrompt || '')
    }
  }, [systemPrompt, isEditing])

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current && isEditing) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [editValue, isEditing])

  const handleStartEdit = useCallback(() => {
    setEditValue(systemPrompt || '')
    setIsEditing(true)
    setIsExpanded(true)
    // Focus textarea after render
    setTimeout(() => textareaRef.current?.focus(), 0)
  }, [systemPrompt])

  const handleCancel = useCallback(() => {
    setEditValue(systemPrompt || '')
    setIsEditing(false)
  }, [systemPrompt])

  const handleSave = useCallback(async (shouldRegenerate: boolean) => {
    await onUpdate(editValue, shouldRegenerate)
    setIsEditing(false)
  }, [editValue, onUpdate])

  const hasPrompt = Boolean(systemPrompt?.trim())
  const hasChanges = editValue !== (systemPrompt || '')

  return (
    <div className={clsx(
      'border rounded-lg transition-all',
      isExpanded || isEditing
        ? 'bg-amber-50/50 dark:bg-amber-900/10 border-amber-200 dark:border-amber-800'
        : hasPrompt
          ? 'bg-gray-50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-700'
          : 'bg-gray-50/50 dark:bg-gray-800/30 border-dashed border-gray-300 dark:border-gray-600'
    )}>
      {/* Header - always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={clsx(
          'w-full flex items-center gap-2 px-3 py-2 text-sm transition-colors',
          hasPrompt
            ? 'text-gray-700 dark:text-gray-300'
            : 'text-gray-500 dark:text-gray-400'
        )}
        disabled={disabled}
      >
        {/* Expand/collapse chevron */}
        <svg
          className={clsx(
            'w-4 h-4 transition-transform shrink-0',
            isExpanded && 'rotate-90'
          )}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>

        {/* Icon */}
        <svg className="w-4 h-4 shrink-0 opacity-60" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>

        {/* Label */}
        <span className="flex-1 text-left font-medium">
          System Prompt
        </span>

        {/* Status indicator */}
        {hasPrompt ? (
          <span className="text-xs text-amber-600 dark:text-amber-400">
            Active
          </span>
        ) : (
          <span className="text-xs text-gray-400 dark:text-gray-500">
            Using model default
          </span>
        )}
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-3 pb-3 space-y-2">
          {isEditing ? (
            <>
              <textarea
                ref={textareaRef}
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                className={clsx(
                  'w-full px-3 py-2 rounded border resize-none',
                  'bg-white dark:bg-gray-900',
                  'border-amber-300 dark:border-amber-700',
                  'focus:outline-none focus:ring-2 focus:ring-amber-500/50',
                  'text-sm text-gray-800 dark:text-gray-200',
                  'placeholder:text-gray-400'
                )}
                placeholder="Enter a system prompt to customize the model's behavior..."
                rows={3}
                disabled={disabled}
              />
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {hasMessages && hasChanges && 'Changing will affect future responses'}
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={handleCancel}
                    className="px-3 py-1.5 text-sm rounded border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-800"
                    disabled={disabled}
                  >
                    Cancel
                  </button>
                  {hasMessages && hasChanges ? (
                    <>
                      <button
                        onClick={() => handleSave(false)}
                        className="px-3 py-1.5 text-sm rounded bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600"
                        disabled={disabled}
                      >
                        Save Only
                      </button>
                      <button
                        onClick={() => handleSave(true)}
                        className="px-3 py-1.5 text-sm rounded bg-amber-500 text-white hover:bg-amber-600"
                        disabled={disabled}
                      >
                        Save & Regenerate
                      </button>
                    </>
                  ) : (
                    <button
                      onClick={() => handleSave(false)}
                      className="px-3 py-1.5 text-sm rounded bg-primary text-white hover:bg-primary/90"
                      disabled={disabled || !hasChanges}
                    >
                      Save
                    </button>
                  )}
                </div>
              </div>
            </>
          ) : (
            <>
              {hasPrompt ? (
                <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                  {systemPrompt}
                </p>
              ) : (
                <p className="text-sm text-gray-500 dark:text-gray-400 italic">
                  No system prompt set. The model will use its default behavior.
                </p>
              )}
              <button
                onClick={handleStartEdit}
                className={clsx(
                  'px-3 py-1.5 text-sm rounded transition-colors',
                  'bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600',
                  'text-gray-700 dark:text-gray-300'
                )}
                disabled={disabled}
              >
                {hasPrompt ? 'Edit' : 'Add System Prompt'}
              </button>
            </>
          )}
        </div>
      )}
    </div>
  )
}
