interface EmptyStateProps {
  type: 'no-model' | 'no-conversation'
}

export function EmptyState({ type }: EmptyStateProps) {
  if (type === 'no-model') {
    return (
      <div className="text-center max-w-md">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 dark:bg-surface-dark flex items-center justify-center">
          <svg className="w-8 h-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
        </div>
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
          No Model Loaded
        </h3>
        <p className="text-gray-500 dark:text-gray-400 text-sm">
          Select a model from the header to start chatting. The model will be loaded into memory for inference.
        </p>
      </div>
    )
  }

  return (
    <div className="text-center max-w-md">
      <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 dark:bg-surface-dark flex items-center justify-center">
        <svg className="w-8 h-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
        </svg>
      </div>
      <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
        Start a New Conversation
      </h3>
      <p className="text-gray-500 dark:text-gray-400 text-sm">
        Click "New Chat" in the sidebar or start typing below to begin.
      </p>
    </div>
  )
}
