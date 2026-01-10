import { useChatStore } from '../../stores/chatStore'
import { useModelStore } from '../../stores/modelStore'
import { useUIStore } from '../../stores/uiStore'
import clsx from 'clsx'

export function Sidebar() {
  const { conversations, activeConversationId, createConversation, setActiveConversation } = useChatStore()
  const { loadedModel } = useModelStore()
  const { setConfirmDelete, isMobile, toggleSidebar } = useUIStore()

  const handleNewConversation = () => {
    if (loadedModel) {
      createConversation(loadedModel.id)
      if (isMobile) toggleSidebar()
    }
  }

  const handleSelectConversation = (id: string) => {
    setActiveConversation(id)
    if (isMobile) toggleSidebar()
  }

  const handleDeleteClick = (e: React.MouseEvent, id: string, title: string) => {
    e.stopPropagation()
    setConfirmDelete({ type: 'conversation', id, title })
  }

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))

    if (days === 0) return 'Today'
    if (days === 1) return 'Yesterday'
    if (days < 7) return `${days} days ago`
    return date.toLocaleDateString()
  }

  // Group conversations by date
  const groupedConversations = conversations.reduce((acc, conv) => {
    const dateKey = formatDate(conv.updatedAt)
    if (!acc[dateKey]) acc[dateKey] = []
    acc[dateKey].push(conv)
    return acc
  }, {} as Record<string, typeof conversations>)

  return (
    <aside className="w-64 h-full bg-gray-50 dark:bg-surface-dark border-r border-gray-200 dark:border-gray-800 flex flex-col">
      {/* New conversation button */}
      <div className="p-3">
        <button
          onClick={handleNewConversation}
          disabled={!loadedModel}
          className={clsx(
            'w-full flex items-center gap-2 px-4 py-2.5 rounded-lg transition-colors',
            loadedModel
              ? 'bg-primary hover:bg-primary-hover text-white'
              : 'bg-gray-200 dark:bg-gray-700 text-gray-400 cursor-not-allowed'
          )}
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          <span className="text-sm font-medium">New Chat</span>
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto">
        {Object.entries(groupedConversations).map(([date, convs]) => (
          <div key={date}>
            <div className="px-4 py-2 text-xs font-medium text-gray-400 dark:text-gray-500 uppercase tracking-wider">
              {date}
            </div>
            {convs.map(conv => (
              <button
                key={conv.id}
                onClick={() => handleSelectConversation(conv.id)}
                className={clsx(
                  'w-full flex items-center gap-3 px-4 py-3 text-left transition-colors group',
                  activeConversationId === conv.id
                    ? 'bg-primary/10 text-primary'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300'
                )}
              >
                <svg className="w-5 h-5 shrink-0 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <span className="flex-1 truncate text-sm">{conv.title}</span>
                <button
                  onClick={(e) => handleDeleteClick(e, conv.id, conv.title)}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-100 dark:hover:bg-red-900/30 text-gray-400 hover:text-red-500 transition-all"
                  aria-label="Delete conversation"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </button>
            ))}
          </div>
        ))}

        {conversations.length === 0 && (
          <div className="p-4 text-center text-gray-400 dark:text-gray-500 text-sm">
            {loadedModel
              ? 'No conversations yet. Start a new chat!'
              : 'Load a model to start chatting.'}
          </div>
        )}
      </div>

      {/* Footer with model info */}
      {loadedModel && (
        <div className="p-3 border-t border-gray-200 dark:border-gray-700">
          <div className="text-xs text-gray-400 dark:text-gray-500">
            <span className="block truncate font-medium text-gray-600 dark:text-gray-300">
              {loadedModel.id}
            </span>
            <span>
              {loadedModel.capabilities.vision && 'Vision '}
              {loadedModel.capabilities.thinking && 'Thinking '}
            </span>
          </div>
        </div>
      )}
    </aside>
  )
}
