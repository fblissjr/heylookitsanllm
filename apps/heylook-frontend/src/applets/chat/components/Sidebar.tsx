import { useRef, useState, useCallback } from 'react'
import { useChatStore } from '../stores/chatStore'
import { useModelStore } from '../../../stores/modelStore'
import { useUIStore } from '../../../stores/uiStore'
import { useSettingsStore } from '../../../stores/settingsStore'
import { exportConversations, importConversations } from '../../../lib/db'
import { useLongPress } from '../../../hooks/useLongPress'
import clsx from 'clsx'
import type { Conversation } from '../../../types/chat'

export function Sidebar() {
  const { conversations, activeConversationId, createConversation, setActiveConversation, loadFromDB } = useChatStore()
  const { loadedModel } = useModelStore()
  const { setConfirmDelete, isMobile, toggleSidebar } = useUIStore()
  const { systemPrompt } = useSettingsStore()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [importStatus, setImportStatus] = useState<string | null>(null)

  // Multi-select state
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [isSelectionMode, setIsSelectionMode] = useState(false)

  const handleNewConversation = () => {
    if (loadedModel) {
      createConversation(loadedModel.id, systemPrompt)
      if (isMobile) toggleSidebar()
    }
  }

  const handleSelectConversation = (id: string) => {
    if (isSelectionMode) {
      toggleSelection(id)
    } else {
      setActiveConversation(id)
      if (isMobile) toggleSidebar()
    }
  }

  const handleDeleteClick = (e: React.MouseEvent, id: string, title: string) => {
    e.stopPropagation()
    setConfirmDelete({ type: 'conversation', id, title })
  }

  // Multi-select handlers
  const toggleSelection = useCallback((id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      // Exit selection mode if nothing selected
      if (next.size === 0) {
        setIsSelectionMode(false)
      }
      return next
    })
  }, [])

  const enterSelectionMode = useCallback((id: string) => {
    setIsSelectionMode(true)
    setSelectedIds(new Set([id]))
  }, [])

  const exitSelectionMode = useCallback(() => {
    setIsSelectionMode(false)
    setSelectedIds(new Set())
  }, [])

  const selectAll = useCallback(() => {
    setSelectedIds(new Set(conversations.map(c => c.id)))
  }, [conversations])

  const handleBulkDelete = useCallback(() => {
    if (selectedIds.size === 0) return

    // Get titles for confirmation
    const selectedConvs = conversations.filter(c => selectedIds.has(c.id))
    const title = selectedConvs.length === 1
      ? selectedConvs[0].title
      : `${selectedConvs.length} conversations`

    setConfirmDelete({
      type: 'bulk',
      ids: Array.from(selectedIds),
      title,
      onComplete: exitSelectionMode,  // Exit selection mode after modal closes
    })
  }, [selectedIds, conversations, setConfirmDelete, exitSelectionMode])

  const handleCheckboxClick = (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    if (!isSelectionMode) {
      enterSelectionMode(id)
    } else {
      toggleSelection(id)
    }
  }

  const handleExport = async () => {
    try {
      const json = await exportConversations()
      const blob = new Blob([json], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `heylook-conversations-${new Date().toISOString().split('T')[0]}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Export failed:', error)
    }
  }

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    try {
      const json = await file.text()
      const count = await importConversations(json)
      await loadFromDB() // Refresh conversations from DB
      setImportStatus(`Imported ${count} conversations`)
      setTimeout(() => setImportStatus(null), 3000)
    } catch (error) {
      console.error('Import failed:', error)
      setImportStatus('Import failed')
      setTimeout(() => setImportStatus(null), 3000)
    }

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
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
      {/* Header - New conversation or Selection mode actions */}
      <div className="p-3">
        {isSelectionMode ? (
          <div className="flex items-center gap-2">
            <button
              onClick={exitSelectionMode}
              className="p-2 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300"
              aria-label="Cancel selection"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <span className="flex-1 text-sm font-medium text-gray-700 dark:text-gray-300">
              {selectedIds.size} selected
            </span>
            <button
              onClick={selectAll}
              className="px-2 py-1 text-xs rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300"
            >
              All
            </button>
          </div>
        ) : (
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
        )}
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto">
        {Object.entries(groupedConversations).map(([date, convs]) => (
          <div key={date}>
            <div className="px-4 py-2 text-xs font-medium text-gray-400 dark:text-gray-500 uppercase tracking-wider">
              {date}
            </div>
            {convs.map(conv => (
              <ConversationItem
                key={conv.id}
                conversation={conv}
                isActive={activeConversationId === conv.id}
                isSelected={selectedIds.has(conv.id)}
                isSelectionMode={isSelectionMode}
                isMobile={isMobile}
                onSelect={handleSelectConversation}
                onDelete={handleDeleteClick}
                onCheckboxClick={handleCheckboxClick}
                onLongPress={enterSelectionMode}
              />
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

      {/* Bulk action bar - appears when items selected */}
      {isSelectionMode && selectedIds.size > 0 && (
        <div className="p-3 border-t border-gray-200 dark:border-gray-700 bg-red-50 dark:bg-red-900/20">
          <button
            onClick={handleBulkDelete}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-red-500 hover:bg-red-600 text-white transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            <span className="text-sm font-medium">
              Delete {selectedIds.size} {selectedIds.size === 1 ? 'conversation' : 'conversations'}
            </span>
          </button>
        </div>
      )}

      {/* Footer with export/import and model info */}
      {!isSelectionMode && (
        <div className="p-3 border-t border-gray-200 dark:border-gray-700 space-y-2">
          {/* Export/Import buttons */}
          <div className="flex gap-2">
            <button
              onClick={handleExport}
              disabled={conversations.length === 0}
              className={clsx(
                'flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
                conversations.length === 0
                  ? 'bg-gray-100 dark:bg-gray-800 text-gray-400 cursor-not-allowed'
                  : 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300'
              )}
              title="Export conversations"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              Export
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300 transition-colors"
              title="Import conversations"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Import
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleImport}
              className="hidden"
            />
          </div>

          {/* Import status message */}
          {importStatus && (
            <div className="text-xs text-center text-accent-green">
              {importStatus}
            </div>
          )}

          {/* Model info */}
          {loadedModel && (
            <div className="text-xs text-gray-400 dark:text-gray-500 pt-1 border-t border-gray-200 dark:border-gray-700">
              <span className="block truncate font-medium text-gray-600 dark:text-gray-300">
                {loadedModel.id}
              </span>
              <span>
                {loadedModel.capabilities.vision && 'Vision '}
                {loadedModel.capabilities.thinking && 'Thinking '}
              </span>
            </div>
          )}
        </div>
      )}
    </aside>
  )
}

// Extracted conversation item component for cleaner long-press handling
interface ConversationItemProps {
  conversation: Conversation
  isActive: boolean
  isSelected: boolean
  isSelectionMode: boolean
  isMobile: boolean
  onSelect: (id: string) => void
  onDelete: (e: React.MouseEvent, id: string, title: string) => void
  onCheckboxClick: (e: React.MouseEvent, id: string) => void
  onLongPress: (id: string) => void
}

function ConversationItem({
  conversation,
  isActive,
  isSelected,
  isSelectionMode,
  isMobile,
  onSelect,
  onDelete,
  onCheckboxClick,
  onLongPress,
}: ConversationItemProps) {
  const longPressHandlers = useLongPress({
    onLongPress: () => onLongPress(conversation.id),
    onClick: () => onSelect(conversation.id),
    delay: 500,
  })

  return (
    <div
      className={clsx(
        'w-full flex items-center gap-2 px-3 py-3 text-left transition-colors group cursor-pointer',
        isSelected
          ? 'bg-primary/20 text-primary'
          : isActive
            ? 'bg-primary/10 text-primary'
            : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300'
      )}
      {...(isMobile ? longPressHandlers : {})}
      onClick={isMobile ? undefined : () => onSelect(conversation.id)}
      onKeyDown={(e) => e.key === 'Enter' && onSelect(conversation.id)}
      role="button"
      tabIndex={0}
    >
      {/* Checkbox - visible in selection mode or on hover (desktop) */}
      <div
        className={clsx(
          'shrink-0 transition-all',
          isSelectionMode
            ? 'opacity-100 w-5'
            : isMobile
              ? 'opacity-0 w-0'
              : 'opacity-0 w-0 group-hover:opacity-100 group-hover:w-5'
        )}
        onClick={(e) => onCheckboxClick(e, conversation.id)}
        onTouchStart={(e) => e.stopPropagation()}
        onKeyDown={(e) => e.key === 'Enter' && onCheckboxClick(e as unknown as React.MouseEvent, conversation.id)}
        role="checkbox"
        aria-checked={isSelected}
        aria-label={`Select ${conversation.title}`}
        tabIndex={isSelectionMode ? 0 : -1}
      >
        <div
          className={clsx(
            'w-4 h-4 rounded border-2 flex items-center justify-center transition-colors',
            isSelected
              ? 'bg-primary border-primary'
              : 'border-gray-400 dark:border-gray-500 hover:border-primary'
          )}
        >
          {isSelected && (
            <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
            </svg>
          )}
        </div>
      </div>

      {/* Chat icon - hidden in selection mode */}
      {!isSelectionMode && (
        <svg className="w-5 h-5 shrink-0 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
        </svg>
      )}

      <span className="flex-1 truncate text-sm">{conversation.title}</span>

      {/* Delete button - hidden in selection mode */}
      {!isSelectionMode && (
        <button
          onClick={(e) => onDelete(e, conversation.id, conversation.title)}
          onTouchStart={(e) => e.stopPropagation()}
          className={clsx(
            'p-1 rounded hover:bg-red-100 dark:hover:bg-red-900/30 text-gray-400 hover:text-red-500 transition-all',
            isMobile ? 'opacity-70' : 'opacity-0 group-hover:opacity-100'
          )}
          aria-label="Delete conversation"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
      )}
    </div>
  )
}
