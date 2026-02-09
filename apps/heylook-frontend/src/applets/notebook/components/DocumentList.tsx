import { useState, useCallback } from 'react'
import { PlusIcon, TrashIcon } from '../../../components/icons'
import type { NotebookDocument } from '../types'
import clsx from 'clsx'

interface DocumentListProps {
  documents: NotebookDocument[]
  activeDocumentId: string | null
  onSelect: (id: string) => void
  onCreate: () => void
  onDelete: (id: string) => void
  onRename: (id: string, title: string) => void
}

function formatRelativeTime(timestamp: number): string {
  const diff = Date.now() - timestamp
  const seconds = Math.floor(diff / 1000)
  if (seconds < 60) return 'just now'
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

export function DocumentList({
  documents,
  activeDocumentId,
  onSelect,
  onCreate,
  onDelete,
  onRename,
}: DocumentListProps) {
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editTitle, setEditTitle] = useState('')

  const handleDoubleClick = useCallback(
    (doc: NotebookDocument) => {
      setEditingId(doc.id)
      setEditTitle(doc.title)
    },
    []
  )

  const handleRenameSubmit = useCallback(
    (id: string) => {
      if (editTitle.trim()) {
        onRename(id, editTitle.trim())
      }
      setEditingId(null)
    },
    [editTitle, onRename]
  )

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
          Documents
        </label>
        <button
          onClick={onCreate}
          className="text-xs text-primary hover:text-primary-hover flex items-center gap-1"
          title="New Document (Cmd+N)"
        >
          <PlusIcon className="w-3 h-3" />
          New
        </button>
      </div>

      {documents.length === 0 ? (
        <p className="text-xs text-gray-400 dark:text-gray-500 py-2">
          No documents yet
        </p>
      ) : (
        <div className="space-y-1">
          {documents.map((doc) => (
            <div
              key={doc.id}
              onClick={() => onSelect(doc.id)}
              onDoubleClick={() => handleDoubleClick(doc)}
              className={clsx(
                'group flex items-center justify-between px-2 py-1.5 rounded-lg cursor-pointer transition-colors',
                doc.id === activeDocumentId
                  ? 'bg-primary/10 text-primary'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
              )}
            >
              <div className="flex-1 min-w-0">
                {editingId === doc.id ? (
                  <input
                    type="text"
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onBlur={() => handleRenameSubmit(doc.id)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleRenameSubmit(doc.id)
                      if (e.key === 'Escape') setEditingId(null)
                    }}
                    autoFocus
                    className="w-full text-xs bg-transparent border-b border-primary outline-none"
                    onClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <>
                    <p className="text-xs font-medium truncate">{doc.title}</p>
                    <p className="text-[10px] text-gray-400 dark:text-gray-500">
                      {formatRelativeTime(doc.modifiedAt)}
                    </p>
                  </>
                )}
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onDelete(doc.id)
                }}
                className="ml-1 p-0.5 rounded text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                title="Delete document"
              >
                <TrashIcon className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
