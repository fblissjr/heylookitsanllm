import { useCallback } from 'react'
import { useUIStore } from '../../stores/uiStore'
import { useChatStore } from '../../stores/chatStore'

export function ConfirmDeleteModal() {
  const { activeModal, confirmDelete, closeModal } = useUIStore()
  const { deleteConversation, deleteMessageWithCascade } = useChatStore()

  const handleConfirm = useCallback(async (regenerate = false) => {
    if (confirmDelete.type === 'conversation' && confirmDelete.id) {
      deleteConversation(confirmDelete.id)
    } else if (confirmDelete.type === 'message' && confirmDelete.id && confirmDelete.conversationId) {
      await deleteMessageWithCascade(confirmDelete.conversationId, confirmDelete.id, regenerate)
    }
    closeModal()
  }, [confirmDelete, deleteConversation, deleteMessageWithCascade, closeModal])

  if (activeModal !== 'deleteMessage' && activeModal !== 'deleteConversation') {
    return null
  }

  const isMessage = confirmDelete.type === 'message'

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-800 rounded-2xl shadow-2xl w-full max-w-sm overflow-hidden border border-gray-700 animate-in fade-in zoom-in-95 duration-200">
        <div className="p-6 text-center">
          {/* Icon */}
          <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-red-800/20 mb-4">
            <svg className="w-6 h-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </div>

          {/* Title */}
          <h3 className="text-lg font-semibold text-white mb-2">
            Delete {isMessage ? 'message' : 'conversation'}?
          </h3>

          {/* Description */}
          <p className="text-sm text-gray-400 mb-4">
            {isMessage
              ? 'This will remove the message from your history. This action cannot be undone.'
              : 'This will permanently delete this conversation and all its messages.'}
          </p>

          {/* Affected content preview */}
          {confirmDelete.title && (
            <div className="p-3 bg-gray-900/40 rounded-lg text-xs text-left border border-gray-700 mb-4">
              <p className="font-medium text-gray-300 mb-1">
                {isMessage ? 'Message:' : 'Conversation:'}
              </p>
              <p className="text-gray-400 truncate">"{confirmDelete.title}"</p>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="grid grid-cols-2 gap-3 px-6 pb-6">
          <button
            onClick={closeModal}
            className="w-full py-2.5 px-4 border border-gray-600 bg-gray-700 text-gray-200 rounded-xl text-sm font-medium hover:bg-gray-600 transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-700"
          >
            Cancel
          </button>
          <button
            onClick={() => handleConfirm(false)}
            className="w-full py-2.5 px-4 bg-red-600 hover:bg-red-700 rounded-xl text-sm font-medium text-white shadow-sm transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
          >
            Delete
          </button>
        </div>

        {/* Extra option for assistant messages - regenerate after delete */}
        {isMessage && (
          <div className="px-6 pb-6 pt-0">
            <button
              onClick={() => handleConfirm(true)}
              className="w-full py-2 px-4 text-sm text-gray-400 hover:text-white hover:bg-gray-700/50 rounded-lg transition-colors"
            >
              Delete and generate new response
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
