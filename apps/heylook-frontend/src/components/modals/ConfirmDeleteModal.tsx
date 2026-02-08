import { useCallback } from 'react'
import { useUIStore } from '../../stores/uiStore'
import { useChatStore } from '../../stores/chatStore'
import { Modal } from '../primitives'
import { TrashIcon } from '../icons'

export function ConfirmDeleteModal() {
  const { activeModal, confirmDelete, closeModal } = useUIStore()
  const { deleteConversation, deleteMessageWithCascade } = useChatStore()

  const handleConfirm = useCallback(async (regenerate = false) => {
    if (confirmDelete.type === 'conversation' && confirmDelete.id) {
      deleteConversation(confirmDelete.id)
    } else if (confirmDelete.type === 'bulk' && confirmDelete.ids) {
      // Delete all selected conversations
      for (const id of confirmDelete.ids) {
        deleteConversation(id)
      }
    } else if (confirmDelete.type === 'message' && confirmDelete.id && confirmDelete.conversationId) {
      await deleteMessageWithCascade(confirmDelete.conversationId, confirmDelete.id, regenerate)
    }
    // Call cleanup callback before closing
    confirmDelete.onComplete?.()
    closeModal()
  }, [confirmDelete, deleteConversation, deleteMessageWithCascade, closeModal])

  const handleCancel = useCallback(() => {
    // Call cleanup callback on cancel too
    confirmDelete.onComplete?.()
    closeModal()
  }, [confirmDelete, closeModal])

  if (activeModal !== 'deleteMessage' && activeModal !== 'deleteConversation') {
    return null
  }

  const isMessage = confirmDelete.type === 'message'
  const isBulk = confirmDelete.type === 'bulk'
  const bulkCount = confirmDelete.ids?.length || 0

  return (
    <Modal>
        <div className="p-6 text-center">
          {/* Icon */}
          <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-red-800/20 mb-4">
            <TrashIcon className="w-6 h-6 text-red-500" />
          </div>

          {/* Title */}
          <h3 className="text-lg font-semibold text-white mb-2">
            Delete {isMessage ? 'message' : isBulk ? `${bulkCount} conversations` : 'conversation'}?
          </h3>

          {/* Description */}
          <p className="text-sm text-gray-400 mb-4">
            {isMessage
              ? 'This will remove the message from your history. This action cannot be undone.'
              : isBulk
                ? `This will permanently delete ${bulkCount} conversations and all their messages.`
                : 'This will permanently delete this conversation and all its messages.'}
          </p>

          {/* Affected content preview */}
          {confirmDelete.title && !isBulk && (
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
            onClick={handleCancel}
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
    </Modal>
  )
}
