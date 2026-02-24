import { useState, useCallback } from 'react'
import { Message } from '../../../types/chat'
import { ModelCapabilities } from '../../../types/models'
import { useChatStore, StreamingState } from '../stores/chatStore'
import { useUIStore } from '../../../stores/uiStore'
import { useModelStore } from '../../../stores/modelStore'
import { MessageMetricsFooter } from './MessageMetricsFooter'
import { MessageDebugModal } from './MessageDebugModal'
import { ThinkingBlock } from '../../../components/composed'
import { MessageActions } from '../../../components/composed/MessageActions'
import { StaleBadge } from '../../../components/primitives/StaleBadge'
import { isMessageStale } from '../../../lib/stale'
import { ComputerIcon, RefreshIcon, CheckIcon } from '../../../components/icons'
import clsx from 'clsx'

interface MessageListProps {
  messages: Message[]
  streaming: StreamingState
  modelCapabilities: ModelCapabilities
}

export function MessageList({ messages, streaming, modelCapabilities }: MessageListProps) {
  // Filter out the message being streamed (it's rendered separately as StreamingMessage)
  const visibleMessages = streaming.isStreaming && streaming.messageId
    ? messages.filter(m => m.id !== streaming.messageId)
    : messages

  return (
    <div className="space-y-6">
      {visibleMessages.map((message, index) => (
        <MessageBubble
          key={message.id}
          message={message}
          index={index}
          totalMessages={visibleMessages.length}
          modelCapabilities={modelCapabilities}
          isStale={isMessageStale(visibleMessages, index)}
        />
      ))}

      {/* Streaming message */}
      {streaming.isStreaming && (
        <StreamingMessage
          content={streaming.content}
          thinking={streaming.thinking}
          showThinking={modelCapabilities.thinking}
        />
      )}
    </div>
  )
}

interface MessageBubbleProps {
  message: Message
  index: number
  totalMessages: number
  modelCapabilities: ModelCapabilities
  isStale: boolean
}

function MessageBubble({ message, index, totalMessages, modelCapabilities, isStale }: MessageBubbleProps) {
  const [isEditing, setIsEditing] = useState(false)
  const [editContent, setEditContent] = useState(message.content)
  const [editThinking, setEditThinking] = useState(message.thinking || '')
  const [showThinking, setShowThinking] = useState(true)
  const [showDebugModal, setShowDebugModal] = useState(false)

  // Get model info for debug modal
  const { loadedModel } = useModelStore()

  const { editMessageAndRegenerate, regenerateFromPosition, continueFromMessage, generateNextTurn } = useChatStore()
  const { setConfirmDelete } = useUIStore()

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(message.content)
  }, [message.content])

  const handleEdit = useCallback(() => {
    setEditContent(message.content)
    setEditThinking(message.thinking || '')
    setIsEditing(true)
  }, [message.content, message.thinking])

  const handleCancelEdit = useCallback(() => {
    setIsEditing(false)
    setEditContent(message.content)
    setEditThinking(message.thinking || '')
  }, [message.content, message.thinking])

  const handleSaveEdit = useCallback(async (shouldRegenerate: boolean) => {
    const conversation = useChatStore.getState().activeConversation()
    if (!conversation) return

    const thinkingArg = message.role === 'assistant' ? editThinking : undefined
    await editMessageAndRegenerate(conversation.id, message.id, editContent, shouldRegenerate, thinkingArg)
    setIsEditing(false)
  }, [editContent, editThinking, message.id, message.role, editMessageAndRegenerate])

  const handleDelete = useCallback(() => {
    const conversation = useChatStore.getState().activeConversation()
    if (!conversation) return

    setConfirmDelete({
      type: 'message',
      id: message.id,
      title: message.content.slice(0, 50) + (message.content.length > 50 ? '...' : ''),
      conversationId: conversation.id,
      messageIndex: index
    })
  }, [message.id, message.content, index, setConfirmDelete])

  const handleRegenerate = useCallback(async () => {
    const conversation = useChatStore.getState().activeConversation()
    if (!conversation) return

    await regenerateFromPosition(conversation.id, index)
  }, [index, regenerateFromPosition])

  const handleContinue = useCallback(async () => {
    const conversation = useChatStore.getState().activeConversation()
    if (!conversation) return

    await continueFromMessage(conversation.id, message.id)
  }, [message.id, continueFromMessage])

  const handleNextTurn = useCallback(async () => {
    const conversation = useChatStore.getState().activeConversation()
    if (!conversation) return

    await generateNextTurn(conversation.id)
  }, [generateNextTurn])

  const handleThinkingSave = useCallback((newThinking: string) => {
    const conversation = useChatStore.getState().activeConversation()
    if (!conversation) return

    editMessageAndRegenerate(conversation.id, message.id, message.content, false, newThinking)
  }, [message.id, message.content, editMessageAndRegenerate])

  if (message.role === 'system') {
    return (
      <div className="flex justify-center">
        <div className="group bg-gray-100 dark:bg-surface-dark px-4 py-2 rounded-lg text-sm text-gray-500 dark:text-gray-400 max-w-[90%]">
          <span className="font-medium">System:</span> {message.content.slice(0, 100)}
          {message.content.length > 100 && '...'}
          {isStale && <StaleBadge className="ml-2" />}
        </div>
      </div>
    )
  }

  if (message.role === 'user') {
    return (
      <div className="flex flex-col items-end gap-2">
        {/* Images if present */}
        {message.images && message.images.length > 0 && (
          <div className="flex flex-wrap gap-2 max-w-[85%]">
            {message.images.map((img, i) => (
              <div key={i} className="relative rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700">
                <img src={img} alt={`Attachment ${i + 1}`} className="max-w-48 max-h-48 object-cover" />
              </div>
            ))}
          </div>
        )}

        {/* Message bubble */}
        <div className="group flex flex-col items-end gap-1 max-w-[85%]">
          {isEditing ? (
            <EditingBubble
              content={editContent}
              onChange={setEditContent}
              onCancel={handleCancelEdit}
              onSave={handleSaveEdit}
              showRegenerateOption={index < totalMessages - 1}
            />
          ) : (
            <>
              <div className="bg-primary text-white px-4 py-3 rounded-2xl rounded-tr-sm shadow-sm">
                <p className="whitespace-pre-wrap">{message.content}</p>
              </div>
              <MessageActions
                role="user"
                onCopy={handleCopy}
                onEdit={handleEdit}
                onDelete={handleDelete}
                isStale={isStale}
              />
            </>
          )}
        </div>

        {/* Timestamp + stale badge */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-gray-400 dark:text-gray-500">
            {formatTime(message.timestamp)}
          </span>
          {isStale && <StaleBadge />}
        </div>
      </div>
    )
  }

  // Assistant message
  return (
    <div className="flex items-start gap-3 max-w-full">
      {/* Avatar */}
      <div className="w-8 h-8 rounded-full bg-emerald-500 flex items-center justify-center shrink-0 shadow-lg shadow-emerald-500/20">
        <ComputerIcon className="w-4 h-4 text-white" />
      </div>

      <div className="flex flex-col gap-2 min-w-0 flex-1">
        {/* Thinking block (if present and model supports it) */}
        {message.thinking && modelCapabilities.thinking && (
          <ThinkingBlock
            content={message.thinking}
            isOpen={showThinking}
            onToggle={() => setShowThinking(!showThinking)}
            thinkingTime={message.performance?.thinkingDuration}
            thinkingTokens={message.performance?.thinkingTokens}
            editable
            onSave={handleThinkingSave}
          />
        )}

        {/* Response content */}
        <div className="group">
          {isEditing ? (
            <div className="w-full space-y-3">
              {/* Thinking textarea for assistant messages */}
              {message.thinking !== undefined && modelCapabilities.thinking && (
                <div className="bg-purple-50 dark:bg-purple-900/10 rounded-xl border-l-4 border-purple-400 p-3">
                  <label className="text-xs font-semibold text-purple-600 dark:text-purple-400 uppercase tracking-wider block mb-2">
                    Thinking
                  </label>
                  <textarea
                    value={editThinking}
                    onChange={(e) => setEditThinking(e.target.value)}
                    className="w-full bg-white dark:bg-surface-darker text-purple-800 dark:text-purple-200 p-3 rounded-lg border border-purple-300 dark:border-purple-700 focus:ring-1 focus:ring-purple-500/50 resize-none min-h-[80px] font-mono text-sm"
                    rows={3}
                  />
                </div>
              )}
              <EditingBubble
                content={editContent}
                onChange={setEditContent}
                onCancel={handleCancelEdit}
                onSave={handleSaveEdit}
                showRegenerateOption={false}
                isAssistant
              />
            </div>
          ) : (
            <>
              <div className="bg-gray-100 dark:bg-surface-dark px-4 py-3 rounded-2xl rounded-tl-sm border border-gray-200 dark:border-gray-700">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <p className="whitespace-pre-wrap">{message.content}</p>
                </div>
              </div>
              <div className="flex items-center justify-between mt-1 px-1">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-gray-400 dark:text-gray-500">
                    {formatTime(message.timestamp)}
                  </span>
                  {isStale && <StaleBadge />}
                </div>
                <MessageActions
                  role="assistant"
                  onCopy={handleCopy}
                  onEdit={handleEdit}
                  onDelete={handleDelete}
                  onRegenerate={handleRegenerate}
                  onContinue={handleContinue}
                  onNextTurn={index === totalMessages - 1 ? handleNextTurn : undefined}
                  isStale={isStale}
                />
              </div>
              {/* Performance metrics footer */}
              {message.performance && (
                <MessageMetricsFooter
                  performance={message.performance}
                  modelId={message.modelId}
                  onShowDebug={() => setShowDebugModal(true)}
                />
              )}
              {/* Debug modal */}
              <MessageDebugModal
                isOpen={showDebugModal}
                onClose={() => setShowDebugModal(false)}
                message={message}
                modelInfo={loadedModel ? {
                  id: loadedModel.id,
                  object: 'model' as const,
                  owned_by: loadedModel.provider ?? 'unknown',
                  provider: loadedModel.provider as 'mlx' | 'mlx_stt' | undefined,
                  capabilities: Object.entries(loadedModel.capabilities)
                    .filter(([, v]) => v)
                    .map(([k]) => k),
                  context_window: loadedModel.contextWindow,
                } : undefined}
              />
            </>
          )}
        </div>
      </div>
    </div>
  )
}

interface EditingBubbleProps {
  content: string
  onChange: (content: string) => void
  onCancel: () => void
  onSave: (shouldRegenerate: boolean) => void
  showRegenerateOption: boolean
  isAssistant?: boolean
}

function EditingBubble({ content, onChange, onCancel, onSave, showRegenerateOption, isAssistant }: EditingBubbleProps) {
  return (
    <div className={clsx(
      'w-full bg-white dark:bg-surface-dark border-2 border-primary/40 rounded-2xl p-3 shadow-lg ring-4 ring-primary/10',
      isAssistant ? 'max-w-full' : 'max-w-[85%] ml-auto'
    )}>
      <div className="flex flex-col gap-3">
        <label className="text-xs font-semibold text-primary uppercase tracking-wider">
          Editing Message
        </label>
        <textarea
          value={content}
          onChange={(e) => onChange(e.target.value)}
          className="w-full bg-gray-50 dark:bg-surface-darker text-gray-800 dark:text-gray-100 p-3 rounded-lg border-none focus:ring-1 focus:ring-primary/50 resize-none min-h-[80px]"
          rows={3}
          autoFocus
        />
        <div className="flex justify-end gap-3">
          <button
            onClick={onCancel}
            className="px-4 py-2 rounded-lg text-sm font-medium text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-white/5 transition-colors"
          >
            Cancel
          </button>
          {showRegenerateOption ? (
            <>
              <button
                onClick={() => onSave(false)}
                className="px-4 py-2 rounded-lg text-sm font-medium bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
              >
                Save Only
              </button>
              <button
                onClick={() => onSave(true)}
                className="px-4 py-2 rounded-lg text-sm font-medium bg-primary text-white hover:bg-primary-hover shadow-md transition-all flex items-center gap-2"
              >
                <RefreshIcon />
                Save & Regenerate
              </button>
            </>
          ) : (
            <button
              onClick={() => onSave(false)}
              className="px-4 py-2 rounded-lg text-sm font-medium bg-primary text-white hover:bg-primary-hover shadow-md transition-all flex items-center gap-2"
            >
              <CheckIcon />
              Save Changes
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

interface StreamingMessageProps {
  content: string
  thinking: string
  showThinking: boolean
}

function StreamingMessage({ content, thinking, showThinking }: StreamingMessageProps) {
  const [isThinkingOpen, setIsThinkingOpen] = useState(true)

  return (
    <div className="flex items-start gap-3 max-w-full">
      {/* Avatar with pulse animation */}
      <div className="w-8 h-8 rounded-full bg-emerald-500 flex items-center justify-center shrink-0 shadow-lg shadow-emerald-500/20 animate-pulse">
        <ComputerIcon className="w-4 h-4 text-white" />
      </div>

      <div className="flex flex-col gap-2 min-w-0 flex-1">
        {/* Thinking block (shown while streaming thinking) */}
        {showThinking && thinking && (
          <ThinkingBlock
            content={thinking + '\u2588'} // Blinking cursor
            isOpen={isThinkingOpen}
            onToggle={() => setIsThinkingOpen(!isThinkingOpen)}
          />
        )}

        {/* Response content */}
        {content && (
          <div className="bg-gray-100 dark:bg-surface-dark px-4 py-3 rounded-2xl rounded-tl-sm border border-gray-200 dark:border-gray-700">
            <p className="whitespace-pre-wrap">
              {content}
              <span className="animate-pulse">{'\u2588'}</span>
            </p>
          </div>
        )}

        {/* Show "Thinking..." if only thinking is streaming */}
        {showThinking && thinking && !content && (
          <div className="text-sm text-gray-400 dark:text-gray-500 italic">
            Thinking...
          </div>
        )}

        {/* Show loading if nothing yet */}
        {!thinking && !content && (
          <div className="bg-gray-100 dark:bg-surface-dark px-4 py-3 rounded-2xl rounded-tl-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 text-gray-400">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function formatTime(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true
  })
}
