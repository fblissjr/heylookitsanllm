import { useEffect, useRef, useCallback } from 'react'
import { useChatStore } from '../stores/chatStore'
import { useModelStore } from '../../../stores/modelStore'
import { useSettingsStore } from '../../../stores/settingsStore'
import { MessageList } from './MessageList'
import { ChatInput } from './ChatInput'
import { EmptyState } from './EmptyState'
import { SystemPromptEditor } from './SystemPromptEditor'

export function ChatView() {
  const { activeConversation, streaming, updateSystemPrompt, createConversation, updateConversationModel } = useChatStore()
  const loadedModel = useModelStore((s) => s.loadedModel)
  const modelStatus = useModelStore((s) => s.modelStatus)
  const models = useModelStore((s) => s.models)
  const systemPrompt = useSettingsStore((s) => s.systemPrompt)
  const scrollRef = useRef<HTMLDivElement>(null)
  const prevModelRef = useRef<string | null>(loadedModel?.id ?? null)

  const conversation = activeConversation()

  // Auto-create conversation when a new model is loaded (replaces onModelLoaded callback)
  useEffect(() => {
    if (loadedModel && loadedModel.id !== prevModelRef.current) {
      if (!activeConversation()) {
        createConversation(loadedModel.id, systemPrompt)
      }
    }
    prevModelRef.current = loadedModel?.id ?? null
  }, [loadedModel, activeConversation, createConversation, systemPrompt])

  // Auto-fix stale model IDs: if the conversation references a model that no longer
  // exists in the server's model list, update it to the currently loaded model.
  useEffect(() => {
    if (!conversation || !loadedModel || models.length === 0) return
    const modelExists = models.some(m => m.id === conversation.defaultModelId)
    if (!modelExists && conversation.defaultModelId) {
      updateConversationModel(conversation.id, loadedModel.id)
    }
  }, [conversation, loadedModel, models, updateConversationModel])

  const handleSystemPromptUpdate = useCallback(async (systemPrompt: string, shouldRegenerate: boolean) => {
    if (!conversation) return
    await updateSystemPrompt(conversation.id, systemPrompt, shouldRegenerate)
  }, [conversation, updateSystemPrompt])

  // Cleanup on unmount -- abort any in-flight stream (matches TokenExplorerView, NotebookView)
  useEffect(() => {
    return () => {
      const { streaming, stopGeneration } = useChatStore.getState()
      if (streaming.isStreaming) {
        stopGeneration()
      }
    }
  }, [])

  // Auto-scroll to bottom when new messages arrive or during streaming
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [conversation?.messages, streaming.content, streaming.thinking])

  // Show empty state if no model loaded
  if (!loadedModel || modelStatus !== 'loaded') {
    return (
      <div className="flex-1 flex items-center justify-center p-4">
        <EmptyState type="no-model" />
      </div>
    )
  }

  // Show empty state if no conversation selected
  if (!conversation) {
    return (
      <div className="flex-1 flex items-center justify-center p-4">
        <EmptyState type="no-conversation" />
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Messages area */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto scroll-touch p-4"
      >
        <div className="max-w-3xl mx-auto space-y-6">
          {/* System Prompt Editor at top of conversation */}
          <SystemPromptEditor
            systemPrompt={conversation.systemPrompt}
            onUpdate={handleSystemPromptUpdate}
            disabled={streaming.isStreaming}
            hasMessages={conversation.messages.length > 0}
          />

          <MessageList
            messages={conversation.messages}
            streaming={streaming}
            modelCapabilities={loadedModel.capabilities}
          />
        </div>
      </div>

      {/* Input area */}
      <ChatInput
        conversationId={conversation.id}
        disabled={streaming.isStreaming}
      />
    </div>
  )
}
