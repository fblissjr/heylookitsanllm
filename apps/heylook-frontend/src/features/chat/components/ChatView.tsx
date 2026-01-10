import { useEffect, useRef } from 'react'
import { useChatStore } from '../../../stores/chatStore'
import { useModelStore } from '../../../stores/modelStore'
import { MessageList } from './MessageList'
import { ChatInput } from './ChatInput'
import { EmptyState } from './EmptyState'

export function ChatView() {
  const { activeConversation, streaming } = useChatStore()
  const { loadedModel, modelStatus } = useModelStore()
  const scrollRef = useRef<HTMLDivElement>(null)

  const conversation = activeConversation()

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
        className="flex-1 overflow-y-auto p-4 space-y-6"
      >
        <MessageList
          messages={conversation.messages}
          streaming={streaming}
          modelCapabilities={loadedModel.capabilities}
        />
      </div>

      {/* Input area */}
      <ChatInput
        conversationId={conversation.id}
        modelId={loadedModel.id}
        hasVision={loadedModel.capabilities.vision}
        disabled={streaming.isStreaming}
      />
    </div>
  )
}
