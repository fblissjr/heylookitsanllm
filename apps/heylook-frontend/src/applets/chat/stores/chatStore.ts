import { create } from 'zustand'
import type { Message, Conversation, StreamingState, EditState } from '../../../types/chat'
import { buildAPIMessages } from '../../../lib/messages'
import { streamChat, type StreamCompletionData } from '../../../api/streaming'
import type { ChatCompletionRequest } from '../../../types/api'
import { useSettingsStore } from '../../../stores/settingsStore'
import * as db from '../../../lib/db'
import { generateId } from '../../../lib/id'

// Re-export StreamingState for components
export type { StreamingState }

const DEFAULT_STREAM_TIMEOUT_MS = 30_000

class ChatStreamManager {
  private controller: AbortController | null = null

  /** Abort any in-flight stream and reset. */
  abort() {
    if (this.controller) {
      this.controller.abort()
      this.controller = null
    }
  }

  /**
   * Run a streaming generation. Handles:
   * - Aborting any previous stream
   * - Creating/cleaning up AbortController
   * - Timeout (configurable via settings)
   * - Routing callbacks to the pinned conversationId
   */
  async run(
    conversationId: string,
    request: ChatCompletionRequest,
    store: {
      appendStreamContent: (content: string, isThinking: boolean, rawEvent?: string) => void
      finalizeStream: (data?: StreamCompletionData, conversationId?: string) => void
    }
  ): Promise<void> {
    this.abort()
    this.controller = new AbortController()

    const targetConversationId = conversationId
    const timeoutMs = useSettingsStore.getState().samplerSettings.streamTimeoutMs ?? DEFAULT_STREAM_TIMEOUT_MS

    // Strip frontend-only settings before sending to API
    const { streamTimeoutMs: _, ...apiRequest } = request as ChatCompletionRequest & { streamTimeoutMs?: number }

    try {
      await streamChat(
        apiRequest,
        {
          onToken: (token, rawEvent) => store.appendStreamContent(token, false, rawEvent),
          onThinking: (thinking, rawEvent) => store.appendStreamContent(thinking, true, rawEvent),
          onComplete: (data) => store.finalizeStream(data, targetConversationId),
          onError: (error) => {
            console.error('Stream error:', error)
            store.finalizeStream(undefined, targetConversationId)
          },
        },
        this.controller.signal,
        timeoutMs
      )
    } finally {
      this.controller = null
    }
  }

  get isActive(): boolean {
    return this.controller !== null
  }
}

const streamManager = new ChatStreamManager()

// Debounced save to avoid too many DB writes
let saveTimeout: ReturnType<typeof setTimeout> | null = null
function debouncedSave(conversation: Conversation) {
  if (saveTimeout) clearTimeout(saveTimeout)
  saveTimeout = setTimeout(() => {
    db.saveConversation(conversation).catch(console.error)
  }, 500)
}

function generateTitle(messages: Message[]): string {
  const firstUserMessage = messages.find(m => m.role === 'user')
  if (firstUserMessage) {
    const content = firstUserMessage.content.slice(0, 50)
    return content.length < firstUserMessage.content.length ? `${content}...` : content
  }
  return 'New Conversation'
}

interface ChatState {
  // Data
  conversations: Conversation[]
  activeConversationId: string | null
  streaming: StreamingState
  editState: EditState

  // Computed
  activeConversation: () => Conversation | undefined

  // Conversation management
  createConversation: (modelId: string, systemPrompt?: string) => string
  setActiveConversation: (id: string | null) => void
  deleteConversation: (id: string) => void
  updateConversationTitle: (id: string, title: string) => void
  updateConversationModel: (id: string, modelId: string) => void
  updateSystemPrompt: (conversationId: string, systemPrompt: string, shouldRegenerate?: boolean) => Promise<void>

  // Message management
  addMessage: (conversationId: string, message: Omit<Message, 'id' | 'timestamp'>) => string
  updateMessage: (conversationId: string, messageId: string, updates: Partial<Message>) => void
  deleteMessage: (conversationId: string, messageId: string) => void

  // Non-linear editing
  deleteMessageAndDownstream: (conversationId: string, messageId: string) => void
  getMessagesUpTo: (conversationId: string, messageId: string) => Message[]

  // API actions
  sendMessage: (conversationId: string, content: string, modelId: string, images?: string[]) => Promise<void>
  stopGeneration: () => void
  regenerateFromPosition: (conversationId: string, position: number) => Promise<void>
  editMessageAndRegenerate: (conversationId: string, messageId: string, newContent: string, shouldRegenerate: boolean, newThinking?: string) => Promise<void>
  deleteMessageWithCascade: (conversationId: string, messageId: string, shouldRegenerateNext?: boolean) => Promise<void>
  continueFromMessage: (conversationId: string, messageId: string) => Promise<void>
  generateNextTurn: (conversationId: string) => Promise<void>

  // Streaming
  setStreaming: (state: Partial<StreamingState>) => void
  appendStreamContent: (content: string, isThinking: boolean, rawEvent?: string) => void
  finalizeStream: (completionData?: StreamCompletionData, targetConversationId?: string) => void

  // Edit mode
  startEditing: (messageId: string, content: string) => void
  updateEditContent: (content: string) => void
  cancelEditing: () => void
  saveEdit: () => { conversationId: string; messageId: string; content: string } | null

  // Persistence
  loadFromDB: () => Promise<void>
  setConversations: (conversations: Conversation[]) => void
  getConversationById: (id: string) => Conversation | undefined
}

export const useChatStore = create<ChatState>((set, get) => ({
  conversations: [],
  activeConversationId: null,
  streaming: {
    isStreaming: false,
    content: '',
    thinking: '',
    messageId: null,
  },
  editState: {
    messageId: null,
    originalContent: '',
    editedContent: '',
  },

  activeConversation: () => {
    const { conversations, activeConversationId } = get()
    return conversations.find(c => c.id === activeConversationId)
  },

  createConversation: (modelId, systemPrompt) => {
    const id = generateId()
    const now = Date.now()
    const newConversation: Conversation = {
      id,
      title: 'New Conversation',
      defaultModelId: modelId,
      messages: [],
      systemPrompt,
      createdAt: now,
      updatedAt: now,
    }
    set(state => ({
      conversations: [newConversation, ...state.conversations],
      activeConversationId: id,
    }))
    // Persist to DB
    db.saveConversation(newConversation).catch(console.error)
    return id
  },

  setActiveConversation: (id) => {
    set({ activeConversationId: id })
  },

  deleteConversation: (id) => {
    set(state => {
      const newConversations = state.conversations.filter(c => c.id !== id)
      const newActiveId = state.activeConversationId === id
        ? (newConversations[0]?.id || null)
        : state.activeConversationId
      return {
        conversations: newConversations,
        activeConversationId: newActiveId,
      }
    })
    // Delete from DB
    db.deleteConversation(id).catch(console.error)
  },

  updateConversationTitle: (id, title) => {
    set(state => ({
      conversations: state.conversations.map(c =>
        c.id === id ? { ...c, title, updatedAt: Date.now() } : c
      ),
    }))
    // Persist to DB
    const conversation = get().conversations.find(c => c.id === id)
    if (conversation) debouncedSave(conversation)
  },

  updateConversationModel: (id, modelId) => {
    set(state => ({
      conversations: state.conversations.map(c =>
        c.id === id ? { ...c, defaultModelId: modelId, updatedAt: Date.now() } : c
      ),
    }))
    const conversation = get().conversations.find(c => c.id === id)
    if (conversation) debouncedSave(conversation)
  },

  updateSystemPrompt: async (conversationId, systemPrompt, shouldRegenerate = false) => {
    const { regenerateFromPosition } = get()
    const conversation = get().conversations.find(c => c.id === conversationId)
    if (!conversation) return

    // Update the system prompt
    set(state => ({
      conversations: state.conversations.map(c =>
        c.id === conversationId ? { ...c, systemPrompt, updatedAt: Date.now() } : c
      ),
    }))

    // Persist to DB
    const updatedConversation = get().conversations.find(c => c.id === conversationId)
    if (updatedConversation) debouncedSave(updatedConversation)

    // If shouldRegenerate and there are assistant messages, regenerate from first one
    if (shouldRegenerate) {
      const firstAssistantIndex = conversation.messages.findIndex(m => m.role === 'assistant')
      if (firstAssistantIndex !== -1) {
        await regenerateFromPosition(conversationId, firstAssistantIndex)
      }
    }
  },

  addMessage: (conversationId, messageData) => {
    const messageId = generateId()
    const message: Message = {
      ...messageData,
      id: messageId,
      timestamp: Date.now(),
    }

    set(state => ({
      conversations: state.conversations.map(c => {
        if (c.id !== conversationId) return c
        const updatedMessages = [...c.messages, message]
        return {
          ...c,
          messages: updatedMessages,
          title: c.messages.length === 0 ? generateTitle(updatedMessages) : c.title,
          updatedAt: Date.now(),
        }
      }),
    }))

    // Persist to DB
    const conversation = get().conversations.find(c => c.id === conversationId)
    if (conversation) debouncedSave(conversation)

    return messageId
  },

  updateMessage: (conversationId, messageId, updates) => {
    set(state => ({
      conversations: state.conversations.map(c => {
        if (c.id !== conversationId) return c
        return {
          ...c,
          messages: c.messages.map(m =>
            m.id === messageId ? { ...m, ...updates } : m
          ),
          updatedAt: Date.now(),
        }
      }),
    }))
    // Persist to DB
    const conversation = get().conversations.find(c => c.id === conversationId)
    if (conversation) debouncedSave(conversation)
  },

  deleteMessage: (conversationId, messageId) => {
    set(state => ({
      conversations: state.conversations.map(c => {
        if (c.id !== conversationId) return c
        return {
          ...c,
          messages: c.messages.filter(m => m.id !== messageId),
          updatedAt: Date.now(),
        }
      }),
    }))
    // Persist to DB
    const conversation = get().conversations.find(c => c.id === conversationId)
    if (conversation) debouncedSave(conversation)
  },

  deleteMessageAndDownstream: (conversationId, messageId) => {
    set(state => ({
      conversations: state.conversations.map(c => {
        if (c.id !== conversationId) return c
        const messageIndex = c.messages.findIndex(m => m.id === messageId)
        if (messageIndex === -1) return c
        return {
          ...c,
          messages: c.messages.slice(0, messageIndex),
          updatedAt: Date.now(),
        }
      }),
    }))
    // Persist to DB
    const conversation = get().conversations.find(c => c.id === conversationId)
    if (conversation) debouncedSave(conversation)
  },

  getMessagesUpTo: (conversationId, messageId) => {
    const conversation = get().conversations.find(c => c.id === conversationId)
    if (!conversation) return []

    const messageIndex = conversation.messages.findIndex(m => m.id === messageId)
    if (messageIndex === -1) return conversation.messages

    return conversation.messages.slice(0, messageIndex)
  },

  // API Actions
  sendMessage: async (conversationId, content, modelId, images) => {
    const { addMessage, setStreaming, appendStreamContent, finalizeStream } = get()
    const conversation = get().conversations.find(c => c.id === conversationId)
    if (!conversation) return

    // Add user message (no modelId for user messages)
    addMessage(conversationId, {
      role: 'user',
      content,
      images,
    })

    // Create placeholder for assistant message with modelId tracking
    const assistantMessageId = addMessage(conversationId, {
      role: 'assistant',
      content: '',
      modelId,
    })

    // Start streaming
    setStreaming({
      isStreaming: true,
      content: '',
      thinking: '',
      messageId: assistantMessageId,
    })

    // Get current settings
    const settings = useSettingsStore.getState()

    // Build messages array for API (exclude the empty placeholder)
    const currentConversation = get().conversations.find(c => c.id === conversationId)
    const apiMessages = buildAPIMessages(
      currentConversation?.messages || [],
      { excludeId: assistantMessageId, systemPrompt: conversation.systemPrompt }
    )

    await streamManager.run(
      conversationId,
      {
        model: modelId,
        messages: apiMessages,
        ...settings.samplerSettings,
      },
      { appendStreamContent, finalizeStream }
    )
  },

  stopGeneration: () => {
    streamManager.abort()
    // If the stream hasn't started yet (abort won't trigger onComplete),
    // force-reset the streaming state
    const { streaming } = get()
    if (streaming.isStreaming && !streamManager.isActive) {
      set({
        streaming: {
          isStreaming: false,
          content: '',
          thinking: '',
          messageId: null,
        },
      })
    }
  },

  regenerateFromPosition: async (conversationId, position) => {
    const { deleteMessageAndDownstream, setStreaming, appendStreamContent, finalizeStream, addMessage } = get()
    const conversation = get().conversations.find(c => c.id === conversationId)
    if (!conversation || position < 0) return

    // Get the message at position
    const messageAtPosition = conversation.messages[position]
    if (!messageAtPosition || messageAtPosition.role !== 'assistant') return

    // Delete from position onwards
    deleteMessageAndDownstream(conversationId, messageAtPosition.id)

    // Get updated conversation after deletion
    const updatedConversation = get().conversations.find(c => c.id === conversationId)
    if (!updatedConversation) return

    // Create new assistant placeholder (use conversation's default model for regeneration)
    const assistantMessageId = addMessage(conversationId, {
      role: 'assistant',
      content: '',
      isRegenerating: true,
      modelId: updatedConversation.defaultModelId,
    })

    // Start streaming
    setStreaming({
      isStreaming: true,
      content: '',
      thinking: '',
      messageId: assistantMessageId,
    })

    const settings = useSettingsStore.getState()

    // Refresh conversation reference after adding the message
    const currentConversation = get().conversations.find(c => c.id === conversationId)
    if (!currentConversation) return

    const apiMessages = buildAPIMessages(
      currentConversation.messages,
      { excludeId: assistantMessageId, systemPrompt: currentConversation.systemPrompt }
    )

    await streamManager.run(
      conversationId,
      {
        model: currentConversation.defaultModelId,
        messages: apiMessages,
        ...settings.samplerSettings,
      },
      { appendStreamContent, finalizeStream }
    )
  },

  editMessageAndRegenerate: async (conversationId, messageId, newContent, shouldRegenerate, newThinking) => {
    const { updateMessage, regenerateFromPosition } = get()
    const conversation = get().conversations.find(c => c.id === conversationId)
    if (!conversation) return

    const messageIndex = conversation.messages.findIndex(m => m.id === messageId)
    if (messageIndex === -1) return

    // Update the message content and mark as edited
    const updates: Partial<Message> = {
      content: newContent,
      editedAt: Date.now(),
    }
    if (newThinking !== undefined) {
      updates.thinking = newThinking
    }
    updateMessage(conversationId, messageId, updates)

    if (shouldRegenerate && messageIndex < conversation.messages.length - 1) {
      // Regenerate from the next position (first assistant message after this)
      const nextAssistantIndex = conversation.messages.findIndex(
        (m, i) => i > messageIndex && m.role === 'assistant'
      )
      if (nextAssistantIndex !== -1) {
        await regenerateFromPosition(conversationId, nextAssistantIndex)
      }
    }
  },

  deleteMessageWithCascade: async (conversationId, messageId, shouldRegenerateNext = false) => {
    const { deleteMessage, regenerateFromPosition, deleteMessageAndDownstream } = get()
    const conversation = get().conversations.find(c => c.id === conversationId)
    if (!conversation) return

    const messageIndex = conversation.messages.findIndex(m => m.id === messageId)
    if (messageIndex === -1) return

    const message = conversation.messages[messageIndex]

    if (shouldRegenerateNext && message.role === 'user') {
      // If deleting a user message and we want to regenerate, delete from here and regenerate
      deleteMessageAndDownstream(conversationId, messageId)
    } else {
      // Just delete this message
      deleteMessage(conversationId, messageId)

      // If we deleted an assistant message and want to regenerate
      if (shouldRegenerateNext && message.role === 'assistant') {
        // Find the position where we need to regenerate
        await regenerateFromPosition(conversationId, messageIndex)
      }
    }
  },

  continueFromMessage: async (conversationId, messageId) => {
    const { setStreaming, appendStreamContent, finalizeStream } = get()
    const conversation = get().conversations.find(c => c.id === conversationId)
    if (!conversation) return

    const targetMessage = conversation.messages.find(m => m.id === messageId)
    if (!targetMessage || targetMessage.role !== 'assistant') return

    // Build API messages INCLUDING this assistant message (no exclusion)
    // Backend sees last message is assistant -> continues from there
    const messageIndex = conversation.messages.findIndex(m => m.id === messageId)
    const messagesUpToTarget = conversation.messages.slice(0, messageIndex + 1)
    const apiMessages = buildAPIMessages(messagesUpToTarget, {
      systemPrompt: conversation.systemPrompt,
    })

    // Start streaming with existing content (we'll append to it)
    setStreaming({
      isStreaming: true,
      content: targetMessage.content,
      thinking: targetMessage.thinking || '',
      messageId,
      startTime: Date.now(),
    })

    const settings = useSettingsStore.getState()

    await streamManager.run(
      conversationId,
      {
        model: conversation.defaultModelId,
        messages: apiMessages,
        ...settings.samplerSettings,
      },
      { appendStreamContent, finalizeStream }
    )
  },

  generateNextTurn: async (conversationId) => {
    const { addMessage, setStreaming, appendStreamContent, finalizeStream } = get()
    const conversation = get().conversations.find(c => c.id === conversationId)
    if (!conversation) return

    // Build API messages from full history
    const apiMessages = buildAPIMessages(conversation.messages, {
      systemPrompt: conversation.systemPrompt,
    })

    // Create new assistant placeholder
    const assistantMessageId = addMessage(conversationId, {
      role: 'assistant',
      content: '',
      modelId: conversation.defaultModelId,
    })

    // Start streaming
    setStreaming({
      isStreaming: true,
      content: '',
      thinking: '',
      messageId: assistantMessageId,
    })

    const settings = useSettingsStore.getState()

    await streamManager.run(
      conversationId,
      {
        model: conversation.defaultModelId,
        messages: apiMessages,
        ...settings.samplerSettings,
      },
      { appendStreamContent, finalizeStream }
    )
  },

  setStreaming: (streamState) => {
    set(state => {
      // When starting a new stream, record the start time
      const newState = { ...state.streaming, ...streamState }
      if (streamState.isStreaming && !state.streaming.isStreaming) {
        newState.startTime = Date.now()
        newState.firstTokenTime = undefined
        newState.rawEvents = []
      }
      return { streaming: newState }
    })
  },

  appendStreamContent: (content, isThinking, rawEvent) => {
    set(state => {
      const now = Date.now()
      const isFirstToken = !state.streaming.firstTokenTime && content.length > 0

      return {
        streaming: {
          ...state.streaming,
          [isThinking ? 'thinking' : 'content']:
            state.streaming[isThinking ? 'thinking' : 'content'] + content,
          // Capture first token time
          firstTokenTime: isFirstToken ? now : state.streaming.firstTokenTime,
          // Store raw events for debugging
          rawEvents: rawEvent
            ? [...(state.streaming.rawEvents || []), rawEvent]
            : state.streaming.rawEvents,
        },
      }
    })
  },

  finalizeStream: (completionData, targetConversationId) => {
    const { streaming } = get()
    const conversationId = targetConversationId || get().activeConversationId
    if (!streaming.messageId || !conversationId) return

    // Extract data from completion response
    const usage = completionData?.usage
    const timing = completionData?.timing
    const generationConfig = completionData?.generationConfig
    const stopReason = completionData?.stopReason

    // Calculate performance metrics
    const now = Date.now()
    const tokenCount = usage?.completion_tokens
    const promptTokens = usage?.prompt_tokens

    // Use server timing if available, fallback to client-side calculation
    const totalDuration = timing?.total_duration_ms ?? (streaming.startTime ? now - streaming.startTime : undefined)
    const thinkingDuration = timing?.thinking_duration_ms

    const timeToFirstToken = streaming.startTime && streaming.firstTokenTime
      ? streaming.firstTokenTime - streaming.startTime
      : undefined
    const tokensPerSecond = tokenCount && totalDuration && totalDuration > 0
      ? (tokenCount / totalDuration) * 1000
      : undefined

    // Update the message with final content, token count, and performance metrics
    get().updateMessage(conversationId, streaming.messageId, {
      content: streaming.content,
      thinking: streaming.thinking || undefined,
      tokenCount,
      isRegenerating: false,
      performance: {
        timeToFirstToken,
        tokensPerSecond,
        totalDuration,
        promptTokens,
        completionTokens: tokenCount,
        // Enhanced metrics from backend
        thinkingTokens: usage?.thinking_tokens,
        contentTokens: usage?.content_tokens,
        thinkingDuration,
        stopReason,
        generationConfig,
      },
      rawStream: streaming.rawEvents,
    })

    // Reset streaming state
    set({
      streaming: {
        isStreaming: false,
        content: '',
        thinking: '',
        messageId: null,
        startTime: undefined,
        firstTokenTime: undefined,
        rawEvents: undefined,
      },
    })
  },

  startEditing: (messageId, content) => {
    set({
      editState: {
        messageId,
        originalContent: content,
        editedContent: content,
      },
    })
  },

  updateEditContent: (content) => {
    set(state => ({
      editState: { ...state.editState, editedContent: content },
    }))
  },

  cancelEditing: () => {
    set({
      editState: {
        messageId: null,
        originalContent: '',
        editedContent: '',
      },
    })
  },

  saveEdit: () => {
    const { editState, activeConversationId } = get()
    if (!editState.messageId || !activeConversationId) return null

    const result = {
      conversationId: activeConversationId,
      messageId: editState.messageId,
      content: editState.editedContent,
    }

    // Update the message
    get().updateMessage(activeConversationId, editState.messageId, {
      content: editState.editedContent,
      isEditing: false,
    })

    // Reset edit state
    set({
      editState: {
        messageId: null,
        originalContent: '',
        editedContent: '',
      },
    })

    return result
  },

  loadFromDB: async () => {
    try {
      const conversations = await db.getAllConversations()
      set({ conversations })
    } catch (error) {
      console.error('Failed to load conversations from DB:', error)
    }
  },

  setConversations: (conversations) => {
    set({ conversations })
  },

  getConversationById: (id) => {
    return get().conversations.find(c => c.id === id)
  },
}))

// Initialize: load conversations from DB on startup
db.getAllConversations()
  .then(conversations => {
    useChatStore.setState({ conversations })
  })
  .catch(console.error)
