import { create } from 'zustand'
import type { Message, Conversation, StreamingState, EditState } from '../types/chat'
import { streamChat } from '../api/streaming'
import { useSettingsStore } from './settingsStore'
import * as db from '../lib/db'

// Re-export StreamingState for components
export type { StreamingState }

// Store abort controller for cancellation
let abortController: AbortController | null = null

// Debounced save to avoid too many DB writes
let saveTimeout: ReturnType<typeof setTimeout> | null = null
function debouncedSave(conversation: Conversation) {
  if (saveTimeout) clearTimeout(saveTimeout)
  saveTimeout = setTimeout(() => {
    db.saveConversation(conversation).catch(console.error)
  }, 500)
}

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
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
  editMessageAndRegenerate: (conversationId: string, messageId: string, newContent: string, shouldRegenerate: boolean) => Promise<void>
  deleteMessageWithCascade: (conversationId: string, messageId: string, shouldRegenerateNext?: boolean) => Promise<void>

  // Streaming
  setStreaming: (state: Partial<StreamingState>) => void
  appendStreamContent: (content: string, isThinking: boolean, rawEvent?: string) => void
  finalizeStream: (tokenCount?: number, promptTokens?: number) => void

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
      modelId,
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

    // Add user message
    addMessage(conversationId, {
      role: 'user',
      content,
      images,
    })

    // Create placeholder for assistant message
    const assistantMessageId = addMessage(conversationId, {
      role: 'assistant',
      content: '',
    })

    // Start streaming
    setStreaming({
      isStreaming: true,
      content: '',
      thinking: '',
      messageId: assistantMessageId,
    })

    // Create abort controller
    abortController = new AbortController()

    // Get current settings
    const settings = useSettingsStore.getState()

    // Build messages array for API
    const apiMessages = get().conversations.find(c => c.id === conversationId)?.messages
      .filter(m => m.id !== assistantMessageId) // Exclude the empty placeholder
      .map(m => ({
        role: m.role as 'system' | 'user' | 'assistant',
        content: m.images && m.images.length > 0
          ? [
              { type: 'text' as const, text: m.content },
              ...m.images.map(img => ({ type: 'image_url' as const, image_url: { url: img } })),
            ]
          : m.content,
      })) || []

    // Add system prompt if set
    if (conversation.systemPrompt) {
      apiMessages.unshift({ role: 'system', content: conversation.systemPrompt })
    }

    await streamChat(
      {
        model: modelId,
        messages: apiMessages,
        ...settings.samplerSettings,
      },
      {
        onToken: (token, rawEvent) => appendStreamContent(token, false, rawEvent),
        onThinking: (thinking, rawEvent) => appendStreamContent(thinking, true, rawEvent),
        onComplete: (usage) => finalizeStream(usage?.completion_tokens, usage?.prompt_tokens),
        onError: (error) => {
          console.error('Stream error:', error)
          finalizeStream()
        },
      },
      abortController.signal
    )
  },

  stopGeneration: () => {
    if (abortController) {
      abortController.abort()
      abortController = null
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

    // Create new assistant placeholder
    const assistantMessageId = addMessage(conversationId, {
      role: 'assistant',
      content: '',
      isRegenerating: true,
    })

    // Start streaming
    setStreaming({
      isStreaming: true,
      content: '',
      thinking: '',
      messageId: assistantMessageId,
    })

    abortController = new AbortController()
    const settings = useSettingsStore.getState()

    // Get updated conversation
    const updatedConversation = get().conversations.find(c => c.id === conversationId)
    if (!updatedConversation) return

    const apiMessages = updatedConversation.messages
      .filter(m => m.id !== assistantMessageId)
      .map(m => ({
        role: m.role as 'system' | 'user' | 'assistant',
        content: m.images && m.images.length > 0
          ? [
              { type: 'text' as const, text: m.content },
              ...m.images.map(img => ({ type: 'image_url' as const, image_url: { url: img } })),
            ]
          : m.content,
      }))

    if (updatedConversation.systemPrompt) {
      apiMessages.unshift({ role: 'system', content: updatedConversation.systemPrompt })
    }

    await streamChat(
      {
        model: updatedConversation.modelId,
        messages: apiMessages,
        ...settings.samplerSettings,
      },
      {
        onToken: (token, rawEvent) => appendStreamContent(token, false, rawEvent),
        onThinking: (thinking, rawEvent) => appendStreamContent(thinking, true, rawEvent),
        onComplete: (usage) => finalizeStream(usage?.completion_tokens, usage?.prompt_tokens),
        onError: (error) => {
          console.error('Regenerate error:', error)
          finalizeStream()
        },
      },
      abortController.signal
    )
  },

  editMessageAndRegenerate: async (conversationId, messageId, newContent, shouldRegenerate) => {
    const { updateMessage, regenerateFromPosition } = get()
    const conversation = get().conversations.find(c => c.id === conversationId)
    if (!conversation) return

    const messageIndex = conversation.messages.findIndex(m => m.id === messageId)
    if (messageIndex === -1) return

    // Update the message content
    updateMessage(conversationId, messageId, { content: newContent })

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

  finalizeStream: (tokenCount, promptTokens) => {
    const { streaming, activeConversationId } = get()
    if (!streaming.messageId || !activeConversationId) return

    // Calculate performance metrics
    const now = Date.now()
    const totalDuration = streaming.startTime ? now - streaming.startTime : undefined
    const timeToFirstToken = streaming.startTime && streaming.firstTokenTime
      ? streaming.firstTokenTime - streaming.startTime
      : undefined
    const tokensPerSecond = tokenCount && totalDuration && totalDuration > 0
      ? (tokenCount / totalDuration) * 1000
      : undefined

    // Update the message with final content, token count, and performance metrics
    get().updateMessage(activeConversationId, streaming.messageId, {
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
