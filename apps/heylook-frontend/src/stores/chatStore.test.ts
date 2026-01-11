import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useChatStore } from './chatStore'
import type { Conversation, Message } from '../types/chat'

// Mock the db module
vi.mock('../lib/db', () => ({
  saveConversation: vi.fn().mockResolvedValue(undefined),
  deleteConversation: vi.fn().mockResolvedValue(undefined),
  getAllConversations: vi.fn().mockResolvedValue([]),
}))

// Mock the streaming API
vi.mock('../api/streaming', () => ({
  streamChat: vi.fn(),
}))

// Mock settingsStore
vi.mock('./settingsStore', () => ({
  useSettingsStore: {
    getState: vi.fn().mockReturnValue({
      samplerSettings: {
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: 2048,
      },
    }),
  },
}))

function createMockConversation(overrides: Partial<Conversation> = {}): Conversation {
  const now = Date.now()
  return {
    id: `conv-${now}`,
    title: 'Test Conversation',
    defaultModelId: 'test-model',
    messages: [],
    createdAt: now,
    updatedAt: now,
    ...overrides,
  }
}

// Helper for future tests - prefixed to avoid unused warning
function _createMockMessage(overrides: Partial<Message> = {}): Message {
  return {
    id: `msg-${Date.now()}`,
    role: 'user',
    content: 'Test message',
    timestamp: Date.now(),
    ...overrides,
  }
}
void _createMockMessage

describe('chatStore', () => {
  beforeEach(() => {
    // Reset store to initial state before each test
    useChatStore.setState({
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
    })
    vi.clearAllMocks()
  })

  describe('initial state', () => {
    it('has empty conversations array', () => {
      const { conversations } = useChatStore.getState()
      expect(conversations).toEqual([])
    })

    it('has null activeConversationId', () => {
      const { activeConversationId } = useChatStore.getState()
      expect(activeConversationId).toBeNull()
    })

    it('has initial streaming state', () => {
      const { streaming } = useChatStore.getState()
      expect(streaming).toEqual({
        isStreaming: false,
        content: '',
        thinking: '',
        messageId: null,
      })
    })

    it('has initial edit state', () => {
      const { editState } = useChatStore.getState()
      expect(editState).toEqual({
        messageId: null,
        originalContent: '',
        editedContent: '',
      })
    })
  })

  describe('conversation management', () => {
    describe('createConversation', () => {
      it('creates a new conversation', () => {
        const { createConversation } = useChatStore.getState()

        const id = createConversation('test-model')

        const { conversations } = useChatStore.getState()
        expect(conversations).toHaveLength(1)
        expect(conversations[0].id).toBe(id)
        expect(conversations[0].defaultModelId).toBe('test-model')
        expect(conversations[0].title).toBe('New Conversation')
        expect(conversations[0].messages).toEqual([])
      })

      it('sets the new conversation as active', () => {
        const { createConversation } = useChatStore.getState()

        const id = createConversation('test-model')

        const { activeConversationId } = useChatStore.getState()
        expect(activeConversationId).toBe(id)
      })

      it('creates conversation with system prompt', () => {
        const { createConversation } = useChatStore.getState()

        createConversation('test-model', 'You are a helpful assistant.')

        const { conversations } = useChatStore.getState()
        expect(conversations[0].systemPrompt).toBe('You are a helpful assistant.')
      })

      it('adds new conversation to the front of the list', () => {
        const { createConversation } = useChatStore.getState()

        const id1 = createConversation('model-1')
        const id2 = createConversation('model-2')

        const { conversations } = useChatStore.getState()
        expect(conversations[0].id).toBe(id2)
        expect(conversations[1].id).toBe(id1)
      })
    })

    describe('setActiveConversation', () => {
      it('sets active conversation by id', () => {
        const { createConversation, setActiveConversation } = useChatStore.getState()

        const id = createConversation('test-model')
        setActiveConversation(null)

        expect(useChatStore.getState().activeConversationId).toBeNull()

        setActiveConversation(id)

        expect(useChatStore.getState().activeConversationId).toBe(id)
      })

      it('can set active conversation to null', () => {
        const { createConversation, setActiveConversation } = useChatStore.getState()

        createConversation('test-model')
        setActiveConversation(null)

        expect(useChatStore.getState().activeConversationId).toBeNull()
      })
    })

    describe('deleteConversation', () => {
      it('removes conversation from list', () => {
        const { createConversation, deleteConversation } = useChatStore.getState()

        const id = createConversation('test-model')
        expect(useChatStore.getState().conversations).toHaveLength(1)

        deleteConversation(id)

        expect(useChatStore.getState().conversations).toHaveLength(0)
      })

      it('selects first remaining conversation when active is deleted', () => {
        const { createConversation, deleteConversation } = useChatStore.getState()

        const id1 = createConversation('model-1')
        const id2 = createConversation('model-2')

        // id2 is now active (most recent)
        expect(useChatStore.getState().activeConversationId).toBe(id2)

        deleteConversation(id2)

        // Should fall back to id1
        expect(useChatStore.getState().activeConversationId).toBe(id1)
      })

      it('sets activeConversationId to null when last conversation deleted', () => {
        const { createConversation, deleteConversation } = useChatStore.getState()

        const id = createConversation('test-model')
        deleteConversation(id)

        expect(useChatStore.getState().activeConversationId).toBeNull()
      })

      it('keeps activeConversationId when non-active conversation deleted', () => {
        const { createConversation, deleteConversation, setActiveConversation } = useChatStore.getState()

        const id1 = createConversation('model-1')
        const id2 = createConversation('model-2')

        // Explicitly set id2 as active
        setActiveConversation(id2)
        deleteConversation(id1)

        expect(useChatStore.getState().activeConversationId).toBe(id2)
      })
    })

    describe('updateConversationTitle', () => {
      it('updates conversation title', () => {
        const { createConversation, updateConversationTitle } = useChatStore.getState()

        const id = createConversation('test-model')
        updateConversationTitle(id, 'Updated Title')

        const { conversations } = useChatStore.getState()
        expect(conversations[0].title).toBe('Updated Title')
      })

      it('updates the updatedAt timestamp', () => {
        const { createConversation, updateConversationTitle } = useChatStore.getState()

        const id = createConversation('test-model')
        const originalUpdatedAt = useChatStore.getState().conversations[0].updatedAt

        updateConversationTitle(id, 'Updated Title')

        const { conversations } = useChatStore.getState()
        // updatedAt should be >= original (same or later)
        expect(conversations[0].updatedAt).toBeGreaterThanOrEqual(originalUpdatedAt)
      })
    })

    describe('activeConversation', () => {
      it('returns undefined when no active conversation', () => {
        const { activeConversation } = useChatStore.getState()
        expect(activeConversation()).toBeUndefined()
      })

      it('returns active conversation when set', () => {
        const { createConversation } = useChatStore.getState()

        const id = createConversation('test-model')

        const active = useChatStore.getState().activeConversation()
        expect(active).toBeDefined()
        expect(active?.id).toBe(id)
      })
    })

    describe('getConversationById', () => {
      it('returns conversation by id', () => {
        const { createConversation } = useChatStore.getState()

        const id = createConversation('test-model')

        const conversation = useChatStore.getState().getConversationById(id)
        expect(conversation).toBeDefined()
        expect(conversation?.id).toBe(id)
      })

      it('returns undefined for non-existent id', () => {
        const { getConversationById } = useChatStore.getState()

        const conversation = getConversationById('non-existent')
        expect(conversation).toBeUndefined()
      })
    })
  })

  describe('message management', () => {
    describe('addMessage', () => {
      it('adds message to conversation', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        const msgId = useChatStore.getState().addMessage(convId, {
          role: 'user',
          content: 'Hello',
        })

        const conversation = useChatStore.getState().getConversationById(convId)
        expect(conversation?.messages).toHaveLength(1)
        expect(conversation?.messages[0].id).toBe(msgId)
        expect(conversation?.messages[0].content).toBe('Hello')
        expect(conversation?.messages[0].role).toBe('user')
      })

      it('generates unique message id', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        const msgId1 = useChatStore.getState().addMessage(convId, { role: 'user', content: 'First' })
        const msgId2 = useChatStore.getState().addMessage(convId, { role: 'user', content: 'Second' })

        expect(msgId1).not.toBe(msgId2)
      })

      it('adds timestamp to message', () => {
        const { createConversation } = useChatStore.getState()

        const beforeTime = Date.now()
        const convId = createConversation('test-model')
        useChatStore.getState().addMessage(convId, { role: 'user', content: 'Test' })
        const afterTime = Date.now()

        const conversation = useChatStore.getState().getConversationById(convId)
        expect(conversation?.messages[0].timestamp).toBeGreaterThanOrEqual(beforeTime)
        expect(conversation?.messages[0].timestamp).toBeLessThanOrEqual(afterTime)
      })

      it('updates conversation title from first user message', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        expect(useChatStore.getState().getConversationById(convId)?.title).toBe('New Conversation')

        useChatStore.getState().addMessage(convId, {
          role: 'user',
          content: 'What is the weather today?',
        })

        expect(useChatStore.getState().getConversationById(convId)?.title).toBe('What is the weather today?')
      })

      it('truncates long messages for title', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        const longContent = 'A'.repeat(100)

        useChatStore.getState().addMessage(convId, {
          role: 'user',
          content: longContent,
        })

        const title = useChatStore.getState().getConversationById(convId)?.title
        expect(title).toBe('A'.repeat(50) + '...')
      })

      it('does not update title after first message', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        useChatStore.getState().addMessage(convId, { role: 'user', content: 'First message' })
        useChatStore.getState().addMessage(convId, { role: 'user', content: 'Second message' })

        expect(useChatStore.getState().getConversationById(convId)?.title).toBe('First message')
      })

      it('adds message with images', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        useChatStore.getState().addMessage(convId, {
          role: 'user',
          content: 'Check this image',
          images: ['data:image/png;base64,abc123'],
        })

        const conversation = useChatStore.getState().getConversationById(convId)
        expect(conversation?.messages[0].images).toEqual(['data:image/png;base64,abc123'])
      })
    })

    describe('updateMessage', () => {
      it('updates message content', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        const msgId = useChatStore.getState().addMessage(convId, { role: 'user', content: 'Original' })

        useChatStore.getState().updateMessage(convId, msgId, { content: 'Updated' })

        const conversation = useChatStore.getState().getConversationById(convId)
        expect(conversation?.messages[0].content).toBe('Updated')
      })

      it('updates message thinking field', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        const msgId = useChatStore.getState().addMessage(convId, { role: 'assistant', content: 'Response' })

        useChatStore.getState().updateMessage(convId, msgId, { thinking: 'Internal reasoning...' })

        const conversation = useChatStore.getState().getConversationById(convId)
        expect(conversation?.messages[0].thinking).toBe('Internal reasoning...')
      })

      it('updates conversation updatedAt timestamp', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        const msgId = useChatStore.getState().addMessage(convId, { role: 'user', content: 'Test' })
        const originalUpdatedAt = useChatStore.getState().getConversationById(convId)?.updatedAt

        useChatStore.getState().updateMessage(convId, msgId, { content: 'Updated' })

        const newUpdatedAt = useChatStore.getState().getConversationById(convId)?.updatedAt
        expect(newUpdatedAt).toBeGreaterThanOrEqual(originalUpdatedAt || 0)
      })
    })

    describe('deleteMessage', () => {
      it('removes message from conversation', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        const msgId = useChatStore.getState().addMessage(convId, { role: 'user', content: 'Test' })

        expect(useChatStore.getState().getConversationById(convId)?.messages).toHaveLength(1)

        useChatStore.getState().deleteMessage(convId, msgId)

        expect(useChatStore.getState().getConversationById(convId)?.messages).toHaveLength(0)
      })

      it('only removes specified message', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        const msgId1 = useChatStore.getState().addMessage(convId, { role: 'user', content: 'First' })
        const msgId2 = useChatStore.getState().addMessage(convId, { role: 'user', content: 'Second' })

        useChatStore.getState().deleteMessage(convId, msgId1)

        const conversation = useChatStore.getState().getConversationById(convId)
        expect(conversation?.messages).toHaveLength(1)
        expect(conversation?.messages[0].id).toBe(msgId2)
      })
    })

    describe('deleteMessageAndDownstream', () => {
      it('deletes message and all messages after it', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        const msgId1 = useChatStore.getState().addMessage(convId, { role: 'user', content: 'First' })
        const msgId2 = useChatStore.getState().addMessage(convId, { role: 'assistant', content: 'Response' })
        useChatStore.getState().addMessage(convId, { role: 'user', content: 'Third' })

        useChatStore.getState().deleteMessageAndDownstream(convId, msgId2)

        const conversation = useChatStore.getState().getConversationById(convId)
        expect(conversation?.messages).toHaveLength(1)
        expect(conversation?.messages[0].id).toBe(msgId1)
      })

      it('handles non-existent message id gracefully', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        useChatStore.getState().addMessage(convId, { role: 'user', content: 'Test' })

        // Should not throw
        useChatStore.getState().deleteMessageAndDownstream(convId, 'non-existent')

        expect(useChatStore.getState().getConversationById(convId)?.messages).toHaveLength(1)
      })
    })

    describe('getMessagesUpTo', () => {
      it('returns messages up to but not including specified message', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        useChatStore.getState().addMessage(convId, { role: 'user', content: 'First' })
        const msgId2 = useChatStore.getState().addMessage(convId, { role: 'assistant', content: 'Second' })
        useChatStore.getState().addMessage(convId, { role: 'user', content: 'Third' })

        const messages = useChatStore.getState().getMessagesUpTo(convId, msgId2)

        expect(messages).toHaveLength(1)
        expect(messages[0].content).toBe('First')
      })

      it('returns all messages if message id not found', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        useChatStore.getState().addMessage(convId, { role: 'user', content: 'First' })
        useChatStore.getState().addMessage(convId, { role: 'user', content: 'Second' })

        const messages = useChatStore.getState().getMessagesUpTo(convId, 'non-existent')

        expect(messages).toHaveLength(2)
      })

      it('returns empty array for non-existent conversation', () => {
        const { getMessagesUpTo } = useChatStore.getState()

        const messages = getMessagesUpTo('non-existent', 'any-id')

        expect(messages).toEqual([])
      })
    })
  })

  describe('streaming state management', () => {
    describe('setStreaming', () => {
      it('updates streaming state partially', () => {
        useChatStore.getState().setStreaming({ isStreaming: true })

        expect(useChatStore.getState().streaming.isStreaming).toBe(true)
        expect(useChatStore.getState().streaming.content).toBe('')
      })

      it('updates multiple streaming fields', () => {
        useChatStore.getState().setStreaming({
          isStreaming: true,
          content: 'Hello',
          messageId: 'msg-123',
        })

        const { streaming } = useChatStore.getState()
        expect(streaming.isStreaming).toBe(true)
        expect(streaming.content).toBe('Hello')
        expect(streaming.messageId).toBe('msg-123')
      })
    })

    describe('appendStreamContent', () => {
      it('appends content to streaming content', () => {
        useChatStore.getState().setStreaming({ isStreaming: true, content: 'Hello' })
        useChatStore.getState().appendStreamContent(' World', false)

        expect(useChatStore.getState().streaming.content).toBe('Hello World')
      })

      it('appends thinking content separately', () => {
        useChatStore.getState().setStreaming({ isStreaming: true, thinking: 'Let me' })
        useChatStore.getState().appendStreamContent(' think...', true)

        expect(useChatStore.getState().streaming.thinking).toBe('Let me think...')
        expect(useChatStore.getState().streaming.content).toBe('')
      })
    })

    describe('finalizeStream', () => {
      it('updates message with final content', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        const msgId = useChatStore.getState().addMessage(convId, { role: 'assistant', content: '' })

        useChatStore.getState().setStreaming({
          isStreaming: true,
          content: 'Final response',
          thinking: 'Internal thoughts',
          messageId: msgId,
        })

        useChatStore.getState().finalizeStream({
          usage: { prompt_tokens: 50, completion_tokens: 100, total_tokens: 150 }
        })

        const conversation = useChatStore.getState().getConversationById(convId)
        expect(conversation?.messages[0].content).toBe('Final response')
        expect(conversation?.messages[0].thinking).toBe('Internal thoughts')
        expect(conversation?.messages[0].tokenCount).toBe(100)
      })

      it('resets streaming state', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        const msgId = useChatStore.getState().addMessage(convId, { role: 'assistant', content: '' })

        useChatStore.getState().setStreaming({
          isStreaming: true,
          content: 'Content',
          messageId: msgId,
        })

        useChatStore.getState().finalizeStream()

        expect(useChatStore.getState().streaming).toEqual({
          isStreaming: false,
          content: '',
          thinking: '',
          messageId: null,
        })
      })

      it('does nothing if no messageId', () => {
        useChatStore.getState().setStreaming({
          isStreaming: true,
          content: 'Content',
          messageId: null,
        })

        useChatStore.getState().finalizeStream()

        // Still has content since finalize didn't run
        expect(useChatStore.getState().streaming.content).toBe('Content')
      })

      it('does nothing if no activeConversationId', () => {
        useChatStore.getState().setActiveConversation(null)
        useChatStore.getState().setStreaming({
          isStreaming: true,
          content: 'Content',
          messageId: 'msg-123',
        })

        useChatStore.getState().finalizeStream()

        // Still has content since finalize didn't run
        expect(useChatStore.getState().streaming.content).toBe('Content')
      })
    })
  })

  describe('edit mode', () => {
    describe('startEditing', () => {
      it('sets edit state with message info', () => {
        useChatStore.getState().startEditing('msg-123', 'Original content')

        expect(useChatStore.getState().editState).toEqual({
          messageId: 'msg-123',
          originalContent: 'Original content',
          editedContent: 'Original content',
        })
      })
    })

    describe('updateEditContent', () => {
      it('updates edited content', () => {
        useChatStore.getState().startEditing('msg-123', 'Original')
        useChatStore.getState().updateEditContent('Modified')

        expect(useChatStore.getState().editState.editedContent).toBe('Modified')
        expect(useChatStore.getState().editState.originalContent).toBe('Original')
      })
    })

    describe('cancelEditing', () => {
      it('resets edit state', () => {
        useChatStore.getState().startEditing('msg-123', 'Content')
        useChatStore.getState().cancelEditing()

        expect(useChatStore.getState().editState).toEqual({
          messageId: null,
          originalContent: '',
          editedContent: '',
        })
      })
    })

    describe('saveEdit', () => {
      it('returns null if no messageId', () => {
        const result = useChatStore.getState().saveEdit()

        expect(result).toBeNull()
      })

      it('returns null if no activeConversationId', () => {
        useChatStore.getState().startEditing('msg-123', 'Content')
        const result = useChatStore.getState().saveEdit()

        expect(result).toBeNull()
      })

      it('updates message and returns edit info', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        const msgId = useChatStore.getState().addMessage(convId, { role: 'user', content: 'Original' })

        useChatStore.getState().startEditing(msgId, 'Original')
        useChatStore.getState().updateEditContent('Modified')
        const result = useChatStore.getState().saveEdit()

        expect(result).toEqual({
          conversationId: convId,
          messageId: msgId,
          content: 'Modified',
        })

        const conversation = useChatStore.getState().getConversationById(convId)
        expect(conversation?.messages[0].content).toBe('Modified')
      })

      it('resets edit state after save', () => {
        const { createConversation } = useChatStore.getState()

        const convId = createConversation('test-model')
        const msgId = useChatStore.getState().addMessage(convId, { role: 'user', content: 'Original' })

        useChatStore.getState().startEditing(msgId, 'Original')
        useChatStore.getState().saveEdit()

        expect(useChatStore.getState().editState).toEqual({
          messageId: null,
          originalContent: '',
          editedContent: '',
        })
      })
    })
  })

  describe('persistence', () => {
    describe('setConversations', () => {
      it('sets conversations array', () => {
        const mockConversations = [
          createMockConversation({ id: 'conv-1' }),
          createMockConversation({ id: 'conv-2' }),
        ]

        useChatStore.getState().setConversations(mockConversations)

        expect(useChatStore.getState().conversations).toEqual(mockConversations)
      })
    })

    describe('loadFromDB', () => {
      it('loads conversations from database', async () => {
        const db = await import('../lib/db')
        const mockConversations = [
          createMockConversation({ id: 'db-conv-1' }),
        ]
        vi.mocked(db.getAllConversations).mockResolvedValue(mockConversations)

        await useChatStore.getState().loadFromDB()

        expect(useChatStore.getState().conversations).toEqual(mockConversations)
      })

      it('handles database errors gracefully', async () => {
        const db = await import('../lib/db')
        const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
        vi.mocked(db.getAllConversations).mockRejectedValue(new Error('DB Error'))

        await useChatStore.getState().loadFromDB()

        expect(consoleSpy).toHaveBeenCalled()
        consoleSpy.mockRestore()
      })
    })
  })

  describe('edge cases', () => {
    it('handles operations on non-existent conversation gracefully', () => {
      // These should not throw
      expect(() => useChatStore.getState().addMessage('non-existent', { role: 'user', content: 'Test' })).not.toThrow()
      expect(() => useChatStore.getState().updateMessage('non-existent', 'msg-id', { content: 'Updated' })).not.toThrow()
      expect(() => useChatStore.getState().deleteMessage('non-existent', 'msg-id')).not.toThrow()
    })

    it('handles multiple rapid conversation creations', () => {
      const ids: string[] = []
      for (let i = 0; i < 10; i++) {
        ids.push(useChatStore.getState().createConversation(`model-${i}`))
      }

      expect(new Set(ids).size).toBe(10) // All unique
      expect(useChatStore.getState().conversations).toHaveLength(10)
    })

    it('handles empty thinking string correctly on finalize', () => {
      const { createConversation } = useChatStore.getState()

      const convId = createConversation('test-model')
      const msgId = useChatStore.getState().addMessage(convId, { role: 'assistant', content: '' })

      useChatStore.getState().setStreaming({
        isStreaming: true,
        content: 'Response',
        thinking: '', // Empty thinking
        messageId: msgId,
      })

      useChatStore.getState().finalizeStream()

      const conversation = useChatStore.getState().getConversationById(convId)
      // Empty string becomes undefined
      expect(conversation?.messages[0].thinking).toBeUndefined()
    })
  })
})
