import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import type { Conversation, Message } from '../types/chat'
import type { PersistedNotebookDocument } from './db'

// Mock data helpers
function createMockMessage(overrides: Partial<Message> = {}): Message {
  return {
    id: `msg-${Math.random().toString(36).slice(2)}`,
    role: 'user',
    content: 'Test message',
    timestamp: Date.now(),
    ...overrides,
  }
}

function createMockConversation(overrides: Partial<Conversation> = {}): Conversation {
  const now = Date.now()
  return {
    id: `conv-${Math.random().toString(36).slice(2)}`,
    title: 'Test Conversation',
    defaultModelId: 'test-model',
    messages: [createMockMessage()],
    createdAt: now,
    updatedAt: now,
    ...overrides,
  }
}

function createMockNotebook(overrides: Partial<PersistedNotebookDocument> = {}): PersistedNotebookDocument {
  const now = Date.now()
  return {
    id: `nb-${Math.random().toString(36).slice(2)}`,
    title: 'Test Notebook',
    content: 'Some content',
    systemPrompt: '',
    images: [],
    createdAt: now,
    modifiedAt: now,
    ...overrides,
  }
}

// In-memory stores keyed by store name
const stores: Record<string, Map<string, unknown>> = {
  conversations: new Map(),
  notebooks: new Map(),
}

// Mock database methods
const mockGetAll = vi.fn((storeName: string) => {
  const store = stores[storeName] ?? new Map()
  return Promise.resolve(Array.from(store.values()))
})

const mockPut = vi.fn((storeName: string, value: { id: string }) => {
  const store = stores[storeName] ?? new Map()
  store.set(value.id, value)
  return Promise.resolve(value.id)
})

const mockDeleteFn = vi.fn((storeName: string, key: string) => {
  const store = stores[storeName] ?? new Map()
  store.delete(key)
  return Promise.resolve()
})

// Mock transaction -- used by importConversations and saveAllNotebookDocuments
const mockTxStore = {
  put: vi.fn((value: { id: string }) => {
    // Transaction writes go to the conversations or notebooks store depending on context
    // importConversations uses 'conversations', saveAllNotebookDocuments uses 'notebooks'
    return Promise.resolve(value.id)
  }),
}

const mockTransaction = vi.fn((_storeName: string) => ({
  store: mockTxStore,
  done: Promise.resolve(),
}))

const mockDB = {
  getAll: mockGetAll,
  put: mockPut,
  delete: mockDeleteFn,
  transaction: mockTransaction,
}

// Mock idb -- openDB returns our mock db
vi.mock('idb', () => ({
  openDB: vi.fn(() => Promise.resolve(mockDB)),
}))

// Import after mocking
import {
  getAllConversations,
  saveConversation,
  deleteConversation,
  exportConversations,
  importConversations,
  getAllNotebookDocuments,
  saveNotebookDocument,
  deleteNotebookDocument,
  saveAllNotebookDocuments,
} from './db'

describe('IndexedDB Wrapper', () => {
  beforeEach(() => {
    stores.conversations.clear()
    stores.notebooks.clear()
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  // ===========================================================================
  // Conversation CRUD
  // ===========================================================================
  describe('saveConversation', () => {
    it('saves a new conversation', async () => {
      const conversation = createMockConversation({ id: 'conv-1' })
      await saveConversation(conversation)

      expect(mockPut).toHaveBeenCalledWith('conversations', conversation)
    })

    it('updates an existing conversation', async () => {
      const conversation = createMockConversation({ id: 'conv-update' })
      await saveConversation(conversation)

      const updated = { ...conversation, title: 'Updated Title', updatedAt: Date.now() + 1000 }
      await saveConversation(updated)

      expect(mockPut).toHaveBeenCalledTimes(2)
      expect(mockPut).toHaveBeenLastCalledWith('conversations', updated)
    })

    it('saves conversation with all fields', async () => {
      const conversation = createMockConversation({
        id: 'conv-full',
        title: 'Full Conversation',
        defaultModelId: 'gpt-4',
        systemPrompt: 'You are a helpful assistant',
        messages: [
          createMockMessage({ role: 'system', content: 'System message' }),
          createMockMessage({ role: 'user', content: 'Hello' }),
          createMockMessage({ role: 'assistant', content: 'Hi there!' }),
        ],
      })

      await saveConversation(conversation)
      expect(mockPut).toHaveBeenCalledWith('conversations', conversation)
    })
  })

  describe('getAllConversations', () => {
    it('returns all conversations sorted by updatedAt descending', async () => {
      const conv1 = createMockConversation({ id: 'conv-1', updatedAt: 1000 })
      const conv2 = createMockConversation({ id: 'conv-2', updatedAt: 3000 })
      const conv3 = createMockConversation({ id: 'conv-3', updatedAt: 2000 })

      stores.conversations.set('conv-1', conv1)
      stores.conversations.set('conv-2', conv2)
      stores.conversations.set('conv-3', conv3)

      const result = await getAllConversations()

      expect(mockGetAll).toHaveBeenCalledWith('conversations')
      expect(result).toHaveLength(3)
      expect(result[0].id).toBe('conv-2')
      expect(result[1].id).toBe('conv-3')
      expect(result[2].id).toBe('conv-1')
    })

    it('returns empty array when no conversations exist', async () => {
      const result = await getAllConversations()
      expect(result).toEqual([])
    })
  })

  describe('deleteConversation', () => {
    it('deletes an existing conversation', async () => {
      stores.conversations.set('conv-delete', createMockConversation({ id: 'conv-delete' }))

      await deleteConversation('conv-delete')

      expect(mockDeleteFn).toHaveBeenCalledWith('conversations', 'conv-delete')
    })

    it('handles deleting non-existent conversation gracefully', async () => {
      await expect(deleteConversation('non-existent')).resolves.not.toThrow()
      expect(mockDeleteFn).toHaveBeenCalledWith('conversations', 'non-existent')
    })
  })

  // ===========================================================================
  // Export / Import
  // ===========================================================================
  describe('exportConversations', () => {
    it('returns valid JSON string of all conversations', async () => {
      stores.conversations.set('conv-1', createMockConversation({ id: 'conv-1', title: 'First' }))
      stores.conversations.set('conv-2', createMockConversation({ id: 'conv-2', title: 'Second' }))

      const result = await exportConversations()
      const parsed = JSON.parse(result)
      expect(Array.isArray(parsed)).toBe(true)
      expect(parsed).toHaveLength(2)
    })

    it('returns formatted JSON (pretty-printed)', async () => {
      stores.conversations.set('conv-1', createMockConversation({ id: 'conv-1' }))

      const result = await exportConversations()
      expect(result).toContain('\n')
      expect(result).toContain('  ')
    })

    it('exports conversations in correct order (most recent first)', async () => {
      stores.conversations.set('conv-1', createMockConversation({ id: 'conv-1', title: 'Oldest', updatedAt: 1000 }))
      stores.conversations.set('conv-2', createMockConversation({ id: 'conv-2', title: 'Newest', updatedAt: 3000 }))

      const result = await exportConversations()
      const parsed = JSON.parse(result) as Conversation[]
      expect(parsed[0].title).toBe('Newest')
      expect(parsed[1].title).toBe('Oldest')
    })

    it('returns empty array JSON when no conversations exist', async () => {
      const result = await exportConversations()
      expect(result).toBe('[]')
    })

    it('preserves all conversation data in export', async () => {
      const conv = createMockConversation({
        id: 'conv-full',
        title: 'Full Export Test',
        defaultModelId: 'test-model',
        systemPrompt: 'System prompt here',
        messages: [
          createMockMessage({ role: 'user', content: 'Hello', thinking: 'user thought' }),
          createMockMessage({ role: 'assistant', content: 'Hi!', thinking: 'assistant thought' }),
        ],
        createdAt: 1000,
        updatedAt: 2000,
      })

      stores.conversations.set('conv-full', conv)

      const result = await exportConversations()
      const parsed = JSON.parse(result) as Conversation[]

      expect(parsed[0]).toEqual(conv)
      expect(parsed[0].systemPrompt).toBe('System prompt here')
      expect(parsed[0].messages).toHaveLength(2)
    })
  })

  describe('importConversations', () => {
    it('parses JSON and saves conversations via transaction', async () => {
      const conversations = [
        createMockConversation({ id: 'import-1', title: 'Imported 1' }),
        createMockConversation({ id: 'import-2', title: 'Imported 2' }),
      ]

      const count = await importConversations(JSON.stringify(conversations))

      expect(count).toBe(2)
      expect(mockTransaction).toHaveBeenCalledWith('conversations', 'readwrite')
      expect(mockTxStore.put).toHaveBeenCalledTimes(2)
    })

    it('returns the count of imported conversations', async () => {
      const conversations = [
        createMockConversation({ id: 'import-1' }),
        createMockConversation({ id: 'import-2' }),
        createMockConversation({ id: 'import-3' }),
      ]

      const count = await importConversations(JSON.stringify(conversations))
      expect(count).toBe(3)
    })

    it('handles empty JSON array', async () => {
      const count = await importConversations('[]')

      expect(count).toBe(0)
      expect(mockTxStore.put).not.toHaveBeenCalled()
    })

    it('throws error for invalid JSON', async () => {
      await expect(importConversations('invalid json {')).rejects.toThrow()
    })

    it('throws error for non-array JSON', async () => {
      await expect(importConversations('{"not": "array"}')).rejects.toThrow('expected an array')
    })

    it('imports conversations with all fields preserved', async () => {
      const conv = createMockConversation({
        id: 'import-full',
        title: 'Full Import',
        defaultModelId: 'imported-model',
        systemPrompt: 'Imported system prompt',
        messages: [createMockMessage({ role: 'user', content: 'Imported message' })],
        createdAt: 5000,
        updatedAt: 6000,
      })

      await importConversations(JSON.stringify([conv]))
      expect(mockTxStore.put).toHaveBeenCalledWith(conv)
    })
  })

  // ===========================================================================
  // Notebook CRUD
  // ===========================================================================
  describe('getAllNotebookDocuments', () => {
    it('returns all notebooks sorted by modifiedAt descending', async () => {
      const nb1 = createMockNotebook({ id: 'nb-1', modifiedAt: 1000 })
      const nb2 = createMockNotebook({ id: 'nb-2', modifiedAt: 3000 })
      const nb3 = createMockNotebook({ id: 'nb-3', modifiedAt: 2000 })

      stores.notebooks.set('nb-1', nb1)
      stores.notebooks.set('nb-2', nb2)
      stores.notebooks.set('nb-3', nb3)

      const result = await getAllNotebookDocuments()

      expect(mockGetAll).toHaveBeenCalledWith('notebooks')
      expect(result).toHaveLength(3)
      expect(result[0].id).toBe('nb-2')
      expect(result[1].id).toBe('nb-3')
      expect(result[2].id).toBe('nb-1')
    })

    it('returns empty array when no notebooks exist', async () => {
      const result = await getAllNotebookDocuments()
      expect(result).toEqual([])
    })
  })

  describe('saveNotebookDocument', () => {
    it('saves a notebook document', async () => {
      const nb = createMockNotebook({ id: 'nb-1' })
      await saveNotebookDocument(nb)
      expect(mockPut).toHaveBeenCalledWith('notebooks', nb)
    })
  })

  describe('deleteNotebookDocument', () => {
    it('deletes a notebook document', async () => {
      await deleteNotebookDocument('nb-1')
      expect(mockDeleteFn).toHaveBeenCalledWith('notebooks', 'nb-1')
    })
  })

  describe('saveAllNotebookDocuments', () => {
    it('saves multiple notebooks in a transaction', async () => {
      const docs = [
        createMockNotebook({ id: 'nb-1' }),
        createMockNotebook({ id: 'nb-2' }),
      ]

      await saveAllNotebookDocuments(docs)

      expect(mockTransaction).toHaveBeenCalledWith('notebooks', 'readwrite')
      expect(mockTxStore.put).toHaveBeenCalledTimes(2)
    })

    it('handles empty array', async () => {
      await saveAllNotebookDocuments([])
      expect(mockTransaction).toHaveBeenCalledWith('notebooks', 'readwrite')
      expect(mockTxStore.put).not.toHaveBeenCalled()
    })
  })

  // ===========================================================================
  // Edge Cases
  // ===========================================================================
  describe('edge cases', () => {
    it('handles conversation with empty messages array', async () => {
      const conv = createMockConversation({ id: 'empty-msgs', messages: [] })
      await saveConversation(conv)
      expect(mockPut).toHaveBeenCalledWith('conversations', conv)
      expect(conv.messages).toEqual([])
    })

    it('handles conversation with special characters in title', async () => {
      const conv = createMockConversation({
        id: 'special-chars',
        title: 'Test <script>alert("xss")</script> & "quotes"',
      })

      stores.conversations.set('special-chars', conv)
      const exported = await exportConversations()
      const parsed = JSON.parse(exported) as Conversation[]

      expect(parsed[0].title).toBe('Test <script>alert("xss")</script> & "quotes"')
    })

    it('handles concurrent saves to same conversation', async () => {
      const conv1 = createMockConversation({ id: 'concurrent', title: 'Version 1' })
      const conv2 = createMockConversation({ id: 'concurrent', title: 'Version 2' })

      await Promise.all([
        saveConversation(conv1),
        saveConversation(conv2),
      ])

      expect(mockPut).toHaveBeenCalledTimes(2)
    })

    it('retries openDB after a rejection', async () => {
      // This test needs a fresh module to start with dbPromise = null.
      // We use vi.resetModules + dynamic import to get a clean slate.
      vi.resetModules()

      const { openDB: mockOpenDB } = await import('idb')
      const mockedOpenDB = vi.mocked(mockOpenDB)

      // First call: openDB rejects
      mockedOpenDB.mockImplementationOnce(() => Promise.reject(new Error('QuotaExceededError')))

      // Second call: openDB succeeds
      mockedOpenDB.mockImplementationOnce(() => Promise.resolve(mockDB as never))

      // Re-import db module to get fresh module state
      const { getAllConversations: freshGetAll } = await import('./db')

      // First call should fail
      await expect(freshGetAll()).rejects.toThrow('QuotaExceededError')

      // Second call should succeed (dbPromise was reset by .catch handler)
      const result = await freshGetAll()
      expect(result).toEqual([])
    })

    it('export round-trip preserves data', async () => {
      const original = createMockConversation({
        id: 'roundtrip',
        title: 'Round Trip',
        messages: [createMockMessage({ content: 'test' })],
      })

      stores.conversations.set('roundtrip', original)
      const exported = await exportConversations()
      const count = await importConversations(exported)

      expect(count).toBe(1)
      expect(mockTxStore.put).toHaveBeenCalledWith(original)
    })
  })
})
