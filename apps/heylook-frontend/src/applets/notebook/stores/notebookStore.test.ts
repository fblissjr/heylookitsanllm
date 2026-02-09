import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { useNotebookStore } from './notebookStore'
import { buildMessages } from './notebookStore'
import type { NotebookDocument, ImageAttachment } from '../types'

// Mock streaming API
const mockStreamChat = vi.fn()
vi.mock('../../../api/streaming', () => ({
  streamChat: (...args: unknown[]) => mockStreamChat(...args),
}))

// Mock model store
vi.mock('../../../stores/modelStore', () => ({
  useModelStore: {
    getState: () => ({
      loadedModel: { id: 'test-model', contextWindow: 4096, capabilities: {} },
    }),
  },
}))

// Mock db module
const mockGetAllNotebookDocuments = vi.fn()
const mockSaveNotebookDocument = vi.fn()
const mockDeleteNotebookDocument = vi.fn()
const mockSaveAllNotebookDocuments = vi.fn()

vi.mock('../../../lib/db', () => ({
  getAllNotebookDocuments: (...args: unknown[]) => mockGetAllNotebookDocuments(...args),
  saveNotebookDocument: (...args: unknown[]) => mockSaveNotebookDocument(...args),
  deleteNotebookDocument: (...args: unknown[]) => mockDeleteNotebookDocument(...args),
  saveAllNotebookDocuments: (...args: unknown[]) => mockSaveAllNotebookDocuments(...args),
}))

// Mock localStorage for migration tests
const localStorageMock = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key]
    }),
    clear: vi.fn(() => {
      store = {}
    }),
  }
})()
Object.defineProperty(globalThis, 'localStorage', { value: localStorageMock })

function makeDoc(overrides?: Partial<NotebookDocument>): NotebookDocument {
  return {
    id: 'doc-1',
    title: 'Test Document',
    content: 'Hello world',
    systemPrompt: '',
    images: [],
    createdAt: 1000,
    modifiedAt: 1000,
    ...overrides,
  }
}

describe('notebookStore', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
    localStorageMock.clear()
    mockGetAllNotebookDocuments.mockResolvedValue([])
    mockSaveNotebookDocument.mockResolvedValue(undefined)
    mockDeleteNotebookDocument.mockResolvedValue(undefined)
    mockSaveAllNotebookDocuments.mockResolvedValue(undefined)
    useNotebookStore.setState({
      documents: [],
      activeDocumentId: null,
      loaded: false,
      generation: null,
      error: null,
      samplerSettings: {},
    })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  describe('initial state', () => {
    it('starts with empty documents', () => {
      expect(useNotebookStore.getState().documents).toEqual([])
    })

    it('starts with no active document', () => {
      expect(useNotebookStore.getState().activeDocumentId).toBeNull()
    })

    it('starts with no generation', () => {
      expect(useNotebookStore.getState().generation).toBeNull()
    })

    it('starts with no error', () => {
      expect(useNotebookStore.getState().error).toBeNull()
    })

    it('starts with loaded false', () => {
      expect(useNotebookStore.getState().loaded).toBe(false)
    })
  })

  describe('createDocument', () => {
    it('creates a document with default title', () => {
      const id = useNotebookStore.getState().createDocument()

      const { documents, activeDocumentId } = useNotebookStore.getState()
      expect(documents).toHaveLength(1)
      expect(documents[0].title).toBe('Untitled')
      expect(documents[0].content).toBe('')
      expect(documents[0].systemPrompt).toBe('')
      expect(documents[0].images).toEqual([])
      expect(activeDocumentId).toBe(id)
    })

    it('creates a document with custom title', () => {
      useNotebookStore.getState().createDocument('My Notes')

      expect(useNotebookStore.getState().documents[0].title).toBe('My Notes')
    })

    it('prepends new documents', () => {
      useNotebookStore.getState().createDocument('First')
      useNotebookStore.getState().createDocument('Second')

      const docs = useNotebookStore.getState().documents
      expect(docs).toHaveLength(2)
      expect(docs[0].title).toBe('Second')
      expect(docs[1].title).toBe('First')
    })

    it('sets new document as active', () => {
      useNotebookStore.getState().createDocument('First')
      const secondId = useNotebookStore.getState().createDocument('Second')

      expect(useNotebookStore.getState().activeDocumentId).toBe(secondId)
    })

    it('triggers debounced save to IDB', () => {
      useNotebookStore.getState().createDocument()
      vi.advanceTimersByTime(600)
      expect(mockSaveNotebookDocument).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'Untitled' })
      )
    })
  })

  describe('updateContent', () => {
    it('updates document content', () => {
      const id = useNotebookStore.getState().createDocument()
      useNotebookStore.getState().updateContent(id, 'New content')

      const doc = useNotebookStore.getState().documents[0]
      expect(doc.content).toBe('New content')
    })

    it('updates modifiedAt timestamp', () => {
      const id = useNotebookStore.getState().createDocument()
      const originalTime = useNotebookStore.getState().documents[0].modifiedAt

      vi.advanceTimersByTime(100)
      useNotebookStore.getState().updateContent(id, 'Updated')

      expect(useNotebookStore.getState().documents[0].modifiedAt).toBeGreaterThan(originalTime)
    })

    it('does not modify other documents', () => {
      const id1 = useNotebookStore.getState().createDocument('Doc 1')
      useNotebookStore.getState().createDocument('Doc 2')

      useNotebookStore.getState().updateContent(id1, 'Changed')

      const docs = useNotebookStore.getState().documents
      const doc2 = docs.find((d) => d.title === 'Doc 2')!
      expect(doc2.content).toBe('')
    })
  })

  describe('updateSystemPrompt', () => {
    it('updates system prompt', () => {
      const id = useNotebookStore.getState().createDocument()
      useNotebookStore.getState().updateSystemPrompt(id, 'You are a poet.')

      expect(useNotebookStore.getState().documents[0].systemPrompt).toBe('You are a poet.')
    })
  })

  describe('updateTitle', () => {
    it('updates document title', () => {
      const id = useNotebookStore.getState().createDocument()
      useNotebookStore.getState().updateTitle(id, 'Renamed')

      expect(useNotebookStore.getState().documents[0].title).toBe('Renamed')
    })
  })

  describe('deleteDocument', () => {
    it('removes the document', () => {
      const id = useNotebookStore.getState().createDocument()
      useNotebookStore.getState().deleteDocument(id)

      expect(useNotebookStore.getState().documents).toHaveLength(0)
    })

    it('selects next document when active is deleted', () => {
      const id1 = useNotebookStore.getState().createDocument('First')
      useNotebookStore.getState().createDocument('Second')

      // Second is active (most recent). Delete it.
      const activeId = useNotebookStore.getState().activeDocumentId!
      useNotebookStore.getState().deleteDocument(activeId)

      expect(useNotebookStore.getState().activeDocumentId).toBe(id1)
    })

    it('sets activeDocumentId to null when last document deleted', () => {
      const id = useNotebookStore.getState().createDocument()
      useNotebookStore.getState().deleteDocument(id)

      expect(useNotebookStore.getState().activeDocumentId).toBeNull()
    })

    it('does not change active if non-active document deleted', () => {
      useNotebookStore.getState().createDocument('First')
      const secondId = useNotebookStore.getState().createDocument('Second')

      // Delete First (not active)
      const firstId = useNotebookStore.getState().documents[1].id
      useNotebookStore.getState().deleteDocument(firstId)

      expect(useNotebookStore.getState().activeDocumentId).toBe(secondId)
    })

    it('calls db.deleteNotebookDocument', () => {
      const id = useNotebookStore.getState().createDocument()
      useNotebookStore.getState().deleteDocument(id)

      expect(mockDeleteNotebookDocument).toHaveBeenCalledWith(id)
    })
  })

  describe('selectDocument', () => {
    it('sets active document', () => {
      const id1 = useNotebookStore.getState().createDocument('First')
      useNotebookStore.getState().createDocument('Second')

      useNotebookStore.getState().selectDocument(id1)

      expect(useNotebookStore.getState().activeDocumentId).toBe(id1)
    })
  })

  describe('addImage', () => {
    it('adds image to document', () => {
      const docId = useNotebookStore.getState().createDocument()
      const image: ImageAttachment = {
        id: 'img-1',
        name: 'screenshot.png',
        dataUrl: 'blob:http://localhost/abc',
      }

      useNotebookStore.getState().addImage(docId, image)

      const doc = useNotebookStore.getState().documents[0]
      expect(doc.images).toHaveLength(1)
      expect(doc.images[0].id).toBe('img-1')
      expect(doc.images[0].name).toBe('screenshot.png')
    })

    it('does not modify other documents', () => {
      const id1 = useNotebookStore.getState().createDocument('Doc 1')
      const id2 = useNotebookStore.getState().createDocument('Doc 2')

      useNotebookStore.getState().addImage(id1, {
        id: 'img-1',
        name: 'test.png',
      })

      const doc2 = useNotebookStore.getState().documents.find((d) => d.id === id2)!
      expect(doc2.images).toHaveLength(0)
    })
  })

  describe('removeImage', () => {
    it('removes image from document', () => {
      const docId = useNotebookStore.getState().createDocument()
      useNotebookStore.getState().addImage(docId, {
        id: 'img-1',
        name: 'test.png',
      })
      useNotebookStore.getState().addImage(docId, {
        id: 'img-2',
        name: 'test2.png',
      })

      useNotebookStore.getState().removeImage(docId, 'img-1')

      const doc = useNotebookStore.getState().documents[0]
      expect(doc.images).toHaveLength(1)
      expect(doc.images[0].id).toBe('img-2')
    })
  })

  describe('buildMessages', () => {
    it('builds text-only message', () => {
      const doc = makeDoc({ content: 'Hello world' })
      const messages = buildMessages(doc, 'Hello')

      expect(messages).toEqual([{ role: 'user', content: 'Hello' }])
    })

    it('includes system prompt when non-empty', () => {
      const doc = makeDoc({ systemPrompt: 'You are a poet.' })
      const messages = buildMessages(doc, 'Write a haiku')

      expect(messages).toHaveLength(2)
      expect(messages[0]).toEqual({
        role: 'system',
        content: 'You are a poet.',
      })
      expect(messages[1]).toEqual({
        role: 'user',
        content: 'Write a haiku',
      })
    })

    it('excludes empty system prompt', () => {
      const doc = makeDoc({ systemPrompt: '' })
      const messages = buildMessages(doc, 'Hello')

      expect(messages).toHaveLength(1)
      expect(messages[0].role).toBe('user')
    })

    it('excludes whitespace-only system prompt', () => {
      const doc = makeDoc({ systemPrompt: '   ' })
      const messages = buildMessages(doc, 'Hello')

      expect(messages).toHaveLength(1)
    })

    it('includes images in multipart content', () => {
      const doc = makeDoc({
        images: [
          { id: 'img-1', name: 'photo.jpg', dataUrl: 'data:image/jpeg;base64,abc' },
        ],
      })
      const messages = buildMessages(doc, 'Describe this')

      expect(messages).toHaveLength(1)
      expect(Array.isArray(messages[0].content)).toBe(true)
      const content = messages[0].content as Array<{ type: string }>
      expect(content).toHaveLength(2) // 1 image + 1 text
      expect(content[0].type).toBe('image_url')
      expect(content[1].type).toBe('text')
    })

    it('includes both system prompt and images', () => {
      const doc = makeDoc({
        systemPrompt: 'Describe images.',
        images: [
          { id: 'img-1', name: 'photo.jpg', dataUrl: 'data:image/jpeg;base64,abc' },
        ],
      })
      const messages = buildMessages(doc, 'What is this?')

      expect(messages).toHaveLength(2)
      expect(messages[0].role).toBe('system')
      expect(Array.isArray(messages[1].content)).toBe(true)
    })

    it('filters images without dataUrl', () => {
      const doc = makeDoc({
        images: [
          { id: 'img-1', name: 'no-url.jpg' }, // no dataUrl
          { id: 'img-2', name: 'has-url.jpg', dataUrl: 'data:image/png;base64,xyz' },
        ],
      })
      const messages = buildMessages(doc, 'Describe')

      const content = messages[0].content as Array<{ type: string }>
      // 1 valid image + 1 text
      expect(content).toHaveLength(2)
    })
  })

  describe('startGeneration', () => {
    it('sets generation state', async () => {
      mockStreamChat.mockImplementation(() => Promise.resolve())

      const docId = useNotebookStore.getState().createDocument()
      useNotebookStore.getState().updateContent(docId, 'Hello world')

      await useNotebookStore.getState().startGeneration(5)

      const { generation } = useNotebookStore.getState()
      expect(generation).toBeDefined()
      expect(generation!.insertPosition).toBe(5)
    })

    it('calls streamChat with correct model and messages', async () => {
      mockStreamChat.mockImplementation(() => Promise.resolve())

      const docId = useNotebookStore.getState().createDocument()
      useNotebookStore.getState().updateContent(docId, 'Hello world')

      await useNotebookStore.getState().startGeneration(5)

      expect(mockStreamChat).toHaveBeenCalledTimes(1)
      const [request] = mockStreamChat.mock.calls[0]
      expect(request.model).toBe('test-model')
      expect(request.messages).toEqual([{ role: 'user', content: 'Hello' }])
    })

    it('sets error when no model loaded', async () => {
      // Temporarily mock no model
      const modelMock = await import('../../../stores/modelStore')
      vi.spyOn(modelMock.useModelStore, 'getState').mockReturnValueOnce({
        loadedModel: null,
      } as ReturnType<typeof modelMock.useModelStore.getState>)

      const docId = useNotebookStore.getState().createDocument()
      useNotebookStore.getState().updateContent(docId, 'Hello')

      await useNotebookStore.getState().startGeneration(5)

      expect(useNotebookStore.getState().error).toBe('No model loaded')
    })

    it('sets error when no text before cursor', async () => {
      const docId = useNotebookStore.getState().createDocument()
      useNotebookStore.getState().updateContent(docId, '   ')

      await useNotebookStore.getState().startGeneration(3)

      expect(useNotebookStore.getState().error).toBe(
        'No text before cursor to continue from'
      )
    })

    it('includes sampler settings in request', async () => {
      mockStreamChat.mockImplementation(() => Promise.resolve())

      const docId = useNotebookStore.getState().createDocument()
      useNotebookStore.getState().updateContent(docId, 'Hello world')
      useNotebookStore.getState().updateSamplerSettings({ temperature: 1.5 })

      await useNotebookStore.getState().startGeneration(5)

      const [request] = mockStreamChat.mock.calls[0]
      expect(request.temperature).toBe(1.5)
    })
  })

  describe('stopGeneration', () => {
    it('marks generation as not generating', () => {
      useNotebookStore.setState({
        generation: { isGenerating: true, insertPosition: 0, generatedLength: 10 },
      })

      useNotebookStore.getState().stopGeneration()

      expect(useNotebookStore.getState().generation!.isGenerating).toBe(false)
    })

    it('does nothing when no generation', () => {
      useNotebookStore.getState().stopGeneration()
      expect(useNotebookStore.getState().generation).toBeNull()
    })
  })

  describe('updateSamplerSettings', () => {
    it('merges settings', () => {
      useNotebookStore.getState().updateSamplerSettings({ temperature: 0.5 })
      useNotebookStore.getState().updateSamplerSettings({ max_tokens: 1024 })

      const { samplerSettings } = useNotebookStore.getState()
      expect(samplerSettings.temperature).toBe(0.5)
      expect(samplerSettings.max_tokens).toBe(1024)
    })
  })

  describe('persistence', () => {
    it('saveToDB writes all documents to IDB', () => {
      useNotebookStore.getState().createDocument('Test')
      useNotebookStore.getState().saveToDB()

      expect(mockSaveAllNotebookDocuments).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({ title: 'Test' }),
        ])
      )
    })

    it('loadFromDB restores documents from IDB', async () => {
      const docs = [makeDoc({ id: 'doc-1', title: 'Saved Doc' })]
      mockGetAllNotebookDocuments.mockResolvedValueOnce(docs)

      await useNotebookStore.getState().loadFromDB()

      const { documents, activeDocumentId, loaded } = useNotebookStore.getState()
      expect(documents).toHaveLength(1)
      expect(documents[0].title).toBe('Saved Doc')
      expect(activeDocumentId).toBe('doc-1')
      expect(loaded).toBe(true)
    })

    it('loadFromDB sets loaded true when IDB is empty', async () => {
      mockGetAllNotebookDocuments.mockResolvedValueOnce([])

      await useNotebookStore.getState().loadFromDB()

      expect(useNotebookStore.getState().loaded).toBe(true)
      expect(useNotebookStore.getState().documents).toEqual([])
    })

    it('loadFromDB sets loaded true even on error', async () => {
      mockGetAllNotebookDocuments.mockRejectedValueOnce(new Error('IDB failure'))
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})

      await useNotebookStore.getState().loadFromDB()

      expect(useNotebookStore.getState().loaded).toBe(true)
      expect(useNotebookStore.getState().documents).toEqual([])
      errorSpy.mockRestore()
    })

    it('strips dataUrl from images when saving via debounce', () => {
      const docId = useNotebookStore.getState().createDocument()
      useNotebookStore.getState().addImage(docId, {
        id: 'img-1',
        name: 'test.png',
        dataUrl: 'blob:http://localhost/abc',
      })

      vi.advanceTimersByTime(600)

      const savedDoc = mockSaveNotebookDocument.mock.calls.find(
        (call) => call[0]?.images?.length > 0
      )?.[0]
      expect(savedDoc).toBeDefined()
      expect(savedDoc.images[0].dataUrl).toBeUndefined()
      expect(savedDoc.images[0].id).toBe('img-1')
      expect(savedDoc.images[0].name).toBe('test.png')
    })

    it('migrates from localStorage when IDB is empty', async () => {
      const legacyDocs = [makeDoc({ id: 'legacy-1', title: 'Legacy Doc' })]
      mockGetAllNotebookDocuments.mockResolvedValueOnce([])
      localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(legacyDocs))

      await useNotebookStore.getState().loadFromDB()

      // Should have written to IDB
      expect(mockSaveAllNotebookDocuments).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({ id: 'legacy-1', title: 'Legacy Doc' }),
        ])
      )
      // Should have removed localStorage key
      expect(localStorageMock.removeItem).toHaveBeenCalledWith('notebook-documents')
      // Should have loaded the docs
      const { documents, loaded } = useNotebookStore.getState()
      expect(documents).toHaveLength(1)
      expect(documents[0].title).toBe('Legacy Doc')
      expect(loaded).toBe(true)
    })

    it('deleteDocument calls db.deleteNotebookDocument', () => {
      const id = useNotebookStore.getState().createDocument()
      useNotebookStore.getState().deleteDocument(id)

      expect(mockDeleteNotebookDocument).toHaveBeenCalledWith(id)
    })
  })
})
