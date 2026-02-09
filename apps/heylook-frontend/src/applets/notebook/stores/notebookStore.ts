import { create } from 'zustand'
import { streamChat } from '../../../api/streaming'
import { useModelStore } from '../../../stores/modelStore'
import { generateId } from '../../../lib/id'
import {
  getAllNotebookDocuments,
  saveNotebookDocument,
  deleteNotebookDocument as dbDeleteDocument,
  saveAllNotebookDocuments,
} from '../../../lib/db'
import type { PersistedNotebookDocument } from '../../../lib/db'
import type { APIMessage, MessageContent } from '../../../types/api'
import type { SamplerSettings } from '../../../types/settings'
import type { NotebookDocument, ImageAttachment, GenerationState } from '../types'

const LOCALSTORAGE_KEY = 'notebook-documents'
const SAVE_DEBOUNCE_MS = 500

let abortController: AbortController | null = null
let saveTimeout: ReturnType<typeof setTimeout> | null = null

/** Build API messages from notebook document state */
export function buildMessages(doc: NotebookDocument, textBeforeCursor: string): APIMessage[] {
  const messages: APIMessage[] = []

  if (doc.systemPrompt.trim()) {
    messages.push({ role: 'system', content: doc.systemPrompt })
  }

  if (doc.images.length > 0) {
    const content: MessageContent[] = [
      ...doc.images
        .filter((img) => img.dataUrl)
        .map((img) => ({
          type: 'image_url' as const,
          image_url: { url: img.dataUrl! },
        })),
      { type: 'text' as const, text: textBeforeCursor },
    ]
    messages.push({ role: 'user', content })
  } else {
    messages.push({ role: 'user', content: textBeforeCursor })
  }

  return messages
}

interface NotebookStoreState {
  documents: NotebookDocument[]
  activeDocumentId: string | null
  loaded: boolean
  generation: GenerationState | null
  error: string | null
  samplerSettings: Partial<SamplerSettings>

  createDocument: (title?: string) => string
  updateContent: (id: string, content: string) => void
  updateSystemPrompt: (id: string, prompt: string) => void
  updateTitle: (id: string, title: string) => void
  deleteDocument: (id: string) => void
  selectDocument: (id: string) => void

  addImage: (docId: string, image: ImageAttachment) => void
  removeImage: (docId: string, imageId: string) => void

  startGeneration: (cursorPosition: number) => Promise<void>
  stopGeneration: () => void

  updateSamplerSettings: (settings: Partial<SamplerSettings>) => void

  saveToDB: () => void
  loadFromDB: () => Promise<void>
}

function serializeDoc(doc: NotebookDocument): PersistedNotebookDocument {
  return {
    id: doc.id,
    title: doc.title,
    content: doc.content,
    systemPrompt: doc.systemPrompt,
    images: doc.images.map(({ id, name }) => ({ id, name })),
    createdAt: doc.createdAt,
    modifiedAt: doc.modifiedAt,
  }
}

function debouncedSave(doc: NotebookDocument) {
  if (saveTimeout) clearTimeout(saveTimeout)
  saveTimeout = setTimeout(() => {
    saveNotebookDocument(serializeDoc(doc)).catch((e) =>
      console.error('Failed to save notebook document:', e)
    )
  }, SAVE_DEBOUNCE_MS)
}

export const useNotebookStore = create<NotebookStoreState>((set, get) => ({
  documents: [],
  activeDocumentId: null,
  loaded: false,
  generation: null,
  error: null,
  samplerSettings: {},

  createDocument: (title) => {
    const id = generateId('doc')
    const now = Date.now()
    const doc: NotebookDocument = {
      id,
      title: title || 'Untitled',
      content: '',
      systemPrompt: '',
      images: [],
      createdAt: now,
      modifiedAt: now,
    }
    set((state) => ({
      documents: [doc, ...state.documents],
      activeDocumentId: id,
    }))
    debouncedSave(doc)
    return id
  },

  updateContent: (id, content) => {
    set((state) => ({
      documents: state.documents.map((d) =>
        d.id === id ? { ...d, content, modifiedAt: Date.now() } : d
      ),
    }))
    const updated = get().documents.find((d) => d.id === id)
    if (updated) debouncedSave(updated)
  },

  updateSystemPrompt: (id, prompt) => {
    set((state) => ({
      documents: state.documents.map((d) =>
        d.id === id ? { ...d, systemPrompt: prompt, modifiedAt: Date.now() } : d
      ),
    }))
    const updated = get().documents.find((d) => d.id === id)
    if (updated) debouncedSave(updated)
  },

  updateTitle: (id, title) => {
    set((state) => ({
      documents: state.documents.map((d) =>
        d.id === id ? { ...d, title, modifiedAt: Date.now() } : d
      ),
    }))
    const updated = get().documents.find((d) => d.id === id)
    if (updated) debouncedSave(updated)
  },

  deleteDocument: (id) => {
    set((state) => {
      const remaining = state.documents.filter((d) => d.id !== id)
      return {
        documents: remaining,
        activeDocumentId:
          state.activeDocumentId === id
            ? remaining[0]?.id || null
            : state.activeDocumentId,
      }
    })
    dbDeleteDocument(id).catch((e) =>
      console.error('Failed to delete notebook document:', e)
    )
  },

  selectDocument: (id) => {
    set({ activeDocumentId: id })
  },

  addImage: (docId, image) => {
    set((state) => ({
      documents: state.documents.map((d) =>
        d.id === docId
          ? { ...d, images: [...d.images, image], modifiedAt: Date.now() }
          : d
      ),
    }))
    const updated = get().documents.find((d) => d.id === docId)
    if (updated) debouncedSave(updated)
  },

  removeImage: (docId, imageId) => {
    set((state) => ({
      documents: state.documents.map((d) =>
        d.id === docId
          ? {
              ...d,
              images: d.images.filter((img) => img.id !== imageId),
              modifiedAt: Date.now(),
            }
          : d
      ),
    }))
    const updated = get().documents.find((d) => d.id === docId)
    if (updated) debouncedSave(updated)
  },

  startGeneration: async (cursorPosition) => {
    const { documents, activeDocumentId, samplerSettings } = get()
    const doc = documents.find((d) => d.id === activeDocumentId)
    if (!doc) return

    const loadedModel = useModelStore.getState().loadedModel
    if (!loadedModel) {
      set({ error: 'No model loaded' })
      return
    }

    // Stop any existing generation
    get().stopGeneration()

    const textBeforeCursor = doc.content.slice(0, cursorPosition)
    if (!textBeforeCursor.trim()) {
      set({ error: 'No text before cursor to continue from' })
      return
    }

    set({
      generation: {
        isGenerating: true,
        insertPosition: cursorPosition,
        generatedLength: 0,
      },
      error: null,
    })

    abortController = new AbortController()

    const messages = buildMessages(doc, textBeforeCursor)

    await streamChat(
      {
        model: loadedModel.id,
        messages,
        ...samplerSettings,
      },
      {
        onToken: (token) => {
          const state = get()
          if (!state.generation || !state.activeDocumentId) return

          const currentDoc = state.documents.find(
            (d) => d.id === state.activeDocumentId
          )
          if (!currentDoc) return

          const pos = state.generation.insertPosition + state.generation.generatedLength
          const newContent =
            currentDoc.content.slice(0, pos) +
            token +
            currentDoc.content.slice(pos)

          set({
            documents: state.documents.map((d) =>
              d.id === state.activeDocumentId
                ? { ...d, content: newContent, modifiedAt: Date.now() }
                : d
            ),
            generation: {
              ...state.generation,
              generatedLength: state.generation.generatedLength + token.length,
            },
          })
        },
        onComplete: () => {
          set((state) => ({
            generation: state.generation
              ? { ...state.generation, isGenerating: false }
              : null,
          }))
          abortController = null
          const { documents: docs, activeDocumentId: activeId } = get()
          const activeDoc = docs.find((d) => d.id === activeId)
          if (activeDoc) debouncedSave(activeDoc)
        },
        onError: (error) => {
          set((state) => ({
            generation: state.generation
              ? { ...state.generation, isGenerating: false }
              : null,
            error: error.message,
          }))
          abortController = null
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
    set((state) => ({
      generation: state.generation
        ? { ...state.generation, isGenerating: false }
        : null,
    }))
  },

  updateSamplerSettings: (settings) => {
    set((state) => ({
      samplerSettings: { ...state.samplerSettings, ...settings },
    }))
  },

  saveToDB: () => {
    const { documents } = get()
    saveAllNotebookDocuments(documents.map(serializeDoc)).catch((e) =>
      console.error('Failed to save notebook documents:', e)
    )
  },

  loadFromDB: async () => {
    try {
      const docs = await getAllNotebookDocuments()
      if (docs.length > 0) {
        set({
          documents: docs,
          activeDocumentId: docs[0]?.id || null,
          loaded: true,
        })
        return
      }

      // Migration: check localStorage for legacy data
      const raw = localStorage.getItem(LOCALSTORAGE_KEY)
      if (raw) {
        const legacy: NotebookDocument[] = JSON.parse(raw)
        const serialized = legacy.map(serializeDoc)
        await saveAllNotebookDocuments(serialized)
        localStorage.removeItem(LOCALSTORAGE_KEY)
        set({
          documents: legacy,
          activeDocumentId: legacy[0]?.id || null,
          loaded: true,
        })
        return
      }

      set({ loaded: true })
    } catch (e) {
      console.error('Failed to load notebook documents:', e)
      set({ loaded: true })
    }
  },
}))
