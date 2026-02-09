import type { SamplerSettings } from '../../types/settings'

export interface NotebookDocument {
  id: string
  title: string
  content: string
  systemPrompt: string
  images: ImageAttachment[]
  createdAt: number
  modifiedAt: number
}

export interface ImageAttachment {
  id: string
  name: string
  /** Runtime only -- created from IndexedDB Blob via URL.createObjectURL() */
  dataUrl?: string
}

/** Stored in IndexedDB separately (Blob, not base64) */
export interface StoredImage {
  id: string
  blob: Blob
  name: string
}

export interface GenerationState {
  isGenerating: boolean
  insertPosition: number
  generatedLength: number
  thinking: string
}

export interface NotebookState {
  // Documents
  documents: NotebookDocument[]
  activeDocumentId: string | null
  loaded: boolean

  // Generation
  generation: GenerationState | null
  error: string | null

  // Settings (per-session, not per-document)
  samplerSettings: Partial<SamplerSettings>

  // Document CRUD
  createDocument: (title?: string) => string
  updateContent: (id: string, content: string) => void
  updateSystemPrompt: (id: string, prompt: string) => void
  updateTitle: (id: string, title: string) => void
  deleteDocument: (id: string) => void
  selectDocument: (id: string) => void

  // Images
  addImage: (docId: string, image: ImageAttachment) => void
  removeImage: (docId: string, imageId: string) => void

  // Generation
  startGeneration: (cursorPosition: number) => Promise<void>
  stopGeneration: () => void

  // Settings
  updateSamplerSettings: (settings: Partial<SamplerSettings>) => void

  // Persistence
  saveToDB: () => void
  loadFromDB: () => Promise<void>
}
