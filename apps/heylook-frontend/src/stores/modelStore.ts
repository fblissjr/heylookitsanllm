import { create } from 'zustand'
import type { Model, ServerCapabilities } from '../types/api'
import type { LoadedModel, ModelCapabilities, ModelStatus } from '../types/models'
import { withDiagnostics } from './diagnosticMiddleware'
import { logger } from '../lib/diagnostics'

interface ModelState {
  // Data
  models: Model[]
  loadedModel: LoadedModel | null
  capabilities: ServerCapabilities | null
  modelStatus: ModelStatus
  error: string | null

  // Actions
  fetchModels: () => Promise<void>
  fetchCapabilities: () => Promise<void>
  setLoadedModel: (model: LoadedModel | null) => void
  setModelStatus: (status: ModelStatus) => void
  setError: (error: string | null) => void
}

function parseCapabilities(caps?: string[]): ModelCapabilities {
  const capSet = new Set(caps || [])
  return {
    chat: capSet.has('chat'),
    vision: capSet.has('vision'),
    thinking: capSet.has('thinking'),
    hidden_states: capSet.has('hidden_states'),
    embeddings: capSet.has('embeddings'),
  }
}

export const useModelStore = create<ModelState>()(withDiagnostics('model', (set) => ({
  models: [],
  loadedModel: null,
  capabilities: null,
  modelStatus: 'unloaded',
  error: null,

  fetchModels: async () => {
    try {
      const response = await fetch('/v1/models')
      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.statusText}`)
      }
      const data = await response.json()
      set({ models: data.data || [], error: null })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to fetch models'
      set({ error: message })
      throw error
    }
  },

  fetchCapabilities: async () => {
    try {
      const response = await fetch('/v1/capabilities')
      if (!response.ok) {
        throw new Error(`Failed to fetch capabilities: ${response.statusText}`)
      }
      const data = await response.json()
      set({ capabilities: data })
    } catch (error) {
      console.error('Failed to fetch capabilities:', error)
    }
  },

  setLoadedModel: (model) => {
    logger.info('model_loaded', 'system', { modelId: model?.id ?? null })
    set({ loadedModel: model, modelStatus: model ? 'loaded' : 'unloaded' })
  },

  setModelStatus: (status) => {
    logger.info('model_status', 'system', { status })
    set({ modelStatus: status })
  },

  setError: (error) => {
    if (error) logger.warn('model_error', 'system', { error })
    set({ error })
  },
})))

// Helper to get capabilities for a specific model
export function getModelCapabilities(model: Model): ModelCapabilities {
  return parseCapabilities(model.capabilities)
}
