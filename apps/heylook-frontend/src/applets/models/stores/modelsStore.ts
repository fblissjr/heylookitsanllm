import { create } from 'zustand'
import { fetchAPI, postAPI } from '../../../api/client'
import type {
  AdminModelConfig,
  ScannedModel,
  ProfileInfo,
  ModelUpdatePayload,
  ModelFilter,
  ImportRequest,
  ScanRequest,
} from '../types'

interface ModelsState {
  // Data
  configs: AdminModelConfig[]
  selectedId: string | null
  searchQuery: string
  filters: ModelFilter
  profiles: ProfileInfo[]

  // Import workflow
  importOpen: boolean
  scanResults: ScannedModel[]
  scanning: boolean
  importing: boolean

  // Loading states
  loading: boolean
  error: string | null
  actionLoading: string | null  // model_id currently being acted on
  profilesLoaded: boolean

  // Actions
  fetchConfigs: () => Promise<void>
  fetchProfiles: () => Promise<void>
  setSelectedId: (id: string | null) => void
  setSearchQuery: (query: string) => void
  setFilters: (filters: Partial<ModelFilter>) => void
  setImportOpen: (open: boolean) => void

  // CRUD
  updateConfig: (id: string, updates: ModelUpdatePayload) => Promise<string[]>
  removeConfig: (id: string) => Promise<void>
  toggleEnabled: (id: string) => Promise<void>

  // Scan / Import
  scanForModels: (request: ScanRequest) => Promise<void>
  importModels: (request: ImportRequest) => Promise<void>

  // Profile
  applyProfile: (ids: string[], profile: string) => Promise<void>

  // Load / Unload
  loadModel: (id: string) => Promise<void>
  unloadModel: (id: string) => Promise<void>
}

const API_BASE = '/v1/admin/models'

export const useModelsStore = create<ModelsState>((set, get) => ({
  configs: [],
  selectedId: null,
  searchQuery: '',
  filters: { provider: [], status: [], capability: [] },
  profiles: [],
  importOpen: false,
  scanResults: [],
  scanning: false,
  importing: false,
  loading: false,
  error: null,
  actionLoading: null,
  profilesLoaded: false,

  fetchConfigs: async () => {
    set({ loading: true, error: null })
    try {
      const data = await fetchAPI<{ models: AdminModelConfig[] }>(API_BASE)
      set({ configs: data.models, loading: false })
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Failed to fetch configs'
      set({ error: msg, loading: false })
    }
  },

  fetchProfiles: async () => {
    try {
      const data = await fetchAPI<{ profiles: ProfileInfo[] }>(`${API_BASE}/profiles`)
      set({ profiles: data.profiles, profilesLoaded: true })
    } catch {
      set({ profilesLoaded: true })
    }
  },

  setSelectedId: (id) => set({ selectedId: id }),
  setSearchQuery: (query) => set({ searchQuery: query }),
  setFilters: (filters) => set((s) => ({ filters: { ...s.filters, ...filters } })),
  setImportOpen: (open) => set({ importOpen: open, scanResults: [], scanning: false }),

  updateConfig: async (id, updates) => {
    try {
      const data = await fetchAPI<{ model: AdminModelConfig; reload_required_fields: string[] }>(
        `${API_BASE}/${encodeURIComponent(id)}`,
        {
          method: 'PATCH',
          body: JSON.stringify(updates),
        }
      )
      set((s) => ({
        configs: s.configs.map((c) => (c.id === id ? data.model : c)),
      }))
      return data.reload_required_fields
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Update failed'
      set({ error: msg })
      throw error
    }
  },

  removeConfig: async (id) => {
    try {
      await fetchAPI(`${API_BASE}/${encodeURIComponent(id)}`, { method: 'DELETE' })
      set((s) => ({
        configs: s.configs.filter((c) => c.id !== id),
        selectedId: s.selectedId === id ? null : s.selectedId,
      }))
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Remove failed'
      set({ error: msg })
      throw error
    }
  },

  toggleEnabled: async (id) => {
    set({ actionLoading: id })
    try {
      const data = await postAPI<AdminModelConfig>(`${API_BASE}/${encodeURIComponent(id)}/toggle`, {})
      set((s) => ({
        configs: s.configs.map((c) => (c.id === id ? { ...c, ...data } : c)),
      }))
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Toggle failed'
      set({ error: msg })
    } finally {
      set({ actionLoading: null })
    }
  },

  scanForModels: async (request) => {
    set({ scanning: true, error: null })
    try {
      const data = await postAPI<{ models: ScannedModel[] }>(`${API_BASE}/scan`, request)
      set({ scanResults: data.models, scanning: false })
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Scan failed'
      set({ error: msg, scanning: false })
    }
  },

  importModels: async (request) => {
    set({ importing: true, error: null })
    try {
      await postAPI(`${API_BASE}/import`, request)
      set({ importing: false, importOpen: false, scanResults: [] })
      // Refresh the list
      await get().fetchConfigs()
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Import failed'
      set({ error: msg, importing: false })
    }
  },

  applyProfile: async (ids, profile) => {
    set({ actionLoading: ids[0] ?? null })
    try {
      await postAPI(`${API_BASE}/bulk-profile`, { model_ids: ids, profile })
      await get().fetchConfigs()
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Apply profile failed'
      set({ error: msg })
    } finally {
      set({ actionLoading: null })
    }
  },

  loadModel: async (id) => {
    set({ actionLoading: id })
    try {
      await postAPI(`${API_BASE}/${encodeURIComponent(id)}/load`, {})
      set((s) => ({
        configs: s.configs.map((c) => (c.id === id ? { ...c, loaded: true } : c)),
      }))
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Load failed'
      set({ error: msg })
    } finally {
      set({ actionLoading: null })
    }
  },

  unloadModel: async (id) => {
    set({ actionLoading: id })
    try {
      await postAPI(`${API_BASE}/${encodeURIComponent(id)}/unload`, {})
      set((s) => ({
        configs: s.configs.map((c) => (c.id === id ? { ...c, loaded: false } : c)),
      }))
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Unload failed'
      set({ error: msg })
    } finally {
      set({ actionLoading: null })
    }
  },
}))
