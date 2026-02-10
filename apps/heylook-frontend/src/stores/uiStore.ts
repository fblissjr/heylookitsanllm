import { create } from 'zustand'

type PanelType = 'settings' | 'models' | 'advanced' | null
type ModalType = 'modelLoad' | 'deleteMessage' | 'deleteConversation' | 'savePreset' | null

interface ConfirmDeleteState {
  type: 'message' | 'conversation' | 'bulk' | null
  id?: string | null
  ids?: string[]  // For bulk delete
  title?: string
  conversationId?: string
  messageIndex?: number
  onComplete?: () => void  // Called after modal closes (confirm or cancel)
}

interface UIState {
  // Panels
  activePanel: PanelType
  isSidebarOpen: boolean
  isSettingsExpanded: Record<string, boolean>

  // Modals
  activeModal: ModalType
  confirmDelete: ConfirmDeleteState

  // Mobile
  isMobile: boolean
  isBottomSheetOpen: boolean

  // Actions - Panels
  setActivePanel: (panel: PanelType) => void
  togglePanel: (panel: PanelType) => void
  toggleSidebar: () => void
  toggleSettingsSection: (section: string) => void

  // Actions - Modals
  openModal: (modal: ModalType) => void
  closeModal: () => void
  setConfirmDelete: (state: ConfirmDeleteState) => void

  // Actions - Mobile
  setIsMobile: (isMobile: boolean) => void
  setBottomSheetOpen: (isOpen: boolean) => void
}

export const useUIStore = create<UIState>((set, get) => ({
  activePanel: null,
  isSidebarOpen: true,
  isSettingsExpanded: {
    sampling: false,
    repetition: false,
    advanced: false,
  },

  activeModal: null,
  confirmDelete: {
    type: null,
    id: null,
  },

  isMobile: false,
  isBottomSheetOpen: false,

  setActivePanel: (panel) => {
    set({ activePanel: panel })
  },

  togglePanel: (panel) => {
    set(state => ({
      activePanel: state.activePanel === panel ? null : panel,
    }))
  },

  toggleSidebar: () => {
    set(state => ({ isSidebarOpen: !state.isSidebarOpen }))
  },

  toggleSettingsSection: (section) => {
    set(state => ({
      isSettingsExpanded: {
        ...state.isSettingsExpanded,
        [section]: !state.isSettingsExpanded[section],
      },
    }))
  },

  openModal: (modal) => {
    set((prev) => ({
      activeModal: modal,
      isSidebarOpen: prev.isMobile && modal ? false : prev.isSidebarOpen,
    }))
  },

  closeModal: () => {
    set({ activeModal: null, confirmDelete: { type: null, id: null } })
  },

  setConfirmDelete: (state) => {
    set({ confirmDelete: state })
    if (state.type) {
      get().openModal('deleteMessage')
    } else {
      set({ activeModal: null })
    }
  },

  setIsMobile: (isMobile) => {
    set({
      isMobile,
      isSidebarOpen: !isMobile, // Close sidebar on mobile by default
    })
  },

  setBottomSheetOpen: (isOpen) => {
    set({ isBottomSheetOpen: isOpen })
  },
}))
