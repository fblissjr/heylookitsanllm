import { describe, it, expect, beforeEach } from 'vitest'
import { useUIStore } from './uiStore'

describe('uiStore', () => {
  beforeEach(() => {
    // Reset store to initial state before each test
    useUIStore.setState({
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
    })
  })

  describe('initial state', () => {
    it('has correct initial panel state', () => {
      const state = useUIStore.getState()
      expect(state.activePanel).toBeNull()
      expect(state.isSidebarOpen).toBe(true)
      expect(state.isSettingsExpanded).toEqual({
        sampling: false,
        repetition: false,
        advanced: false,
      })
    })

    it('has correct initial modal state', () => {
      const state = useUIStore.getState()
      expect(state.activeModal).toBeNull()
      expect(state.confirmDelete).toEqual({
        type: null,
        id: null,
      })
    })

    it('has correct initial mobile state', () => {
      const state = useUIStore.getState()
      expect(state.isMobile).toBe(false)
      expect(state.isBottomSheetOpen).toBe(false)
    })
  })

  describe('panel actions', () => {
    describe('setActivePanel', () => {
      it('sets the active panel', () => {
        const { setActivePanel } = useUIStore.getState()

        setActivePanel('settings')

        expect(useUIStore.getState().activePanel).toBe('settings')
      })

      it('sets different panel types', () => {
        const { setActivePanel } = useUIStore.getState()

        setActivePanel('models')
        expect(useUIStore.getState().activePanel).toBe('models')

        setActivePanel('advanced')
        expect(useUIStore.getState().activePanel).toBe('advanced')
      })

      it('clears panel when set to null', () => {
        const { setActivePanel } = useUIStore.getState()

        setActivePanel('settings')
        setActivePanel(null)

        expect(useUIStore.getState().activePanel).toBeNull()
      })
    })

    describe('togglePanel', () => {
      it('toggles panel on when not active', () => {
        const { togglePanel } = useUIStore.getState()

        togglePanel('settings')

        expect(useUIStore.getState().activePanel).toBe('settings')
      })

      it('toggles panel off when already active', () => {
        const { setActivePanel, togglePanel } = useUIStore.getState()

        setActivePanel('settings')
        togglePanel('settings')

        expect(useUIStore.getState().activePanel).toBeNull()
      })

      it('switches to different panel when another is active', () => {
        const { setActivePanel, togglePanel } = useUIStore.getState()

        setActivePanel('settings')
        togglePanel('models')

        expect(useUIStore.getState().activePanel).toBe('models')
      })
    })
  })

  describe('sidebar', () => {
    describe('toggleSidebar', () => {
      it('closes sidebar when open', () => {
        const { toggleSidebar } = useUIStore.getState()

        toggleSidebar()

        expect(useUIStore.getState().isSidebarOpen).toBe(false)
      })

      it('opens sidebar when closed', () => {
        useUIStore.setState({ isSidebarOpen: false })
        const { toggleSidebar } = useUIStore.getState()

        toggleSidebar()

        expect(useUIStore.getState().isSidebarOpen).toBe(true)
      })

      it('toggles multiple times correctly', () => {
        const { toggleSidebar } = useUIStore.getState()

        toggleSidebar() // true -> false
        expect(useUIStore.getState().isSidebarOpen).toBe(false)

        toggleSidebar() // false -> true
        expect(useUIStore.getState().isSidebarOpen).toBe(true)

        toggleSidebar() // true -> false
        expect(useUIStore.getState().isSidebarOpen).toBe(false)
      })
    })
  })

  describe('settings sections', () => {
    describe('toggleSettingsSection', () => {
      it('expands a collapsed section', () => {
        const { toggleSettingsSection } = useUIStore.getState()

        toggleSettingsSection('sampling')

        expect(useUIStore.getState().isSettingsExpanded.sampling).toBe(true)
      })

      it('collapses an expanded section', () => {
        useUIStore.setState({
          isSettingsExpanded: { sampling: true, repetition: false, advanced: false },
        })
        const { toggleSettingsSection } = useUIStore.getState()

        toggleSettingsSection('sampling')

        expect(useUIStore.getState().isSettingsExpanded.sampling).toBe(false)
      })

      it('does not affect other sections', () => {
        const { toggleSettingsSection } = useUIStore.getState()

        toggleSettingsSection('sampling')

        const state = useUIStore.getState()
        expect(state.isSettingsExpanded.sampling).toBe(true)
        expect(state.isSettingsExpanded.repetition).toBe(false)
        expect(state.isSettingsExpanded.advanced).toBe(false)
      })

      it('handles multiple section toggles independently', () => {
        const { toggleSettingsSection } = useUIStore.getState()

        toggleSettingsSection('sampling')
        toggleSettingsSection('repetition')

        const state = useUIStore.getState()
        expect(state.isSettingsExpanded.sampling).toBe(true)
        expect(state.isSettingsExpanded.repetition).toBe(true)
        expect(state.isSettingsExpanded.advanced).toBe(false)
      })

      it('handles new section names dynamically', () => {
        const { toggleSettingsSection } = useUIStore.getState()

        toggleSettingsSection('custom_section')

        expect(useUIStore.getState().isSettingsExpanded.custom_section).toBe(true)
      })
    })
  })

  describe('modals', () => {
    describe('openModal', () => {
      it('opens modelLoad modal', () => {
        const { openModal } = useUIStore.getState()

        openModal('modelLoad')

        expect(useUIStore.getState().activeModal).toBe('modelLoad')
      })

      it('opens deleteMessage modal', () => {
        const { openModal } = useUIStore.getState()

        openModal('deleteMessage')

        expect(useUIStore.getState().activeModal).toBe('deleteMessage')
      })

      it('opens deleteConversation modal', () => {
        const { openModal } = useUIStore.getState()

        openModal('deleteConversation')

        expect(useUIStore.getState().activeModal).toBe('deleteConversation')
      })

      it('opens savePreset modal', () => {
        const { openModal } = useUIStore.getState()

        openModal('savePreset')

        expect(useUIStore.getState().activeModal).toBe('savePreset')
      })

      it('switches between modals', () => {
        const { openModal } = useUIStore.getState()

        openModal('modelLoad')
        expect(useUIStore.getState().activeModal).toBe('modelLoad')

        openModal('savePreset')
        expect(useUIStore.getState().activeModal).toBe('savePreset')
      })
    })

    describe('closeModal', () => {
      it('closes an open modal', () => {
        const { openModal, closeModal } = useUIStore.getState()

        openModal('modelLoad')
        closeModal()

        expect(useUIStore.getState().activeModal).toBeNull()
      })

      it('clears confirmDelete state when closing', () => {
        useUIStore.setState({
          activeModal: 'deleteMessage',
          confirmDelete: {
            type: 'message',
            id: 'msg-123',
            conversationId: 'conv-456',
            messageIndex: 5,
          },
        })
        const { closeModal } = useUIStore.getState()

        closeModal()

        const state = useUIStore.getState()
        expect(state.activeModal).toBeNull()
        expect(state.confirmDelete).toEqual({ type: null, id: null })
      })
    })
  })

  describe('confirmDelete', () => {
    describe('setConfirmDelete', () => {
      it('sets message delete state and opens modal', () => {
        const { setConfirmDelete } = useUIStore.getState()

        setConfirmDelete({
          type: 'message',
          id: 'msg-123',
          conversationId: 'conv-456',
          messageIndex: 3,
        })

        const state = useUIStore.getState()
        expect(state.confirmDelete).toEqual({
          type: 'message',
          id: 'msg-123',
          conversationId: 'conv-456',
          messageIndex: 3,
        })
        expect(state.activeModal).toBe('deleteMessage')
      })

      it('sets conversation delete state and opens modal', () => {
        const { setConfirmDelete } = useUIStore.getState()

        setConfirmDelete({
          type: 'conversation',
          id: 'conv-789',
          title: 'Test Conversation',
        })

        const state = useUIStore.getState()
        expect(state.confirmDelete).toEqual({
          type: 'conversation',
          id: 'conv-789',
          title: 'Test Conversation',
        })
        expect(state.activeModal).toBe('deleteMessage')
      })

      it('clears modal when type is null', () => {
        // First set a delete state
        useUIStore.setState({
          confirmDelete: { type: 'message', id: 'msg-123' },
          activeModal: 'deleteMessage',
        })
        const { setConfirmDelete } = useUIStore.getState()

        setConfirmDelete({ type: null, id: null })

        const state = useUIStore.getState()
        expect(state.confirmDelete).toEqual({ type: null, id: null })
        expect(state.activeModal).toBeNull()
      })

      it('handles all optional fields', () => {
        const { setConfirmDelete } = useUIStore.getState()

        setConfirmDelete({
          type: 'message',
          id: 'msg-full',
          title: 'Message Title',
          conversationId: 'conv-abc',
          messageIndex: 10,
        })

        const state = useUIStore.getState()
        expect(state.confirmDelete.type).toBe('message')
        expect(state.confirmDelete.id).toBe('msg-full')
        expect(state.confirmDelete.title).toBe('Message Title')
        expect(state.confirmDelete.conversationId).toBe('conv-abc')
        expect(state.confirmDelete.messageIndex).toBe(10)
      })
    })
  })

  describe('mobile', () => {
    describe('setIsMobile', () => {
      it('sets mobile mode and closes sidebar', () => {
        const { setIsMobile } = useUIStore.getState()

        setIsMobile(true)

        const state = useUIStore.getState()
        expect(state.isMobile).toBe(true)
        expect(state.isSidebarOpen).toBe(false)
      })

      it('sets desktop mode and opens sidebar', () => {
        useUIStore.setState({ isMobile: true, isSidebarOpen: false })
        const { setIsMobile } = useUIStore.getState()

        setIsMobile(false)

        const state = useUIStore.getState()
        expect(state.isMobile).toBe(false)
        expect(state.isSidebarOpen).toBe(true)
      })

      it('toggling between mobile and desktop updates sidebar accordingly', () => {
        const { setIsMobile } = useUIStore.getState()

        // Start in desktop mode (default)
        expect(useUIStore.getState().isSidebarOpen).toBe(true)

        // Switch to mobile
        setIsMobile(true)
        expect(useUIStore.getState().isSidebarOpen).toBe(false)

        // Switch back to desktop
        setIsMobile(false)
        expect(useUIStore.getState().isSidebarOpen).toBe(true)
      })
    })

    describe('setBottomSheetOpen', () => {
      it('opens bottom sheet', () => {
        const { setBottomSheetOpen } = useUIStore.getState()

        setBottomSheetOpen(true)

        expect(useUIStore.getState().isBottomSheetOpen).toBe(true)
      })

      it('closes bottom sheet', () => {
        useUIStore.setState({ isBottomSheetOpen: true })
        const { setBottomSheetOpen } = useUIStore.getState()

        setBottomSheetOpen(false)

        expect(useUIStore.getState().isBottomSheetOpen).toBe(false)
      })
    })
  })
})
