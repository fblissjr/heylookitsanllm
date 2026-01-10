import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { Sidebar } from './Sidebar'
import type { Conversation } from '../../types/chat'
import type { LoadedModel } from '../../types/models'

// Mock functions
const mockCreateConversation = vi.fn()
const mockSetActiveConversation = vi.fn()
const mockSetConfirmDelete = vi.fn()
const mockToggleSidebar = vi.fn()

// Default mock states
const defaultLoadedModel: LoadedModel = {
  id: 'test-model',
  capabilities: { chat: true, vision: false, thinking: false, hidden_states: false, embeddings: false },
  contextWindow: 4096,
}

const defaultChatState = {
  conversations: [] as Conversation[],
  activeConversationId: null as string | null,
  createConversation: mockCreateConversation,
  setActiveConversation: mockSetActiveConversation,
}

const defaultModelState = {
  loadedModel: defaultLoadedModel as LoadedModel | null,
}

const defaultUIState = {
  setConfirmDelete: mockSetConfirmDelete,
  isMobile: false,
  toggleSidebar: mockToggleSidebar,
}

vi.mock('../../stores/chatStore', () => ({
  useChatStore: vi.fn(() => defaultChatState),
}))

vi.mock('../../stores/modelStore', () => ({
  useModelStore: vi.fn(() => defaultModelState),
}))

vi.mock('../../stores/uiStore', () => ({
  useUIStore: vi.fn(() => defaultUIState),
}))

import { useChatStore } from '../../stores/chatStore'
import { useModelStore } from '../../stores/modelStore'
import { useUIStore } from '../../stores/uiStore'

// Helper to create mock conversations with a fixed date
function createMockConversation(overrides: Partial<Conversation> = {}): Conversation {
  const baseTime = new Date('2024-06-15T12:00:00.000Z').getTime()
  return {
    id: `conv-${Math.random().toString(36).substr(2, 9)}`,
    title: 'Test Conversation',
    modelId: 'test-model',
    messages: [],
    createdAt: baseTime,
    updatedAt: baseTime,
    ...overrides,
  }
}

describe('Sidebar', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(useChatStore).mockReturnValue(defaultChatState)
    vi.mocked(useModelStore).mockReturnValue(defaultModelState)
    vi.mocked(useUIStore).mockReturnValue(defaultUIState)

    // Mock Date for consistent date formatting tests
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2024-06-15T12:00:00.000Z'))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  describe('rendering', () => {
    it('renders the sidebar aside element', () => {
      render(<Sidebar />)

      const aside = document.querySelector('aside.w-64')
      expect(aside).toBeInTheDocument()
    })

    it('renders the New Chat button', () => {
      render(<Sidebar />)

      expect(screen.getByText('New Chat')).toBeInTheDocument()
    })
  })

  describe('New Chat button', () => {
    it('is enabled when a model is loaded', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: defaultLoadedModel,
      })

      render(<Sidebar />)

      const newChatButton = screen.getByText('New Chat').closest('button')
      expect(newChatButton).not.toBeDisabled()
    })

    it('is disabled when no model is loaded', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: null,
      })

      render(<Sidebar />)

      const newChatButton = screen.getByText('New Chat').closest('button')
      expect(newChatButton).toBeDisabled()
    })

    it('calls createConversation with model id when clicked', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: defaultLoadedModel,
      })

      render(<Sidebar />)

      const newChatButton = screen.getByText('New Chat').closest('button')
      fireEvent.click(newChatButton!)

      expect(mockCreateConversation).toHaveBeenCalledWith('test-model')
    })

    it('toggles sidebar after creating conversation on mobile', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: defaultLoadedModel,
      })
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        isMobile: true,
      })

      render(<Sidebar />)

      const newChatButton = screen.getByText('New Chat').closest('button')
      fireEvent.click(newChatButton!)

      expect(mockToggleSidebar).toHaveBeenCalled()
    })

    it('does not toggle sidebar after creating conversation on desktop', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: defaultLoadedModel,
      })
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        isMobile: false,
      })

      render(<Sidebar />)

      const newChatButton = screen.getByText('New Chat').closest('button')
      fireEvent.click(newChatButton!)

      expect(mockToggleSidebar).not.toHaveBeenCalled()
    })

    it('does not create conversation when no model loaded', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: null,
      })

      render(<Sidebar />)

      const newChatButton = screen.getByText('New Chat').closest('button')
      fireEvent.click(newChatButton!)

      expect(mockCreateConversation).not.toHaveBeenCalled()
    })

    it('has disabled styling when no model is loaded', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: null,
      })

      render(<Sidebar />)

      const newChatButton = screen.getByText('New Chat').closest('button')
      expect(newChatButton).toHaveClass('cursor-not-allowed')
    })
  })

  describe('empty state', () => {
    it('shows "Start a new chat" message when model loaded but no conversations', () => {
      vi.mocked(useChatStore).mockReturnValue({
        ...defaultChatState,
        conversations: [],
      })
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: defaultLoadedModel,
      })

      render(<Sidebar />)

      expect(screen.getByText('No conversations yet. Start a new chat!')).toBeInTheDocument()
    })

    it('shows "Load a model" message when no model loaded', () => {
      vi.mocked(useChatStore).mockReturnValue({
        ...defaultChatState,
        conversations: [],
      })
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: null,
      })

      render(<Sidebar />)

      expect(screen.getByText('Load a model to start chatting.')).toBeInTheDocument()
    })
  })

  describe('conversation list', () => {
    it('renders conversation items', () => {
      const conversations = [
        createMockConversation({ id: 'conv-1', title: 'First Chat' }),
        createMockConversation({ id: 'conv-2', title: 'Second Chat' }),
      ]
      vi.mocked(useChatStore).mockReturnValue({
        ...defaultChatState,
        conversations,
      })

      render(<Sidebar />)

      expect(screen.getByText('First Chat')).toBeInTheDocument()
      expect(screen.getByText('Second Chat')).toBeInTheDocument()
    })

    it('highlights active conversation', () => {
      const conversations = [
        createMockConversation({ id: 'conv-1', title: 'Active Chat' }),
        createMockConversation({ id: 'conv-2', title: 'Other Chat' }),
      ]
      vi.mocked(useChatStore).mockReturnValue({
        ...defaultChatState,
        conversations,
        activeConversationId: 'conv-1',
      })

      render(<Sidebar />)

      const activeButton = screen.getByText('Active Chat').closest('button')
      expect(activeButton).toHaveClass('bg-primary/10')
      expect(activeButton).toHaveClass('text-primary')
    })

    it('calls setActiveConversation when clicking a conversation', () => {
      const conversations = [
        createMockConversation({ id: 'conv-1', title: 'Test Chat' }),
      ]
      vi.mocked(useChatStore).mockReturnValue({
        ...defaultChatState,
        conversations,
      })

      render(<Sidebar />)

      // Click on the conversation text/span, not the outer button
      const conversationText = screen.getByText('Test Chat')
      const conversationButton = conversationText.closest('button')
      fireEvent.click(conversationButton!)

      expect(mockSetActiveConversation).toHaveBeenCalledWith('conv-1')
    })

    it('toggles sidebar after selecting conversation on mobile', () => {
      const conversations = [
        createMockConversation({ id: 'conv-1', title: 'Test Chat' }),
      ]
      vi.mocked(useChatStore).mockReturnValue({
        ...defaultChatState,
        conversations,
      })
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        isMobile: true,
      })

      render(<Sidebar />)

      const conversationText = screen.getByText('Test Chat')
      const conversationButton = conversationText.closest('button')
      fireEvent.click(conversationButton!)

      expect(mockToggleSidebar).toHaveBeenCalled()
    })
  })

  describe('delete button', () => {
    it('renders delete button for each conversation', () => {
      const conversations = [
        createMockConversation({ id: 'conv-1', title: 'Test Chat' }),
      ]
      vi.mocked(useChatStore).mockReturnValue({
        ...defaultChatState,
        conversations,
      })

      render(<Sidebar />)

      const deleteButton = screen.getByLabelText('Delete conversation')
      expect(deleteButton).toBeInTheDocument()
    })

    it('calls setConfirmDelete when delete button is clicked', () => {
      const conversations = [
        createMockConversation({ id: 'conv-1', title: 'Chat to Delete' }),
      ]
      vi.mocked(useChatStore).mockReturnValue({
        ...defaultChatState,
        conversations,
      })

      render(<Sidebar />)

      const deleteButton = screen.getByLabelText('Delete conversation')
      fireEvent.click(deleteButton)

      expect(mockSetConfirmDelete).toHaveBeenCalledWith({
        type: 'conversation',
        id: 'conv-1',
        title: 'Chat to Delete',
      })
    })

    it('does not trigger setActiveConversation when clicking delete', () => {
      const conversations = [
        createMockConversation({ id: 'conv-1', title: 'Test Chat' }),
      ]
      vi.mocked(useChatStore).mockReturnValue({
        ...defaultChatState,
        conversations,
      })

      render(<Sidebar />)

      const deleteButton = screen.getByLabelText('Delete conversation')
      fireEvent.click(deleteButton)

      // Delete click should call setConfirmDelete but not setActiveConversation
      expect(mockSetConfirmDelete).toHaveBeenCalled()
      expect(mockSetActiveConversation).not.toHaveBeenCalled()
    })
  })

  describe('date grouping', () => {
    it('groups conversations by "Today"', () => {
      const now = Date.now()
      const conversations = [
        createMockConversation({ id: 'conv-1', title: 'Today Chat', updatedAt: now }),
      ]
      vi.mocked(useChatStore).mockReturnValue({
        ...defaultChatState,
        conversations,
      })

      render(<Sidebar />)

      expect(screen.getByText('Today')).toBeInTheDocument()
    })

    it('groups conversations by "Yesterday"', () => {
      const yesterday = Date.now() - 24 * 60 * 60 * 1000
      const conversations = [
        createMockConversation({ id: 'conv-1', title: 'Yesterday Chat', updatedAt: yesterday }),
      ]
      vi.mocked(useChatStore).mockReturnValue({
        ...defaultChatState,
        conversations,
      })

      render(<Sidebar />)

      expect(screen.getByText('Yesterday')).toBeInTheDocument()
    })

    it('groups conversations by "X days ago"', () => {
      const threeDaysAgo = Date.now() - 3 * 24 * 60 * 60 * 1000
      const conversations = [
        createMockConversation({ id: 'conv-1', title: 'Old Chat', updatedAt: threeDaysAgo }),
      ]
      vi.mocked(useChatStore).mockReturnValue({
        ...defaultChatState,
        conversations,
      })

      render(<Sidebar />)

      expect(screen.getByText('3 days ago')).toBeInTheDocument()
    })

    it('shows formatted date for conversations older than 7 days', () => {
      const tenDaysAgo = Date.now() - 10 * 24 * 60 * 60 * 1000
      const conversations = [
        createMockConversation({ id: 'conv-1', title: 'Very Old Chat', updatedAt: tenDaysAgo }),
      ]
      vi.mocked(useChatStore).mockReturnValue({
        ...defaultChatState,
        conversations,
      })

      render(<Sidebar />)

      // Should show a formatted date string
      // The exact format depends on locale, so just check it's not "X days ago"
      expect(screen.queryByText(/days ago/)).not.toBeInTheDocument()
    })

    it('groups multiple conversations under same date', () => {
      const now = Date.now()
      const conversations = [
        createMockConversation({ id: 'conv-1', title: 'Chat 1', updatedAt: now }),
        createMockConversation({ id: 'conv-2', title: 'Chat 2', updatedAt: now - 1000 }),
      ]
      vi.mocked(useChatStore).mockReturnValue({
        ...defaultChatState,
        conversations,
      })

      render(<Sidebar />)

      // Should only have one "Today" header
      const todayHeaders = screen.getAllByText('Today')
      expect(todayHeaders).toHaveLength(1)

      // Both conversations should be rendered
      expect(screen.getByText('Chat 1')).toBeInTheDocument()
      expect(screen.getByText('Chat 2')).toBeInTheDocument()
    })
  })

  describe('footer with loaded model info', () => {
    it('shows footer when model is loaded', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: defaultLoadedModel,
      })

      render(<Sidebar />)

      expect(screen.getByText('test-model')).toBeInTheDocument()
    })

    it('does not show footer when no model is loaded', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: null,
      })
      vi.mocked(useChatStore).mockReturnValue({
        ...defaultChatState,
        conversations: [createMockConversation({ title: 'Test' })],
      })

      render(<Sidebar />)

      // Footer border-t element should not exist
      const footer = document.querySelector('.border-t.border-gray-200')
      expect(footer).not.toBeInTheDocument()
    })

    it('shows Vision capability when model has vision', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: {
          ...defaultLoadedModel,
          capabilities: { ...defaultLoadedModel.capabilities, vision: true },
        },
      })

      render(<Sidebar />)

      expect(screen.getByText(/Vision/)).toBeInTheDocument()
    })

    it('shows Thinking capability when model has thinking', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: {
          ...defaultLoadedModel,
          capabilities: { ...defaultLoadedModel.capabilities, thinking: true },
        },
      })

      render(<Sidebar />)

      expect(screen.getByText(/Thinking/)).toBeInTheDocument()
    })

    it('shows both Vision and Thinking when model has both', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: {
          ...defaultLoadedModel,
          capabilities: { ...defaultLoadedModel.capabilities, vision: true, thinking: true },
        },
      })

      render(<Sidebar />)

      expect(screen.getByText(/Vision/)).toBeInTheDocument()
      expect(screen.getByText(/Thinking/)).toBeInTheDocument()
    })
  })

  describe('button styling', () => {
    it('new chat button has enabled styling when model loaded', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: defaultLoadedModel,
      })

      render(<Sidebar />)

      const newChatButton = screen.getByText('New Chat').closest('button')
      expect(newChatButton).toHaveClass('bg-primary')
    })

    it('new chat button has disabled styling when no model', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: null,
      })

      render(<Sidebar />)

      const newChatButton = screen.getByText('New Chat').closest('button')
      expect(newChatButton).toHaveClass('bg-gray-200')
      expect(newChatButton).toHaveClass('text-gray-400')
    })
  })
})
