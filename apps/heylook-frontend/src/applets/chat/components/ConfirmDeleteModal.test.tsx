import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ConfirmDeleteModal } from './ConfirmDeleteModal'

// Mock the stores
const mockCloseModal = vi.fn()
const mockDeleteConversation = vi.fn()
const mockDeleteMessageWithCascade = vi.fn()

const defaultUIState = {
  activeModal: null as string | null,
  confirmDelete: {
    type: null as 'message' | 'conversation' | null,
    id: null as string | null,
    title: undefined as string | undefined,
    conversationId: undefined as string | undefined,
    messageIndex: undefined as number | undefined,
  },
  closeModal: mockCloseModal,
}

const defaultChatState = {
  deleteConversation: mockDeleteConversation,
  deleteMessageWithCascade: mockDeleteMessageWithCascade,
}

vi.mock('../../../stores/uiStore', () => ({
  useUIStore: vi.fn(() => defaultUIState),
}))

vi.mock('../stores/chatStore', () => ({
  useChatStore: vi.fn(() => defaultChatState),
}))

// Import after mocks are defined
import { useUIStore } from '../../../stores/uiStore'
import { useChatStore } from '../stores/chatStore'

describe('ConfirmDeleteModal', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(useUIStore).mockReturnValue(defaultUIState)
    vi.mocked(useChatStore).mockReturnValue(defaultChatState)
  })

  describe('conditional rendering', () => {
    it('returns null when activeModal is null', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activeModal: null,
      })

      const { container } = render(<ConfirmDeleteModal />)
      expect(container.firstChild).toBeNull()
    })

    it('returns null when activeModal is not deleteMessage or deleteConversation', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activeModal: 'modelLoad',
      })

      const { container } = render(<ConfirmDeleteModal />)
      expect(container.firstChild).toBeNull()
    })

    it('renders when activeModal is deleteMessage', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activeModal: 'deleteMessage',
        confirmDelete: {
          type: 'message',
          id: 'msg-123',
          title: 'Test message',
          conversationId: 'conv-123',
          messageIndex: 0,
        },
      })

      render(<ConfirmDeleteModal />)
      expect(screen.getByText('Delete message?')).toBeInTheDocument()
    })

    it('renders when activeModal is deleteConversation', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activeModal: 'deleteConversation',
        confirmDelete: {
          type: 'conversation',
          id: 'conv-123',
          title: 'Test conversation',
          conversationId: undefined,
          messageIndex: undefined,
        },
      })

      render(<ConfirmDeleteModal />)
      expect(screen.getByText('Delete conversation?')).toBeInTheDocument()
    })
  })

  describe('message deletion content', () => {
    beforeEach(() => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activeModal: 'deleteMessage',
        confirmDelete: {
          type: 'message',
          id: 'msg-123',
          title: 'Test message content',
          conversationId: 'conv-123',
          messageIndex: 0,
        },
      })
    })

    it('shows message-specific title', () => {
      render(<ConfirmDeleteModal />)
      expect(screen.getByText('Delete message?')).toBeInTheDocument()
    })

    it('shows message-specific description', () => {
      render(<ConfirmDeleteModal />)
      expect(screen.getByText('This will remove the message from your history. This action cannot be undone.')).toBeInTheDocument()
    })

    it('shows message preview with correct label', () => {
      render(<ConfirmDeleteModal />)
      expect(screen.getByText('Message:')).toBeInTheDocument()
      expect(screen.getByText('"Test message content"')).toBeInTheDocument()
    })

    it('shows "Delete and generate new response" button for message type', () => {
      render(<ConfirmDeleteModal />)
      expect(screen.getByRole('button', { name: 'Delete and generate new response' })).toBeInTheDocument()
    })
  })

  describe('conversation deletion content', () => {
    beforeEach(() => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activeModal: 'deleteConversation',
        confirmDelete: {
          type: 'conversation',
          id: 'conv-123',
          title: 'My important conversation',
          conversationId: undefined,
          messageIndex: undefined,
        },
      })
    })

    it('shows conversation-specific title', () => {
      render(<ConfirmDeleteModal />)
      expect(screen.getByText('Delete conversation?')).toBeInTheDocument()
    })

    it('shows conversation-specific description', () => {
      render(<ConfirmDeleteModal />)
      expect(screen.getByText('This will permanently delete this conversation and all its messages.')).toBeInTheDocument()
    })

    it('shows conversation preview with correct label', () => {
      render(<ConfirmDeleteModal />)
      expect(screen.getByText('Conversation:')).toBeInTheDocument()
      expect(screen.getByText('"My important conversation"')).toBeInTheDocument()
    })

    it('does not show "Delete and generate new response" button for conversation type', () => {
      render(<ConfirmDeleteModal />)
      expect(screen.queryByRole('button', { name: 'Delete and generate new response' })).not.toBeInTheDocument()
    })
  })

  describe('button interactions', () => {
    it('calls closeModal when Cancel button is clicked', async () => {
      const user = userEvent.setup()
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activeModal: 'deleteMessage',
        confirmDelete: {
          type: 'message',
          id: 'msg-123',
          title: 'Test',
          conversationId: 'conv-123',
          messageIndex: 0,
        },
      })

      render(<ConfirmDeleteModal />)

      const cancelButton = screen.getByRole('button', { name: 'Cancel' })
      await user.click(cancelButton)

      expect(mockCloseModal).toHaveBeenCalled()
    })

    it('calls deleteConversation and closeModal when Delete is clicked for conversation', async () => {
      const user = userEvent.setup()
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activeModal: 'deleteConversation',
        confirmDelete: {
          type: 'conversation',
          id: 'conv-456',
          title: 'Test',
          conversationId: undefined,
          messageIndex: undefined,
        },
      })

      render(<ConfirmDeleteModal />)

      const deleteButton = screen.getByRole('button', { name: 'Delete' })
      await user.click(deleteButton)

      expect(mockDeleteConversation).toHaveBeenCalledWith('conv-456')
      expect(mockCloseModal).toHaveBeenCalled()
    })

    it('calls deleteMessageWithCascade with regenerate=false when Delete is clicked for message', async () => {
      const user = userEvent.setup()
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activeModal: 'deleteMessage',
        confirmDelete: {
          type: 'message',
          id: 'msg-789',
          title: 'Test',
          conversationId: 'conv-123',
          messageIndex: 0,
        },
      })

      render(<ConfirmDeleteModal />)

      const deleteButton = screen.getByRole('button', { name: 'Delete' })
      await user.click(deleteButton)

      expect(mockDeleteMessageWithCascade).toHaveBeenCalledWith('conv-123', 'msg-789', false)
      expect(mockCloseModal).toHaveBeenCalled()
    })

    it('calls deleteMessageWithCascade with regenerate=true when "Delete and generate new response" is clicked', async () => {
      const user = userEvent.setup()
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activeModal: 'deleteMessage',
        confirmDelete: {
          type: 'message',
          id: 'msg-789',
          title: 'Test',
          conversationId: 'conv-123',
          messageIndex: 0,
        },
      })

      render(<ConfirmDeleteModal />)

      const regenerateButton = screen.getByRole('button', { name: 'Delete and generate new response' })
      await user.click(regenerateButton)

      expect(mockDeleteMessageWithCascade).toHaveBeenCalledWith('conv-123', 'msg-789', true)
      expect(mockCloseModal).toHaveBeenCalled()
    })
  })

  describe('preview section', () => {
    it('does not show preview section when no title is provided', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activeModal: 'deleteMessage',
        confirmDelete: {
          type: 'message',
          id: 'msg-123',
          title: undefined,
          conversationId: 'conv-123',
          messageIndex: 0,
        },
      })

      render(<ConfirmDeleteModal />)

      expect(screen.queryByText('Message:')).not.toBeInTheDocument()
      expect(screen.queryByText('Conversation:')).not.toBeInTheDocument()
    })

    it('shows preview section when title is provided', () => {
      vi.mocked(useUIStore).mockReturnValue({
        ...defaultUIState,
        activeModal: 'deleteConversation',
        confirmDelete: {
          type: 'conversation',
          id: 'conv-123',
          title: 'Some title',
          conversationId: undefined,
          messageIndex: undefined,
        },
      })

      render(<ConfirmDeleteModal />)

      expect(screen.getByText('Conversation:')).toBeInTheDocument()
      expect(screen.getByText('"Some title"')).toBeInTheDocument()
    })
  })
})
