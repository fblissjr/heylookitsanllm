import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MessageList } from './MessageList'
import type { Message } from '../../../types/chat'
import type { StreamingState } from '../stores/chatStore'
import type { ModelCapabilities } from '../../../types/models'

// Mock the stores
const mockEditMessageAndRegenerate = vi.fn()
const mockRegenerateFromPosition = vi.fn()
const mockSetConfirmDelete = vi.fn()

const mockActiveConversation = {
  id: 'conv-123',
  title: 'Test Conversation',
  defaultModelId: 'model-abc',
  messages: [],
  createdAt: Date.now(),
  updatedAt: Date.now(),
}

vi.mock('../stores/chatStore', () => ({
  useChatStore: Object.assign(
    vi.fn(() => ({
      editMessageAndRegenerate: mockEditMessageAndRegenerate,
      regenerateFromPosition: mockRegenerateFromPosition,
    })),
    {
      getState: vi.fn(() => ({
        activeConversation: () => mockActiveConversation,
      })),
    }
  ),
}))

vi.mock('../../../stores/uiStore', () => ({
  useUIStore: vi.fn(() => ({
    setConfirmDelete: mockSetConfirmDelete,
  })),
}))

vi.mock('../../../stores/modelStore', () => ({
  useModelStore: vi.fn(() => ({
    loadedModel: {
      id: 'test-model',
      provider: 'mlx',
      capabilities: {
        chat: true,
        vision: false,
        thinking: false,
        hidden_states: false,
        embeddings: false,
      },
      contextWindow: 4096,
    },
  })),
}))


describe('MessageList', () => {
  const defaultStreamingState: StreamingState = {
    isStreaming: false,
    content: '',
    thinking: '',
    messageId: null,
  }

  const defaultModelCapabilities: ModelCapabilities = {
    chat: true,
    vision: false,
    thinking: false,
    hidden_states: false,
    embeddings: false,
  }

  const createMessage = (overrides: Partial<Message> = {}): Message => ({
    id: `msg-${Math.random().toString(36).substr(2, 9)}`,
    role: 'user',
    content: 'Test message content',
    timestamp: Date.now(),
    ...overrides,
  })

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('rendering messages', () => {
    it('renders empty list when no messages', () => {
      render(
        <MessageList
          messages={[]}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      // Should render the container but no message bubbles
      const container = document.querySelector('.space-y-6')
      expect(container).toBeInTheDocument()
      expect(container?.children.length).toBe(0)
    })

    it('renders user message with content', () => {
      const messages = [createMessage({ role: 'user', content: 'Hello from user' })]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      expect(screen.getByText('Hello from user')).toBeInTheDocument()
    })

    it('renders assistant message with content', () => {
      const messages = [
        createMessage({ role: 'assistant', content: 'Hello from assistant' }),
      ]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      expect(screen.getByText('Hello from assistant')).toBeInTheDocument()
    })

    it('renders system message with truncated content', () => {
      const longContent = 'A'.repeat(150)
      const messages = [createMessage({ role: 'system', content: longContent })]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      // System messages are truncated to 100 characters
      expect(screen.getByText('System:')).toBeInTheDocument()
      expect(screen.getByText(/A{100}\.\.\./)).toBeInTheDocument()
    })

    it('renders multiple messages in order', () => {
      const messages = [
        createMessage({ role: 'user', content: 'First message' }),
        createMessage({ role: 'assistant', content: 'Second message' }),
        createMessage({ role: 'user', content: 'Third message' }),
      ]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      expect(screen.getByText('First message')).toBeInTheDocument()
      expect(screen.getByText('Second message')).toBeInTheDocument()
      expect(screen.getByText('Third message')).toBeInTheDocument()
    })

    it('renders user message with images', () => {
      const messages = [
        createMessage({
          role: 'user',
          content: 'Check this image',
          images: ['data:image/png;base64,abc123'],
        }),
      ]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      expect(screen.getByText('Check this image')).toBeInTheDocument()
      const img = screen.getByAltText('Attachment 1')
      expect(img).toBeInTheDocument()
      expect(img).toHaveAttribute('src', 'data:image/png;base64,abc123')
    })

    it('renders assistant message with token count in metrics footer', () => {
      const messages = [
        createMessage({
          role: 'assistant',
          content: 'Response',
          performance: {
            completionTokens: 42,
            tokensPerSecond: 25.5,
          },
        }),
      ]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      // Both mobile and desktop layouts render the tokens (hidden via CSS)
      const tokenElements = screen.getAllByText(/42 tokens/)
      expect(tokenElements.length).toBeGreaterThan(0)
    })
  })

  describe('thinking blocks', () => {
    it('does not show thinking block when model does not support thinking', () => {
      const messages = [
        createMessage({
          role: 'assistant',
          content: 'Response',
          thinking: 'Some thinking process',
        }),
      ]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={{ ...defaultModelCapabilities, thinking: false }}
        />
      )

      expect(screen.queryByText('Thinking')).not.toBeInTheDocument()
    })

    it('shows thinking block when model supports thinking and message has thinking', () => {
      const messages = [
        createMessage({
          role: 'assistant',
          content: 'Response',
          thinking: 'Some thinking process',
        }),
      ]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={{ ...defaultModelCapabilities, thinking: true }}
        />
      )

      expect(screen.getByText('Thinking')).toBeInTheDocument()
    })

    it('toggles thinking block visibility on click', async () => {
      const user = userEvent.setup()
      const messages = [
        createMessage({
          role: 'assistant',
          content: 'Response',
          thinking: 'Some thinking process',
        }),
      ]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={{ ...defaultModelCapabilities, thinking: true }}
        />
      )

      const summary = screen.getByText('Thinking')

      // Initially closed (default showThinking is false)
      expect(screen.queryByText('Some thinking process')).not.toBeInTheDocument()

      // Click to open
      await user.click(summary)
      expect(screen.getByText('Some thinking process')).toBeInTheDocument()

      // Click to close
      await user.click(summary)
      expect(screen.queryByText('Some thinking process')).not.toBeInTheDocument()
    })
  })

  describe('streaming state', () => {
    it('shows streaming message when streaming is active', () => {
      const streamingState: StreamingState = {
        isStreaming: true,
        content: 'Streaming content...',
        thinking: '',
        messageId: 'msg-streaming',
      }

      render(
        <MessageList
          messages={[]}
          streaming={streamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      expect(screen.getByText(/Streaming content\.\.\./)).toBeInTheDocument()
    })

    it('shows loading animation when streaming but no content yet', () => {
      const streamingState: StreamingState = {
        isStreaming: true,
        content: '',
        thinking: '',
        messageId: 'msg-streaming',
      }

      render(
        <MessageList
          messages={[]}
          streaming={streamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      // Check for bouncing dots animation
      const dots = document.querySelectorAll('.animate-bounce')
      expect(dots.length).toBe(3)
    })

    it('shows thinking indicator when only thinking is streaming', () => {
      const streamingState: StreamingState = {
        isStreaming: true,
        content: '',
        thinking: 'Thinking hard...',
        messageId: 'msg-streaming',
      }

      render(
        <MessageList
          messages={[]}
          streaming={streamingState}
          modelCapabilities={{ ...defaultModelCapabilities, thinking: true }}
        />
      )

      expect(screen.getByText('Thinking...')).toBeInTheDocument()
    })

    it('shows pulsing avatar when streaming', () => {
      const streamingState: StreamingState = {
        isStreaming: true,
        content: 'Response',
        thinking: '',
        messageId: 'msg-streaming',
      }

      render(
        <MessageList
          messages={[]}
          streaming={streamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      const pulsingAvatar = document.querySelector('.animate-pulse')
      expect(pulsingAvatar).toBeInTheDocument()
    })
  })

  describe('message actions', () => {
    it('shows action buttons for user message', () => {
      const messages = [createMessage({ role: 'user', content: 'User message' })]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      // Action buttons should exist but be invisible initially
      expect(screen.getByTitle('Copy')).toBeInTheDocument()
      expect(screen.getByTitle('Edit')).toBeInTheDocument()
      expect(screen.getByTitle('Delete')).toBeInTheDocument()
    })

    it('shows regenerate button for assistant messages', () => {
      const messages = [
        createMessage({ role: 'assistant', content: 'Assistant message' }),
      ]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      expect(screen.getByTitle('Regenerate')).toBeInTheDocument()
    })

    it('does not show regenerate button for user messages', () => {
      const messages = [createMessage({ role: 'user', content: 'User message' })]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      expect(screen.queryByTitle('Regenerate')).not.toBeInTheDocument()
    })

    it('has a clickable copy button', async () => {
      const user = userEvent.setup()
      const messages = [
        createMessage({ role: 'user', content: 'Content to copy' }),
      ]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      const copyButton = screen.getByTitle('Copy')
      // Clicking should not throw - the actual clipboard call is tested via the component implementation
      await user.click(copyButton)
      expect(copyButton).toBeInTheDocument()
    })

    it('opens delete confirmation when delete button is clicked', async () => {
      const user = userEvent.setup()
      const messageId = 'msg-to-delete'
      const messages = [
        createMessage({ id: messageId, role: 'user', content: 'Delete me' }),
      ]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      const deleteButton = screen.getByTitle('Delete')
      await user.click(deleteButton)

      expect(mockSetConfirmDelete).toHaveBeenCalledWith({
        type: 'message',
        id: messageId,
        title: 'Delete me',
        conversationId: 'conv-123',
        messageIndex: 0,
      })
    })
  })

  describe('editing', () => {
    it('shows edit UI when edit button is clicked', async () => {
      const user = userEvent.setup()
      const messages = [createMessage({ role: 'user', content: 'Edit me' })]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      const editButton = screen.getByTitle('Edit')
      await user.click(editButton)

      expect(screen.getByText('Editing Message')).toBeInTheDocument()
      expect(screen.getByDisplayValue('Edit me')).toBeInTheDocument()
    })

    it('shows cancel button in edit mode', async () => {
      const user = userEvent.setup()
      const messages = [createMessage({ role: 'user', content: 'Edit me' })]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      const editButton = screen.getByTitle('Edit')
      await user.click(editButton)

      expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument()
    })

    it('cancels editing when cancel is clicked', async () => {
      const user = userEvent.setup()
      const messages = [createMessage({ role: 'user', content: 'Edit me' })]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      const editButton = screen.getByTitle('Edit')
      await user.click(editButton)

      const cancelButton = screen.getByRole('button', { name: 'Cancel' })
      await user.click(cancelButton)

      expect(screen.queryByText('Editing Message')).not.toBeInTheDocument()
      expect(screen.getByText('Edit me')).toBeInTheDocument()
    })

    it('shows save and regenerate options for middle messages', async () => {
      const user = userEvent.setup()
      const messages = [
        createMessage({ role: 'user', content: 'First message' }),
        createMessage({ role: 'assistant', content: 'Response' }),
        createMessage({ role: 'user', content: 'Second message' }),
      ]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      // Click edit on the first user message
      const editButtons = screen.getAllByTitle('Edit')
      await user.click(editButtons[0])

      // Should show both "Save Only" and "Save & Regenerate" options
      expect(screen.getByRole('button', { name: 'Save Only' })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /Save & Regenerate/ })).toBeInTheDocument()
    })

    it('shows only save button for last message', async () => {
      const user = userEvent.setup()
      const messages = [
        createMessage({ role: 'user', content: 'Only message' }),
      ]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      const editButton = screen.getByTitle('Edit')
      await user.click(editButton)

      // Should show only "Save Changes" button
      expect(screen.getByRole('button', { name: /Save Changes/ })).toBeInTheDocument()
      expect(screen.queryByRole('button', { name: 'Save Only' })).not.toBeInTheDocument()
    })
  })

  describe('timestamp formatting', () => {
    it('shows formatted timestamp for messages', () => {
      // Create a message at a specific time
      const timestamp = new Date('2024-01-15T14:30:00').getTime()
      const messages = [createMessage({ role: 'user', content: 'Test', timestamp })]

      render(
        <MessageList
          messages={messages}
          streaming={defaultStreamingState}
          modelCapabilities={defaultModelCapabilities}
        />
      )

      // The exact format depends on locale, but should contain time
      expect(screen.getByText(/\d{1,2}:\d{2}\s*[AP]M/i)).toBeInTheDocument()
    })
  })
})
