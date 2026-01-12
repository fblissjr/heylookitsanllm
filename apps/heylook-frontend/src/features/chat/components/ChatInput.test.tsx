import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ChatInput } from './ChatInput'

// Mock the chatStore
const mockSendMessage = vi.fn()
const mockStopGeneration = vi.fn()

const defaultMockChatState = {
  sendMessage: mockSendMessage,
  stopGeneration: mockStopGeneration,
  streaming: {
    isStreaming: false,
    content: '',
    thinking: '',
    messageId: null,
  },
}

vi.mock('../../../stores/chatStore', () => ({
  useChatStore: vi.fn(() => defaultMockChatState),
}))

// Mock the modelStore
const defaultMockModelState = {
  models: [
    { id: 'model-abc', capabilities: ['chat'], owned_by: 'test' },
    { id: 'model-vision', capabilities: ['chat', 'vision'], owned_by: 'test' },
  ],
  loadedModel: { id: 'model-abc', capabilities: { chat: true, vision: false, thinking: false, hidden_states: false, embeddings: false }, contextWindow: 4096 },
}

vi.mock('../../../stores/modelStore', () => ({
  useModelStore: vi.fn(() => defaultMockModelState),
  getModelCapabilities: vi.fn((model) => {
    const caps = model.capabilities || []
    return {
      chat: caps.includes('chat'),
      vision: caps.includes('vision'),
      thinking: caps.includes('thinking'),
      hidden_states: caps.includes('hidden_states'),
      embeddings: caps.includes('embeddings'),
    }
  }),
}))

// Import the mocks after defining them
import { useChatStore } from '../../../stores/chatStore'
import { useModelStore } from '../../../stores/modelStore'

describe('ChatInput', () => {
  const defaultProps = {
    conversationId: 'conv-123',
    defaultModelId: 'model-abc',
    disabled: false,
  }

  beforeEach(() => {
    vi.clearAllMocks()
    // Reset to default state before each test
    vi.mocked(useChatStore).mockReturnValue(defaultMockChatState)
    vi.mocked(useModelStore).mockReturnValue(defaultMockModelState)
  })

  describe('rendering', () => {
    it('renders the textarea with correct placeholder', () => {
      render(<ChatInput {...defaultProps} />)

      const textarea = screen.getByPlaceholderText('Message...')
      expect(textarea).toBeInTheDocument()
    })

    it('renders placeholder mentioning images when selected model has vision', () => {
      // Mock a vision model being selected
      vi.mocked(useModelStore).mockReturnValue({
        ...defaultMockModelState,
        loadedModel: { id: 'model-vision', capabilities: { chat: true, vision: true, thinking: false, hidden_states: false, embeddings: false }, contextWindow: 4096 },
      })
      render(<ChatInput {...defaultProps} defaultModelId="model-vision" />)

      const textarea = screen.getByPlaceholderText('Message... (paste or drag images)')
      expect(textarea).toBeInTheDocument()
    })

    it('renders the send button', () => {
      render(<ChatInput {...defaultProps} />)

      const sendButton = screen.getByTitle('Send message')
      expect(sendButton).toBeInTheDocument()
    })

    it('shows add image button when selected model has vision', () => {
      vi.mocked(useModelStore).mockReturnValue({
        ...defaultMockModelState,
        loadedModel: { id: 'model-vision', capabilities: { chat: true, vision: true, thinking: false, hidden_states: false, embeddings: false }, contextWindow: 4096 },
      })
      render(<ChatInput {...defaultProps} defaultModelId="model-vision" />)

      const addImageButton = screen.getByTitle('Add image')
      expect(addImageButton).toBeInTheDocument()
    })

    it('hides add image button when selected model does not have vision', () => {
      render(<ChatInput {...defaultProps} />)

      const addImageButton = screen.queryByTitle('Add image')
      expect(addImageButton).not.toBeInTheDocument()
    })

    it('renders disclaimer text', () => {
      render(<ChatInput {...defaultProps} />)

      expect(screen.getByText('AI can make mistakes. Check important info.')).toBeInTheDocument()
    })
  })

  describe('user input', () => {
    it('allows typing in the textarea', async () => {
      const user = userEvent.setup()
      render(<ChatInput {...defaultProps} />)

      const textarea = screen.getByPlaceholderText('Message...')
      await user.type(textarea, 'Hello, world!')

      expect(textarea).toHaveValue('Hello, world!')
    })

    it('disables textarea when disabled prop is true', () => {
      render(<ChatInput {...defaultProps} disabled />)

      const textarea = screen.getByPlaceholderText('Message...')
      expect(textarea).toBeDisabled()
    })
  })

  describe('submission', () => {
    it('calls sendMessage when send button is clicked with text', async () => {
      const user = userEvent.setup()
      render(<ChatInput {...defaultProps} />)

      const textarea = screen.getByPlaceholderText('Message...')
      await user.type(textarea, 'Test message')

      const sendButton = screen.getByTitle('Send message')
      await user.click(sendButton)

      expect(mockSendMessage).toHaveBeenCalledWith(
        'conv-123',
        'Test message',
        'model-abc',
        undefined
      )
    })

    it('clears input after submission', async () => {
      const user = userEvent.setup()
      render(<ChatInput {...defaultProps} />)

      const textarea = screen.getByPlaceholderText('Message...')
      await user.type(textarea, 'Test message')

      const sendButton = screen.getByTitle('Send message')
      await user.click(sendButton)

      expect(textarea).toHaveValue('')
    })

    it('does not submit empty message', async () => {
      const user = userEvent.setup()
      render(<ChatInput {...defaultProps} />)

      const sendButton = screen.getByTitle('Send message')
      await user.click(sendButton)

      expect(mockSendMessage).not.toHaveBeenCalled()
    })

    it('does not submit whitespace-only message', async () => {
      const user = userEvent.setup()
      render(<ChatInput {...defaultProps} />)

      const textarea = screen.getByPlaceholderText('Message...')
      await user.type(textarea, '   ')

      const sendButton = screen.getByTitle('Send message')
      await user.click(sendButton)

      expect(mockSendMessage).not.toHaveBeenCalled()
    })

    it('does not submit when disabled', () => {
      render(<ChatInput {...defaultProps} disabled />)

      // Textarea is disabled, but we can test the button state
      const sendButton = screen.getByTitle('Send message')
      expect(sendButton).toBeDisabled()
    })
  })

  describe('keyboard handling', () => {
    it('submits on Enter without shift', async () => {
      const user = userEvent.setup()
      render(<ChatInput {...defaultProps} />)

      const textarea = screen.getByPlaceholderText('Message...')
      await user.type(textarea, 'Test message')
      await user.keyboard('{Enter}')

      expect(mockSendMessage).toHaveBeenCalledWith(
        'conv-123',
        'Test message',
        'model-abc',
        undefined
      )
    })

    it('does not submit on Shift+Enter (allows newline)', async () => {
      const user = userEvent.setup()
      render(<ChatInput {...defaultProps} />)

      const textarea = screen.getByPlaceholderText('Message...')
      await user.type(textarea, 'Line 1')
      await user.keyboard('{Shift>}{Enter}{/Shift}')
      await user.type(textarea, 'Line 2')

      expect(mockSendMessage).not.toHaveBeenCalled()
      expect(textarea).toHaveValue('Line 1\nLine 2')
    })
  })

  describe('streaming state', () => {
    it('shows stop button when streaming', () => {
      vi.mocked(useChatStore).mockReturnValue({
        sendMessage: mockSendMessage,
        stopGeneration: mockStopGeneration,
        streaming: {
          isStreaming: true,
          content: 'Streaming content...',
          thinking: '',
          messageId: 'msg-123',
        },
      })

      render(<ChatInput {...defaultProps} />)

      const stopButton = screen.getByTitle('Stop generation')
      expect(stopButton).toBeInTheDocument()
    })

    it('calls stopGeneration when stop button is clicked', async () => {
      const user = userEvent.setup()
      vi.mocked(useChatStore).mockReturnValue({
        sendMessage: mockSendMessage,
        stopGeneration: mockStopGeneration,
        streaming: {
          isStreaming: true,
          content: 'Streaming content...',
          thinking: '',
          messageId: 'msg-123',
        },
      })

      render(<ChatInput {...defaultProps} />)

      const stopButton = screen.getByTitle('Stop generation')
      await user.click(stopButton)

      expect(mockStopGeneration).toHaveBeenCalled()
    })
  })

  describe('send button state', () => {
    it('send button is disabled when no text and no images', () => {
      render(<ChatInput {...defaultProps} />)

      const sendButton = screen.getByTitle('Send message')
      expect(sendButton).toBeDisabled()
    })

    it('send button is enabled when there is text', async () => {
      const user = userEvent.setup()
      render(<ChatInput {...defaultProps} />)

      const textarea = screen.getByPlaceholderText('Message...')
      await user.type(textarea, 'Some text')

      const sendButton = screen.getByTitle('Send message')
      expect(sendButton).not.toBeDisabled()
    })
  })
})
