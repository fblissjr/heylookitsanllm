import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ChatInput } from './ChatInput'
import { mockFileReader } from '../../../test/mocks'

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

vi.mock('../stores/chatStore', () => ({
  useChatStore: vi.fn(() => defaultMockChatState),
}))

// Mock the modelStore -- loadedModel drives vision and model ID
const defaultMockModelState = {
  loadedModel: { id: 'model-abc', capabilities: { chat: true, vision: false, thinking: false, hidden_states: false, embeddings: false }, contextWindow: 4096 },
}

vi.mock('../../../stores/modelStore', () => ({
  useModelStore: vi.fn(() => defaultMockModelState),
}))

// Import the mocks after defining them
import { useChatStore } from '../stores/chatStore'
import { useModelStore } from '../../../stores/modelStore'

describe('ChatInput', () => {
  const defaultProps = {
    conversationId: 'conv-123',
    disabled: false,
  }

  let createObjectURLCounter = 0

  beforeEach(() => {
    vi.clearAllMocks()
    createObjectURLCounter = 0
    // Reset to default state before each test
    vi.mocked(useChatStore).mockReturnValue(defaultMockChatState)
    vi.mocked(useModelStore).mockReturnValue(defaultMockModelState)
    // Mock URL APIs for image tests
    globalThis.URL.createObjectURL = vi.fn(() => `blob:mock-url-${++createObjectURLCounter}`)
    globalThis.URL.revokeObjectURL = vi.fn()
  })

  describe('rendering', () => {
    it('renders the textarea with correct placeholder', () => {
      render(<ChatInput {...defaultProps} />)

      const textarea = screen.getByPlaceholderText('Message...')
      expect(textarea).toBeInTheDocument()
    })

    it('renders placeholder mentioning images when loaded model has vision', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: { id: 'model-vision', capabilities: { chat: true, vision: true, thinking: false, hidden_states: false, embeddings: false }, contextWindow: 4096 },
      })
      render(<ChatInput {...defaultProps} />)

      const textarea = screen.getByPlaceholderText('Message... (paste or drag images)')
      expect(textarea).toBeInTheDocument()
    })

    it('renders the send button', () => {
      render(<ChatInput {...defaultProps} />)

      const sendButton = screen.getByTitle('Send message')
      expect(sendButton).toBeInTheDocument()
    })

    it('shows add image button when loaded model has vision', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: { id: 'model-vision', capabilities: { chat: true, vision: true, thinking: false, hidden_states: false, embeddings: false }, contextWindow: 4096 },
      })
      render(<ChatInput {...defaultProps} />)

      const addImageButton = screen.getByTitle('Add image')
      expect(addImageButton).toBeInTheDocument()
    })

    it('hides add image button when loaded model does not have vision', () => {
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
    it('calls sendMessage with loadedModel.id when send button is clicked', async () => {
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

  // Helpers for image tests
  const visionModelState = {
    loadedModel: { id: 'model-vision', capabilities: { chat: true, vision: true, thinking: false, hidden_states: false, embeddings: false }, contextWindow: 4096 },
  }

  function createImageFile(name = 'photo.png', type = 'image/png') {
    return new File(['pixels'], name, { type })
  }

  function createDropEvent(files: File[]) {
    const dataTransfer = {
      files,
      items: files.map(f => ({ kind: 'file', type: f.type, getAsFile: () => f })),
      types: ['Files'],
    }
    return { dataTransfer }
  }

  function createPasteEvent(items: Array<{ type: string; getAsFile: () => File | null }>) {
    return {
      clipboardData: { items },
    }
  }

  describe('image handling', () => {
    it('adds image preview when file dropped on vision model', () => {
      vi.mocked(useModelStore).mockReturnValue(visionModelState)
      render(<ChatInput {...defaultProps} />)

      const container = screen.getByPlaceholderText('Message... (paste or drag images)').closest('div')!
      const file = createImageFile()
      fireEvent.drop(container, createDropEvent([file]))

      expect(screen.getByAltText('Upload 1')).toBeInTheDocument()
    })

    it('does not add image when model has no vision', () => {
      render(<ChatInput {...defaultProps} />)

      const container = screen.getByPlaceholderText('Message...').closest('div')!
      const file = createImageFile()
      fireEvent.drop(container, createDropEvent([file]))

      expect(screen.queryByAltText('Upload 1')).not.toBeInTheDocument()
    })

    it('filters out non-image files', () => {
      vi.mocked(useModelStore).mockReturnValue(visionModelState)
      render(<ChatInput {...defaultProps} />)

      const container = screen.getByPlaceholderText('Message... (paste or drag images)').closest('div')!
      const textFile = new File(['hello'], 'notes.txt', { type: 'text/plain' })
      fireEvent.drop(container, createDropEvent([textFile]))

      expect(screen.queryByAltText('Upload 1')).not.toBeInTheDocument()
    })

    it('calls URL.createObjectURL for each image file', () => {
      vi.mocked(useModelStore).mockReturnValue(visionModelState)
      render(<ChatInput {...defaultProps} />)

      const container = screen.getByPlaceholderText('Message... (paste or drag images)').closest('div')!
      const file1 = createImageFile('a.png')
      const file2 = createImageFile('b.png')
      fireEvent.drop(container, createDropEvent([file1, file2]))

      expect(globalThis.URL.createObjectURL).toHaveBeenCalledTimes(2)
    })

    it('shows remove button for each image preview', () => {
      vi.mocked(useModelStore).mockReturnValue(visionModelState)
      render(<ChatInput {...defaultProps} />)

      const container = screen.getByPlaceholderText('Message... (paste or drag images)').closest('div')!
      fireEvent.drop(container, createDropEvent([createImageFile()]))

      expect(screen.getByLabelText('Remove image 1')).toBeInTheDocument()
    })

    it('removes image and revokes URL on remove click', async () => {
      const user = userEvent.setup()
      vi.mocked(useModelStore).mockReturnValue(visionModelState)
      render(<ChatInput {...defaultProps} />)

      const container = screen.getByPlaceholderText('Message... (paste or drag images)').closest('div')!
      fireEvent.drop(container, createDropEvent([createImageFile()]))

      expect(screen.getByAltText('Upload 1')).toBeInTheDocument()

      await user.click(screen.getByLabelText('Remove image 1'))

      expect(screen.queryByAltText('Upload 1')).not.toBeInTheDocument()
      expect(globalThis.URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock-url-1')
    })

    it('can add multiple images', () => {
      vi.mocked(useModelStore).mockReturnValue(visionModelState)
      render(<ChatInput {...defaultProps} />)

      const container = screen.getByPlaceholderText('Message... (paste or drag images)').closest('div')!
      fireEvent.drop(container, createDropEvent([createImageFile('a.png'), createImageFile('b.png')]))

      expect(screen.getByAltText('Upload 1')).toBeInTheDocument()
      expect(screen.getByAltText('Upload 2')).toBeInTheDocument()
    })
  })

  describe('paste image handling', () => {
    it('adds image when pasting from clipboard', () => {
      vi.mocked(useModelStore).mockReturnValue(visionModelState)
      render(<ChatInput {...defaultProps} />)

      const textarea = screen.getByPlaceholderText('Message... (paste or drag images)')
      const file = createImageFile()
      fireEvent.paste(textarea, createPasteEvent([
        { type: 'image/png', getAsFile: () => file },
      ]))

      expect(screen.getByAltText('Upload 1')).toBeInTheDocument()
    })

    it('does not add images when pasting text-only', () => {
      vi.mocked(useModelStore).mockReturnValue(visionModelState)
      render(<ChatInput {...defaultProps} />)

      const textarea = screen.getByPlaceholderText('Message... (paste or drag images)')
      fireEvent.paste(textarea, createPasteEvent([
        { type: 'text/plain', getAsFile: () => null },
      ]))

      expect(screen.queryByAltText('Upload 1')).not.toBeInTheDocument()
    })

    it('handles null from getAsFile safely', () => {
      vi.mocked(useModelStore).mockReturnValue(visionModelState)
      render(<ChatInput {...defaultProps} />)

      const textarea = screen.getByPlaceholderText('Message... (paste or drag images)')
      fireEvent.paste(textarea, createPasteEvent([
        { type: 'image/png', getAsFile: () => null },
      ]))

      expect(screen.queryByAltText('Upload 1')).not.toBeInTheDocument()
    })
  })

  describe('drag and drop', () => {
    it('sets dragging state on dragOver when vision model active', () => {
      vi.mocked(useModelStore).mockReturnValue(visionModelState)
      render(<ChatInput {...defaultProps} />)

      const container = screen.getByPlaceholderText('Message... (paste or drag images)').closest('div')!
      fireEvent.dragOver(container)

      expect(screen.getByText('Drop images here')).toBeInTheDocument()
    })

    it('does not set dragging state without vision model', () => {
      render(<ChatInput {...defaultProps} />)

      const container = screen.getByPlaceholderText('Message...').closest('div')!
      fireEvent.dragOver(container)

      expect(screen.queryByText('Drop images here')).not.toBeInTheDocument()
    })

    it('clears dragging state on drop', () => {
      vi.mocked(useModelStore).mockReturnValue(visionModelState)
      render(<ChatInput {...defaultProps} />)

      const container = screen.getByPlaceholderText('Message... (paste or drag images)').closest('div')!
      fireEvent.dragOver(container)
      expect(screen.getByText('Drop images here')).toBeInTheDocument()

      fireEvent.drop(container, createDropEvent([createImageFile()]))
      expect(screen.queryByText('Drop images here')).not.toBeInTheDocument()
    })

    it('clears dragging state on dragLeave', () => {
      vi.mocked(useModelStore).mockReturnValue(visionModelState)
      render(<ChatInput {...defaultProps} />)

      const container = screen.getByPlaceholderText('Message... (paste or drag images)').closest('div')!
      fireEvent.dragOver(container)
      expect(screen.getByText('Drop images here')).toBeInTheDocument()

      fireEvent.dragLeave(container)
      expect(screen.queryByText('Drop images here')).not.toBeInTheDocument()
    })
  })

  describe('image submission', () => {
    it('enables send button when images present even without text', () => {
      vi.mocked(useModelStore).mockReturnValue(visionModelState)
      render(<ChatInput {...defaultProps} />)

      const container = screen.getByPlaceholderText('Message... (paste or drag images)').closest('div')!
      fireEvent.drop(container, createDropEvent([createImageFile()]))

      const sendButton = screen.getByTitle('Send message')
      expect(sendButton).not.toBeDisabled()
    })

    it('submits with base64 data from images', async () => {
      const restore = mockFileReader()

      const user = userEvent.setup()
      vi.mocked(useModelStore).mockReturnValue(visionModelState)
      render(<ChatInput {...defaultProps} />)

      const container = screen.getByPlaceholderText('Message... (paste or drag images)').closest('div')!
      fireEvent.drop(container, createDropEvent([createImageFile()]))

      const textarea = screen.getByPlaceholderText('Message... (paste or drag images)')
      await user.type(textarea, 'Look at this')
      await user.click(screen.getByTitle('Send message'))

      expect(mockSendMessage).toHaveBeenCalledWith(
        'conv-123',
        'Look at this',
        'model-vision',
        ['data:image/png;base64,abc123']
      )

      restore()
    })

    it('clears images after successful submission', async () => {
      const restore = mockFileReader()

      const user = userEvent.setup()
      vi.mocked(useModelStore).mockReturnValue(visionModelState)
      render(<ChatInput {...defaultProps} />)

      const container = screen.getByPlaceholderText('Message... (paste or drag images)').closest('div')!
      fireEvent.drop(container, createDropEvent([createImageFile()]))
      expect(screen.getByAltText('Upload 1')).toBeInTheDocument()

      const textarea = screen.getByPlaceholderText('Message... (paste or drag images)')
      await user.type(textarea, 'test')
      await user.click(screen.getByTitle('Send message'))

      expect(screen.queryByAltText('Upload 1')).not.toBeInTheDocument()

      restore()
    })

    it('revokes object URLs on submission', async () => {
      const restore = mockFileReader()

      const user = userEvent.setup()
      vi.mocked(useModelStore).mockReturnValue(visionModelState)
      render(<ChatInput {...defaultProps} />)

      const container = screen.getByPlaceholderText('Message... (paste or drag images)').closest('div')!
      fireEvent.drop(container, createDropEvent([createImageFile()]))

      const textarea = screen.getByPlaceholderText('Message... (paste or drag images)')
      await user.type(textarea, 'test')
      await user.click(screen.getByTitle('Send message'))

      expect(globalThis.URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock-url-1')

      restore()
    })
  })
})
