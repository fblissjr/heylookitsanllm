import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ChatView } from './ChatView'
import type { Conversation } from '../../../types/chat'
import type { LoadedModel, ModelStatus } from '../../../types/models'

// Mock child components
vi.mock('./MessageList', () => ({
  MessageList: vi.fn(({ messages, streaming, modelCapabilities }) => (
    <div data-testid="message-list">
      <span data-testid="messages-count">{messages.length}</span>
      <span data-testid="streaming-status">{streaming.isStreaming ? 'streaming' : 'idle'}</span>
      <span data-testid="model-vision">{modelCapabilities.vision ? 'has-vision' : 'no-vision'}</span>
      <span data-testid="model-thinking">{modelCapabilities.thinking ? 'has-thinking' : 'no-thinking'}</span>
    </div>
  )),
}))

vi.mock('./ChatInput', () => ({
  ChatInput: vi.fn(({ conversationId, modelId, hasVision, disabled }) => (
    <div data-testid="chat-input">
      <span data-testid="input-conversation-id">{conversationId}</span>
      <span data-testid="input-model-id">{modelId}</span>
      <span data-testid="input-has-vision">{hasVision ? 'vision-enabled' : 'vision-disabled'}</span>
      <span data-testid="input-disabled">{disabled ? 'disabled' : 'enabled'}</span>
    </div>
  )),
}))

vi.mock('./EmptyState', () => ({
  EmptyState: vi.fn(({ type }) => (
    <div data-testid="empty-state" data-type={type}>
      {type === 'no-model' ? 'No Model Loaded' : 'Start a New Conversation'}
    </div>
  )),
}))

// Mock stores
const mockActiveConversation = vi.fn()
const defaultChatState = {
  activeConversation: mockActiveConversation,
  streaming: {
    isStreaming: false,
    content: '',
    thinking: '',
    messageId: null,
  },
}

const defaultModelState: {
  loadedModel: LoadedModel | null
  modelStatus: ModelStatus
} = {
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
  modelStatus: 'loaded',
}

vi.mock('../../../stores/chatStore', () => ({
  useChatStore: vi.fn(() => defaultChatState),
}))

vi.mock('../../../stores/modelStore', () => ({
  useModelStore: vi.fn(() => defaultModelState),
}))

// Import after mocks
import { useChatStore } from '../../../stores/chatStore'
import { useModelStore } from '../../../stores/modelStore'
import { MessageList } from './MessageList'
import { ChatInput } from './ChatInput'

describe('ChatView', () => {
  const mockConversation: Conversation = {
    id: 'conv-123',
    title: 'Test Conversation',
    defaultModelId: 'test-model',
    messages: [
      {
        id: 'msg-1',
        role: 'user',
        content: 'Hello',
        timestamp: Date.now(),
      },
      {
        id: 'msg-2',
        role: 'assistant',
        content: 'Hi there!',
        timestamp: Date.now(),
      },
    ],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockActiveConversation.mockReturnValue(mockConversation)
    vi.mocked(useChatStore).mockReturnValue({
      ...defaultChatState,
      activeConversation: mockActiveConversation,
    })
    vi.mocked(useModelStore).mockReturnValue(defaultModelState)
  })

  describe('empty states', () => {
    it('shows EmptyState type="no-model" when no model loaded', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: null,
        modelStatus: 'unloaded',
      })

      render(<ChatView />)

      const emptyState = screen.getByTestId('empty-state')
      expect(emptyState).toBeInTheDocument()
      expect(emptyState).toHaveAttribute('data-type', 'no-model')
      expect(screen.getByText('No Model Loaded')).toBeInTheDocument()
    })

    it('shows EmptyState type="no-model" when modelStatus is not loaded', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: {
          id: 'test-model',
          capabilities: {
            chat: true,
            vision: false,
            thinking: false,
            hidden_states: false,
            embeddings: false,
          },
          contextWindow: 4096,
        },
        modelStatus: 'loading',
      })

      render(<ChatView />)

      const emptyState = screen.getByTestId('empty-state')
      expect(emptyState).toHaveAttribute('data-type', 'no-model')
    })

    it('shows EmptyState type="no-conversation" when no conversation selected', () => {
      mockActiveConversation.mockReturnValue(undefined)
      vi.mocked(useChatStore).mockReturnValue({
        ...defaultChatState,
        activeConversation: mockActiveConversation,
      })

      render(<ChatView />)

      const emptyState = screen.getByTestId('empty-state')
      expect(emptyState).toBeInTheDocument()
      expect(emptyState).toHaveAttribute('data-type', 'no-conversation')
      expect(screen.getByText('Start a New Conversation')).toBeInTheDocument()
    })
  })

  describe('conversation view', () => {
    it('renders MessageList and ChatInput when conversation exists', () => {
      render(<ChatView />)

      expect(screen.getByTestId('message-list')).toBeInTheDocument()
      expect(screen.getByTestId('chat-input')).toBeInTheDocument()
      expect(screen.queryByTestId('empty-state')).not.toBeInTheDocument()
    })

    it('does not show empty state when conversation exists', () => {
      render(<ChatView />)

      expect(screen.queryByTestId('empty-state')).not.toBeInTheDocument()
    })
  })

  describe('props passed to MessageList', () => {
    it('passes messages from conversation to MessageList', () => {
      render(<ChatView />)

      expect(screen.getByTestId('messages-count')).toHaveTextContent('2')
    })

    it('passes streaming state to MessageList', () => {
      render(<ChatView />)

      expect(screen.getByTestId('streaming-status')).toHaveTextContent('idle')
    })

    it('passes streaming state when actively streaming', () => {
      vi.mocked(useChatStore).mockReturnValue({
        activeConversation: mockActiveConversation,
        streaming: {
          isStreaming: true,
          content: 'Streaming...',
          thinking: '',
          messageId: 'msg-stream',
        },
      })

      render(<ChatView />)

      expect(screen.getByTestId('streaming-status')).toHaveTextContent('streaming')
    })

    it('passes modelCapabilities to MessageList', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: {
          id: 'vision-model',
          capabilities: {
            chat: true,
            vision: true,
            thinking: true,
            hidden_states: false,
            embeddings: false,
          },
          contextWindow: 8192,
        },
        modelStatus: 'loaded',
      })

      render(<ChatView />)

      expect(screen.getByTestId('model-vision')).toHaveTextContent('has-vision')
      expect(screen.getByTestId('model-thinking')).toHaveTextContent('has-thinking')
    })

    it('verifies MessageList is called with correct props', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: {
          id: 'test-model',
          capabilities: {
            chat: true,
            vision: false,
            thinking: false,
            hidden_states: false,
            embeddings: false,
          },
          contextWindow: 4096,
        },
        modelStatus: 'loaded',
      })

      render(<ChatView />)

      expect(MessageList).toHaveBeenCalled()
      const callArgs = vi.mocked(MessageList).mock.calls[0][0]
      expect(callArgs.messages).toEqual(mockConversation.messages)
      expect(callArgs.streaming.isStreaming).toBe(false)
      expect(callArgs.modelCapabilities.chat).toBe(true)
      expect(callArgs.modelCapabilities.vision).toBe(false)
    })
  })

  describe('props passed to ChatInput', () => {
    it('passes conversationId to ChatInput', () => {
      render(<ChatView />)

      expect(screen.getByTestId('input-conversation-id')).toHaveTextContent('conv-123')
    })

    it('passes modelId to ChatInput', () => {
      render(<ChatView />)

      expect(screen.getByTestId('input-model-id')).toHaveTextContent('test-model')
    })

    it('passes hasVision=false when model does not support vision', () => {
      render(<ChatView />)

      expect(screen.getByTestId('input-has-vision')).toHaveTextContent('vision-disabled')
    })

    it('passes hasVision=true when model supports vision', () => {
      vi.mocked(useModelStore).mockReturnValue({
        loadedModel: {
          id: 'vision-model',
          capabilities: {
            chat: true,
            vision: true,
            thinking: false,
            hidden_states: false,
            embeddings: false,
          },
          contextWindow: 4096,
        },
        modelStatus: 'loaded',
      })

      render(<ChatView />)

      expect(screen.getByTestId('input-has-vision')).toHaveTextContent('vision-enabled')
    })

    it('passes disabled=false when not streaming', () => {
      render(<ChatView />)

      expect(screen.getByTestId('input-disabled')).toHaveTextContent('enabled')
    })

    it('passes disabled=true when streaming', () => {
      vi.mocked(useChatStore).mockReturnValue({
        activeConversation: mockActiveConversation,
        streaming: {
          isStreaming: true,
          content: 'Generating...',
          thinking: '',
          messageId: 'msg-123',
        },
      })

      render(<ChatView />)

      expect(screen.getByTestId('input-disabled')).toHaveTextContent('disabled')
    })

    it('verifies ChatInput is called with correct props', () => {
      vi.mocked(useChatStore).mockReturnValue({
        activeConversation: mockActiveConversation,
        streaming: {
          isStreaming: true,
          content: '',
          thinking: '',
          messageId: null,
        },
      })

      render(<ChatView />)

      expect(ChatInput).toHaveBeenCalled()
      const callArgs = vi.mocked(ChatInput).mock.calls[0][0]
      expect(callArgs.conversationId).toBe('conv-123')
      expect(callArgs.modelId).toBe('test-model')
      expect(callArgs.hasVision).toBe(false)
      expect(callArgs.disabled).toBe(true)
    })
  })

  describe('scrolling behavior', () => {
    it('renders scroll container with correct ref', () => {
      render(<ChatView />)

      // The scroll container should exist
      const scrollContainer = document.querySelector('.overflow-y-auto')
      expect(scrollContainer).toBeInTheDocument()
    })
  })
})
