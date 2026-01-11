import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { fetchAPI, postAPI, APIError } from './client'
import { streamChat, chatCompletion, type StreamCallbacks, type StreamCompletionData } from './streaming'
import {
  getModels,
  getCapabilities,
  chatComplete,
  reloadModels,
  chatWithImages,
  batchChat,
} from './endpoints'
import type {
  Model,
  ModelListResponse,
  ServerCapabilities,
  ChatCompletionRequest,
  ChatCompletionResponse,
  Usage,
} from '../types/api'

// Mock global fetch
const mockFetch = vi.fn()
global.fetch = mockFetch

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  vi.restoreAllMocks()
})

// =============================================================================
// APIError Class Tests
// =============================================================================
describe('APIError', () => {
  it('creates error with status and statusText', () => {
    const error = new APIError(404, 'Not Found')
    expect(error.status).toBe(404)
    expect(error.statusText).toBe('Not Found')
    expect(error.message).toBe('API Error 404: Not Found')
    expect(error.name).toBe('APIError')
    expect(error.body).toBeUndefined()
  })

  it('creates error with body', () => {
    const body = { detail: 'Model not found' }
    const error = new APIError(404, 'Not Found', body)
    expect(error.body).toEqual(body)
  })

  it('extends Error class', () => {
    const error = new APIError(500, 'Internal Server Error')
    expect(error).toBeInstanceOf(Error)
    expect(error).toBeInstanceOf(APIError)
  })
})

// =============================================================================
// fetchAPI Tests
// =============================================================================
describe('fetchAPI', () => {
  describe('successful requests', () => {
    it('returns JSON response on success', async () => {
      const mockData = { id: 'test', value: 123 }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockData),
      })

      const result = await fetchAPI<typeof mockData>('/api/test')
      expect(result).toEqual(mockData)
      expect(mockFetch).toHaveBeenCalledWith('/api/test', {
        headers: {
          'Content-Type': 'application/json',
        },
      })
    })

    it('merges custom headers with defaults', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({}),
      })

      await fetchAPI('/api/test', {
        headers: {
          'Authorization': 'Bearer token',
          'X-Custom': 'value',
        },
      })

      expect(mockFetch).toHaveBeenCalledWith('/api/test', {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer token',
          'X-Custom': 'value',
        },
      })
    })

    it('passes additional options', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({}),
      })

      const signal = new AbortController().signal
      await fetchAPI('/api/test', {
        method: 'DELETE',
        signal,
      })

      expect(mockFetch).toHaveBeenCalledWith('/api/test', expect.objectContaining({
        method: 'DELETE',
        signal,
      }))
    })
  })

  describe('error responses', () => {
    it('throws APIError with JSON body on non-ok response', async () => {
      const errorBody = { detail: 'Validation failed' }
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 422,
        statusText: 'Unprocessable Entity',
        json: () => Promise.resolve(errorBody),
      })

      await expect(fetchAPI('/api/test')).rejects.toThrow(APIError)

      try {
        await fetchAPI('/api/test')
      } catch (error) {
        // fetchAPI is called twice, so we need fresh mock for second call
      }

      // Reset and test again properly
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 422,
        statusText: 'Unprocessable Entity',
        json: () => Promise.resolve(errorBody),
      })

      try {
        await fetchAPI('/api/test')
      } catch (error) {
        expect(error).toBeInstanceOf(APIError)
        const apiError = error as APIError
        expect(apiError.status).toBe(422)
        expect(apiError.statusText).toBe('Unprocessable Entity')
        expect(apiError.body).toEqual(errorBody)
      }
    })

    it('throws APIError with text body when JSON parsing fails', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: () => Promise.reject(new Error('Invalid JSON')),
        text: () => Promise.resolve('Server crashed'),
      })

      try {
        await fetchAPI('/api/test')
      } catch (error) {
        expect(error).toBeInstanceOf(APIError)
        const apiError = error as APIError
        expect(apiError.status).toBe(500)
        expect(apiError.body).toBe('Server crashed')
      }
    })

    it('throws network error for fetch failures', async () => {
      mockFetch.mockRejectedValueOnce(new TypeError('fetch failed'))

      await expect(fetchAPI('/api/test')).rejects.toThrow('Network error: Unable to connect to server')
    })

    it('re-throws APIError without wrapping', async () => {
      const originalError = new APIError(401, 'Unauthorized')
      mockFetch.mockRejectedValueOnce(originalError)

      try {
        await fetchAPI('/api/test')
      } catch (error) {
        expect(error).toBe(originalError)
      }
    })

    it('re-throws unknown errors', async () => {
      const unknownError = new Error('Something unexpected')
      mockFetch.mockRejectedValueOnce(unknownError)

      try {
        await fetchAPI('/api/test')
      } catch (error) {
        expect(error).toBe(unknownError)
      }
    })
  })
})

// =============================================================================
// postAPI Tests
// =============================================================================
describe('postAPI', () => {
  it('sends POST request with JSON body', async () => {
    const requestData = { name: 'test', value: 42 }
    const responseData = { id: 1, ...requestData }

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(responseData),
    })

    const result = await postAPI<typeof responseData>('/api/items', requestData)

    expect(result).toEqual(responseData)
    expect(mockFetch).toHaveBeenCalledWith('/api/items', {
      method: 'POST',
      body: JSON.stringify(requestData),
      headers: {
        'Content-Type': 'application/json',
      },
    })
  })

  it('merges additional options', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({}),
    })

    await postAPI('/api/items', { data: 'test' }, {
      headers: { 'X-Custom': 'header' },
    })

    expect(mockFetch).toHaveBeenCalledWith('/api/items', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ data: 'test' }),
      headers: {
        'Content-Type': 'application/json',
        'X-Custom': 'header',
      },
    }))
  })

  it('handles complex nested data', async () => {
    const complexData = {
      messages: [
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi there!' },
      ],
      options: {
        temperature: 0.7,
        max_tokens: 100,
      },
    }

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ success: true }),
    })

    await postAPI('/api/chat', complexData)

    expect(mockFetch).toHaveBeenCalledWith('/api/chat', expect.objectContaining({
      body: JSON.stringify(complexData),
    }))
  })
})

// =============================================================================
// Endpoint Functions Tests
// =============================================================================
describe('endpoints', () => {
  describe('getModels', () => {
    it('fetches and returns models array', async () => {
      const mockModels: Model[] = [
        { id: 'model-1', object: 'model', owned_by: 'local', provider: 'mlx' },
        { id: 'model-2', object: 'model', owned_by: 'local', provider: 'llama_cpp' },
      ]
      const mockResponse: ModelListResponse = {
        object: 'list',
        data: mockModels,
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })

      const result = await getModels()
      expect(result).toEqual(mockModels)
      expect(mockFetch).toHaveBeenCalledWith('/v1/models', expect.any(Object))
    })

    it('returns empty array when data is undefined', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ object: 'list' }),
      })

      const result = await getModels()
      expect(result).toEqual([])
    })
  })

  describe('getCapabilities', () => {
    it('fetches server capabilities', async () => {
      const mockCapabilities: Partial<ServerCapabilities> = {
        server_version: '1.0.0',
        features: {
          streaming: true,
          model_caching: { enabled: true, cache_size: 2, eviction_policy: 'lru' },
          vision_models: true,
          concurrent_requests: false,
          supported_image_formats: ['jpg', 'png'],
        },
        limits: {
          max_tokens: 4096,
          max_images_per_request: 5,
          max_request_size_mb: 10,
          timeout_seconds: 300,
        },
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockCapabilities),
      })

      const result = await getCapabilities()
      expect(result).toEqual(mockCapabilities)
      expect(mockFetch).toHaveBeenCalledWith('/v1/capabilities', expect.any(Object))
    })
  })

  describe('chatComplete', () => {
    it('sends chat completion request', async () => {
      const request: ChatCompletionRequest = {
        model: 'test-model',
        messages: [{ role: 'user', content: 'Hello' }],
        temperature: 0.7,
      }

      const mockResponse: ChatCompletionResponse = {
        id: 'chatcmpl-123',
        object: 'chat.completion',
        created: Date.now(),
        model: 'test-model',
        choices: [{
          index: 0,
          message: { role: 'assistant', content: 'Hi there!' },
          finish_reason: 'stop',
        }],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })

      const result = await chatComplete(request)
      expect(result).toEqual(mockResponse)

      const [url, options] = mockFetch.mock.calls[0]
      expect(url).toBe('/v1/chat/completions')
      expect(JSON.parse(options.body)).toEqual({
        ...request,
        stream: false,
      })
    })
  })

  describe('reloadModels', () => {
    it('sends reload request', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ status: 'ok' }),
      })

      await reloadModels()

      const [url, options] = mockFetch.mock.calls[0]
      expect(url).toBe('/v1/admin/reload')
      expect(options.method).toBe('POST')
    })
  })

  describe('chatWithImages', () => {
    it('sends multipart form data with images', async () => {
      const mockResponse: ChatCompletionResponse = {
        id: 'chatcmpl-img-123',
        object: 'chat.completion',
        created: Date.now(),
        model: 'vision-model',
        choices: [{
          index: 0,
          message: { role: 'assistant', content: 'I see an image' },
          finish_reason: 'stop',
        }],
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })

      const imageFile = new File(['fake image data'], 'test.jpg', { type: 'image/jpeg' })
      const result = await chatWithImages('vision-model', 'What is in this image?', [imageFile])

      expect(result).toEqual(mockResponse)
      expect(mockFetch).toHaveBeenCalledWith(
        '/v1/chat/completions/multipart',
        expect.objectContaining({
          method: 'POST',
          body: expect.any(FormData),
        })
      )

      // Verify FormData contents
      const formData = mockFetch.mock.calls[0][1].body as FormData
      expect(formData.get('model')).toBe('vision-model')
      expect(formData.get('resize_max')).toBe('1024')
      expect(formData.get('image_quality')).toBe('85')
    })

    it('includes additional params in form data', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          id: 'test',
          object: 'chat.completion',
          created: Date.now(),
          model: 'vision-model',
          choices: [],
        }),
      })

      const imageFile = new File(['data'], 'test.jpg', { type: 'image/jpeg' })
      await chatWithImages('vision-model', 'Describe', [imageFile], {
        temperature: 0.5,
        max_tokens: 256,
      })

      const formData = mockFetch.mock.calls[0][1].body as FormData
      expect(formData.get('temperature')).toBe('0.5')
      expect(formData.get('max_tokens')).toBe('256')
    })

    it('throws error on non-ok response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: () => Promise.resolve({ detail: 'Invalid image format' }),
      })

      const imageFile = new File(['data'], 'test.jpg', { type: 'image/jpeg' })

      await expect(chatWithImages('model', 'prompt', [imageFile])).rejects.toThrow('Invalid image format')
    })

    it('handles JSON parse error in error response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: () => Promise.reject(new Error('Invalid JSON')),
      })

      const imageFile = new File(['data'], 'test.jpg', { type: 'image/jpeg' })

      // When JSON parsing fails, the code falls back to statusText
      await expect(chatWithImages('model', 'prompt', [imageFile])).rejects.toThrow('Internal Server Error')
    })
  })

  describe('batchChat', () => {
    it('sends batch chat request with multiple prompts', async () => {
      const mockResponse: ChatCompletionResponse = {
        id: 'batch-123',
        object: 'chat.completion',
        created: Date.now(),
        model: 'test-model',
        choices: [
          { index: 0, message: { role: 'assistant', content: 'Response 1' }, finish_reason: 'stop' },
          { index: 1, message: { role: 'assistant', content: 'Response 2' }, finish_reason: 'stop' },
        ],
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })

      const prompts = ['Question 1', 'Question 2']
      const result = await batchChat('test-model', prompts, { temperature: 0.5 })

      expect(result).toEqual(mockResponse)

      const [url, options] = mockFetch.mock.calls[0]
      expect(url).toBe('/v1/batch/chat/completions')

      const body = JSON.parse(options.body)
      expect(body.model).toBe('test-model')
      expect(body.messages).toEqual([
        [{ role: 'user', content: 'Question 1' }],
        [{ role: 'user', content: 'Question 2' }],
      ])
      expect(body.temperature).toBe(0.5)
    })
  })
})

// =============================================================================
// Streaming Tests
// =============================================================================
describe('streaming', () => {
  // Helper to create a mock ReadableStream from SSE data
  function createMockStream(chunks: string[]): ReadableStream<Uint8Array> {
    const encoder = new TextEncoder()
    let index = 0

    return new ReadableStream({
      pull(controller) {
        if (index < chunks.length) {
          controller.enqueue(encoder.encode(chunks[index]))
          index++
        } else {
          controller.close()
        }
      },
    })
  }

  describe('streamChat', () => {
    it('processes SSE tokens and calls onToken', async () => {
      const tokens: string[] = []
      const callbacks: StreamCallbacks = {
        onToken: (token) => tokens.push(token),
        onComplete: vi.fn(),
        onError: vi.fn(),
      }

      const sseChunks = [
        'data: {"id":"1","choices":[{"delta":{"content":"Hello"}}]}\n\n',
        'data: {"id":"1","choices":[{"delta":{"content":" World"}}]}\n\n',
        'data: [DONE]\n\n',
      ]

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: createMockStream(sseChunks),
      })

      const request: ChatCompletionRequest = {
        model: 'test-model',
        messages: [{ role: 'user', content: 'Hi' }],
      }

      await streamChat(request, callbacks)

      expect(tokens).toEqual(['Hello', ' World'])
      expect(callbacks.onComplete).toHaveBeenCalled()
      expect(callbacks.onError).not.toHaveBeenCalled()
    })

    it('handles thinking tokens', async () => {
      const thinking: string[] = []
      const tokens: string[] = []
      const callbacks: StreamCallbacks = {
        onToken: (token) => tokens.push(token),
        onThinking: (t) => thinking.push(t),
        onComplete: vi.fn(),
        onError: vi.fn(),
      }

      const sseChunks = [
        'data: {"id":"1","choices":[{"delta":{"thinking":"Let me think"}}]}\n\n',
        'data: {"id":"1","choices":[{"delta":{"thinking":"..."}}]}\n\n',
        'data: {"id":"1","choices":[{"delta":{"content":"The answer is 42"}}]}\n\n',
        'data: [DONE]\n\n',
      ]

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: createMockStream(sseChunks),
      })

      const request: ChatCompletionRequest = {
        model: 'test-model',
        messages: [{ role: 'user', content: 'Think about it' }],
        enable_thinking: true,
      }

      await streamChat(request, callbacks)

      expect(thinking).toEqual(['Let me think', '...'])
      expect(tokens).toEqual(['The answer is 42'])
    })

    it('extracts usage from final chunk', async () => {
      const usageResults: (StreamCompletionData | undefined)[] = []
      const callbacks: StreamCallbacks = {
        onToken: vi.fn(),
        onComplete: (data) => { usageResults.push(data) },
        onError: vi.fn(),
      }

      const usage: Usage = {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
      }

      const sseChunks = [
        'data: {"id":"1","choices":[{"delta":{"content":"Done"}}]}\n\n',
        `data: {"id":"1","choices":[],"usage":${JSON.stringify(usage)}}\n\n`,
        'data: [DONE]\n\n',
      ]

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: createMockStream(sseChunks),
      })

      await streamChat(
        { model: 'test', messages: [] },
        callbacks
      )

      // onComplete is called twice: once when usage chunk is received, once at end of stream
      // The first call should have the usage wrapped in StreamCompletionData
      expect(usageResults[0]).toEqual({
        usage,
        timing: undefined,
        generationConfig: undefined,
        stopReason: undefined,
      })
    })

    it('handles chunked SSE data (split across reads)', async () => {
      const tokens: string[] = []
      const callbacks: StreamCallbacks = {
        onToken: (token) => tokens.push(token),
        onComplete: vi.fn(),
        onError: vi.fn(),
      }

      // Data split across multiple chunks
      const sseChunks = [
        'data: {"id":"1","choices":[{"delta":{"content":"He',
        'llo"}}]}\n\ndata: {"id":"1","choices":[{"delta":{"content":" W',
        'orld"}}]}\n\ndata: [DONE]\n\n',
      ]

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: createMockStream(sseChunks),
      })

      await streamChat(
        { model: 'test', messages: [] },
        callbacks
      )

      expect(tokens).toEqual(['Hello', ' World'])
    })

    it('handles HTTP error response', async () => {
      const callbacks: StreamCallbacks = {
        onToken: vi.fn(),
        onComplete: vi.fn(),
        onError: vi.fn(),
      }

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        json: () => Promise.resolve({ detail: 'Model not loaded' }),
      })

      await streamChat(
        { model: 'test', messages: [] },
        callbacks
      )

      expect(callbacks.onError).toHaveBeenCalled()
      const error = (callbacks.onError as ReturnType<typeof vi.fn>).mock.calls[0][0]
      expect(error.message).toContain('Model not loaded')
    })

    it('handles missing response body', async () => {
      const callbacks: StreamCallbacks = {
        onToken: vi.fn(),
        onComplete: vi.fn(),
        onError: vi.fn(),
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: null,
      })

      await streamChat(
        { model: 'test', messages: [] },
        callbacks
      )

      expect(callbacks.onError).toHaveBeenCalled()
      const error = (callbacks.onError as ReturnType<typeof vi.fn>).mock.calls[0][0]
      expect(error.message).toContain('No response body')
    })

    it('handles abort signal gracefully', async () => {
      const callbacks: StreamCallbacks = {
        onToken: vi.fn(),
        onComplete: vi.fn(),
        onError: vi.fn(),
      }

      // Create proper AbortError (DOMException with AbortError name)
      const abortError = new Error('The operation was aborted')
      abortError.name = 'AbortError'
      mockFetch.mockRejectedValueOnce(abortError)

      const controller = new AbortController()
      controller.abort()

      await streamChat(
        { model: 'test', messages: [] },
        callbacks,
        controller.signal
      )

      // When abort happens, onComplete is called and onError is not
      expect(callbacks.onComplete).toHaveBeenCalled()
      expect(callbacks.onError).not.toHaveBeenCalled()
    })

    it('calls onError for unknown errors', async () => {
      const callbacks: StreamCallbacks = {
        onToken: vi.fn(),
        onComplete: vi.fn(),
        onError: vi.fn(),
      }

      mockFetch.mockRejectedValueOnce('string error')

      await streamChat(
        { model: 'test', messages: [] },
        callbacks
      )

      expect(callbacks.onError).toHaveBeenCalledWith(expect.any(Error))
      const error = (callbacks.onError as ReturnType<typeof vi.fn>).mock.calls[0][0]
      expect(error.message).toBe('Unknown streaming error')
    })

    it('skips invalid JSON in SSE chunks', async () => {
      const tokens: string[] = []
      const callbacks: StreamCallbacks = {
        onToken: (token) => tokens.push(token),
        onComplete: vi.fn(),
        onError: vi.fn(),
      }

      const sseChunks = [
        'data: {"id":"1","choices":[{"delta":{"content":"Valid"}}]}\n\n',
        'data: {invalid json}\n\n',
        'data: {"id":"1","choices":[{"delta":{"content":" Text"}}]}\n\n',
        'data: [DONE]\n\n',
      ]

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: createMockStream(sseChunks),
      })

      await streamChat(
        { model: 'test', messages: [] },
        callbacks
      )

      expect(tokens).toEqual(['Valid', ' Text'])
      expect(callbacks.onError).not.toHaveBeenCalled()
    })

    it('skips empty lines and non-data lines', async () => {
      const tokens: string[] = []
      const callbacks: StreamCallbacks = {
        onToken: (token) => tokens.push(token),
        onComplete: vi.fn(),
        onError: vi.fn(),
      }

      const sseChunks = [
        ': comment line\n',
        '\n',
        '   \n',
        'event: message\n',
        'data: {"id":"1","choices":[{"delta":{"content":"Token"}}]}\n\n',
        'data: [DONE]\n\n',
      ]

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: createMockStream(sseChunks),
      })

      await streamChat(
        { model: 'test', messages: [] },
        callbacks
      )

      expect(tokens).toEqual(['Token'])
    })

    it('sends stream: true and stream_options in request', async () => {
      const callbacks: StreamCallbacks = {
        onToken: vi.fn(),
        onComplete: vi.fn(),
        onError: vi.fn(),
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: createMockStream(['data: [DONE]\n\n']),
      })

      const request: ChatCompletionRequest = {
        model: 'test-model',
        messages: [{ role: 'user', content: 'Hi' }],
        temperature: 0.7,
      }

      await streamChat(request, callbacks)

      const [, options] = mockFetch.mock.calls[0]
      const body = JSON.parse(options.body)
      expect(body.stream).toBe(true)
      expect(body.stream_options).toEqual({ include_usage: true })
      expect(body.temperature).toBe(0.7)
    })
  })

  describe('chatCompletion (non-streaming)', () => {
    it('returns content and usage', async () => {
      const mockResponse = {
        choices: [{
          message: {
            role: 'assistant',
            content: 'Hello there!',
          },
        }],
        usage: {
          prompt_tokens: 5,
          completion_tokens: 3,
          total_tokens: 8,
        },
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })

      const result = await chatCompletion({
        model: 'test',
        messages: [{ role: 'user', content: 'Hi' }],
      })

      expect(result.content).toBe('Hello there!')
      expect(result.usage).toEqual(mockResponse.usage)
    })

    it('returns thinking content when present', async () => {
      const mockResponse = {
        choices: [{
          message: {
            role: 'assistant',
            content: 'The answer is 42',
            thinking: 'Let me calculate...',
          },
        }],
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })

      const result = await chatCompletion({
        model: 'test',
        messages: [{ role: 'user', content: 'What is the answer?' }],
        enable_thinking: true,
      })

      expect(result.content).toBe('The answer is 42')
      expect(result.thinking).toBe('Let me calculate...')
    })

    it('sends stream: false in request', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          choices: [{ message: { content: 'response' } }],
        }),
      })

      await chatCompletion({
        model: 'test',
        messages: [],
      })

      const [, options] = mockFetch.mock.calls[0]
      const body = JSON.parse(options.body)
      expect(body.stream).toBe(false)
    })

    it('throws error on non-ok response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: () => Promise.resolve({ detail: 'Invalid request' }),
      })

      await expect(chatCompletion({
        model: 'test',
        messages: [],
      })).rejects.toThrow('Invalid request')
    })

    it('handles empty choices array', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ choices: [] }),
      })

      const result = await chatCompletion({
        model: 'test',
        messages: [],
      })

      expect(result.content).toBe('')
    })

    it('handles missing message in choice', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          choices: [{ index: 0 }],
        }),
      })

      const result = await chatCompletion({
        model: 'test',
        messages: [],
      })

      expect(result.content).toBe('')
    })
  })
})

// =============================================================================
// Request/Response Transformation Tests
// =============================================================================
describe('request/response transformations', () => {
  describe('message content types', () => {
    it('handles string content', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          choices: [{ message: { content: 'response' } }],
        }),
      })

      await chatComplete({
        model: 'test',
        messages: [{ role: 'user', content: 'Simple text' }],
      })

      const body = JSON.parse(mockFetch.mock.calls[0][1].body)
      expect(body.messages[0].content).toBe('Simple text')
    })

    it('handles multimodal content array', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          choices: [{ message: { content: 'I see the image' } }],
        }),
      })

      await chatComplete({
        model: 'vision-model',
        messages: [{
          role: 'user',
          content: [
            { type: 'text', text: 'What is this?' },
            { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,abc123' } },
          ],
        }],
      })

      const body = JSON.parse(mockFetch.mock.calls[0][1].body)
      expect(body.messages[0].content).toEqual([
        { type: 'text', text: 'What is this?' },
        { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,abc123' } },
      ])
    })
  })

  describe('sampler parameters', () => {
    it('includes all sampler params in request', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          choices: [{ message: { content: 'response' } }],
        }),
      })

      const params = {
        temperature: 0.8,
        max_tokens: 500,
        top_p: 0.9,
        top_k: 40,
        min_p: 0.05,
        repetition_penalty: 1.1,
        seed: 42,
        stop: ['END', 'STOP'],
      }

      await chatComplete({
        model: 'test',
        messages: [{ role: 'user', content: 'test' }],
        ...params,
      })

      const body = JSON.parse(mockFetch.mock.calls[0][1].body)
      expect(body.temperature).toBe(0.8)
      expect(body.max_tokens).toBe(500)
      expect(body.top_p).toBe(0.9)
      expect(body.top_k).toBe(40)
      expect(body.min_p).toBe(0.05)
      expect(body.repetition_penalty).toBe(1.1)
      expect(body.seed).toBe(42)
      expect(body.stop).toEqual(['END', 'STOP'])
    })
  })

  describe('special request options', () => {
    it('includes logprobs options', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          choices: [{
            message: { content: 'response' },
            logprobs: { content: [{ token: 'response', logprob: -0.5, token_id: 1, bytes: [] }] },
          }],
        }),
      })

      await chatComplete({
        model: 'test',
        messages: [{ role: 'user', content: 'test' }],
        logprobs: true,
        top_logprobs: 5,
      })

      const body = JSON.parse(mockFetch.mock.calls[0][1].body)
      expect(body.logprobs).toBe(true)
      expect(body.top_logprobs).toBe(5)
    })

    it('includes enable_thinking option', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          choices: [{ message: { content: 'response', thinking: 'internal thoughts' } }],
        }),
      })

      await chatComplete({
        model: 'qwen3-model',
        messages: [{ role: 'user', content: 'think about this' }],
        enable_thinking: true,
      })

      const body = JSON.parse(mockFetch.mock.calls[0][1].body)
      expect(body.enable_thinking).toBe(true)
    })
  })
})

// =============================================================================
// Edge Cases and Error Scenarios
// =============================================================================
describe('edge cases', () => {
  it('handles timeout errors', async () => {
    const timeoutError = new Error('The operation timed out')
    timeoutError.name = 'TimeoutError'
    mockFetch.mockRejectedValueOnce(timeoutError)

    await expect(fetchAPI('/api/slow')).rejects.toThrow('The operation timed out')
  })

  it('handles null response body in JSON', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(null),
    })

    const result = await fetchAPI('/api/null')
    expect(result).toBeNull()
  })

  it('handles empty string response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(''),
    })

    const result = await fetchAPI('/api/empty')
    expect(result).toBe('')
  })

  it('handles very long responses', async () => {
    const longContent = 'x'.repeat(100000)
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ content: longContent }),
    })

    const result = await fetchAPI<{ content: string }>('/api/long')
    expect(result.content.length).toBe(100000)
  })

  it('handles special characters in error messages', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
      json: () => Promise.resolve({ detail: 'Invalid chars: <script>alert("xss")</script>' }),
    })

    try {
      await fetchAPI('/api/test')
    } catch (error) {
      expect((error as APIError).body).toEqual({
        detail: 'Invalid chars: <script>alert("xss")</script>',
      })
    }
  })

  it('handles unicode in responses', async () => {
    const unicodeContent = 'Hello! Bonjour! Hallo! Konnichiwa!'
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ content: unicodeContent }),
    })

    const result = await fetchAPI<{ content: string }>('/api/unicode')
    expect(result.content).toBe(unicodeContent)
  })
})
