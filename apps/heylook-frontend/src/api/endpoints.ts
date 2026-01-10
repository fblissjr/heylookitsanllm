// Typed API endpoint functions

import { fetchAPI, postAPI } from './client'
import type {
  Model,
  ModelListResponse,
  ServerCapabilities,
  ChatCompletionRequest,
  ChatCompletionResponse,
} from '../types/api'

// Models
export async function getModels(): Promise<Model[]> {
  const response = await fetchAPI<ModelListResponse>('/v1/models')
  return response.data || []
}

// Capabilities
export async function getCapabilities(): Promise<ServerCapabilities> {
  return fetchAPI<ServerCapabilities>('/v1/capabilities')
}

// Chat (non-streaming)
export async function chatComplete(
  request: ChatCompletionRequest
): Promise<ChatCompletionResponse> {
  return postAPI<ChatCompletionResponse>('/v1/chat/completions', {
    ...request,
    stream: false,
  })
}

// Admin
export async function reloadModels(): Promise<void> {
  await postAPI('/v1/admin/reload', {})
}

// Multipart image upload
export async function chatWithImages(
  model: string,
  prompt: string,
  images: File[],
  params: Partial<ChatCompletionRequest> = {}
): Promise<ChatCompletionResponse> {
  const formData = new FormData()

  // Build messages with placeholders
  interface MessageContentItem {
    type: string
    text?: string
    image_url?: { url: string }
  }
  const content: MessageContentItem[] = [
    { type: 'text', text: prompt }
  ]

  images.forEach(() => {
    content.push({
      type: 'image_url',
      image_url: { url: '__RAW_IMAGE__' }
    })
  })

  const messages = [{ role: 'user', content }]

  formData.append('model', model)
  formData.append('messages', JSON.stringify(messages))

  // Add parameters
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      formData.append(key, String(value))
    }
  })

  // Add image files
  images.forEach((file, index) => {
    formData.append('images', file, `image-${index}.jpg`)
  })

  // Add default resize options
  formData.append('resize_max', '1024')
  formData.append('image_quality', '85')

  const response = await fetch('/v1/chat/completions/multipart', {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  return response.json()
}

// Batch processing
export async function batchChat(
  model: string,
  prompts: string[],
  params: Partial<ChatCompletionRequest> = {}
): Promise<ChatCompletionResponse> {
  const messages = prompts.map(prompt => [
    { role: 'user' as const, content: prompt }
  ])

  return postAPI<ChatCompletionResponse>('/v1/batch/chat/completions', {
    model,
    messages,
    ...params,
  })
}
