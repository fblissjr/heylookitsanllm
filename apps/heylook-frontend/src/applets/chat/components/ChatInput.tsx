import { useState, useRef, useCallback, useEffect, KeyboardEvent, ChangeEvent, DragEvent } from 'react'
import { useChatStore } from '../stores/chatStore'
import { useModelStore } from '../../../stores/modelStore'
import clsx from 'clsx'

interface ImagePreview {
  file: File
  previewUrl: string
}

interface ChatInputProps {
  conversationId: string
  disabled: boolean
}

export function ChatInput({ conversationId, disabled }: ChatInputProps) {
  const [message, setMessage] = useState('')
  const [images, setImages] = useState<ImagePreview[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const { sendMessage, stopGeneration, streaming } = useChatStore()
  const loadedModel = useModelStore((s) => s.loadedModel)

  // Get vision capability from the globally loaded model
  const hasVision = loadedModel?.capabilities?.vision ?? false

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px'
    }
  }, [message])

  // Convert File to base64 data URL
  const fileToBase64 = useCallback((file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  }, [])

  const handleSubmit = useCallback(async () => {
    const trimmedMessage = message.trim()
    if (!trimmedMessage && images.length === 0) return
    if (disabled) return
    if (!loadedModel) return

    // Convert images to base64 only when sending (lazy conversion)
    let base64Images: string[] | undefined
    if (images.length > 0) {
      base64Images = await Promise.all(images.map(img => fileToBase64(img.file)))
    }

    // Clean up preview URLs before clearing state
    images.forEach(img => URL.revokeObjectURL(img.previewUrl))

    await sendMessage(conversationId, trimmedMessage, loadedModel.id, base64Images)
    setMessage('')
    setImages([])
  }, [message, images, disabled, conversationId, loadedModel, sendMessage, fileToBase64])

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }, [handleSubmit])

  const handleStop = useCallback(() => {
    stopGeneration()
  }, [stopGeneration])

  // Image handling - use object URLs for instant preview (no base64 conversion)
  const processFiles = useCallback((files: FileList | File[]) => {
    if (!hasVision) return

    const newImages: ImagePreview[] = []
    for (const file of Array.from(files)) {
      if (!file.type.startsWith('image/')) continue

      // Create object URL for instant preview (much faster than base64)
      const previewUrl = URL.createObjectURL(file)
      newImages.push({ file, previewUrl })
    }

    setImages(prev => [...prev, ...newImages])
  }, [hasVision])

  const handleFileSelect = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      processFiles(e.target.files)
      e.target.value = '' // Reset so same file can be selected again
    }
  }, [processFiles])

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const items = e.clipboardData.items
    const files: File[] = []

    for (const item of Array.from(items)) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile()
        if (file) files.push(file)
      }
    }

    if (files.length > 0) {
      processFiles(files)
    }
  }, [processFiles])

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault()
    if (hasVision) setIsDragging(true)
  }, [hasVision])

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files) {
      processFiles(e.dataTransfer.files)
    }
  }, [processFiles])

  const removeImage = useCallback((index: number) => {
    setImages(prev => {
      // Revoke the object URL to free memory
      const imageToRemove = prev[index]
      if (imageToRemove) {
        URL.revokeObjectURL(imageToRemove.previewUrl)
      }
      return prev.filter((_, i) => i !== index)
    })
  }, [])

  return (
    <div className="p-4 pb-[max(1rem,env(safe-area-inset-bottom))] bg-white dark:bg-background-dark border-t border-gray-200 dark:border-gray-800">
      {/* Image previews */}
      {images.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-3">
          {images.map((img, i) => (
            <div key={i} className="relative group">
              <img
                src={img.previewUrl}
                alt={`Upload ${i + 1}`}
                className="w-20 h-20 object-cover rounded-lg border border-gray-200 dark:border-gray-700"
              />
              <button
                onClick={() => removeImage(i)}
                className="absolute -top-1.5 -right-1.5 w-6 h-6 min-w-[24px] min-h-[24px] bg-red-500 text-white rounded-full flex items-center justify-center opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity"
                aria-label={`Remove image ${i + 1}`}
              >
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Input container */}
      <div
        className={clsx(
          'relative flex items-end gap-2 bg-gray-100 dark:bg-surface-dark rounded-3xl p-2 border transition-all',
          isDragging
            ? 'border-primary border-dashed bg-primary/5'
            : 'border-transparent focus-within:border-gray-300 dark:focus-within:border-gray-600'
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* Attachment button */}
        {hasVision && (
          <button
            onClick={() => fileInputRef.current?.click()}
            className="p-2 rounded-full text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors shrink-0"
            title="Add image"
            aria-label="Add image"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
        )}

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          onChange={handleFileSelect}
          className="hidden"
        />

        {/* Text input */}
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          placeholder={hasVision ? "Message... (paste or drag images)" : "Message..."}
          disabled={disabled}
          className={clsx(
            'flex-1 bg-transparent border-none text-gray-800 dark:text-white placeholder-gray-500 focus:ring-0 resize-none py-3 px-2 max-h-[200px] leading-relaxed',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
          rows={1}
        />

        {/* Send / Stop button */}
        {streaming.isStreaming ? (
          <button
            onClick={handleStop}
            className="p-2.5 min-w-[44px] min-h-[44px] flex items-center justify-center rounded-full bg-red-500 hover:bg-red-600 text-white transition-colors shrink-0 shadow-md"
            title="Stop generation"
            aria-label="Stop generation"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="6" width="12" height="12" rx="2" />
            </svg>
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            disabled={disabled || (!message.trim() && images.length === 0)}
            className={clsx(
              'p-2.5 min-w-[44px] min-h-[44px] flex items-center justify-center rounded-full transition-all shrink-0 shadow-md',
              !disabled && (message.trim() || images.length > 0)
                ? 'bg-primary hover:bg-primary-hover text-white'
                : 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed'
            )}
            title="Send message"
            aria-label="Send message"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
            </svg>
          </button>
        )}
      </div>

      {/* Drag overlay */}
      {isDragging && (
        <div className="absolute inset-0 bg-primary/10 rounded-3xl border-2 border-dashed border-primary flex items-center justify-center pointer-events-none">
          <span className="text-primary font-medium">Drop images here</span>
        </div>
      )}

      {/* Disclaimer */}
      <p className="text-[10px] text-gray-400 dark:text-gray-600 text-center mt-2">
        AI can make mistakes. Check important info.
      </p>
    </div>
  )
}
