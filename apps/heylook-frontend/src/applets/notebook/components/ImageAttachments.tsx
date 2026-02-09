import { useCallback, useRef } from 'react'
import { CloseIcon, PlusIcon } from '../../../components/icons'
import { generateId } from '../../../lib/id'
import type { ImageAttachment } from '../types'

interface ImageAttachmentsProps {
  images: ImageAttachment[]
  onAdd: (image: ImageAttachment) => void
  onRemove: (imageId: string) => void
}

export function ImageAttachments({ images, onAdd, onRemove }: ImageAttachmentsProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const processFile = useCallback(
    (file: File) => {
      const id = generateId('img')
      const reader = new FileReader()
      reader.onload = () => {
        onAdd({
          id,
          name: file.name,
          dataUrl: reader.result as string,
        })
      }
      reader.readAsDataURL(file)
    },
    [onAdd]
  )

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files
      if (!files) return
      for (const file of Array.from(files)) {
        if (file.type.startsWith('image/')) {
          processFile(file)
        }
      }
      // Reset so same file can be re-selected
      e.target.value = ''
    },
    [processFile]
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      const files = e.dataTransfer.files
      for (const file of Array.from(files)) {
        if (file.type.startsWith('image/')) {
          processFile(file)
        }
      }
    },
    [processFile]
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  return (
    <div
      className="space-y-2"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
          Images
        </label>
        <button
          onClick={() => fileInputRef.current?.click()}
          className="text-xs text-primary hover:text-primary-hover flex items-center gap-1"
        >
          <PlusIcon className="w-3 h-3" />
          Attach
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          onChange={handleFileChange}
          className="hidden"
        />
      </div>

      {images.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {images.map((img) => (
            <div
              key={img.id}
              className="relative group w-16 h-16 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 bg-gray-100 dark:bg-gray-800"
            >
              {img.dataUrl ? (
                <img
                  src={img.dataUrl}
                  alt={img.name}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-gray-400 text-[10px]">
                  {img.name.slice(0, 8)}
                </div>
              )}
              <button
                onClick={() => onRemove(img.id)}
                className="absolute top-0.5 right-0.5 w-4 h-4 rounded-full bg-black/60 text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <CloseIcon className="w-2.5 h-2.5" />
              </button>
            </div>
          ))}
        </div>
      ) : (
        <div
          className="border border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-3 text-center cursor-pointer hover:border-gray-400 dark:hover:border-gray-500 transition-colors"
          onClick={() => fileInputRef.current?.click()}
        >
          <p className="text-xs text-gray-400 dark:text-gray-500">
            Drop images here or click to attach
          </p>
          <p className="text-[10px] text-gray-300 dark:text-gray-600 mt-0.5">
            Provides context for vision models
          </p>
        </div>
      )}
    </div>
  )
}

export { type ImageAttachmentsProps }
