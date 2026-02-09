import { useState, useCallback } from 'react'
import { ChevronDownIcon } from '../../../components/icons'
import { SamplerControls } from '../../../components/composed/SamplerControls'
import { useModelStore } from '../../../stores/modelStore'
import { useUIStore } from '../../../stores/uiStore'
import { DEFAULT_SAMPLER_SETTINGS } from '../../../types/settings'
import type { SamplerSettings } from '../../../types/settings'
import { useNotebookStore } from '../stores/notebookStore'
import { ImageAttachments } from './ImageAttachments'
import { DocumentList } from './DocumentList'
import type { ImageAttachment } from '../types'
import clsx from 'clsx'

export function LeftPanel() {
  const [showSampler, setShowSampler] = useState(false)

  const documents = useNotebookStore((s) => s.documents)
  const activeDocumentId = useNotebookStore((s) => s.activeDocumentId)
  const samplerSettings = useNotebookStore((s) => s.samplerSettings)
  const createDocument = useNotebookStore((s) => s.createDocument)
  const selectDocument = useNotebookStore((s) => s.selectDocument)
  const deleteDocument = useNotebookStore((s) => s.deleteDocument)
  const updateTitle = useNotebookStore((s) => s.updateTitle)
  const updateSystemPrompt = useNotebookStore((s) => s.updateSystemPrompt)
  const addImage = useNotebookStore((s) => s.addImage)
  const removeImage = useNotebookStore((s) => s.removeImage)
  const updateSamplerSettings = useNotebookStore((s) => s.updateSamplerSettings)

  const loadedModel = useModelStore((s) => s.loadedModel)
  const setActivePanel = useUIStore((s) => s.setActivePanel)

  const activeDoc = activeDocumentId
    ? documents.find((d) => d.id === activeDocumentId)
    : null

  const handleSystemPromptChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      if (!activeDocumentId) return
      updateSystemPrompt(activeDocumentId, e.target.value)
    },
    [activeDocumentId, updateSystemPrompt]
  )

  const handleAddImage = useCallback(
    (image: ImageAttachment) => {
      if (!activeDocumentId) return
      addImage(activeDocumentId, image)
    },
    [activeDocumentId, addImage]
  )

  const handleRemoveImage = useCallback(
    (imageId: string) => {
      if (!activeDocumentId) return
      removeImage(activeDocumentId, imageId)
    },
    [activeDocumentId, removeImage]
  )

  const handleSamplerUpdate = useCallback(
    <K extends keyof SamplerSettings>(key: K, value: SamplerSettings[K]) => {
      updateSamplerSettings({ [key]: value })
    },
    [updateSamplerSettings]
  )

  // Merge partial settings with defaults for SamplerControls
  const mergedSettings: SamplerSettings = {
    ...DEFAULT_SAMPLER_SETTINGS,
    ...samplerSettings,
  }

  return (
    <div className="w-72 flex-shrink-0 border-r border-gray-200 dark:border-gray-700 flex flex-col overflow-hidden bg-gray-50 dark:bg-surface-darker">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
          Notebook
        </h1>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
          Base-model text continuation
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        {/* Model */}
        <div>
          <label className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
            Model
          </label>
          <button
            onClick={() => setActivePanel('models')}
            className="mt-1 w-full text-left px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-sm hover:border-gray-300 dark:hover:border-gray-600 transition-colors"
          >
            {loadedModel ? (
              <span className="text-primary font-medium truncate block">
                {loadedModel.id}
              </span>
            ) : (
              <span className="text-gray-400">Select a model...</span>
            )}
          </button>
        </div>

        {/* System Prompt */}
        {activeDoc && (
          <div>
            <label className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              System Prompt (sent as-is)
            </label>
            <textarea
              value={activeDoc.systemPrompt}
              onChange={handleSystemPromptChange}
              placeholder="Optional. Leave empty for raw continuation."
              rows={3}
              className="mt-1 w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-sm text-gray-900 dark:text-gray-100 placeholder:text-gray-400 resize-none focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
        )}

        {/* Images */}
        {activeDoc && (
          <ImageAttachments
            images={activeDoc.images}
            onAdd={handleAddImage}
            onRemove={handleRemoveImage}
          />
        )}

        {/* Sampler Settings */}
        <div>
          <button
            onClick={() => setShowSampler(!showSampler)}
            className="w-full flex items-center justify-between text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
          >
            <span>Sampler Settings</span>
            <ChevronDownIcon
              className={clsx(
                'w-3.5 h-3.5 transition-transform',
                showSampler && 'rotate-180'
              )}
            />
          </button>
          {showSampler && (
            <div className="mt-3">
              <SamplerControls
                settings={mergedSettings}
                onUpdate={handleSamplerUpdate}
              />
            </div>
          )}
        </div>

        {/* Document List */}
        <DocumentList
          documents={documents}
          activeDocumentId={activeDocumentId}
          onSelect={selectDocument}
          onCreate={() => createDocument()}
          onDelete={deleteDocument}
          onRename={updateTitle}
        />
      </div>
    </div>
  )
}
