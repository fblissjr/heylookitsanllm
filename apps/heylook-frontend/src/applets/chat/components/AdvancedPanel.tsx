import { useState } from 'react'
import { useSettingsStore } from '../../../stores/settingsStore'
import { useUIStore } from '../../../stores/uiStore'
import { useChatStore } from '../stores/chatStore'
import { CloseIcon } from '../../../components/icons'
import type { Preset } from '../../../types/settings'
import clsx from 'clsx'

export function AdvancedPanel() {
  const {
    systemPrompt,
    setSystemPrompt,
    getAllPresets,
    loadPreset,
    savePreset,
    deletePreset,
    activeSystemPromptPresetId,
  } = useSettingsStore()
  const { setActivePanel } = useUIStore()
  const { activeConversation } = useChatStore()
  const [isCreatingPreset, setIsCreatingPreset] = useState(false)
  const [newPresetName, setNewPresetName] = useState('')
  const [localPrompt, setLocalPrompt] = useState(systemPrompt)

  const presets = getAllPresets('system_prompt')
  const activePreset = presets.find(p => p.id === activeSystemPromptPresetId)
  const currentConversation = activeConversation()

  const handlePromptChange = (value: string) => {
    setLocalPrompt(value)
  }

  const handleApplyPrompt = () => {
    setSystemPrompt(localPrompt)
  }

  const handleSelectPreset = (preset: Preset) => {
    loadPreset(preset)
    setLocalPrompt(preset.data.prompt as string)
  }

  const handleSavePreset = () => {
    if (newPresetName.trim()) {
      // Apply the local prompt first
      setSystemPrompt(localPrompt)
      savePreset('system_prompt', newPresetName.trim())
      setNewPresetName('')
      setIsCreatingPreset(false)
    }
  }

  const handleDeletePreset = (preset: Preset) => {
    if (!preset.isBuiltIn) {
      deletePreset(preset.id)
    }
  }

  // Check if prompt has been modified from preset
  const isModified = activePreset
    ? localPrompt !== (activePreset.data.prompt as string)
    : localPrompt !== systemPrompt

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            System Prompt
          </h2>
          <button
            onClick={() => setActivePanel(null)}
            className="p-1 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800"
          >
            <CloseIcon />
          </button>
        </div>
        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
          Set instructions for how the AI should behave
        </p>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Presets */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Presets
          </label>
          <div className="flex flex-wrap gap-2">
            {presets.map(preset => (
              <button
                key={preset.id}
                onClick={() => handleSelectPreset(preset)}
                className={clsx(
                  'px-3 py-1.5 text-xs rounded-full transition-colors flex items-center gap-1',
                  activePreset?.id === preset.id
                    ? 'bg-primary text-white'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                )}
              >
                {preset.name}
                {!preset.isBuiltIn && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      handleDeletePreset(preset)
                    }}
                    className="ml-1 p-0.5 rounded hover:bg-red-500/20"
                    title="Delete preset"
                  >
                    <CloseIcon className="w-3 h-3" />
                  </button>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* System Prompt Editor */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Prompt
            {isModified && (
              <span className="ml-2 text-xs text-amber-500">(modified)</span>
            )}
          </label>
          <textarea
            value={localPrompt}
            onChange={(e) => handlePromptChange(e.target.value)}
            placeholder="Enter instructions for the AI..."
            className="w-full h-40 px-3 py-2 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:ring-2 focus:ring-primary/50 focus:border-primary resize-none"
          />
        </div>

        {/* Current Conversation Info */}
        {currentConversation && (
          <div className="p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
              Current conversation uses:
            </p>
            <p className="text-sm text-gray-700 dark:text-gray-300 truncate">
              {currentConversation.systemPrompt || 'No system prompt set'}
            </p>
            <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
              Note: Changes apply to new conversations
            </p>
          </div>
        )}

        {/* Save as Preset */}
        {isCreatingPreset ? (
          <div className="flex gap-2">
            <input
              type="text"
              value={newPresetName}
              onChange={(e) => setNewPresetName(e.target.value)}
              placeholder="Preset name..."
              className="flex-1 px-3 py-2 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100"
              autoFocus
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSavePreset()
                if (e.key === 'Escape') setIsCreatingPreset(false)
              }}
            />
            <button
              onClick={handleSavePreset}
              disabled={!newPresetName.trim()}
              className="px-3 py-2 text-sm rounded-lg bg-primary text-white hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Save
            </button>
            <button
              onClick={() => setIsCreatingPreset(false)}
              className="px-3 py-2 text-sm rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={() => setIsCreatingPreset(true)}
            className="w-full px-3 py-2 text-sm rounded-lg border border-dashed border-gray-300 dark:border-gray-600 text-gray-500 dark:text-gray-400 hover:border-primary hover:text-primary transition-colors"
          >
            + Save as Preset
          </button>
        )}
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700">
        <button
          onClick={handleApplyPrompt}
          disabled={!isModified && localPrompt === systemPrompt}
          className={clsx(
            'w-full px-4 py-2 rounded-lg text-sm font-medium transition-colors',
            isModified || localPrompt !== systemPrompt
              ? 'bg-primary text-white hover:bg-primary-hover'
              : 'bg-gray-100 dark:bg-gray-800 text-gray-400 cursor-not-allowed'
          )}
        >
          Apply to New Conversations
        </button>
      </div>
    </div>
  )
}
