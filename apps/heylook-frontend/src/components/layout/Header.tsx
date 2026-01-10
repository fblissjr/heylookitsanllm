import { useModelStore } from '../../stores/modelStore'
import { useUIStore } from '../../stores/uiStore'

export function Header() {
  const { models, loadedModel, modelStatus } = useModelStore()
  const { toggleSidebar, togglePanel, activePanel } = useUIStore()

  const currentModel = loadedModel
    ? models.find(m => m.id === loadedModel.id)
    : null

  const getStatusColor = () => {
    switch (modelStatus) {
      case 'loaded':
        return 'bg-accent-green'
      case 'loading':
        return 'bg-amber-500 animate-pulse'
      default:
        return 'bg-gray-500'
    }
  }

  return (
    <header className="shrink-0 flex items-center justify-between px-4 py-3 bg-background-light dark:bg-background-dark border-b border-gray-200 dark:border-gray-800 z-20">
      {/* Left: Menu button */}
      <button
        onClick={toggleSidebar}
        className="p-2 rounded-lg text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-surface-dark transition-colors"
        aria-label="Toggle sidebar"
      >
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      {/* Center: Model selector */}
      <button
        onClick={() => togglePanel('models')}
        className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-gray-100 dark:bg-surface-dark border border-transparent hover:border-gray-300 dark:hover:border-gray-600 transition-colors"
      >
        <span className={`w-2 h-2 rounded-full ${getStatusColor()}`} />
        <span className="text-sm font-medium text-gray-700 dark:text-gray-200 max-w-[150px] truncate">
          {currentModel?.id || 'Select Model'}
        </span>
        <svg className="w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Right: Settings buttons */}
      <div className="flex items-center gap-1">
        {/* Advanced settings (system prompt, templates) */}
        <button
          onClick={() => togglePanel('advanced')}
          className={`p-2 rounded-lg transition-colors ${
            activePanel === 'advanced'
              ? 'bg-primary/20 text-primary'
              : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-surface-dark'
          }`}
          aria-label="Advanced settings"
          title="System Prompt & Templates"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        </button>

        {/* Sampler settings */}
        <button
          onClick={() => togglePanel('settings')}
          className={`p-2 rounded-lg transition-colors relative ${
            activePanel === 'settings'
              ? 'bg-primary/20 text-primary'
              : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-surface-dark'
          }`}
          aria-label="Sampler settings"
          title="Generation Parameters"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
          </svg>
          {/* Indicator dot when settings are modified */}
          <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 bg-primary rounded-full" />
        </button>
      </div>
    </header>
  )
}
