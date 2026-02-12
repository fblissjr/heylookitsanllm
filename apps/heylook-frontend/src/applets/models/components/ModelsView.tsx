import { useEffect } from 'react'
import { AppletLayout } from '../../../components/layout/AppletLayout'
import { useModelsStore } from '../stores/modelsStore'
import { ModelList } from './ModelList'
import { ModelDetail } from './ModelDetail'
import { ModelImporter } from './ModelImporter'

export function ModelsView() {
  const fetchConfigs = useModelsStore((s) => s.fetchConfigs)
  const fetchProfiles = useModelsStore((s) => s.fetchProfiles)
  const error = useModelsStore((s) => s.error)
  const loading = useModelsStore((s) => s.loading)

  useEffect(() => {
    fetchConfigs()
    fetchProfiles()
  }, [fetchConfigs, fetchProfiles])

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  return (
    <>
      <AppletLayout leftPanel={<ModelList />} leftPanelWidth="w-72">
        <div className="h-full bg-white dark:bg-background-dark">
          {error && (
            <div className="px-4 py-2 bg-red-500/10 border-b border-red-500/20 text-xs text-red-400">
              {error}
            </div>
          )}
          <ModelDetail />
        </div>
      </AppletLayout>
      <ModelImporter />
    </>
  )
}
