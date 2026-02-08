import { useBatchStore } from '../stores/batchStore'
import { BatchCreateForm } from './BatchCreateForm'
import { BatchDashboard } from './BatchDashboard'

export function BatchView() {
  const view = useBatchStore((s) => s.view)

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {view === 'create' ? <BatchCreateForm /> : <BatchDashboard />}
    </div>
  )
}
