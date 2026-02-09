import { EmptyState as EmptyStateBase } from '../../../components/primitives'
import { ComputerIcon, ChatBubbleIcon } from '../../../components/icons'

interface ChatEmptyStateProps {
  type: 'no-model' | 'no-conversation'
}

const variants = {
  'no-model': {
    icon: <ComputerIcon className="w-8 h-8 text-gray-400" />,
    title: 'No Model Loaded',
    description: 'Select a model from the header to start chatting. The model will be loaded into memory for inference.',
  },
  'no-conversation': {
    icon: <ChatBubbleIcon className="w-8 h-8 text-gray-400" />,
    title: 'Start a New Conversation',
    description: 'Click "New Chat" in the sidebar or start typing below to begin.',
  },
} as const

export function EmptyState({ type }: ChatEmptyStateProps) {
  const { icon, title, description } = variants[type]
  return <EmptyStateBase icon={icon} title={title} description={description} />
}
