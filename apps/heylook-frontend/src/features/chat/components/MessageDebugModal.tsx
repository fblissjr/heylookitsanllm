import { useState } from 'react'
import type { Message } from '../../../types/chat'
import type { Model } from '../../../types/api'

interface MessageDebugModalProps {
  isOpen: boolean
  onClose: () => void
  message: Message
  modelInfo?: Model
}

// Accordion section component
function Section({
  title,
  children,
  defaultOpen = false,
}: {
  title: string
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className="border-b border-gray-200 dark:border-gray-700 last:border-b-0">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between py-3 px-4 text-left hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
      >
        <span className="font-medium text-gray-900 dark:text-gray-100">{title}</span>
        <svg
          className={`w-4 h-4 text-gray-500 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {isOpen && (
        <div className="px-4 pb-4 text-sm">
          {children}
        </div>
      )}
    </div>
  )
}

// Key-value row for metrics
function MetricRow({ label, value, mono = false }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div className="flex justify-between py-1">
      <span className="text-gray-500 dark:text-gray-400">{label}</span>
      <span className={`text-gray-900 dark:text-gray-100 ${mono ? 'font-mono' : ''}`}>
        {value}
      </span>
    </div>
  )
}

// Format duration
function formatDuration(ms?: number): string {
  if (ms === undefined) return '-'
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

export function MessageDebugModal({ isOpen, onClose, message, modelInfo }: MessageDebugModalProps) {
  if (!isOpen) return null

  const perf = message.performance
  const genConfig = perf?.generationConfig

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-50"
        onClick={onClose}
      />

      {/* Desktop: centered modal */}
      <div className="hidden sm:block fixed inset-0 z-50 pointer-events-none">
        <div className="flex items-center justify-center min-h-screen p-4">
          <div
            className="bg-white dark:bg-surface-dark rounded-lg shadow-xl max-w-xl w-full max-h-[80vh] overflow-hidden pointer-events-auto"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Message Debug Info
              </h2>
              <button
                onClick={onClose}
                className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                aria-label="Close"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Content */}
            <div className="overflow-y-auto max-h-[calc(80vh-60px)]">
              {/* Model Info */}
              <Section title="Model Info" defaultOpen>
                <MetricRow label="Model ID" value={message.modelId || modelInfo?.id || '-'} mono />
                {modelInfo?.context_window && (
                  <MetricRow label="Context Window" value={modelInfo.context_window.toLocaleString()} mono />
                )}
                {modelInfo?.capabilities && modelInfo.capabilities.length > 0 && (
                  <MetricRow label="Capabilities" value={modelInfo.capabilities.join(', ')} />
                )}
                {modelInfo?.provider && (
                  <MetricRow label="Provider" value={modelInfo.provider} />
                )}
              </Section>

              {/* Generation Config */}
              {genConfig && (
                <Section title="Generation Config" defaultOpen>
                  {genConfig.temperature !== undefined && (
                    <MetricRow label="Temperature" value={genConfig.temperature} mono />
                  )}
                  {genConfig.top_p !== undefined && (
                    <MetricRow label="Top P" value={genConfig.top_p} mono />
                  )}
                  {genConfig.top_k !== undefined && (
                    <MetricRow label="Top K" value={genConfig.top_k} mono />
                  )}
                  {genConfig.min_p !== undefined && (
                    <MetricRow label="Min P" value={genConfig.min_p} mono />
                  )}
                  {genConfig.max_tokens !== undefined && (
                    <MetricRow label="Max Tokens" value={genConfig.max_tokens.toLocaleString()} mono />
                  )}
                  {genConfig.enable_thinking !== undefined && (
                    <MetricRow label="Thinking Mode" value={genConfig.enable_thinking ? 'Enabled' : 'Disabled'} />
                  )}
                </Section>
              )}

              {/* Performance Stats */}
              {perf && (
                <Section title="Performance Stats" defaultOpen>
                  <MetricRow label="Time to First Token" value={formatDuration(perf.timeToFirstToken)} mono />
                  <MetricRow
                    label="Tokens/Second"
                    value={perf.tokensPerSecond ? `${perf.tokensPerSecond.toFixed(1)} tok/s` : '-'}
                    mono
                  />
                  <MetricRow label="Total Duration" value={formatDuration(perf.totalDuration)} mono />
                  <div className="border-t border-gray-100 dark:border-gray-800 my-2" />
                  <MetricRow
                    label="Prompt Tokens"
                    value={perf.promptTokens?.toLocaleString() ?? '-'}
                    mono
                  />
                  <MetricRow
                    label="Completion Tokens"
                    value={perf.completionTokens?.toLocaleString() ?? '-'}
                    mono
                  />
                  {perf.thinkingTokens !== undefined && (
                    <MetricRow
                      label="Thinking Tokens"
                      value={perf.thinkingTokens.toLocaleString()}
                      mono
                    />
                  )}
                  {perf.contentTokens !== undefined && (
                    <MetricRow
                      label="Content Tokens"
                      value={perf.contentTokens.toLocaleString()}
                      mono
                    />
                  )}
                  {perf.thinkingDuration !== undefined && (
                    <MetricRow
                      label="Thinking Duration"
                      value={formatDuration(perf.thinkingDuration)}
                      mono
                    />
                  )}
                  {perf.cached !== undefined && (
                    <MetricRow
                      label="Prefix Cached"
                      value={
                        <span className={perf.cached ? 'text-green-600 dark:text-green-400' : ''}>
                          {perf.cached ? 'Yes' : 'No'}
                        </span>
                      }
                    />
                  )}
                </Section>
              )}

              {/* Stop Reason */}
              {perf?.stopReason && (
                <Section title="Stop Reason">
                  <div className="text-gray-900 dark:text-gray-100 font-mono">
                    {perf.stopReason}
                  </div>
                </Section>
              )}

              {/* Raw Stream Events */}
              {message.rawStream && message.rawStream.length > 0 && (
                <Section title={`Raw Stream (${message.rawStream.length} events)`}>
                  <div className="max-h-48 overflow-y-auto bg-gray-50 dark:bg-gray-900 rounded p-2 font-mono text-xs">
                    {message.rawStream.map((event, i) => (
                      <div key={i} className="text-gray-600 dark:text-gray-400 break-all">
                        {event}
                      </div>
                    ))}
                  </div>
                </Section>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Mobile: bottom sheet */}
      <div className="sm:hidden fixed inset-x-0 bottom-0 z-50">
        <div
          className="bg-white dark:bg-surface-dark rounded-t-xl shadow-xl max-h-[80vh] overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Drag handle */}
          <div className="flex justify-center py-2">
            <div className="w-10 h-1 bg-gray-300 dark:bg-gray-600 rounded-full" />
          </div>

          {/* Header */}
          <div className="flex items-center justify-between px-4 pb-2">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Message Debug
            </h2>
            <button
              onClick={onClose}
              className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500"
              aria-label="Close"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Content - reuse same sections */}
          <div className="overflow-y-auto max-h-[calc(80vh-60px)] border-t border-gray-200 dark:border-gray-700">
            {/* Model Info */}
            <Section title="Model Info" defaultOpen>
              <MetricRow label="Model ID" value={message.modelId || modelInfo?.id || '-'} mono />
              {modelInfo?.context_window && (
                <MetricRow label="Context Window" value={modelInfo.context_window.toLocaleString()} mono />
              )}
            </Section>

            {/* Generation Config */}
            {genConfig && (
              <Section title="Generation Config">
                {genConfig.temperature !== undefined && (
                  <MetricRow label="Temperature" value={genConfig.temperature} mono />
                )}
                {genConfig.top_p !== undefined && (
                  <MetricRow label="Top P" value={genConfig.top_p} mono />
                )}
                {genConfig.enable_thinking !== undefined && (
                  <MetricRow label="Thinking" value={genConfig.enable_thinking ? 'On' : 'Off'} />
                )}
              </Section>
            )}

            {/* Performance Stats */}
            {perf && (
              <Section title="Performance" defaultOpen>
                <MetricRow label="TTFT" value={formatDuration(perf.timeToFirstToken)} mono />
                <MetricRow
                  label="Speed"
                  value={perf.tokensPerSecond ? `${perf.tokensPerSecond.toFixed(1)} tok/s` : '-'}
                  mono
                />
                <MetricRow
                  label="Tokens"
                  value={perf.completionTokens?.toLocaleString() ?? '-'}
                  mono
                />
              </Section>
            )}

            {/* Stop Reason */}
            {perf?.stopReason && (
              <Section title="Stop Reason">
                <span className="font-mono">{perf.stopReason}</span>
              </Section>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
