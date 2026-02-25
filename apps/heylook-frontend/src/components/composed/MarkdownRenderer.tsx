import { useState, useCallback, memo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface MarkdownRendererProps {
  content: string
}

export const MarkdownRenderer = memo(function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none text-gray-800 dark:text-gray-100">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          pre({ children }) {
            return <div className="relative group my-4">{children}</div>
          },
          code({ className, children }) {
            const match = /language-(\w+)/.exec(className || '')
            const isBlock = Boolean(match) || (typeof children === 'string' && children.includes('\n'))

            if (!isBlock) {
              return (
                <code className="bg-gray-200 dark:bg-gray-800 rounded px-1.5 py-0.5 text-sm font-mono text-primary">
                  {children}
                </code>
              )
            }

            const language = match ? match[1] : 'text'
            const codeStr = String(children).replace(/\n$/, '')

            return (
              <div className="rounded-lg overflow-hidden border border-gray-300 dark:border-gray-700 bg-gray-900">
                <div className="flex items-center justify-between px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-300 dark:border-gray-700">
                  <span className="text-xs font-mono text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    {language}
                  </span>
                  <CopyButton content={codeStr} />
                </div>
                <pre className="overflow-x-auto p-4 m-0 text-sm leading-relaxed">
                  <code className="text-gray-100 font-mono">{codeStr}</code>
                </pre>
              </div>
            )
          },
          table({ children }) {
            return (
              <div className="overflow-x-auto my-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700 m-0">
                  {children}
                </table>
              </div>
            )
          },
          thead({ children }) {
            return <thead className="bg-gray-50 dark:bg-gray-800/50">{children}</thead>
          },
          th({ children }) {
            return (
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                {children}
              </th>
            )
          },
          td({ children }) {
            return (
              <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300 border-t border-gray-200 dark:border-gray-700">
                {children}
              </td>
            )
          },
          a({ children, href }) {
            return (
              <a
                className="text-primary hover:text-primary-hover hover:underline transition-colors"
                href={href}
                target="_blank"
                rel="noopener noreferrer"
              >
                {children}
              </a>
            )
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
})

function CopyButton({ content }: { content: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }, [content])

  return (
    <button
      onClick={handleCopy}
      title="Copy code"
      className="p-1.5 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400 transition-colors"
      aria-label="Copy code"
    >
      {copied ? (
        <svg className="w-4 h-4 text-accent-green" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
      ) : (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
      )}
    </button>
  )
}
