import { useState, useCallback, useRef } from 'react'

interface PromptInputProps {
  prompts: string[]
  onPromptsChange: (prompts: string[]) => void
}

const DELIMITER = '---'

function parsePrompts(text: string): string[] {
  return text
    .split(DELIMITER)
    .map((p) => p.trim())
    .filter((p) => p.length > 0)
}

export function PromptInput({ prompts, onPromptsChange }: PromptInputProps) {
  const [rawText, setRawText] = useState(prompts.join(`\n${DELIMITER}\n`))
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleTextChange = useCallback(
    (text: string) => {
      setRawText(text)
      onPromptsChange(parsePrompts(text))
    },
    [onPromptsChange]
  )

  const handleImport = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (!file) return

      const text = await file.text()
      // If the file already uses --- delimiters, use as-is; otherwise treat each line as a prompt
      const hasDelimiters = text.includes(DELIMITER)
      const importedText = hasDelimiters ? text : text.split('\n').filter((l) => l.trim()).join(`\n${DELIMITER}\n`)
      handleTextChange(importedText)

      // Reset input so same file can be re-imported
      if (fileInputRef.current) fileInputRef.current.value = ''
    },
    [handleTextChange]
  )

  const promptCount = prompts.length

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Prompts
        </label>
        <div className="flex items-center gap-2">
          {promptCount > 0 && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary font-medium">
              {promptCount} prompt{promptCount !== 1 ? 's' : ''}
            </span>
          )}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="text-xs px-2 py-1 rounded bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"
          >
            Import TXT
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.md"
            onChange={handleImport}
            className="hidden"
          />
        </div>
      </div>
      <textarea
        value={rawText}
        onChange={(e) => handleTextChange(e.target.value)}
        placeholder={`Enter prompts separated by ${DELIMITER}\n\nExample:\nWhat is quantum computing?\n${DELIMITER}\nExplain machine learning\n${DELIMITER}\nDescribe neural networks`}
        rows={8}
        className="w-full px-3 py-2 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-600 resize-y focus:ring-2 focus:ring-primary focus:border-transparent"
      />
      <p className="text-xs text-gray-400 dark:text-gray-500">
        Separate multiple prompts with <code className="px-1 py-0.5 rounded bg-gray-100 dark:bg-gray-800">---</code> or import a text file
      </p>
    </div>
  )
}
