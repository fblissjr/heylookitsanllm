import { useCallback } from 'react'

interface BatchPromptInputProps {
  value: string
  onChange: (value: string) => void
}

/** Parse prompts separated by --- lines */
export function parsePrompts(raw: string): string[] {
  return raw
    .split(/^---$/m)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
}

export function BatchPromptInput({ value, onChange }: BatchPromptInputProps) {
  const prompts = parsePrompts(value)

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      onChange(e.target.value)
    },
    [onChange]
  )

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Prompts
        </label>
        <span className="text-xs text-gray-400">
          {prompts.length} prompt{prompts.length !== 1 ? 's' : ''}
        </span>
      </div>
      <textarea
        value={value}
        onChange={handleChange}
        rows={6}
        placeholder={'Enter prompts separated by ---\n\nFirst prompt here\n---\nSecond prompt here'}
        className="w-full px-3 py-2 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 placeholder-gray-400 resize-y"
      />
      <p className="text-[10px] text-gray-400">
        Separate multiple prompts with --- on its own line
      </p>
    </div>
  )
}
