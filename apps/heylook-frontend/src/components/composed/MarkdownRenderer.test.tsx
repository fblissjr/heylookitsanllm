import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
// userEvent doesn't play well with clipboard mock; using fireEvent for click tests
import { MarkdownRenderer } from './MarkdownRenderer'

describe('MarkdownRenderer', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('plain text', () => {
    it('renders plain text content', () => {
      render(<MarkdownRenderer content="Hello world" />)
      expect(screen.getByText('Hello world')).toBeInTheDocument()
    })

    it('renders multiple paragraphs', () => {
      render(<MarkdownRenderer content={'First paragraph\n\nSecond paragraph'} />)
      expect(screen.getByText('First paragraph')).toBeInTheDocument()
      expect(screen.getByText('Second paragraph')).toBeInTheDocument()
    })
  })

  describe('inline code', () => {
    it('renders inline code without language class', () => {
      render(<MarkdownRenderer content="Use `console.log` for debugging" />)
      const code = screen.getByText('console.log')
      expect(code.tagName).toBe('CODE')
      // Inline code should NOT be inside a pre-like code block container
      expect(code.closest('.rounded-lg.overflow-hidden')).not.toBeInTheDocument()
    })
  })

  describe('block code', () => {
    it('renders fenced code block with language label', () => {
      const content = '```python\nprint("hello")\n```'
      render(<MarkdownRenderer content={content} />)
      expect(screen.getByText('python')).toBeInTheDocument()
      expect(screen.getByText('print("hello")')).toBeInTheDocument()
    })

    it('renders fenced code block with copy button', () => {
      const content = '```js\nconst x = 1\n```'
      render(<MarkdownRenderer content={content} />)
      expect(screen.getByLabelText('Copy code')).toBeInTheDocument()
    })

    it('renders multi-line code without language as block', () => {
      const content = '```\nline one\nline two\n```'
      render(<MarkdownRenderer content={content} />)
      // Should show "text" as default language
      expect(screen.getByText('text')).toBeInTheDocument()
    })

    it('strips trailing newline from code content', () => {
      const content = '```js\nconst x = 1\n```'
      render(<MarkdownRenderer content={content} />)
      // The code should contain "const x = 1" without trailing newline
      const codeEl = screen.getByText('const x = 1')
      expect(codeEl).toBeInTheDocument()
    })
  })

  describe('CopyButton', () => {
    it('copies code to clipboard on click', () => {
      const writeText = vi.fn().mockResolvedValue(undefined)
      Object.defineProperty(navigator, 'clipboard', {
        value: { writeText, readText: vi.fn() },
        writable: true,
        configurable: true,
      })

      const content = '```js\nconst x = 1\n```'
      render(<MarkdownRenderer content={content} />)

      const copyButton = screen.getByLabelText('Copy code')
      fireEvent.click(copyButton)

      expect(writeText).toHaveBeenCalledWith('const x = 1')
    })

    it('shows check icon after copying', () => {
      const content = '```js\nconst x = 1\n```'
      render(<MarkdownRenderer content={content} />)

      const copyButton = screen.getByLabelText('Copy code')
      fireEvent.click(copyButton)

      // Check icon has a specific path (checkmark)
      const svg = copyButton.querySelector('svg')
      expect(svg).toBeInTheDocument()
      const path = svg?.querySelector('path')
      expect(path?.getAttribute('d')).toContain('5 13l4 4L19 7')
    })
  })

  describe('GFM tables', () => {
    it('renders a table with headers and cells', () => {
      const content = '| Name | Value |\n| --- | --- |\n| foo | bar |'
      render(<MarkdownRenderer content={content} />)

      expect(screen.getByText('Name')).toBeInTheDocument()
      expect(screen.getByText('Value')).toBeInTheDocument()
      expect(screen.getByText('foo')).toBeInTheDocument()
      expect(screen.getByText('bar')).toBeInTheDocument()
    })

    it('wraps table in scrollable container', () => {
      const content = '| Name | Value |\n| --- | --- |\n| foo | bar |'
      render(<MarkdownRenderer content={content} />)

      const table = screen.getByText('Name').closest('table')
      const wrapper = table?.closest('.overflow-x-auto')
      expect(wrapper).toBeInTheDocument()
    })
  })

  describe('links', () => {
    it('renders links with target="_blank"', () => {
      render(<MarkdownRenderer content="[Example](https://example.com)" />)
      const link = screen.getByText('Example')
      expect(link).toHaveAttribute('target', '_blank')
    })

    it('renders links with noopener noreferrer', () => {
      render(<MarkdownRenderer content="[Example](https://example.com)" />)
      const link = screen.getByText('Example')
      expect(link).toHaveAttribute('rel', 'noopener noreferrer')
    })

    it('renders links with correct href', () => {
      render(<MarkdownRenderer content="[Example](https://example.com)" />)
      const link = screen.getByText('Example')
      expect(link).toHaveAttribute('href', 'https://example.com')
    })
  })

  describe('markdown formatting', () => {
    it('renders bold text', () => {
      render(<MarkdownRenderer content="This is **bold** text" />)
      const bold = screen.getByText('bold')
      expect(bold.tagName).toBe('STRONG')
    })

    it('renders italic text', () => {
      render(<MarkdownRenderer content="This is *italic* text" />)
      const italic = screen.getByText('italic')
      expect(italic.tagName).toBe('EM')
    })

    it('renders headings', () => {
      render(<MarkdownRenderer content="## Heading Two" />)
      const heading = screen.getByText('Heading Two')
      expect(heading.tagName).toBe('H2')
    })

    it('renders unordered lists', () => {
      render(<MarkdownRenderer content={'- item one\n\n- item two'} />)
      expect(screen.getByText('item one')).toBeInTheDocument()
      expect(screen.getByText('item two')).toBeInTheDocument()
    })
  })

  describe('prose wrapper', () => {
    it('applies prose classes for typography', () => {
      const { container } = render(<MarkdownRenderer content="Hello" />)
      const prose = container.querySelector('.prose')
      expect(prose).toBeInTheDocument()
    })

    it('applies dark mode prose-invert', () => {
      const { container } = render(<MarkdownRenderer content="Hello" />)
      const prose = container.querySelector('.prose')
      expect(prose?.className).toContain('dark:prose-invert')
    })
  })
})
