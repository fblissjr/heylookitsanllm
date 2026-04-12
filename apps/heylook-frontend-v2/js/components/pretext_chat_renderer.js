import {
  CODE_BLOCK_PADDING_X,
  CODE_BLOCK_PADDING_Y,
  CODE_LINE_HEIGHT,
  materializeTemplateBlocks,
} from './pretext_chat_model.js'

/**
 * Project visible message rows into the canvas using virtualization.
 *
 * @param {object} frame - ConversationFrame from the model
 * @param {number} start - first visible message index
 * @param {number} end - one-past-last visible message index
 * @param {boolean} needsRelayout - whether all bubbles need re-rendering
 * @param {HTMLElement} canvas - the scrollable canvas element to append rows into
 * @param {Array} rows - mutable cache array of { bubble, row } objects (indexed by message)
 * @param {number} mountedStart - previous start index (from caller state)
 * @param {number} mountedEnd - previous end index (from caller state)
 * @returns {{ mountedStart: number, mountedEnd: number }} updated range for caller to store
 */
export function projectVisibleRows(
  frame,
  start,
  end,
  needsRelayout,
  canvas,
  rows,
  mountedStart,
  mountedEnd,
) {
  const previousStart = mountedStart
  const previousEnd = mountedEnd
  const overlapStart = Math.max(start, previousStart)
  const overlapEnd = Math.min(end, previousEnd)

  for (let index = previousStart; index < Math.min(previousEnd, start); index++) {
    const node = rows[index]
    if (node === undefined) continue
    node.row.remove()
    rows[index] = undefined
  }

  for (let index = Math.max(previousStart, end); index < previousEnd; index++) {
    const node = rows[index]
    if (node === undefined) continue
    node.row.remove()
    rows[index] = undefined
  }

  if (overlapStart >= overlapEnd) {
    for (let index = start; index < end; index++) {
      const cachedRow = prepareRow(frame.messages[index], index, needsRelayout, rows)
      projectMessageNode(cachedRow, frame.messages[index].frame, frame.messages[index].top)
      if (cachedRow.row.parentNode === null) canvas.append(cachedRow.row)
    }
  } else {
    let anchorRow = rows[overlapStart]?.row ?? null
    for (let index = overlapStart - 1; index >= start; index--) {
      const cachedRow = prepareRow(frame.messages[index], index, needsRelayout, rows)
      projectMessageNode(cachedRow, frame.messages[index].frame, frame.messages[index].top)
      if (anchorRow === null) {
        if (cachedRow.row.parentNode === null) canvas.append(cachedRow.row)
      } else if (cachedRow.row.parentNode !== canvas || cachedRow.row.nextSibling !== anchorRow) {
        canvas.insertBefore(cachedRow.row, anchorRow)
      }
      anchorRow = cachedRow.row
    }

    for (let index = overlapStart; index < overlapEnd; index++) {
      const cachedRow = prepareRow(frame.messages[index], index, needsRelayout, rows)
      projectMessageNode(cachedRow, frame.messages[index].frame, frame.messages[index].top)
    }

    for (let index = overlapEnd; index < end; index++) {
      const cachedRow = prepareRow(frame.messages[index], index, needsRelayout, rows)
      projectMessageNode(cachedRow, frame.messages[index].frame, frame.messages[index].top)
      if (cachedRow.row.parentNode === null) canvas.append(cachedRow.row)
    }
  }

  return { mountedStart: start, mountedEnd: end }
}

/**
 * Prepare (or reuse) a cached row for a message at the given index.
 *
 * @param {object} message - ChatMessageInstance
 * @param {number} index - message index
 * @param {boolean} needsRelayout - whether to re-render contents
 * @param {Array} rows - mutable cache array
 * @returns {{ bubble: HTMLDivElement, row: HTMLElement }}
 */
export function prepareRow(message, index, needsRelayout, rows) {
  let cachedRow = rows[index]
  if (cachedRow === undefined) {
    cachedRow = createMessageShell(message.frame.role)
    rows[index] = cachedRow
    renderMessageContents(cachedRow.bubble, message)
    return cachedRow
  }
  if (needsRelayout) renderMessageContents(cachedRow.bubble, message)
  return cachedRow
}

/**
 * Create the outer DOM shell for a chat message.
 *
 * @param {string} role - message role (e.g. "user", "assistant")
 * @returns {{ bubble: HTMLDivElement, row: HTMLElement }}
 */
export function createMessageShell(role) {
  const row = document.createElement('article')
  row.className = `msg msg--${role}`

  const bubble = document.createElement('div')
  bubble.className = 'msg-bubble'

  row.append(bubble)
  return { bubble, row }
}

/**
 * Render all blocks of a message into a bubble element.
 *
 * @param {HTMLDivElement} bubble
 * @param {object} message - ChatMessageInstance
 */
export function renderMessageContents(bubble, message) {
  const blocks = materializeTemplateBlocks(message)
  const fragment = document.createDocumentFragment()
  for (let index = 0; index < blocks.length; index++) {
    fragment.append(renderBlock(blocks[index], message.frame.contentInsetX))
  }
  bubble.replaceChildren(fragment)
}

/**
 * Position a cached message row using absolute coordinates.
 *
 * @param {{ bubble: HTMLDivElement, row: HTMLElement }} cachedRow
 * @param {object} frame - TemplateFrame
 * @param {number} top - vertical offset in px
 */
export function projectMessageNode(cachedRow, frame, top) {
  cachedRow.row.style.top = `${top}px`
  cachedRow.row.style.height = `${frame.totalHeight}px`
  cachedRow.bubble.style.width = `${frame.frameWidth}px`
  cachedRow.bubble.style.height = `${frame.bubbleHeight}px`
}

/**
 * Render a single block (inline, code, or rule).
 *
 * @param {object} block - BlockLayout
 * @param {number} contentInsetX
 * @returns {HTMLElement}
 */
export function renderBlock(block, contentInsetX) {
  switch (block.kind) {
    case 'inline':
      return renderInlineBlock(block, contentInsetX)
    case 'code':
      return renderCodeBlock(block, contentInsetX)
    case 'rule':
      return renderRuleBlock(block, contentInsetX)
  }
}

/**
 * Render an inline text block with line rows and fragments.
 *
 * @param {object} block - BlockLayout with kind 'inline'
 * @param {number} contentInsetX
 * @returns {HTMLElement}
 */
export function renderInlineBlock(block, contentInsetX) {
  const wrapper = createBlockShell(block, 'block block--inline', contentInsetX)

  for (let lineIndex = 0; lineIndex < block.lines.length; lineIndex++) {
    const line = block.lines[lineIndex]
    const row = document.createElement('div')
    row.className = 'line-row'
    row.style.height = `${block.lineHeight}px`
    row.style.left = `${contentInsetX + block.contentLeft}px`
    row.style.top = `${lineIndex * block.lineHeight}px`

    for (let fragmentIndex = 0; fragmentIndex < line.fragments.length; fragmentIndex++) {
      row.append(renderInlineFragment(line.fragments[fragmentIndex]))
    }
    wrapper.append(row)
  }

  return wrapper
}

/**
 * Render a fenced code block.
 *
 * @param {object} block - BlockLayout with kind 'code'
 * @param {number} contentInsetX
 * @returns {HTMLElement}
 */
export function renderCodeBlock(block, contentInsetX) {
  const wrapper = createBlockShell(block, 'block block--code-shell', contentInsetX)

  const codeBox = document.createElement('div')
  codeBox.className = 'code-box'
  codeBox.style.left = `${contentInsetX + block.contentLeft}px`
  codeBox.style.width = `${block.width}px`
  codeBox.style.height = `${block.height}px`

  for (let lineIndex = 0; lineIndex < block.lines.length; lineIndex++) {
    const line = block.lines[lineIndex]
    const row = document.createElement('div')
    row.className = 'code-line'
    row.style.left = `${CODE_BLOCK_PADDING_X}px`
    row.style.top = `${CODE_BLOCK_PADDING_Y + lineIndex * CODE_LINE_HEIGHT}px`
    row.textContent = line.text
    codeBox.append(row)
  }

  wrapper.append(codeBox)
  return wrapper
}

/**
 * Render a horizontal rule block.
 *
 * @param {object} block - BlockLayout with kind 'rule'
 * @param {number} contentInsetX
 * @returns {HTMLElement}
 */
export function renderRuleBlock(block, contentInsetX) {
  const wrapper = createBlockShell(block, 'block block--rule-shell', contentInsetX)
  const rule = document.createElement('div')
  rule.className = 'rule-line'
  rule.style.left = `${contentInsetX + block.contentLeft}px`
  rule.style.top = `${Math.floor(block.height / 2)}px`
  rule.style.width = `${block.width}px`
  wrapper.append(rule)
  return wrapper
}

/**
 * Create the base wrapper div for any block type.
 *
 * @param {object} block - BlockLayout
 * @param {string} className
 * @param {number} contentInsetX
 * @returns {HTMLDivElement}
 */
export function createBlockShell(block, className, contentInsetX) {
  const wrapper = document.createElement('div')
  wrapper.className = className
  wrapper.style.top = `${block.top}px`
  wrapper.style.height = `${block.height}px`

  appendRails(wrapper, block, contentInsetX)
  appendMarker(wrapper, block, contentInsetX)
  return wrapper
}

/**
 * Append blockquote rail divs to a block wrapper.
 *
 * @param {HTMLDivElement} wrapper
 * @param {object} block - BlockLayout
 * @param {number} contentInsetX
 */
export function appendRails(wrapper, block, contentInsetX) {
  for (let index = 0; index < block.quoteRailLefts.length; index++) {
    const rail = document.createElement('div')
    rail.className = 'quote-rail'
    rail.style.left = `${contentInsetX + block.quoteRailLefts[index]}px`
    wrapper.append(rail)
  }
}

/**
 * Append a list marker (bullet/number) to a block wrapper.
 *
 * @param {HTMLDivElement} wrapper
 * @param {object} block - BlockLayout
 * @param {number} contentInsetX
 */
export function appendMarker(wrapper, block, contentInsetX) {
  if (block.markerText === null || block.markerLeft === null || block.markerClassName === null) return

  const marker = document.createElement('span')
  marker.className = block.markerClassName
  marker.style.left = `${contentInsetX + block.markerLeft}px`
  marker.style.top = `${markerTop(block)}px`
  marker.textContent = block.markerText
  wrapper.append(marker)
}

/**
 * Compute the vertical offset for a list marker based on block kind.
 *
 * @param {object} block - BlockLayout
 * @returns {number}
 */
export function markerTop(block) {
  switch (block.kind) {
    case 'code':
      return CODE_BLOCK_PADDING_Y
    case 'inline':
      return Math.max(0, Math.round((block.lineHeight - 12) / 2))
    case 'rule':
      return 0
  }
}

/**
 * Render a single inline text fragment (span or anchor).
 *
 * @param {object} fragment - InlineFragmentLayout
 * @returns {HTMLElement}
 */
export function renderInlineFragment(fragment) {
  const node = fragment.href === null
    ? document.createElement('span')
    : document.createElement('a')

  node.className = fragment.className
  if (fragment.leadingGap > 0) {
    node.style.marginLeft = `${fragment.leadingGap}px`
  }
  node.textContent = fragment.text

  if (node instanceof HTMLAnchorElement && fragment.href !== null) {
    node.href = fragment.href
    node.target = '_blank'
    node.rel = 'noreferrer'
  }

  return node
}
