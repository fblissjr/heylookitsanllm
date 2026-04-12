import { marked } from '../vendor/marked.esm.js'

import {
  layoutWithLines,
  measureLineStats,
  measureNaturalWidth,
  prepareWithSegments,
} from '../vendor/pretext/layout.js'
import {
  materializeRichInlineLineRange,
  measureRichInlineStats,
  prepareRichInline,
  walkRichInlineLineRanges,
} from '../vendor/pretext/rich-inline.js'

export const MIN_CHAT_WIDTH = 360
export const DEFAULT_CHAT_WIDTH = 640
export const MAX_CHAT_WIDTH = 860
export const CHAT_VIEWPORT_HEIGHT = 560
export const OCCLUSION_BANNER_HEIGHT = 61
export const PAGE_MARGIN = 28
export const MESSAGE_SIDE_PADDING = 22

const COMPACT_OCCLUSION_BANNER_HEIGHT = 43
const COMPACT_OCCLUSION_VIEWPORT_HEIGHT = 460
const CHAT_TOP_PADDING_OFFSET = 14
const CHAT_BOTTOM_PADDING_OFFSET = 10
const MESSAGE_GAP = 12
const BUBBLE_MAX_RATIO = 0.78
export const BUBBLE_PADDING_X = 16
const BUBBLE_PADDING_Y = 10
const BODY_LINE_HEIGHT = 22
const HEADING_ONE_LINE_HEIGHT = 28
const HEADING_TWO_LINE_HEIGHT = 25
const HARD_BREAK_GAP = 4
const BLOCK_GAP = 12
const RICH_BLOCK_GAP = 2
const LIST_ITEM_GAP = 4
const LIST_NESTING_INDENT = 18
const BLOCKQUOTE_INDENT = 18
const LIST_MARKER_GAP = 10
export const CODE_LINE_HEIGHT = 18
export const CODE_BLOCK_PADDING_X = 12
export const CODE_BLOCK_PADDING_Y = 8
const RULE_HEIGHT = 18
const RAIL_OFFSET = 5
const SANS_FAMILY = '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
const SERIF_FAMILY = '"Iowan Old Style", Georgia, "Times New Roman", serif'
const MONO_FAMILY = '"SF Mono", ui-monospace, Menlo, Monaco, monospace'
const INLINE_CODE_FONT = `600 12px ${MONO_FAMILY}`
const INLINE_CODE_EXTRA_WIDTH = 12
const IMAGE_FONT = `700 11px ${SANS_FAMILY}`
const IMAGE_EXTRA_WIDTH = 14
const MARKER_FONT = `600 11px ${MONO_FAMILY}`

const EMPTY_MARK_STATE = {
  bold: false,
  italic: false,
  strike: false,
  href: null,
}

const markerWidthCache = new Map()

export function prepareMessageTemplate(markdown, role) {
  return {
    blocks: parseMarkdownBlocks(markdown),
    role,
  }
}

export function getMaxChatWidth(viewportWidth) {
  return Math.max(240, Math.min(MAX_CHAT_WIDTH, viewportWidth - PAGE_MARGIN * 2))
}

export function buildConversationFrame(
  templates,
  chatWidth,
  occlusionBannerHeight = OCCLUSION_BANNER_HEIGHT,
) {
  const messageCount = templates.length
  const laneWidth = Math.max(120, chatWidth - MESSAGE_SIDE_PADDING * 2)
  const userFrameWidth = Math.min(laneWidth, Math.max(240, Math.floor(chatWidth * BUBBLE_MAX_RATIO)))
  const assistantFrameWidth = laneWidth
  const messages = new Array(messageCount)
  const chatTopPadding = occlusionBannerHeight + CHAT_TOP_PADDING_OFFSET
  const chatBottomPadding = occlusionBannerHeight + CHAT_BOTTOM_PADDING_OFFSET

  let y = chatTopPadding
  for (let ordinal = 0; ordinal < messageCount; ordinal++) {
    const template = templates[ordinal]
    const contentInsetX = template.role === 'assistant' ? 0 : BUBBLE_PADDING_X
    const frameWidth = template.role === 'assistant' ? assistantFrameWidth : userFrameWidth
    const contentWidth = Math.max(120, frameWidth - contentInsetX * 2)
    const messageFrame = layoutTemplateFrame(template, frameWidth, contentWidth, contentInsetX)
    const top = y
    const bottom = top + messageFrame.totalHeight

    messages[ordinal] = {
      bottom,
      frame: messageFrame,
      prepared: template,
      top,
    }
    y = bottom
    y += MESSAGE_GAP
  }

  const totalHeight =
    messages.length === 0
      ? chatTopPadding + chatBottomPadding
      : y - MESSAGE_GAP + chatBottomPadding

  return {
    chatWidth,
    messages,
    occlusionBannerHeight,
    totalHeight,
  }
}

export function getOcclusionBannerHeight(viewportHeight) {
  return viewportHeight <= COMPACT_OCCLUSION_VIEWPORT_HEIGHT
    ? COMPACT_OCCLUSION_BANNER_HEIGHT
    : OCCLUSION_BANNER_HEIGHT
}

export function findVisibleRange(
  frame,
  scrollTop,
  viewportHeight,
  topOcclusionHeight,
  bottomOcclusionHeight,
) {
  if (frame.messages.length === 0) return { start: 0, end: 0 }

  const minY = Math.max(0, scrollTop + topOcclusionHeight)
  const maxY = Math.max(minY, scrollTop + viewportHeight - bottomOcclusionHeight)
  let low = 0
  let high = frame.messages.length

  while (low < high) {
    const mid = (low + high) >> 1
    if (frame.messages[mid].bottom > minY) {
      high = mid
    } else {
      low = mid + 1
    }
  }
  const start = low

  low = start
  high = frame.messages.length
  while (low < high) {
    const mid = (low + high) >> 1
    if (frame.messages[mid].top >= maxY) {
      high = mid
    } else {
      low = mid + 1
    }
  }

  return { start, end: low }
}

export function formatPixelCount(value) {
  return `${Math.round(value).toLocaleString()}px`
}

function parseMarkdownBlocks(markdown) {
  const tokens = marked.lexer(markdown, { gfm: true })
  return parseBlockTokens(tokens, { listDepth: 0, quoteDepth: 0 })
}

function parseBlockTokens(tokens, ctx) {
  const blocks = []

  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index]

    switch (token.type) {
      case 'space':
      case 'def': {
        continue
      }

      case 'paragraph': {
        appendBlockGroup(blocks, buildInlineBlocks(token.tokens ?? [], 'body', ctx), BLOCK_GAP)
        continue
      }

      case 'heading': {
        appendBlockGroup(
          blocks,
          buildInlineBlocks(token.tokens ?? [], headingVariant(token.depth), ctx),
          BLOCK_GAP + 4,
        )
        continue
      }

      case 'code': {
        appendBlockGroup(blocks, [buildCodeBlock(token.text, ctx)], RICH_BLOCK_GAP)
        continue
      }

      case 'list': {
        appendBlockGroup(blocks, buildListBlocks(token, ctx), BLOCK_GAP)
        continue
      }

      case 'blockquote': {
        appendBlockGroup(
          blocks,
          parseBlockTokens(token.tokens ?? [], {
            listDepth: ctx.listDepth,
            quoteDepth: ctx.quoteDepth + 1,
          }),
          RICH_BLOCK_GAP,
        )
        continue
      }

      case 'hr': {
        appendBlockGroup(blocks, [buildRuleBlock(ctx)], BLOCK_GAP + 2)
        continue
      }

      case 'table': {
        appendBlockGroup(blocks, [buildCodeBlock(formatTable(token), ctx)], RICH_BLOCK_GAP)
        continue
      }

      case 'html': {
        const htmlText = token.text.trim().length > 0 ? token.text : token.raw
        const isPre = 'pre' in token && token.pre === true
        if (token.block || isPre) {
          appendBlockGroup(blocks, [buildCodeBlock(htmlText, ctx)], RICH_BLOCK_GAP)
        } else {
          appendBlockGroup(blocks, buildPlainTextBlocks(htmlText, 'body', ctx), BLOCK_GAP)
        }
        continue
      }

      case 'text': {
        if (Array.isArray(token.tokens) && token.tokens.length > 0) {
          appendBlockGroup(blocks, buildInlineBlocks(token.tokens, 'body', ctx), BLOCK_GAP)
        } else {
          appendBlockGroup(blocks, buildPlainTextBlocks(token.text, 'body', ctx), BLOCK_GAP)
        }
        continue
      }

      default: {
        const fallbackText = fallbackTextForToken(token)
        if (fallbackText.length > 0) {
          appendBlockGroup(blocks, buildPlainTextBlocks(fallbackText, 'body', ctx), BLOCK_GAP)
        }
      }
    }
  }

  return blocks
}

function buildListBlocks(token, ctx) {
  const blocks = []
  const itemCtx = {
    listDepth: ctx.listDepth + 1,
    quoteDepth: ctx.quoteDepth,
  }

  for (let index = 0; index < token.items.length; index++) {
    const item = token.items[index]
    let itemBlocks = parseBlockTokens(item.tokens, itemCtx)
    if (itemBlocks.length === 0) {
      itemBlocks = buildPlainTextBlocks(item.text, 'body', itemCtx)
    }

    decorateListItemBlocks(itemBlocks, resolveListMarkerText(token, item, index), resolveListMarkerClassName(token, item))
    appendBlockGroup(blocks, itemBlocks, LIST_ITEM_GAP)
  }

  return blocks
}

function decorateListItemBlocks(
  blocks,
  markerText,
  markerClassName,
) {
  if (blocks.length === 0) return

  const markerArea = measureMarkerWidth(markerText) + LIST_MARKER_GAP
  for (let index = 0; index < blocks.length; index++) {
    blocks[index] = shiftBlock(blocks[index], markerArea)
  }

  const firstBlock = blocks[0]
  blocks[0] = {
    ...firstBlock,
    markerClassName,
    markerLeft: firstBlock.contentLeft - markerArea,
    markerText,
  }
}

function buildPlainTextBlocks(
  text,
  variant,
  ctx,
) {
  const piece = createTextPiece(text, EMPTY_MARK_STATE, variant)
  if (piece === null) return []
  return buildPreparedInlineBlocks([[piece]], variant, ctx)
}

function buildInlineBlocks(
  tokens,
  variant,
  ctx,
) {
  const lines = collectInlinePieceLines(tokens, variant)
  return buildPreparedInlineBlocks(lines, variant, ctx)
}

function buildPreparedInlineBlocks(
  lines,
  variant,
  ctx,
) {
  const blocks = []

  for (let index = 0; index < lines.length; index++) {
    const block = buildPreparedInlineBlock(lines[index], variant, ctx)
    if (block === null) continue
    blocks.push({
      ...block,
      marginTop: blocks.length === 0 ? 0 : HARD_BREAK_GAP,
    })
  }

  return blocks
}

function buildPreparedInlineBlock(
  pieces,
  variant,
  ctx,
) {
  if (pieces.length === 0) return null

  return {
    ...createBlockBase(ctx),
    classNames: pieces.map(piece => piece.className),
    flow: prepareRichInline(pieces.map(piece => ({
      text: piece.text,
      font: piece.font,
      break: piece.breakMode,
      extraWidth: piece.extraWidth,
    }))),
    hrefs: pieces.map(piece => piece.href),
    kind: 'inline',
    lineHeight: lineHeightForVariant(variant),
  }
}

function buildCodeBlock(text, ctx) {
  return {
    ...createBlockBase(ctx),
    kind: 'code',
    lineHeight: CODE_LINE_HEIGHT,
    prepared: prepareWithSegments(stripSingleTrailingNewline(text), `500 12px ${MONO_FAMILY}`, {
      whiteSpace: 'pre-wrap',
    }),
  }
}

function buildRuleBlock(ctx) {
  return {
    ...createBlockBase(ctx),
    height: RULE_HEIGHT,
    kind: 'rule',
  }
}

function createBlockBase(ctx) {
  const listIndent = Math.max(0, ctx.listDepth - 1) * LIST_NESTING_INDENT
  const contentLeft = listIndent + ctx.quoteDepth * BLOCKQUOTE_INDENT
  const quoteRailLefts = []

  for (let depth = 0; depth < ctx.quoteDepth; depth++) {
    quoteRailLefts.push(listIndent + depth * BLOCKQUOTE_INDENT + RAIL_OFFSET)
  }

  return {
    contentLeft,
    marginTop: 0,
    markerClassName: null,
    markerLeft: null,
    markerText: null,
    quoteRailLefts,
  }
}

function collectInlinePieceLines(
  tokens,
  variant,
) {
  const lines = [[]]

  function currentLine() {
    return lines[lines.length - 1]
  }

  function pushLineBreak() {
    lines.push([])
  }

  function pushPiece(piece) {
    if (piece === null) return
    const line = currentLine()
    const previous = line[line.length - 1]
    if (previous !== undefined && canMergeInlinePieces(previous, piece)) {
      previous.text += piece.text
      return
    }
    line.push(piece)
  }

  function walk(tokenList, marks) {
    for (let index = 0; index < tokenList.length; index++) {
      const token = tokenList[index]

      switch (token.type) {
        case 'text': {
          if (Array.isArray(token.tokens) && token.tokens.length > 0) {
            walk(token.tokens, marks)
          } else {
            pushPiece(createTextPiece(token.text, marks, variant))
          }
          continue
        }

        case 'escape': {
          pushPiece(createTextPiece(token.text, marks, variant))
          continue
        }

        case 'strong': {
          walk(token.tokens ?? [], { ...marks, bold: true })
          continue
        }

        case 'em': {
          walk(token.tokens ?? [], { ...marks, italic: true })
          continue
        }

        case 'del': {
          walk(token.tokens ?? [], { ...marks, strike: true })
          continue
        }

        case 'codespan': {
          pushPiece(createCodePiece(token.text))
          continue
        }

        case 'link': {
          walk(token.tokens ?? [], { ...marks, href: token.href })
          continue
        }

        case 'image': {
          pushPiece(createImagePiece(token.text.length > 0 ? token.text : token.href))
          continue
        }

        case 'br': {
          pushLineBreak()
          continue
        }

        case 'checkbox': {
          pushPiece(createTextPiece(token.checked ? '[x] ' : '[ ] ', marks, variant))
          continue
        }

        case 'html': {
          pushPiece(createTextPiece(token.text, marks, variant))
          continue
        }

        default: {
          const fallback = fallbackTextForToken(token)
          if (fallback.length > 0) {
            pushPiece(createTextPiece(fallback, marks, variant))
          }
        }
      }
    }
  }

  walk(tokens, EMPTY_MARK_STATE)

  while (lines.length > 0 && lines[lines.length - 1].length === 0) {
    lines.pop()
  }

  return lines
}

function createTextPiece(
  text,
  marks,
  variant,
) {
  if (text.length === 0) return null

  return {
    breakMode: 'normal',
    className: resolveTextClassName(variant, marks),
    extraWidth: 0,
    font: resolveTextFont(variant, marks),
    href: marks.href,
    text,
  }
}

function createCodePiece(text) {
  if (text.length === 0) return null

  return {
    breakMode: 'normal',
    className: 'frag frag--code',
    extraWidth: INLINE_CODE_EXTRA_WIDTH,
    font: INLINE_CODE_FONT,
    href: null,
    text,
  }
}

function createImagePiece(text) {
  return {
    breakMode: 'never',
    className: 'frag frag--chip',
    extraWidth: IMAGE_EXTRA_WIDTH,
    font: IMAGE_FONT,
    href: null,
    text: text.length > 0 ? text : 'image',
  }
}

function canMergeInlinePieces(a, b) {
  return (
    a.breakMode === b.breakMode &&
    a.className === b.className &&
    a.extraWidth === b.extraWidth &&
    a.font === b.font &&
    a.href === b.href
  )
}

function resolveTextFont(variant, marks) {
  const italicPrefix = marks.italic ? 'italic ' : ''

  switch (variant) {
    case 'heading-1': {
      const weight = marks.bold ? 800 : 700
      return `${italicPrefix}${weight} 20px ${SERIF_FAMILY}`
    }

    case 'heading-2': {
      const weight = marks.bold ? 800 : 700
      return `${italicPrefix}${weight} 17px ${SERIF_FAMILY}`
    }

    case 'body': {
      const weight = marks.bold ? 700 : marks.href === null ? 400 : 500
      return `${italicPrefix}${weight} 14px ${SANS_FAMILY}`
    }
  }
}

function resolveTextClassName(variant, marks) {
  let className = 'frag'

  switch (variant) {
    case 'heading-1':
      className += ' frag--heading-1'
      break
    case 'heading-2':
      className += ' frag--heading-2'
      break
    case 'body':
      className += ' frag--body'
      break
  }

  if (marks.href !== null) className += ' is-link'
  if (marks.bold) className += ' is-strong'
  if (marks.italic) className += ' is-em'
  if (marks.strike) className += ' is-del'
  return className
}

function headingVariant(depth) {
  if (depth <= 1) return 'heading-1'
  if (depth === 2) return 'heading-2'
  return 'body'
}

function lineHeightForVariant(variant) {
  switch (variant) {
    case 'heading-1':
      return HEADING_ONE_LINE_HEIGHT
    case 'heading-2':
      return HEADING_TWO_LINE_HEIGHT
    case 'body':
      return BODY_LINE_HEIGHT
  }
}

function appendBlockGroup(
  target,
  group,
  firstMargin,
) {
  if (group.length === 0) return

  for (let index = 0; index < group.length; index++) {
    const block = group[index]
    target.push({
      ...block,
      marginTop: index === 0 ? (target.length === 0 ? 0 : firstMargin) : block.marginTop,
    })
  }
}

function shiftBlock(block, delta) {
  return {
    ...block,
    contentLeft: block.contentLeft + delta,
  }
}

function resolveListMarkerText(
  list,
  item,
  index,
) {
  if (item.task) return item.checked ? '\u2611' : '\u2610'
  if (list.ordered) {
    const start = typeof list.start === 'number' ? list.start : 1
    return `${start + index}.`
  }
  return '\u2022'
}

function resolveListMarkerClassName(
  list,
  item,
) {
  if (item.task) return 'block-marker block-marker--task'
  return list.ordered
    ? 'block-marker block-marker--ordered'
    : 'block-marker block-marker--bullet'
}

function measureMarkerWidth(text) {
  const cached = markerWidthCache.get(text)
  if (cached !== undefined) return cached

  const width = measureNaturalWidth(prepareWithSegments(text, MARKER_FONT))
  markerWidthCache.set(text, width)
  return width
}

function fallbackTextForToken(token) {
  if ('text' in token && typeof token.text === 'string') return token.text
  return token.raw ?? ''
}

function formatTable(token) {
  const header = token.header.map(cell => inlineTokensToPlainText(cell.tokens)).join(' | ')
  const divider = token.header.map(() => '---').join(' | ')
  const rows = token.rows.map(row => row.map(cell => inlineTokensToPlainText(cell.tokens)).join(' | '))
  return [header, divider, ...rows].join('\n')
}

function inlineTokensToPlainText(tokens) {
  let text = ''
  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index]
    switch (token.type) {
      case 'strong':
      case 'em':
      case 'del':
      case 'link':
        text += inlineTokensToPlainText(token.tokens ?? [])
        break
      case 'codespan':
      case 'escape':
      case 'text':
      case 'html':
        text += token.text
        break
      case 'br':
        text += '\n'
        break
      case 'image':
        text += token.text
        break
      default:
        text += fallbackTextForToken(token)
    }
  }
  return text
}

function stripSingleTrailingNewline(text) {
  return text.endsWith('\n') ? text.slice(0, -1) : text
}

function layoutTemplateFrame(
  template,
  maxFrameWidth,
  maxContentWidth,
  contentInsetX,
) {
  let y = BUBBLE_PADDING_Y
  const blocks = []
  let usedContentWidth = 0

  for (let index = 0; index < template.blocks.length; index++) {
    const block = template.blocks[index]
    y += block.marginTop
    const blockFrame = layoutBlockFrame(block, maxContentWidth, y)
    blocks.push(blockFrame)
    y += blockFrame.height
    usedContentWidth = Math.max(usedContentWidth, getUsedBlockWidth(blockFrame))
  }

  const bubbleHeight = y + BUBBLE_PADDING_Y
  const frameWidth = template.role === 'assistant'
    ? maxFrameWidth
    : Math.min(maxFrameWidth, contentInsetX * 2 + Math.max(1, usedContentWidth))
  return {
    blocks,
    bubbleHeight,
    contentInsetX,
    frameWidth,
    layoutContentWidth: maxContentWidth,
    role: template.role,
    totalHeight: bubbleHeight,
  }
}

function layoutBlockFrame(
  block,
  contentWidth,
  top,
) {
  switch (block.kind) {
    case 'inline': {
      const lineWidth = Math.max(1, contentWidth - block.contentLeft)
      const { lineCount, maxLineWidth } = measureRichInlineStats(block.flow, lineWidth)
      return {
        contentLeft: block.contentLeft,
        height: lineCount * block.lineHeight,
        kind: 'inline',
        lineHeight: block.lineHeight,
        markerClassName: block.markerClassName,
        markerLeft: block.markerLeft,
        markerText: block.markerText,
        quoteRailLefts: block.quoteRailLefts,
        top,
        usedWidth: maxLineWidth,
      }
    }

    case 'code': {
      const boxWidth = Math.max(1, contentWidth - block.contentLeft)
      const innerWidth = Math.max(1, boxWidth - CODE_BLOCK_PADDING_X * 2)
      const { lineCount, maxLineWidth } = measureLineStats(block.prepared, innerWidth)
      return {
        contentLeft: block.contentLeft,
        height: lineCount * block.lineHeight + CODE_BLOCK_PADDING_Y * 2,
        kind: 'code',
        lineHeight: block.lineHeight,
        markerClassName: block.markerClassName,
        markerLeft: block.markerLeft,
        markerText: block.markerText,
        quoteRailLefts: block.quoteRailLefts,
        top,
        width: maxLineWidth + CODE_BLOCK_PADDING_X * 2,
      }
    }

    case 'rule': {
      return {
        contentLeft: block.contentLeft,
        height: block.height,
        kind: 'rule',
        markerClassName: block.markerClassName,
        markerLeft: block.markerLeft,
        markerText: block.markerText,
        quoteRailLefts: block.quoteRailLefts,
        top,
        width: Math.max(1, contentWidth - block.contentLeft),
      }
    }
  }
}

function getUsedBlockWidth(block) {
  switch (block.kind) {
    case 'inline':
      return block.contentLeft + block.usedWidth
    case 'code':
      return block.contentLeft + block.width
    case 'rule':
      return block.contentLeft + block.width
  }
}

export function materializeTemplateBlocks(message) {
  return message.prepared.blocks.map((block, index) =>
    materializeBlockLayout(block, message.frame.blocks[index], message.frame.layoutContentWidth),
  )
}

function materializeBlockLayout(
  block,
  frame,
  contentWidth,
) {
  switch (frame.kind) {
    case 'inline': {
      if (block.kind !== 'inline') throw new Error('Inline block/frame mismatch')
      const lineWidth = Math.max(1, contentWidth - frame.contentLeft)
      const lines = []
      walkRichInlineLineRanges(block.flow, lineWidth, range => {
        const line = materializeRichInlineLineRange(block.flow, range)
        lines.push({
          fragments: line.fragments.map(fragment => ({
            className: block.classNames[fragment.itemIndex],
            href: block.hrefs[fragment.itemIndex] ?? null,
            leadingGap: fragment.gapBefore,
            text: fragment.text,
          })),
          width: line.width,
        })
      })

      return {
        contentLeft: frame.contentLeft,
        height: frame.height,
        kind: 'inline',
        lineHeight: frame.lineHeight,
        lines,
        markerClassName: frame.markerClassName,
        markerLeft: frame.markerLeft,
        markerText: frame.markerText,
        quoteRailLefts: frame.quoteRailLefts,
        top: frame.top,
        usedWidth: frame.usedWidth,
      }
    }

    case 'code': {
      if (block.kind !== 'code') throw new Error('Code block/frame mismatch')
      const boxWidth = Math.max(1, contentWidth - frame.contentLeft)
      const innerWidth = Math.max(1, boxWidth - CODE_BLOCK_PADDING_X * 2)
      const layout = layoutWithLines(block.prepared, innerWidth, frame.lineHeight)
      return {
        contentLeft: frame.contentLeft,
        height: frame.height,
        kind: 'code',
        lines: layout.lines,
        markerClassName: frame.markerClassName,
        markerLeft: frame.markerLeft,
        markerText: frame.markerText,
        quoteRailLefts: frame.quoteRailLefts,
        top: frame.top,
        usedWidth: frame.width,
        width: frame.width,
      }
    }

    case 'rule': {
      if (block.kind !== 'rule') throw new Error('Rule block/frame mismatch')
      return {
        contentLeft: frame.contentLeft,
        height: frame.height,
        kind: 'rule',
        markerClassName: frame.markerClassName,
        markerLeft: frame.markerLeft,
        markerText: frame.markerText,
        quoteRailLefts: frame.quoteRailLefts,
        top: frame.top,
        width: frame.width,
      }
    }
  }
}
